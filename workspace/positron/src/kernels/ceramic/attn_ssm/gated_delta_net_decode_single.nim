# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ──────────────  GDN decode_single (one gated-delta-rule step per threadgroup)  ───────────────────────────────

## One decode step (T = 1) of the gated delta-rule recurrence (arXiv:2412.06464):
##
##   S ← S·exp2(g·log2e) + k ⊗ (β·(v − (S·exp2(g·log2e))·k))    y ← S'·(q·Dk^-0.5)
##
  ## | contract       | value                                                                                                                                                  |
  ## | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
  ## | state math     | all fp32 and never rounds, one 8-row state tile per threadgroup, no inter-threadgroup sync                                                             |
  ## | q, k           | (B·Hk, Dk) element dtype, already l2-normalized (l2norm stays host-side)                                                                               |
  ## | v, beta        | (B·Hv, Dv) and (B·Hv,) element dtype, g is (B·Hv,) f32 log-decay                                                                                       |
  ## | y              | (B·Hv, Dv) element dtype, one round-to-nearest-even                                                                                                    |
  ## | element dtype  | compile-time element type of one body (`gdnDecodeStepTile`'s `T` generic, inferred from the pointers)                                                  |
  ## | head mapping   | value head bh reads key head `(bh mod Hv) div hkRatio + (bh div Hv)·Hk`, hkRatio = Hv div Hk                                                           |
  ## | batch          | the head axis, one launch at grid (Dv div TileR, B·Hv) over per-sequence stacked inputs is the batched decode step                                     |
  ## | decay / q̃     | exp2(g·log2e), the log2e factor is the shared `math_consts.Log2e`, Dk^-0.5 folded into q in f32 (rsqrt-multiply form, Metal has no exp device builtin) |
  ## | g precondition | finite and ≤ 0 by construction, no kernel clamp, a violating g explodes the persistent f32 state                                                       |

#
## Register-tile naming convention, shared by the gdn and kda kernels:
##
## - `<x>T`, the element-dtype register tile of operand x, loaded from memory
## - `<x>32`, the fp32 register tile of the same operand, an fp32-storage
##   operand loads straight into its `32` form, an element-dtype operand
##   widens its `T` form into the `32` form
##
## - Entries are consumer-side, a `metal:` block wraps the grid-driven proc with concrete
##   static (Dk, Dv, TileR), one call-site line per static binding set
## - The engine's monomorphization key erases static bindings, calls sharing a call-site line collapse into one body
## - The decode mega kernel composes the tile core `gatedDeltaDecodeStepTileAt` inline instead,
##   the kda module's grid-driven entry forwards into it with decayChannel = true and qDivQScale = true
##
## Binding and state ABI:
## - hosts binding through the Metal engine's no-copy path get in-place state
##   updates and visible y writes from one run
## - any other binding copies and the y writes are lost
## - `state` is the engine's output buffer, `y` is written by the kernel
##
## - the state's ABI is (B·Hv, Dv, Dk) f32, dense row-major, head-major over
##   (sequence, value head), one unrounded fp32 tile per (bh, Dv-row-block)
## - the f32 state buffer persists across steps and launches with no in-kernel reset,
##   the host owns the layout and the lifetime
## - rebinding the state to a 16-bit dtype or a strided view silently
##   corrupts the recurrence

from ../math_consts import Log2e
import workspace/crucible
import workspace/ceramic
import ../tile_widen

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Module-local device helpers ─────────────────────────────────────

# ─── Core tile procs (inline-tile property) ──────────────────────────

# tiles-allow gatedDeltaDecodeStepTileAt carries the row-bounded y-store walk, it needs the bounded
# tile-IO store primitive (row-guarded store over register tiles)
proc gatedDeltaDecodeStepTileAt*[T; U; B](
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[T],             # (B·Hv, Dv) element dtype core output
    k: ptr UncheckedArray[U],             # (B·Hk, Dk) k/q storage dtype U, gdn element dtype, kda f32, post-l2norm
    q: ptr UncheckedArray[U],             # (B·Hk, Dk) k/q storage dtype U, gdn element dtype, kda f32, post-l2norm
    v: ptr UncheckedArray[T],             # (B·Hv, Dv) element dtype
    g: ptr UncheckedArray[float32],       # gdn (B·Hv,) one f32 log decay per value head, kda (B·Hk, Dk) one per key channel
    beta: ptr UncheckedArray[B],          # (B·Hv,) one delta weighting per value head, gdn element dtype, kda f32
    qScale: float32,                      # √Dk, the host's f64 sqrt cast to f32, dead under the rsqrt q̃ form
    Hv, Hk, hkRatio: int32,
    dvBlock, bh: int32,
    decayChannel: static bool = false,    # the decay form, false the gdn scalar g broadcast over the state tile, true the kda per-channel g tile
    qDivQScale: static bool = false,      # the q̃ form, false multiply by the device-side rsqrt(Dk), true divide by the runtime qScale
    Dk, Dv, TileR: static int) {.device.} =
  ## One (bh, TileR-row) state tile of the shared gated-delta decode step (gdn and kda forms)
  ## at the caller's coordinates
  ##
  ## - `decayChannel = false`, the gdn form, S ← S·exp2(g·log2e), one scalar log-decay per value head
  ## - `decayChannel = true`, the kda form, S ← S·Diag(exp2(g·log2e)), one log-decay per key channel
  ## - the q̃ forms, `qDivQScale = false` q̃ = q·rsqrt(Dk) device-side in f32, `true` q̃ = q/qScale with the runtime qScale, the host's f64 √Dk
  ##
  ## | parameter     | shape, dtype, layout                                                                                                                                             | producer                                                                                                                                                         | unit                |
  ## | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
  ## | state         | (B·Hv, Dv, Dk) f32, dense row-major, head-major over (sequence, value head), the persistent recurrence state, written back in place                              | the previous step's launch writes it, this kernel reads and rewrites it in place, the buffer persists with no in-kernel reset, the host owns layout and lifetime | f32, never rounds   |
  ## | y             | (B·Hv, Dv) El, row-major, the step's output, one round-to-nearest-even per element                                                                               | this kernel                                                                                                                                                      | El                  |
  ## | k             | row-major, the key vector of the key head this tile serves, post-l2norm (the l2norm stays host-side), gdn element dtype, kda f32                                 | host-computed                                                                                                                                                    | U                   |
  ## | q             | row-major, the query vector of the same key head, post-l2norm, host-computed, the q̃ fold happens device-side in f32 per the q̃ form                             | host-computed (the fold device-side)                                                                                                                             | U                   |
  ## | v             | (B·Hv, Dv) El, row-major, the value vector of the value head this tile serves                                                                                    | host-computed                                                                                                                                                    | El                  |
  ## | g             | f32 log decay, gdn (B·Hv,) per value head, kda (B·Hk, Dk) per key channel, finite and ≤ 0 by construction, no kernel clamp, a violating g explodes the f32 state | host-computed                                                                                                                                                    | log2-decay exponent |
  ## | beta          | (B·Hv,) one per value head, the delta weighting, gdn element dtype, kda f32                                                                                      | host-computed                                                                                                                                                    | dimensionless       |
  ## | qScale        | √Dk, the host's f64 sqrt cast to f32, dead under the rsqrt q̃ form                                                                                               | host-computed (the divide device-side)                                                                                                                           | dimensionless       |
  ## | Hv, Hk        | value and key head counts, host-derived from the model config                                                                                                    | host-computed                                                                                                                                                    | heads               |
  ## | hkRatio       | Hv div Hk, the GQA head ratio                                                                                                                                    | host-computed                                                                                                                                                    | dimensionless       |
  ## | Dk, Dv, TileR | static tile geometry (head dim, value dim, the row block height)                                                                                                 | compile-time                                                                                                                                                     | elements            |
  ## | decayChannel  | the decay form, false the gdn scalar g, true the kda per-channel g tile                                                                                          | compile-time                                                                                                                                                     | form                |
  ## | qDivQScale    | the q̃ form, false the gdn rsqrt-multiply, true the kda divide by the runtime qScale                                                                             | compile-time                                                                                                                                                     | form                |
  ## | dvBlock, bh   | the Dv div TileR row-block index and the (sequence, value head) flat head index                                                                                  | device-computed grid coordinates                                                                                                                                 | elements            |
  ##
  ## - the element dtype is an unconstrained compile-time generic, U the k/q storage dtype,
  ##   B the beta storage dtype, both inferred from the call
  ## - every fork keeps each form's exact op sequence and operand order, no shared
  ##   expression re-associates and no op crosses a rounding point
  ## - the element-dtype loads and the one element-dtype y rounding are the only
  ##   dtype-dependent steps, the tile walk is dtype-mechanical
  ##
  ##   S ← S·exp2(g·log2e) + k ⊗ (β·(v − (S·exp2(g·log2e))·k))    y ← S'·(q·Dk^-0.5)
  ##
  ## Contract:
  ## - all state arithmetic is fp32, the state never rounds
  ## - precondition, Hk > 0, Hv an exact multiple of Hk and hkRatio = Hv div Hk
  ##
  ## - the decay applies before the kv read (the recurrence's step order)
  ## - the state stores in place, f32, no rounding
  ##
  ##   kv_mem[row] = Σ_dk decayed[row][dk]·k[dk]
  ##   delta[row] = β·(v[row] − kv_mem[row])
  ##   y[row] = Σ_dk S'[row][dk]·(q[dk]·Dk^-0.5), one element-dtype round
  ##
  ## Y write goes to the lanes whose fragment column is 0, one lane per state row.
  ##
  ## - `bh` is the (sequence, value head) row block, `dvBlock` the Dv/TileR row block
  ## - grid-driven wrapper, receiving the threadgroup coordinates from the grid
  ## - generic only over the element dtype and the static shape, every (Dk, Dv, TileR)
  ##   binding needs its own call-site line
  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store covers one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
  let hk = (bh mod Hv) div (Hv div Hk) + (bh div Hv) * Hk
  let headLin = bh * Dv * Dk
  let yLin = bh * Dv
  let kLin = hk * Dk

  let glState = state.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dk, 1))
  let glK = k.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glQ = q.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glV = v.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, 0))

  var s: rt_l(float32, TileR, Dk)
  var k32: rt_l(float32, TileR, Dk)
  var q32: rt_l(float32, TileR, Dk)
  var kT: rt_l(T, TileR, Dk)
  var qT: rt_l(T, TileR, Dk)
  var vT: rt_l(T, TileR, 8)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))
  when U is float32:
    # kda f32 storage, the 32-forms load straight in, no widening,
    # the T forms stay dead under this binding
    k32.loadTile(glK, (kLin, 0, 0, 0))
    q32.loadTile(glQ, (kLin, 0, 0, 0))
  else:
    # gdn element-dtype storage, the T tiles load now, the 32 forms widen
    # from them at the gdn widen sites (k after the decay, q after the delta)
    kT.loadTile(glK, (kLin, 0, 0, 0))
    qT.loadTile(glQ, (kLin, 0, 0, 0))
  vT.loadTile(glV, (yLin, 0, dvBlock, 0))

  when decayChannel:
    # kda per-channel decay, BEFORE the kv read (the recurrence's step order).
    # The g tile broadcasts one key head's log-decay row over the tile rows,
    # the exp2 form (see the module doc), one tile mul into the state.
    var gT: rt_l(float32, TileR, Dk)
    let glG = g.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
    gT.loadTile(glG, (kLin, 0, 0, 0))
    gT.mul(gT, Log2e)
    gT.exp2(gT)
    s.mul(s, gT)
  else:
    # gdn scalar decay, one log-decay per value head
    let dec = exp2(g[bh] * Log2e)
    s.mul(s, dec)

  when U is not float32:
    k32.widen(kT)

  # kv_mem[row] = Σ_dk decayed[row][dk]·k[dk] over the decayed state, the k
  # tile broadcasts one key vector over the tile rows, one row sum per lane
  var prod: rt_l(float32, TileR, Dk)
  prod.mul(s, k32)
  var kvVec: rv(float32, TileR, Dk)
  kvVec.row_sum(prod)
  let kvMem = kvVec.rowScalar()

  let v32 = vT.laneScalar().float32
  let delta = beta[bh].float32 * (v32 - kvMem)

  s.addScaled(k32, delta)
  when U is not float32:
    q32.widen(qT)

  var oProd: rt_l(float32, TileR, Dk)
  when qDivQScale:
    oProd.map2(s, q32, x * (y / qScale))
  else:
    let scale = rsqrt(float32(Dk))
    oProd.map2(s, q32, x * (y * scale))
  var oVec: rv(float32, TileR, Dk)
  oVec.row_sum(oProd)
  let oVal = oVec.rowScalar()

  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (int(thread_index_in_threadgroup), 0)).toIntVal()
  let rowIn = cell mod APPLE_8x8x8_F32.getM()
  let colIn = cell div APPLE_8x8x8_F32.getM()
  if colIn == 0:
      y[yLin + dvBlock * APPLE_8x8x8_F32.getN() + int32(rowIn)] = roundToNearestEven[T](oVal)
  glState.storeTile(s, (headLin, 0, dvBlock, 0))

proc gdnDecodeStepTile*[T](
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[T],             # (B·Hv, Dv) element dtype core output
    k: ptr UncheckedArray[T],             # (B·Hk, Dk) element dtype, post-l2norm
    q: ptr UncheckedArray[T],             # (B·Hk, Dk) element dtype, post-l2norm
    v: ptr UncheckedArray[T],             # (B·Hv, Dv) element dtype
    g: ptr UncheckedArray[float32],       # (B·Hv,) f32 log decay
    beta: ptr UncheckedArray[T],          # (B·Hv,) element dtype, one per value head
    Hv, Hk, hkRatio: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## Grid-driven form of `gatedDeltaDecodeStepTileAt` in its gdn binding, the caller's `metal:` entry wraps this proc.
  ## Grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8, gdn form, scalar g decay, rsqrt-multiply q̃.
  ##
  ## | parameter     | shape, dtype, layout                                                                                                                                                      | producer                                                                                                                                                         | unit                |
  ## | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
  ## | state         | (B·Hv, Dv, Dk) f32, dense row-major, head-major over (sequence, value head), the persistent recurrence state, written back in place                                       | the previous step's launch writes it, this kernel reads and rewrites it in place, the buffer persists with no in-kernel reset, the host owns layout and lifetime | f32, never rounds   |
  ## | y             | (B·Hv, Dv) El, row-major, the step's output, one round-to-nearest-even per element                                                                                        | this kernel                                                                                                                                                      | El                  |
  ## | k             | (B·Hk, Dk) El, row-major, the key vector of the key head this tile serves, post-l2norm (the l2norm stays host-side)                                                       | host-computed                                                                                                                                                    | El                  |
  ## | q             | (B·Hk, Dk) El, row-major, the query vector of the same key head, post-l2norm, host-computed, the Dk^-0.5 fold happens device-side in f32                                  | host-computed (the fold device-side)                                                                                                                             | El                  |
  ## | v             | (B·Hv, Dv) El, row-major, the value vector of the value head this tile serves                                                                                             | host-computed                                                                                                                                                    | El                  |
  ## | g             | (B·Hv,) f32 log decay, one per value head, unit the log2-decay exponent, finite and ≤ 0 by construction, no kernel clamp, a violating g explodes the persistent f32 state | host-computed                                                                                                                                                    | log2-decay exponent |
  ## | beta          | (B·Hv,) El, one per value head, the delta weighting                                                                                                                       | host-computed                                                                                                                                                    | dimensionless       |
  ## | Hv, Hk        | value and key head counts, host-derived from the model config                                                                                                             | host-computed                                                                                                                                                    | heads               |
  ## | hkRatio       | Hv div Hk, the GQA head ratio                                                                                                                                             | host-computed                                                                                                                                                    | dimensionless       |
  ## | Dk, Dv, TileR | static tile geometry (head dim, value dim, the row block height)                                                                                                          | compile-time                                                                                                                                                     | elements            |
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  gatedDeltaDecodeStepTileAt(state, y, k, q, v, g, beta, 0'f32, Hv, Hk, hkRatio,
    dvBlock, bh, false, false, Dk, Dv, TileR)
