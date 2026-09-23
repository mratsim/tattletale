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

##
## - Entries are consumer-side, a `metal:` block wraps the grid-driven proc with concrete
##   static (Dk, Dv, TileR), one call-site line per static binding set
## - The engine's monomorphization key erases static bindings, calls sharing a call-site line collapse into one body
## - The decode mega kernel composes the tile core `gdnDecodeStepTileAt` inline instead
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

proc gdnDecodeStepTileAt*[T](
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[T],             # (B·Hv, Dv) element dtype core output
    k: ptr UncheckedArray[T],             # (B·Hk, Dk) element dtype, post-l2norm
    q: ptr UncheckedArray[T],             # (B·Hk, Dk) element dtype, post-l2norm
    v: ptr UncheckedArray[T],             # (B·Hv, Dv) element dtype
    g: ptr UncheckedArray[float32],       # (B·Hv,) f32 log decay
    beta: ptr UncheckedArray[T],          # (B·Hv,) element dtype, one per value head
    Hv, Hk, hkRatio: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## One (bh, TileR-row) state tile of the gated delta-rule decode step at the caller's coordinates
  ##
  ## - the element dtype is an unconstrained compile-time generic
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
  let hk = ((bh mod Hv) div hkRatio) + ((bh div Hv) * Hk)
  let headLin = bh * Dv * Dk
  let yLin = bh * Dv
  let kLin = hk * Dk

  let glState = state.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dk, 1))
  let glK = k.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glQ = q.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glV = v.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, 0))

  var s: rt_l(float32, TileR, Dk)
  var kT: rt_l(T, TileR, Dk)
  var qT: rt_l(T, TileR, Dk)
  var vT: rt_l(T, TileR, 8)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))
  kT.loadTile(glK, (kLin, 0, 0, 0))
  qT.loadTile(glQ, (kLin, 0, 0, 0))
  vT.loadTile(glV, (yLin, 0, dvBlock, 0))

  let dec = exp2(g[bh] * Log2e)
  s.mul(s, dec)

  # kv_mem[row] = Σ_dk decayed[row][dk]·k[dk] over the decayed state, the k
  # tile broadcasts one key vector over the tile rows, one row sum per lane
  var k32: rt_l(float32, TileR, Dk)
  k32.widen(kT)
  var prod: rt_l(float32, TileR, Dk)
  prod.mul(s, k32)
  var kvVec: rv(float32, TileR, Dk)
  kvVec.row_sum(prod)
  let kvMem = kvVec.rowScalar()

  let v32 = vT.laneScalar().float32
  let delta = beta[bh].float32 * (v32 - kvMem)

  s.addScaled(k32, delta)

  let scale = rsqrt(float32(Dk))
  var q32: rt_l(float32, TileR, Dk)
  q32.widen(qT)
  var oProd: rt_l(float32, TileR, Dk)
  oProd.map2(s, q32, x * (y * scale))
  var oVec: rv(float32, TileR, Dk)
  oVec.row_sum(oProd)
  let oVal = oVec.rowScalar()

  let rowIn = laneRowOf(APPLE_8x8x8_F32)
  let colIn = laneColOf(APPLE_8x8x8_F32)
  if colIn == 0:
      y[yLin + dvBlock * 8 + int32(rowIn)] = roundToRne[T](oVal)
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
  ## Grid-driven form of `gdnDecodeStepTileAt`, the caller's `metal:` entry wraps this proc.
  ## - grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  gdnDecodeStepTileAt(state, y, k, q, v, g, beta, Hv, Hk, hkRatio,
    dvBlock, bh, Dk, Dv, TileR)
