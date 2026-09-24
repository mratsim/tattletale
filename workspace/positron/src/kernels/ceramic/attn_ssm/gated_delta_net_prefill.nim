# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ───────────────  GDN prefill (chunked scan over T tokens, one launch)  ───────────────

## Prefill (T > 1) of the gated delta-rule recurrence (arXiv:2412.06464), one launch
## walking the tokens in chunks of ChunkC over the in-block cumulative log decay cumulogdecay:
##
## | term            | formula                                                                                    |
## | --------------- | ------------------------------------------------------------------------------------------ |
## | pairdecay(t, s) | exp2((cumulogdecay[t] − cumulogdecay[s])·log2e)                                            |
## | u_t             | β_t·(v_t − exp(cumulogdecay[t])·(S_carry·k_t)) − β_t·Σ_{s<t} pairdecay(t, s)·(k_t·k_s)·u_s |
## | y_t             | exp(cumulogdecay[t])·(S_carry·q̃_t) + Σ_{s≤t} pairdecay(t, s)·(q̃_t·k_s)·u_s               |
## | carry           | S = exp(cumulogdecay[end])·S_carry + Σ_s pairdecay(end, s)·k_s ⊗ u_s                       |
##
## | contract      | value                                                                                                                                       |
## | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
## | state math    | all fp32 and never rounds, one 8-row state tile per threadgroup, register-resident across the chunk walk                                    |
## | q, k          | (B·Hk, T, Dk) element dtype, already l2-normalized (l2norm stays host-side)                                                                 |
## | v, beta       | (B·Hv, T, Dv) and (B·Hv, T) element dtype, g is (B·Hv, T) f32 log-decay                                                                     |
## | y             | (B·Hv, T, Dv) element dtype, one round-to-nearest-even per element                                                                          |
## | element dtype | one compile-time element type (`gdnPrefillChunkScan`'s `El` generic)                                                                        |
## | head mapping  | value head bh reads key head `(bh mod Hv) div hkRatio + (bh div Hv)·Hk`, hkRatio = Hv div Hk                                                |
## | chunk axis    | tokens are walked in chunks of ChunkC, the u solve sequential in t inside a chunk, chunks sequential on the register state                  |
## | decay / q̃    | exp2(g·log2e), log2e is the shared `math_consts.Log2e`, Dk^-0.5 folded into q in f32 (rsqrt-multiply form, Metal has no exp device builtin) |
##
## | contract       | value                                                                                                                                                                   |
## | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | g precondition | finite and ≤ 0 (−exp(A_log)·softplus ≤ 0 by construction), cumulogdecay inherits the sign, the kernel applies no clamp, a violating g explodes the persistent f32 state |

#
## Register-tile naming convention, shared by the gdn and kda kernels:
##
## - `<x>T`, the element-dtype register tile of operand x, loaded from memory
## - `<x>32`, the fp32 register tile of the same operand, an fp32-storage
##   operand loads straight into its `32` form, an element-dtype operand
##   widens its `T` form into the `32` form
##
## | provenance | source                                                                                                                                                            |
## | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | schedule   | the fp64 chunked WY/UT reassociation: per-chunk cumulogdecay, the A-matrix u solve in token order, one decayed carry read plus the u outer products |

##
## Implementation shape:
##   per chunk:  t 0 → t 1 → … → t ChunkC-1   u solve, then the carry update
##   chunks:     0 → 1 → … → N-1              each carry feeds the next chunk
## - each lane computes its own state row's scalars (the solve, y, the u contributions),
##   so the u vectors live in a per-lane local array, no inter-threadgroup data movement needed
## - the k·k and q̃·k dot products run as one broadcast-tile pass per (t, s) pair,
##   every lane reads the same row-identical row sum, lanes agree bit-exactly
## - the pair (t, s) passes recompute the key tiles per pair, chunks of 64 stay inside
##   the register budget (per-lane state and u arrays are the only residents, no (ChunkC, Dk) working tile)
##
## Entries are consumer-side:
## - a `metal` block wraps the grid-driven proc with concrete static (Dk, Dv, ChunkC),
##   one call-site line per static binding set
## - the engine's monomorphization key erases static bindings, calls sharing a call-site line collapse into one body
##
## Binding and state ABI:
## - hosts binding through the Metal engine's no-copy path get in-place state
##   updates and visible y writes from one run
##
## - any other binding copies and the y writes are lost
## - `state` is the engine's output buffer, `y` is written by the kernel
##
## - the state's ABI is (B·Hv, Dv, Dk) f32, dense row-major, head-major over
##   (sequence, value head), one unrounded fp32 tile per (bh, Dv-row-block)
## - the f32 state buffer persists across steps and launches with no in-kernel reset,
##   the host owns the layout and the lifetime
## - rebinding the state to a 16-bit dtype or a strided view silently corrupts the recurrence

from ../math_consts import Log2e
import workspace/crucible
import workspace/ceramic

from ../tile_widen import widen

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Core tile procs (inline-tile property) ──────────────────────────

# tiles-allow gdnPrefillChunkScanAt carries the row-bounded y-store walk, it needs the bounded
# tile-IO store primitive (row-guarded store over register tiles)
proc gdnPrefillChunkScanAt*[El](
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[El],           # (B·Hv, T, Dv) element dtype
    k: ptr UncheckedArray[El],           # (B·Hk, T, Dk) element dtype, post-l2norm
    q: ptr UncheckedArray[El],           # (B·Hk, T, Dk) element dtype, post-l2norm
    v: ptr UncheckedArray[El],           # (B·Hv, T, Dv) element dtype
    g: ptr UncheckedArray[float32],       # (B·Hv, T) f32 log decay
    beta: ptr UncheckedArray[El],        # (B·Hv, T) element dtype
    Hv, Hk, hkRatio, T: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## One (bh, TileR-row) state tile of the chunked GDN prefill scan, the whole
  ## token sequence walked over the register state at the caller's coordinates
  ## (the element dtype an unconstrained compile-time generic):
  ##
  ##   t 0 → t 1 → … → t ChunkC-1   u solve per t, sums over s ascending
  ##
  ## Parameters, pointers naming their dtypes, shapes bound at the call:
  ##
  ## | parameter             | shape, dtype, layout                                                                                                                                                                         | producer                                                                                                             | unit                |
  ## | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------- |
  ## | state                 | (B·Hv, Dv, Dk) f32, dense row-major, head-major over (sequence, value head), the persistent recurrence state written back in place (S_carry entering the chunk, the carry update leaving it) | the previous chunk's launch writes it, this kernel reads and rewrites it in place, the host owns layout and lifetime | f32, never rounds   |
  ## | y                     | (B·Hv, T, Dv) El, row-major, the whole token sequence's output, one round-to-nearest-even per element, stored row-bounded (out-of-range rows drop)                                           | this kernel                                                                                                          | El                  |
  ## | k                     | (B·Hk, T, Dk) El, row-major, the keys, post-l2norm (the l2norm stays host-side)                                                                                                              | host-computed                                                                                                        | El                  |
  ## | q                     | (B·Hk, T, Dk) El, row-major, the queries, post-l2norm, host-computed, q̃ scales by the device-side f32 1/√Dk, never rounded to the element dtype                                             | host-computed (the scale device-side)                                                                                | El                  |
  ## | v                     | (B·Hv, T, Dv) El, row-major, the values                                                                                                                                                      | host-computed                                                                                                        | El                  |
  ## | g                     | (B·Hv, T) f32 log decay, one decay per VALUE head, the GDN scalar decay, not a per-key-channel matrix                                                                                        | host-computed                                                                                                        | natural-log decay exponent |
  ## | beta                  | (B·Hv, T) El, one per value head and token, the delta weighting                                                                                                                              | host-computed                                                                                                        | dimensionless       |
  ## | Hv, Hk                | value and key head counts, host-derived from the model config                                                                                                                                | host-computed                                                                                                        | heads               |
  ## | hkRatio               | Hv div Hk, the GQA head ratio                                                                                                                                                                | host-computed                                                                                                        | dimensionless       |
  ## | T                     | the token count, host-derived from the sequence length                                                                                                                                       | host-computed                                                                                                        | tokens              |
  ## | Dk, Dv, TileR, ChunkC | static tile geometry (head dim, value dim, the row block height, the chunk length), the per-lane cumulogdecay/u local arrays are sized by ChunkC (≤ 64)                                      | compile-time                                                                                                         | elements            |
  ## | dvBlock, bh           | the Dv div TileR row-block index and the (sequence, value head) flat head index                                                                                                              | device-computed grid coordinates                                                                                     | elements            |
  ##
  ## Contract:
  ## - all state arithmetic is fp32, the state never rounds before the final in-place store
  ## - the state entering a chunk is S_carry, untouched until the chunk-end carry update
  ## - the u solve is sequential in t, the sums over s walk s ascending
  ##
  ## - the element-dtype loads and the one element-dtype y round per element are
  ##   the only dtype-dependent steps, the tile walk is dtype-mechanical
  ##
  ## - precondition, Hk > 0, Hv an exact multiple of Hk and hkRatio = Hv div Hk
  ## - the g precondition, finite and ≤ 0 by construction, no kernel clamp,
  ##   a violating g explodes the f32 state
  ##
  ## - grid-driven wrapper, receiving the threadgroup coordinates from the grid
  ## - generic only over the element dtype and the static shape, every (Dk, Dv, ChunkC)
  ##   binding needs its own call-site line
  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store covers one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
    doAssert ChunkC <= 64, "the per-lane cumulogdecay/u local arrays are sized by ChunkC"

  let hk = (bh mod Hv) div (Hv div Hk) + (bh div Hv) * Hk
  let headLin = bh * Dv * Dk
  let seqLin = bh * T * Dv
  let kHeadLin = hk * T * Dk

  let glState = state.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dk, 1))
  let glK = k.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glQ = q.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))

  var s: rt_l(float32, TileR, Dk)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))

  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (int(thread_index_in_threadgroup), 0)).toIntVal()
  let rowIn = cell mod APPLE_8x8x8_F32.getM()
  let colIn = cell div APPLE_8x8x8_F32.getM()
  let scale = rsqrt(float32(Dk))

  var cumulogdecay: array[ChunkC, float32]   # in-block cumulative log decay, identical on every lane
  var uLoc: array[ChunkC, float32]   # this lane's state row's solve vector u_s[rowIn]

  var c0 = int32(0)
  while c0 < T:
    let cLen = min(ChunkC, int(T - c0))
    # In-block cumulative log decay.
    var gsum = 0'f32
    for i in 0 ..< cLen:
      gsum += g[bh * T + c0 + int32(i)]
      cumulogdecay[i] = gsum

    for t in 0 ..< cLen:
      let gt = c0 + int32(t)
      let kLinT = kHeadLin + gt * Dk
      var kT: rt_l(El, TileR, Dk)
      kT.loadTile(glK, (kLinT, 0, 0, 0))
      var k32: rt_l(float32, TileR, Dk)
      k32.widen(kT)

      # G_t = exp(cumulogdecay[t])·(S_carry·k_t), the decayed carry read against the key
      var kProd: rt_l(float32, TileR, Dk)
      kProd.mul(s, k32)
      var kVec: rv(float32, TileR, Dk)
      kVec.row_sum(kProd)
      let decayT = exp2(cumulogdecay[t] * Log2e)
      let gRead = decayT * kVec.rowScalar()

      let bt = beta[bh * T + gt].float32
      let v32 = v[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)].float32

      # u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} pairdecay(t, s)·(k_t·k_s)·u_s, solved in token order
      var uacc = 0'f32
      for sIdx in 0 ..< t:
        var ksT: rt_l(El, TileR, Dk)
        ksT.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var ks32: rt_l(float32, TileR, Dk)
        ks32.widen(ksT)
        var kkProd: rt_l(float32, TileR, Dk)
        kkProd.mul(k32, ks32)
        var kkVec: rv(float32, TileR, Dk)
        kkVec.row_sum(kkProd)
        let pdts = exp2((cumulogdecay[t] - cumulogdecay[sIdx]) * Log2e)
        uacc += pdts * kkVec.rowScalar() * uLoc[sIdx]
      let ut = bt * (v32 - gRead) - bt * uacc
      uLoc[t] = ut

      # y_t = exp(cumulogdecay[t])·(S_carry·q̃_t) + Σ_{s≤t} pairdecay(t, s)·(q̃_t·k_s)·u_s
      var qT: rt_l(El, TileR, Dk)
      qT.loadTile(glQ, (kLinT, 0, 0, 0))
      var q32: rt_l(float32, TileR, Dk)
      q32.widen(qT)
      q32.mul(q32, scale)

      var qProd: rt_l(float32, TileR, Dk)
      qProd.mul(s, q32)
      var qVec: rv(float32, TileR, Dk)
      qVec.row_sum(qProd)
      var yVal = decayT * qVec.rowScalar()
      for sIdx in 0 .. t:
        var ksT: rt_l(El, TileR, Dk)
        ksT.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var ks32: rt_l(float32, TileR, Dk)
        ks32.widen(ksT)
        var qkProd: rt_l(float32, TileR, Dk)
        qkProd.mul(q32, ks32)
        var qkVec: rv(float32, TileR, Dk)
        qkVec.row_sum(qkProd)
        let pdts = exp2((cumulogdecay[t] - cumulogdecay[sIdx]) * Log2e)
        yVal += pdts * qkVec.rowScalar() * uLoc[sIdx]

      if colIn == 0:
          y[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)] = roundToNearestEven[El](yVal)

    # Carry out of the chunk:
    # S = exp(cumulogdecay[end])·S_carry + Σ_s pairdecay(end, s)·k_s ⊗ u_s
    let decayEnd = exp2(cumulogdecay[cLen - 1] * Log2e)
    s.mul(s, decayEnd)
    for sIdx in 0 ..< cLen:
      let ws = exp2((cumulogdecay[cLen - 1] - cumulogdecay[sIdx]) * Log2e) * uLoc[sIdx]
      var ksT: rt_l(El, TileR, Dk)
      ksT.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
      var ks32: rt_l(float32, TileR, Dk)
      ks32.widen(ksT)
      s.addScaled(ks32, ws)
    c0 += int32(cLen)

  glState.storeTile(s, (headLin, 0, dvBlock, 0))


proc gdnPrefillChunkScan*[El](
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[El],           # (B·Hv, T, Dv) element dtype
    k: ptr UncheckedArray[El],           # (B·Hk, T, Dk) element dtype, post-l2norm
    q: ptr UncheckedArray[El],           # (B·Hk, T, Dk) element dtype, post-l2norm
    v: ptr UncheckedArray[El],           # (B·Hv, T, Dv) element dtype
    g: ptr UncheckedArray[float32],       # (B·Hv, T) f32 log decay
    beta: ptr UncheckedArray[El],        # (B·Hv, T) element dtype
    Hv, Hk, hkRatio, T: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## Grid-driven form of `gdnPrefillChunkScanAt`, the caller's `metal:` entry wraps this proc.
  ## Grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8.
  ##
  ## Parameters, pointers naming their dtypes, shapes bound at the call, grid coordinates arriving from the grid:
  ##
  ## | parameter             | shape, dtype, layout                                                                                                                                                                         | producer                                                                                                             | unit                |
  ## | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------- |
  ## | state                 | (B·Hv, Dv, Dk) f32, dense row-major, head-major over (sequence, value head), the persistent recurrence state written back in place (S_carry entering the chunk, the carry update leaving it) | the previous chunk's launch writes it, this kernel reads and rewrites it in place, the host owns layout and lifetime | f32, never rounds   |
  ## | y                     | (B·Hv, T, Dv) El, row-major, the whole token sequence's output, one round-to-nearest-even per element, stored row-bounded (out-of-range rows drop)                                           | this kernel                                                                                                          | El                  |
  ## | k                     | (B·Hk, T, Dk) El, row-major, the keys, post-l2norm (the l2norm stays host-side)                                                                                                              | host-computed                                                                                                        | El                  |
  ## | q                     | (B·Hk, T, Dk) El, row-major, the queries, post-l2norm, host-computed, q̃ scales by the device-side f32 1/√Dk, never rounded to the element dtype                                             | host-computed (the scale device-side)                                                                                | El                  |
  ## | v                     | (B·Hv, T, Dv) El, row-major, the values                                                                                                                                                      | host-computed                                                                                                        | El                  |
  ## | g                     | (B·Hv, T) f32 log decay, one decay per VALUE head, the GDN scalar decay, not a per-key-channel matrix                                                                                        | host-computed                                                                                                        | natural-log decay exponent |
  ## | beta                  | (B·Hv, T) El, one per value head and token, the delta weighting                                                                                                                              | host-computed                                                                                                        | dimensionless       |
  ## | Hv, Hk                | value and key head counts, host-derived from the model config                                                                                                                                | host-computed                                                                                                        | heads               |
  ## | hkRatio               | Hv div Hk, the GQA head ratio                                                                                                                                                                | host-computed                                                                                                        | dimensionless       |
  ## | T                     | the token count, host-derived from the sequence length                                                                                                                                       | host-computed                                                                                                        | tokens              |
  ## | Dk, Dv, TileR, ChunkC | static tile geometry (head dim, value dim, the row block height, the chunk length), the per-lane cumulogdecay/u local arrays are sized by ChunkC (≤ 64)                                      | compile-time                                                                                                         | elements            |
  ## | dvBlock, bh           | the Dv div TileR row-block index and the (sequence, value head) flat head index                                                                                                              | device-computed grid coordinates                                                                                     | elements            |
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  gdnPrefillChunkScanAt(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, T,
    dvBlock, bh, Dk, Dv, TileR, ChunkC)