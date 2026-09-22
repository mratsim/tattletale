# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ───────────────  GDN prefill (chunked scan over T tokens, one launch)  ───────────────

## Prefill (T > 1) of the gated delta-rule recurrence (arXiv:2412.06464), one launch
## walking the tokens in chunks of ChunkC over the in-block cumulative log decay cumg:
##
## | term            | formula                                                                            |
## | --------------- | ---------------------------------------------------------------------------------- |
## | pairdecay(t, s) | exp2((cumg[t] − cumg[s])·log2e)                                                    |
## | u_t             | β_t·(v_t − exp(cumg[t])·(S_carry·k_t)) − β_t·Σ_{s<t} pairdecay(t, s)·(k_t·k_s)·u_s |
## | y_t             | exp(cumg[t])·(S_carry·q̃_t) + Σ_{s≤t} pairdecay(t, s)·(q̃_t·k_s)·u_s               |
## | carry           | S = exp(cumg[end])·S_carry + Σ_s pairdecay(end, s)·k_s ⊗ u_s                       |
##
## | contract     | value                                                                                                                              |
## | ------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
## | state math   | all fp32 and never rounds, one 8-row state tile per threadgroup, register-resident across the chunk walk                           |
## | q, k         | (B·Hk, T, Dk) family dtype, already l2-normalized (l2norm stays host-side)                                                         |
## | v, beta      | (B·Hv, T, Dv) and (B·Hv, T) family dtype, g is (B·Hv, T) f32 log-decay                                                             |
## | y            | (B·Hv, T, Dv) family dtype, one round-to-nearest-even per element                                                                  |
## | family dtype | fp16 primary (`gdnPrefillChunkScanF16`), bf16 the range-robust fallback (`gdnPrefillChunkScanBf16`)                                |
## | head mapping | value head bh reads key head `(bh mod Hv) div hkRatio + (bh div Hv)·Hk`, hkRatio = Hv div Hk                                       |
## | chunk axis   | tokens are walked in chunks of ChunkC, the u solve sequential in t inside a chunk, chunks sequential on the register state         |
## | decay / q̃   | exp2(g·log2e), log2e = 1.4426950408889634'f32, Dk^-0.5 folded into q in f32 (rsqrt-multiply form, Metal has no exp device builtin) |
##
## | provenance | source                                                                                                                                                    |
## | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | schedule   | the naive WY/UT reference `gdnPrefillChunked` in workspace/positron/tests/naive/naive_gdn.nim, the same cumg, pairdecay, solve and carry formulas at fp32 |
## | tiles      | the WIP spelling state_space/gdn/gdn_prefill.nim (20260912-positron-taxonomy worktree), kernel design mined, test shapes not carried over                 |

##
## Implementation shape:
## - each lane computes its own state row's scalars (the solve, y, the u contributions),
##   so the u vectors live in a per-lane local array, no inter-threadgroup data movement needed
## - the k·k and q̃·k dot products run as one broadcast-tile pass per (t, s) pair,
##   every lane reads the same row-identical row sum, lanes agree bit-exactly
## - the pair (t, s) passes recompute the key tiles per pair, chunks of 64 stay inside
##   the register budget (per-lane state and u arrays are the only residents, no (ChunkC, Dk) working tile)
##
## Entries are consumer-side:
## - a `metal:` block wraps the grid-driven proc with concrete static (Dk, Dv, ChunkC),
##   one call-site line per static binding set
## - the engine's monomorphization key erases static bindings, calls sharing a call-site line collapse into one body
##
## Binding note:
## - hosts binding through the Metal engine's no-copy path get in-place state updates
##   and visible y writes from one run (page-aligned pointer, page-multiple byte length)
## - `state` is the engine's output buffer, `y` is written by the kernel
## - any other binding copies and the y writes are lost

import workspace/crucible
import workspace/ceramic

from ../../../tile_widen import widenBf16, widenF16

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Core tile procs (inline-tile property) ──────────────────────────

proc gdnPrefillChunkScanBf16At*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[bfloat16],      # (B·Hv, T, Dv) bf16
    k: ptr UncheckedArray[bfloat16],      # (B·Hk, T, Dk) bf16, post-l2norm
    q: ptr UncheckedArray[bfloat16],      # (B·Hk, T, Dk) bf16, post-l2norm
    v: ptr UncheckedArray[bfloat16],      # (B·Hv, T, Dv) bf16
    g: ptr UncheckedArray[float32],       # (B·Hv, T) f32 log decay
    beta: ptr UncheckedArray[bfloat16],   # (B·Hv, T) bf16
    Hv, Hk, hkRatio, T: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## One (bh, TileR-row) state tile of the chunked GDN prefill scan, the whole
  ## token sequence walked over the register state at the caller's coordinates:
  ##
  ## Contract:
  ## - all state arithmetic is fp32, the state never rounds before the final in-place store
  ## - the state entering a chunk is S_carry, untouched until the chunk-end carry update
  ## - the u solve is sequential in t, the sums over s walk s ascending
  ##
  ## `bh` is the (sequence, value head) row block, `dvBlock` the Dv/TileR row block.
  ## The grid-driven wrapper passes the threadgroup coordinates.
  ## Generic only over the static shape, every (Dk, Dv, ChunkC) binding needs its own call-site line.
  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store covers one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
    doAssert ChunkC <= 64, "the per-lane cumg/u local arrays are sized by ChunkC"
  const log2e = 1.4426950408889634'f32
  const rowTiles = TileR div atom.getM()
  const colTiles = Dk div atom.getN()
  const vpt = atom.getVpt()

  let hk = ((bh mod Hv) div hkRatio) + ((bh div Hv) * Hk)
  let headLin = bh * Dv * Dk
  let seqLin = bh * T * Dv
  let kHeadLin = hk * T * Dk

  let glState = state.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dk, 1))
  let glK = k.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glQ = q.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))

  var s: rt_l(float32, TileR, Dk)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))

  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (lane, 0)).toIntVal()
  let rowIn = cell mod 8
  let colIn = cell div 8
  let scale = rsqrt(float32(Dk))

  var cumg: array[ChunkC, float32]   # in-block cumulative log decay, identical on every lane
  var uLoc: array[ChunkC, float32]   # this lane's state row's solve vector u_s[rowIn]

  var c0 = int32(0)
  while c0 < T:
    let cLen = min(ChunkC, int(T - c0))
    # In-block cumulative log decay.
    var gsum = 0'f32
    for i in 0 ..< cLen:
      gsum += g[bh * T + c0 + int32(i)]
      cumg[i] = gsum

    for t in 0 ..< cLen:
      let gt = c0 + int32(t)
      let kLinT = kHeadLin + gt * Dk
      var kT: rt_l(bfloat16, TileR, Dk)
      kT.loadTile(glK, (kLinT, 0, 0, 0))
      var k32: rt_l(float32, TileR, Dk)
      k32.widenBf16(kT)

      # G_t = exp(cumg[t])·(S_carry·k_t), the decayed carry read against the key
      var kProd: rt_l(float32, TileR, Dk)
      kProd.mul(s, k32)
      var kVec: rv(float32, TileR, Dk)
      kVec.row_sum(kProd)
      let decayT = exp2(cumg[t] * log2e)
      let gRead = decayT * kVec.data[0]

      let bt = beta[bh * T + gt].float32
      let v32 = v[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)].float32

      # u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} pairdecay(t, s)·(k_t·k_s)·u_s, solved in token order
      var uacc = 0'f32
      for sIdx in 0 ..< t:
        var ksT: rt_l(bfloat16, TileR, Dk)
        ksT.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var ks32: rt_l(float32, TileR, Dk)
        ks32.widenBf16(ksT)
        var kkProd: rt_l(float32, TileR, Dk)
        kkProd.mul(k32, ks32)
        var kkVec: rv(float32, TileR, Dk)
        kkVec.row_sum(kkProd)
        let pdts = exp2((cumg[t] - cumg[sIdx]) * log2e)
        uacc += pdts * kkVec.data[0] * uLoc[sIdx]
      let ut = bt * (v32 - gRead) - bt * uacc
      uLoc[t] = ut

      # y_t = exp(cumg[t])·(S_carry·q̃_t) + Σ_{s≤t} pairdecay(t, s)·(q̃_t·k_s)·u_s
      var qT: rt_l(bfloat16, TileR, Dk)
      qT.loadTile(glQ, (kLinT, 0, 0, 0))
      var q32: rt_l(float32, TileR, Dk)
      q32.widenBf16(qT)
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            q32.frags[n][m].frag[f] = q32.frags[n][m].frag[f] * scale

      var qProd: rt_l(float32, TileR, Dk)
      qProd.mul(s, q32)
      var qVec: rv(float32, TileR, Dk)
      qVec.row_sum(qProd)
      var yVal = decayT * qVec.data[0]
      for sIdx in 0 .. t:
        var ksT: rt_l(bfloat16, TileR, Dk)
        ksT.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var ks32: rt_l(float32, TileR, Dk)
        ks32.widenBf16(ksT)
        var qkProd: rt_l(float32, TileR, Dk)
        qkProd.mul(q32, ks32)
        var qkVec: rv(float32, TileR, Dk)
        qkVec.row_sum(qkProd)
        let pdts = exp2((cumg[t] - cumg[sIdx]) * log2e)
        yVal += pdts * qkVec.data[0] * uLoc[sIdx]

      if colIn == 0:
        y[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)] = yVal.bfloat16

    # Carry out of the chunk:
    # S = exp(cumg[end])·S_carry + Σ_s pairdecay(end, s)·k_s ⊗ u_s
    let decayEnd = exp2(cumg[cLen - 1] * log2e)
    s.mul(s, decayEnd)
    for sIdx in 0 ..< cLen:
      let ws = exp2((cumg[cLen - 1] - cumg[sIdx]) * log2e) * uLoc[sIdx]
      var ksT: rt_l(bfloat16, TileR, Dk)
      ksT.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
      var ks32: rt_l(float32, TileR, Dk)
      ks32.widenBf16(ksT)
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            s.frags[n][m].frag[f] = s.frags[n][m].frag[f] + ks32.frags[n][m].frag[f] * ws
    c0 += int32(cLen)

  glState.storeTile(s, (headLin, 0, dvBlock, 0))

proc gdnPrefillChunkScanBf16*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[bfloat16],      # (B·Hv, T, Dv) bf16
    k: ptr UncheckedArray[bfloat16],      # (B·Hk, T, Dk) bf16, post-l2norm
    q: ptr UncheckedArray[bfloat16],      # (B·Hk, T, Dk) bf16, post-l2norm
    v: ptr UncheckedArray[bfloat16],      # (B·Hv, T, Dv) bf16
    g: ptr UncheckedArray[float32],       # (B·Hv, T) f32 log decay
    beta: ptr UncheckedArray[bfloat16],   # (B·Hv, T) bf16
    Hv, Hk, hkRatio, T: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## Grid-driven form of `gdnPrefillChunkScanBf16At`, the caller's `metal:` entry wraps this proc.
  ## - grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  gdnPrefillChunkScanBf16At(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, T,
    dvBlock, bh, Dk, Dv, TileR, ChunkC)

proc gdnPrefillChunkScanF16At*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[float16],       # (B·Hv, T, Dv) fp16
    k: ptr UncheckedArray[float16],       # (B·Hk, T, Dk) fp16, post-l2norm
    q: ptr UncheckedArray[float16],       # (B·Hk, T, Dk) fp16, post-l2norm
    v: ptr UncheckedArray[float16],       # (B·Hv, T, Dv) fp16
    g: ptr UncheckedArray[float32],       # (B·Hv, T) f32 log decay
    beta: ptr UncheckedArray[float16],    # (B·Hv, T) fp16
    Hv, Hk, hkRatio, T: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## `gdnPrefillChunkScanBf16At` with the fp16 family dtype, the same fp32 state arithmetic,
  ## fp16 loads and one fp16 y rounding per element.
  ##
  ## The 8×8×8 fp16 atom shares the bf16 lane→element geometry, tile walk, geometry contract and static asserts are identical.
  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store covers one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
    doAssert ChunkC <= 64, "the per-lane cumg/u local arrays are sized by ChunkC"
  const log2e = 1.4426950408889634'f32
  const rowTiles = TileR div atom.getM()
  const colTiles = Dk div atom.getN()
  const vpt = atom.getVpt()

  let hk = ((bh mod Hv) div hkRatio) + ((bh div Hv) * Hk)
  let headLin = bh * Dv * Dk
  let seqLin = bh * T * Dv
  let kHeadLin = hk * T * Dk

  let glState = state.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dk, 1))
  let glK = k.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glQ = q.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))

  var s: rt_l(float32, TileR, Dk)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))

  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (lane, 0)).toIntVal()
  let rowIn = cell mod 8
  let colIn = cell div 8
  let scale = rsqrt(float32(Dk))

  var cumg: array[ChunkC, float32]   # in-block cumulative log decay, identical on every lane
  var uLoc: array[ChunkC, float32]   # this lane's state row's solve vector u_s[rowIn]

  var c0 = int32(0)
  while c0 < T:
    let cLen = min(ChunkC, int(T - c0))
    # In-block cumulative log decay.
    var gsum = 0'f32
    for i in 0 ..< cLen:
      gsum += g[bh * T + c0 + int32(i)]
      cumg[i] = gsum

    for t in 0 ..< cLen:
      let gt = c0 + int32(t)
      let kLinT = kHeadLin + gt * Dk
      var kT: rt_l(float16, TileR, Dk)
      kT.loadTile(glK, (kLinT, 0, 0, 0))
      var k32: rt_l(float32, TileR, Dk)
      k32.widenF16(kT)

      # G_t = exp(cumg[t])·(S_carry·k_t), the decayed carry read against the key
      var kProd: rt_l(float32, TileR, Dk)
      kProd.mul(s, k32)
      var kVec: rv(float32, TileR, Dk)
      kVec.row_sum(kProd)
      let decayT = exp2(cumg[t] * log2e)
      let gRead = decayT * kVec.data[0]

      let bt = beta[bh * T + gt].float32
      let v32 = v[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)].float32

      # u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} pairdecay(t, s)·(k_t·k_s)·u_s, solved in token order
      var uacc = 0'f32
      for sIdx in 0 ..< t:
        var ksT: rt_l(float16, TileR, Dk)
        ksT.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var ks32: rt_l(float32, TileR, Dk)
        ks32.widenF16(ksT)
        var kkProd: rt_l(float32, TileR, Dk)
        kkProd.mul(k32, ks32)
        var kkVec: rv(float32, TileR, Dk)
        kkVec.row_sum(kkProd)
        let pdts = exp2((cumg[t] - cumg[sIdx]) * log2e)
        uacc += pdts * kkVec.data[0] * uLoc[sIdx]
      let ut = bt * (v32 - gRead) - bt * uacc
      uLoc[t] = ut

      # y_t = exp(cumg[t])·(S_carry·q̃_t) + Σ_{s≤t} pairdecay(t, s)·(q̃_t·k_s)·u_s
      var qT: rt_l(float16, TileR, Dk)
      qT.loadTile(glQ, (kLinT, 0, 0, 0))
      var q32: rt_l(float32, TileR, Dk)
      q32.widenF16(qT)
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            q32.frags[n][m].frag[f] = q32.frags[n][m].frag[f] * scale

      var qProd: rt_l(float32, TileR, Dk)
      qProd.mul(s, q32)
      var qVec: rv(float32, TileR, Dk)
      qVec.row_sum(qProd)
      var yVal = decayT * qVec.data[0]
      for sIdx in 0 .. t:
        var ksT: rt_l(float16, TileR, Dk)
        ksT.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var ks32: rt_l(float32, TileR, Dk)
        ks32.widenF16(ksT)
        var qkProd: rt_l(float32, TileR, Dk)
        qkProd.mul(q32, ks32)
        var qkVec: rv(float32, TileR, Dk)
        qkVec.row_sum(qkProd)
        let pdts = exp2((cumg[t] - cumg[sIdx]) * log2e)
        yVal += pdts * qkVec.data[0] * uLoc[sIdx]

      if colIn == 0:
        y[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)] = yVal.float16

    # Carry out of the chunk:
    # S = exp(cumg[end])·S_carry + Σ_s pairdecay(end, s)·k_s ⊗ u_s
    let decayEnd = exp2(cumg[cLen - 1] * log2e)
    s.mul(s, decayEnd)
    for sIdx in 0 ..< cLen:
      let ws = exp2((cumg[cLen - 1] - cumg[sIdx]) * log2e) * uLoc[sIdx]
      var ksT: rt_l(float16, TileR, Dk)
      ksT.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
      var ks32: rt_l(float32, TileR, Dk)
      ks32.widenF16(ksT)
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            s.frags[n][m].frag[f] = s.frags[n][m].frag[f] + ks32.frags[n][m].frag[f] * ws
    c0 += int32(cLen)

  glState.storeTile(s, (headLin, 0, dvBlock, 0))

proc gdnPrefillChunkScanF16*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[float16],       # (B·Hv, T, Dv) fp16
    k: ptr UncheckedArray[float16],       # (B·Hk, T, Dk) fp16, post-l2norm
    q: ptr UncheckedArray[float16],       # (B·Hk, T, Dk) fp16, post-l2norm
    v: ptr UncheckedArray[float16],       # (B·Hv, T, Dv) fp16
    g: ptr UncheckedArray[float32],       # (B·Hv, T) f32 log decay
    beta: ptr UncheckedArray[float16],    # (B·Hv, T) fp16
    Hv, Hk, hkRatio, T: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## Grid-driven form of `gdnPrefillChunkScanF16At`, the caller's `metal:` entry wraps this proc.
  ## - grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  gdnPrefillChunkScanF16At(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, T,
    dvBlock, bh, Dk, Dv, TileR, ChunkC)
