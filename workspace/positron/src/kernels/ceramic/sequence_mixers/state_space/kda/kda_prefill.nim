# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ───────────────  KDA prefill (chunked scan over T tokens, one launch)  ───────────────

## Prefill (T > 1) of the Kimi Delta Attention recurrence (arXiv:2510.26692), one launch
## on the ceramic Tile API walking the tokens in chunks of ChunkC over the per-channel
## cumulative log decay cumg (one decay per KEY channel):
##
## | term      | formula                                                                                        |
## | --------- | ---------------------------------------------------------------------------------------------- |
## | pairdecay | exp2((cumg[t, dk] − cumg[s, dk])·log2e), per key channel dk (the difference form)              |
## | u_t       | β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s, A[t, s] = Σ_dk pairdecay(t, s)[dk]·k_t[dk]·k_s[dk]  |
## | G_t       | Σ_dk exp(cumg[t, dk])·k_t[dk]·S_carry[r, dk], the decayed carry read BEFORE the kv contraction |
## | y_t       | H_t + Σ_{s≤t} B[t, s]·u_s[r], B[t, s] = Σ_dk pairdecay(t, s)[dk]·q̃_t[dk]·k_s[dk]              |
## | H_t       | Σ_dk exp(cumg[t, dk])·q̃_t[dk]·S_carry[r, dk]                                                  |
## | carry     | S[r, dk] = exp(cumg[end, dk])·S_carry[r, dk] + Σ_s pairdecay(end, s)[dk]·k_s[dk]·u_s[r]        |
##
## | contract     | value                                                                                                                          |
## | ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
## | state math   | all fp32 and never rounds, one 8-row state tile per threadgroup, register-resident across the chunk walk                       |
## | q, k, g, β   | (B·Hk, T, Dk) f32 q/k/g post-l2norm and (B·Hv, T) f32 beta, never rounded to family (the recorded per-channel family contract) |
## | cumg         | (B·Hk, T, Dk) f32 per-channel cumulative log decay, the host prefix of g (the GateForm formula stays host-side)                |
## | v, y         | (B·Hv, T, Dv) family dtype each, y gets one round-to-nearest-even per element                                                  |
## | family dtype | fp16 primary (`kdaPrefillChunkScanF16`), bf16 the range-robust fallback (`kdaPrefillChunkScanBf16`)                            |
## | head mapping | value head bh reads key head `(bh mod Hv) div hkRatio + (bh div Hv)·Hk`, hkRatio = Hv div Hk                                   |
## | chunk axis   | tokens are walked in chunks of ChunkC, the u solve sequential in t inside a chunk, chunks sequential on the register state     |
## | decay / q̃   | exp2(cumg·log2e) per channel in-device, log2e = 1.4426950408889634'f32, q̃ = per-element division by the runtime f32 qScale    |
##
## | provenance | source                                                                                                                                                        |
## | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | schedule   | the naive WY/UT reference `kdaPrefillChunked` in workspace/positron/tests/naive/naive_kda.nim, the same per-channel cumg, pairdecay, solve and carry formulas |
## | tiles      | the WIP spelling state_space/kda/kda_prefill.nim (20260912-positron-taxonomy worktree) and the landed GDN chunk scan state_space/gdn/gdn_prefill.nim          |

##
## Implementation shape:
## - each lane computes its own state row's scalars (the solve, y, the u contributions),
##   so the u vectors live in a per-lane local array, no inter-threadgroup data movement needed
## - the per-token decay dT = exp2(cumg_t·log2e) folds into the carry reads once
##   per token, over the same broadcast Tile ops as the per-channel factors
## - the pair decay exp2((cumg_t − cumg_s)·log2e) folds into the A/B dots per
##   (t, s) pair, the chunk-end decay into the carry
##
## Pair decay's difference form removes the dT·invd_s overflow:
##
## - exp2(x)·exp2(y) = exp2(x + y)
## - cumg decreases along t, so the argument is ≤ 0 and no intermediate exceeds 1
## - the factorized spelling dT·invd_s overflows exp2 once |cumg_s| ≳ 88.7,
##   the resulting Inf × dT → 0 product NaNs the carry and the persistent state
##
## - the k·k and q̃·k dot products run as one broadcast-tile pass per (t, s) pair,
##   every lane reads the same row-identical row sum, lanes agree bit-exactly
## - the pair (t, s) passes reload the key and cumg tiles per pair, chunks of 64 stay inside
##   the register budget (per-lane state and u arrays are the only residents, no (ChunkC, Dk) working tile)
##
##   per chunk:   cumg tiles ─→ per token t: dT, G_t read ─→ u_t solve ─→ y_t store
##                (t in token order)                           └─────────┐
##            └──→ S ← dEnd ⊙ S_carry + Σ_s (dEnd·invd_s ⊙ k_s) [x] u_s
##
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

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

const PairdecaySabotage* {.booldefine.} = false
  ## Compile-time sabotage switch for the pairdecay overflow fixture
  ##
  ## - restores the factorized spelling dT·exp2(−cumg_s·log2e) at all six
  ##   pairdecay sites, the two forms are algebraically identical
  ## - the factorized form overflows exp2 once |cumg_s| > 128/log2e ≈ 88.7,
  ##   the Inf·dT product NaNs the state (see the per-site notes)
  ## - default builds leave the factorized spelling out

# ─── Core tile procs (inline-tile property) ──────────────────────────

proc kdaPrefillChunkScanBf16At*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[bfloat16],      # (B·Hv, T, Dv) bf16
    k: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    q: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    cumg: ptr UncheckedArray[float32],    # (B·Hk, T, Dk) f32 per-channel cumulative log decay
    v: ptr UncheckedArray[bfloat16],      # (B·Hv, T, Dv) bf16
    beta: ptr UncheckedArray[float32],    # (B·Hv, T) f32 beta
    qScale: float32,                      # √Dk, host-computed f64→f32 cast
    Hv, Hk, hkRatio, T: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## One (bh, TileR-row) state tile of the chunked KDA prefill scan, the whole
  ## token sequence walked over the register state at the caller's coordinates
  ##
  ## Contract:
  ## - all state arithmetic is fp32, the state never rounds before the final in-place store
  ## - the state entering a chunk is S_carry, untouched until the chunk-end carry update
  ## - the per-channel decay applies BEFORE the kv reads, one decay per KEY channel,
  ##   the carry reads contracting the decayed state, the pair decays from the chunk's
  ##   per-channel cumulative log decay
  ##
  ## - the u solve is sequential in t, the sums over s walk s ascending
  ##
  ##   per token:   dT ─→ G_t, H_t reads ─→ u_t solve ─→ y_t store
  ##   chunk end   S ← dEnd ⊙ S_carry + Σ_s (dEnd·invd_s ⊙ k_s) [x] u_s
  ##
  ## `bh` is the (sequence, value head) row block, `dvBlock` the Dv/TileR row block.
  ## The grid-driven wrapper passes the threadgroup coordinates.
  ## Generic only over the static shape, every (Dk, Dv, ChunkC) binding needs its own call-site line.
  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store covers one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
    doAssert ChunkC <= 64, "the per-lane u local array is sized by ChunkC"
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
  let glCumg = cumg.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))

  var s: rt_l(float32, TileR, Dk)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))

  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (lane, 0)).toIntVal()
  let rowIn = cell mod 8
  let colIn = cell div 8

  var uLoc: array[ChunkC, float32]   # this lane's state row's solve vector u_s[rowIn]

  var c0 = int32(0)
  while c0 < T:
    let cLen = min(ChunkC, int(T - c0))
    for t in 0 ..< cLen:
      let gt = c0 + int32(t)
      let kLinT = kHeadLin + gt * Dk
      var k32: rt_l(float32, TileR, Dk)
      k32.loadTile(glK, (kLinT, 0, 0, 0))
      var cumgT: rt_l(float32, TileR, Dk)
      cumgT.loadTile(glCumg, (kLinT, 0, 0, 0))

      # Per-token decay dT = exp2(cumg_t·log2e), one factor per key channel,
      # the exp2 form (see the module doc).
      var dT: rt_l(float32, TileR, Dk)
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            dT.frags[n][m].frag[f] = cumgT.frags[n][m].frag[f]
      dT.mul(dT, log2e)
      exp2(dT, dT)

      # G_t = Σ_dk dT[dk]·k_t[dk]·S_carry[row][dk], the decayed carry read
      # against the key, BEFORE the kv contraction (the recurrence's step order)
      var gProd: rt_l(float32, TileR, Dk)
      gProd.mul(s, k32)
      gProd.mul(gProd, dT)
      var gVec: rv(float32, TileR, Dk)
      gVec.row_sum(gProd)
      let gRead = gVec.data[0]

      let bt = beta[bh * T + gt]
      let v32 = v[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)].float32

      # u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s, solved in token order
      var uacc = 0'f32
      for sIdx in 0 ..< t:
        var ks32: rt_l(float32, TileR, Dk)
        ks32.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var cumgS: rt_l(float32, TileR, Dk)
        cumgS.loadTile(glCumg, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        # pairdecay(t, s)[dk] = exp2((cumg_t[dk] − cumg_s[dk])·log2e) per key channel,
        # the difference form (dT·exp2(−cumg_s·log2e) in algebra).
        #
        # - the argument stays ≤ 0 (cumg decreases along t), no intermediate
        #   exceeds 1, exp2 cannot overflow
        # - the factorized spelling dT·exp2(−cumg_s·log2e) overflows exp2 once
        #   |cumg_s| ≳ 88.7, the resulting Inf × dT → 0 product NaNs the carry
        #   and the persistent state
        var pdT: rt_l(float32, TileR, Dk)
        when PairdecaySabotage:
          # the factorized spelling dT·exp2(−cumg_s·log2e), exp2 gives Inf
          # once |cumg_s| > 128/log2e ≈ 88.7, the Inf·dT product NaNs the state
          for n in 0 ..< rowTiles:
            for m in 0 ..< colTiles:
              for f in 0 ..< vpt:
                pdT.frags[n][m].frag[f] = dT.frags[n][m].frag[f] *
                  exp2(-cumgS.frags[n][m].frag[f] * log2e)
        else:
          for n in 0 ..< rowTiles:
            for m in 0 ..< colTiles:
              for f in 0 ..< vpt:
                pdT.frags[n][m].frag[f] = exp2(
                  (cumgT.frags[n][m].frag[f] - cumgS.frags[n][m].frag[f]) * log2e)
        var kkProd: rt_l(float32, TileR, Dk)
        kkProd.mul(k32, ks32)
        kkProd.mul(kkProd, pdT)
        var kkVec: rv(float32, TileR, Dk)
        kkVec.row_sum(kkProd)
        uacc += kkVec.data[0] * uLoc[sIdx]
      let ut = bt * (v32 - gRead) - bt * uacc
      uLoc[t] = ut

      # y_t = H_t + Σ_{s≤t} B[t, s]·u_s, H_t the decayed carry read against q̃
      var q32: rt_l(float32, TileR, Dk)
      q32.loadTile(glQ, (kLinT, 0, 0, 0))
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            q32.frags[n][m].frag[f] = q32.frags[n][m].frag[f] / qScale

      var hProd: rt_l(float32, TileR, Dk)
      hProd.mul(s, q32)
      hProd.mul(hProd, dT)
      var hVec: rv(float32, TileR, Dk)
      hVec.row_sum(hProd)
      var yVal = hVec.data[0]
      for sIdx in 0 .. t:
        var ks32: rt_l(float32, TileR, Dk)
        ks32.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var cumgS: rt_l(float32, TileR, Dk)
        cumgS.loadTile(glCumg, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var pdT: rt_l(float32, TileR, Dk)
        # pairdecay(t, s)[dk] = exp2((cumg_t[dk] − cumg_s[dk])·log2e) per key channel,
        # the difference form (dT·exp2(−cumg_s·log2e) in algebra).
        when PairdecaySabotage:
          # the factorized spelling dT·exp2(−cumg_s·log2e), exp2 gives Inf
          # once |cumg_s| > 128/log2e ≈ 88.7, the Inf·dT product NaNs the state
          for n in 0 ..< rowTiles:
            for m in 0 ..< colTiles:
              for f in 0 ..< vpt:
                pdT.frags[n][m].frag[f] = dT.frags[n][m].frag[f] *
                  exp2(-cumgS.frags[n][m].frag[f] * log2e)
        else:
          for n in 0 ..< rowTiles:
            for m in 0 ..< colTiles:
              for f in 0 ..< vpt:
                pdT.frags[n][m].frag[f] = exp2(
                  (cumgT.frags[n][m].frag[f] - cumgS.frags[n][m].frag[f]) * log2e)
        var qkProd: rt_l(float32, TileR, Dk)
        qkProd.mul(q32, ks32)
        qkProd.mul(qkProd, pdT)
        var qkVec: rv(float32, TileR, Dk)
        qkVec.row_sum(qkProd)
        yVal += qkVec.data[0] * uLoc[sIdx]

      if colIn == 0:
        y[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)] = yVal.bfloat16

    # Carry out of the chunk, per key channel:
    # S = dEnd ⊙ S_carry + Σ_s (dEnd·invd_s ⊙ k_s) ⊗ u_s
    let gtEnd = c0 + int32(cLen - 1)
    var cumgEnd: rt_l(float32, TileR, Dk)
    cumgEnd.loadTile(glCumg, (kHeadLin + gtEnd * Dk, 0, 0, 0))
    var dEnd: rt_l(float32, TileR, Dk)
    dEnd.mul(cumgEnd, log2e)
    exp2(dEnd, dEnd)
    s.mul(s, dEnd)
    for sIdx in 0 ..< cLen:
      var ks32: rt_l(float32, TileR, Dk)
      ks32.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
      var cumgS: rt_l(float32, TileR, Dk)
      cumgS.loadTile(glCumg, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
      # pairdecay(end, s)[dk] = exp2((cumg_end[dk] − cumg_s[dk])·log2e) per key channel,
      # the difference form (the token pairdecay note carries the overflow bound)
      var pdEnd: rt_l(float32, TileR, Dk)
      when PairdecaySabotage:
        # the factorized spelling dEnd·exp2(−cumg_s·log2e), the token
        # pairdecay note carries the overflow bound
        for n in 0 ..< rowTiles:
          for m in 0 ..< colTiles:
            for f in 0 ..< vpt:
              pdEnd.frags[n][m].frag[f] = dEnd.frags[n][m].frag[f] *
                exp2(-cumgS.frags[n][m].frag[f] * log2e)
      else:
        for n in 0 ..< rowTiles:
          for m in 0 ..< colTiles:
            for f in 0 ..< vpt:
              pdEnd.frags[n][m].frag[f] = exp2(
                (cumgEnd.frags[n][m].frag[f] - cumgS.frags[n][m].frag[f]) * log2e)
      pdEnd.mul(pdEnd, ks32)
      let ws = uLoc[sIdx]
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            s.frags[n][m].frag[f] = s.frags[n][m].frag[f] + pdEnd.frags[n][m].frag[f] * ws
    c0 += int32(cLen)

  glState.storeTile(s, (headLin, 0, dvBlock, 0))

proc kdaPrefillChunkScanBf16*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[bfloat16],      # (B·Hv, T, Dv) bf16
    k: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    q: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    cumg: ptr UncheckedArray[float32],    # (B·Hk, T, Dk) f32 per-channel cumulative log decay
    v: ptr UncheckedArray[bfloat16],      # (B·Hv, T, Dv) bf16
    beta: ptr UncheckedArray[float32],    # (B·Hv, T) f32 beta
    qScale: float32,                      # √Dk, host-computed f64→f32 cast
    Hv, Hk, hkRatio, T: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## Grid-driven form of `kdaPrefillChunkScanBf16At`, the caller's `metal:` entry wraps this proc.
  ## - grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  kdaPrefillChunkScanBf16At(state, y, k, q, cumg, v, beta, qScale, Hv, Hk, hkRatio, T,
    dvBlock, bh, Dk, Dv, TileR, ChunkC)

proc kdaPrefillChunkScanF16At*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[float16],       # (B·Hv, T, Dv) fp16
    k: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    q: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    cumg: ptr UncheckedArray[float32],    # (B·Hk, T, Dk) f32 per-channel cumulative log decay
    v: ptr UncheckedArray[float16],       # (B·Hv, T, Dv) fp16
    beta: ptr UncheckedArray[float32],    # (B·Hv, T) f32 beta
    qScale: float32,                      # √Dk, host-computed f64→f32 cast
    Hv, Hk, hkRatio, T: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## `kdaPrefillChunkScanBf16At` with the fp16 family dtype, the same fp32 state arithmetic,
  ## fp16 loads and one fp16 y rounding per element.
  ##
  ## The 8×8×8 fp16 atom shares the bf16 lane→element geometry, tile walk, geometry contract and static asserts are identical.
  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store covers one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
    doAssert ChunkC <= 64, "the per-lane u local array is sized by ChunkC"
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
  let glCumg = cumg.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))

  var s: rt_l(float32, TileR, Dk)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))

  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (lane, 0)).toIntVal()
  let rowIn = cell mod 8
  let colIn = cell div 8

  var uLoc: array[ChunkC, float32]   # this lane's state row's solve vector u_s[rowIn]

  var c0 = int32(0)
  while c0 < T:
    let cLen = min(ChunkC, int(T - c0))
    for t in 0 ..< cLen:
      let gt = c0 + int32(t)
      let kLinT = kHeadLin + gt * Dk
      var k32: rt_l(float32, TileR, Dk)
      k32.loadTile(glK, (kLinT, 0, 0, 0))
      var cumgT: rt_l(float32, TileR, Dk)
      cumgT.loadTile(glCumg, (kLinT, 0, 0, 0))

      # Per-token decay dT = exp2(cumg_t·log2e), one factor per key channel,
      # the exp2 form (see the module doc).
      var dT: rt_l(float32, TileR, Dk)
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            dT.frags[n][m].frag[f] = cumgT.frags[n][m].frag[f]
      dT.mul(dT, log2e)
      exp2(dT, dT)

      # G_t = Σ_dk dT[dk]·k_t[dk]·S_carry[row][dk], the decayed carry read
      # against the key, BEFORE the kv contraction (the recurrence's step order)
      var gProd: rt_l(float32, TileR, Dk)
      gProd.mul(s, k32)
      gProd.mul(gProd, dT)
      var gVec: rv(float32, TileR, Dk)
      gVec.row_sum(gProd)
      let gRead = gVec.data[0]

      let bt = beta[bh * T + gt]
      let v32 = v[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)].float32

      # u_t = β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s, solved in token order
      var uacc = 0'f32
      for sIdx in 0 ..< t:
        var ks32: rt_l(float32, TileR, Dk)
        ks32.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var cumgS: rt_l(float32, TileR, Dk)
        cumgS.loadTile(glCumg, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        # pairdecay(t, s)[dk] = exp2((cumg_t[dk] − cumg_s[dk])·log2e) per key channel,
        # the difference form (dT·exp2(−cumg_s·log2e) in algebra).
        #
        # - the argument stays ≤ 0 (cumg decreases along t), no intermediate
        #   exceeds 1, exp2 cannot overflow
        # - the factorized spelling dT·exp2(−cumg_s·log2e) overflows exp2 once
        #   |cumg_s| ≳ 88.7, the resulting Inf × dT → 0 product NaNs the carry
        #   and the persistent state
        var pdT: rt_l(float32, TileR, Dk)
        when PairdecaySabotage:
          # the factorized spelling dT·exp2(−cumg_s·log2e), exp2 gives Inf
          # once |cumg_s| > 128/log2e ≈ 88.7, the Inf·dT product NaNs the state
          for n in 0 ..< rowTiles:
            for m in 0 ..< colTiles:
              for f in 0 ..< vpt:
                pdT.frags[n][m].frag[f] = dT.frags[n][m].frag[f] *
                  exp2(-cumgS.frags[n][m].frag[f] * log2e)
        else:
          for n in 0 ..< rowTiles:
            for m in 0 ..< colTiles:
              for f in 0 ..< vpt:
                pdT.frags[n][m].frag[f] = exp2(
                  (cumgT.frags[n][m].frag[f] - cumgS.frags[n][m].frag[f]) * log2e)
        var kkProd: rt_l(float32, TileR, Dk)
        kkProd.mul(k32, ks32)
        kkProd.mul(kkProd, pdT)
        var kkVec: rv(float32, TileR, Dk)
        kkVec.row_sum(kkProd)
        uacc += kkVec.data[0] * uLoc[sIdx]
      let ut = bt * (v32 - gRead) - bt * uacc
      uLoc[t] = ut

      # y_t = H_t + Σ_{s≤t} B[t, s]·u_s, H_t the decayed carry read against q̃
      var q32: rt_l(float32, TileR, Dk)
      q32.loadTile(glQ, (kLinT, 0, 0, 0))
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            q32.frags[n][m].frag[f] = q32.frags[n][m].frag[f] / qScale

      var hProd: rt_l(float32, TileR, Dk)
      hProd.mul(s, q32)
      hProd.mul(hProd, dT)
      var hVec: rv(float32, TileR, Dk)
      hVec.row_sum(hProd)
      var yVal = hVec.data[0]
      for sIdx in 0 .. t:
        var ks32: rt_l(float32, TileR, Dk)
        ks32.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var cumgS: rt_l(float32, TileR, Dk)
        cumgS.loadTile(glCumg, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var pdT: rt_l(float32, TileR, Dk)
        # pairdecay(t, s)[dk] = exp2((cumg_t[dk] − cumg_s[dk])·log2e) per key channel,
        # the difference form (dT·exp2(−cumg_s·log2e) in algebra).
        when PairdecaySabotage:
          # the factorized spelling dT·exp2(−cumg_s·log2e), exp2 gives Inf
          # once |cumg_s| > 128/log2e ≈ 88.7, the Inf·dT product NaNs the state
          for n in 0 ..< rowTiles:
            for m in 0 ..< colTiles:
              for f in 0 ..< vpt:
                pdT.frags[n][m].frag[f] = dT.frags[n][m].frag[f] *
                  exp2(-cumgS.frags[n][m].frag[f] * log2e)
        else:
          for n in 0 ..< rowTiles:
            for m in 0 ..< colTiles:
              for f in 0 ..< vpt:
                pdT.frags[n][m].frag[f] = exp2(
                  (cumgT.frags[n][m].frag[f] - cumgS.frags[n][m].frag[f]) * log2e)
        var qkProd: rt_l(float32, TileR, Dk)
        qkProd.mul(q32, ks32)
        qkProd.mul(qkProd, pdT)
        var qkVec: rv(float32, TileR, Dk)
        qkVec.row_sum(qkProd)
        yVal += qkVec.data[0] * uLoc[sIdx]

      if colIn == 0:
        y[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)] = yVal.float16

    # Carry out of the chunk, per key channel:
    # S = dEnd ⊙ S_carry + Σ_s (dEnd·invd_s ⊙ k_s) ⊗ u_s
    let gtEnd = c0 + int32(cLen - 1)
    var cumgEnd: rt_l(float32, TileR, Dk)
    cumgEnd.loadTile(glCumg, (kHeadLin + gtEnd * Dk, 0, 0, 0))
    var dEnd: rt_l(float32, TileR, Dk)
    dEnd.mul(cumgEnd, log2e)
    exp2(dEnd, dEnd)
    s.mul(s, dEnd)
    for sIdx in 0 ..< cLen:
      var ks32: rt_l(float32, TileR, Dk)
      ks32.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
      var cumgS: rt_l(float32, TileR, Dk)
      cumgS.loadTile(glCumg, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
      # pairdecay(end, s)[dk] = exp2((cumg_end[dk] − cumg_s[dk])·log2e) per key channel,
      # the difference form (the token pairdecay note carries the overflow bound)
      var pdEnd: rt_l(float32, TileR, Dk)
      when PairdecaySabotage:
        # the factorized spelling dEnd·exp2(−cumg_s·log2e), the token
        # pairdecay note carries the overflow bound
        for n in 0 ..< rowTiles:
          for m in 0 ..< colTiles:
            for f in 0 ..< vpt:
              pdEnd.frags[n][m].frag[f] = dEnd.frags[n][m].frag[f] *
                exp2(-cumgS.frags[n][m].frag[f] * log2e)
      else:
        for n in 0 ..< rowTiles:
          for m in 0 ..< colTiles:
            for f in 0 ..< vpt:
              pdEnd.frags[n][m].frag[f] = exp2(
                (cumgEnd.frags[n][m].frag[f] - cumgS.frags[n][m].frag[f]) * log2e)
      pdEnd.mul(pdEnd, ks32)
      let ws = uLoc[sIdx]
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            s.frags[n][m].frag[f] = s.frags[n][m].frag[f] + pdEnd.frags[n][m].frag[f] * ws
    c0 += int32(cLen)

  glState.storeTile(s, (headLin, 0, dvBlock, 0))

proc kdaPrefillChunkScanF16*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[float16],       # (B·Hv, T, Dv) fp16
    k: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    q: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    cumg: ptr UncheckedArray[float32],    # (B·Hk, T, Dk) f32 per-channel cumulative log decay
    v: ptr UncheckedArray[float16],       # (B·Hv, T, Dv) fp16
    beta: ptr UncheckedArray[float32],    # (B·Hv, T) f32 beta
    qScale: float32,                      # √Dk, host-computed f64→f32 cast
    Hv, Hk, hkRatio, T: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## Grid-driven form of `kdaPrefillChunkScanF16At`, the caller's `metal:` entry wraps this proc.
  ## - grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  kdaPrefillChunkScanF16At(state, y, k, q, cumg, v, beta, qScale, Hv, Hk, hkRatio, T,
    dvBlock, bh, Dk, Dv, TileR, ChunkC)
