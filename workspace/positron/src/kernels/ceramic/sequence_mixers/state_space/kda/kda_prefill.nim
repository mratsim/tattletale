# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ───────────────  KDA prefill (chunked scan over T tokens, one launch)  ───────────────

## Prefill (T > 1) of the Kimi Delta Attention recurrence (arXiv:2510.26692), one launch
## on the ceramic Tile API walking the tokens in chunks of ChunkC over the per-channel
## cumulative log decay cumulogdecay (one decay per KEY channel):
##
## | term      | formula                                                                                                |
## | --------- | ------------------------------------------------------------------------------------------------------ |
## | pairdecay | exp2((cumulogdecay[t, dk] − cumulogdecay[s, dk])·log2e), per key channel dk (the difference form)      |
## | u_t       | β_t·(v_t − G_t) − β_t·Σ_{s<t} A[t, s]·u_s, A[t, s] = Σ_dk pairdecay(t, s)[dk]·k_t[dk]·k_s[dk]          |
## | G_t       | Σ_dk exp(cumulogdecay[t, dk])·k_t[dk]·S_carry[r, dk], the decayed carry read BEFORE the kv contraction |
## | y_t       | H_t + Σ_{s≤t} B[t, s]·u_s[r], B[t, s] = Σ_dk pairdecay(t, s)[dk]·q̃_t[dk]·k_s[dk]                      |
## | H_t       | Σ_dk exp(cumulogdecay[t, dk])·q̃_t[dk]·S_carry[r, dk]                                                  |
## | carry     | S[r, dk] = exp(cumulogdecay[end, dk])·S_carry[r, dk] + Σ_s pairdecay(end, s)[dk]·k_s[dk]·u_s[r]        |
##
## | contract     | value                                                                                                                                        |
## | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
## | state math   | all fp32 and never rounds, one 8-row state tile per threadgroup, register-resident across the chunk walk                                     |
## | q, k, g, β   | (B·Hk, T, Dk) f32 q/k/g post-l2norm and (B·Hv, T) f32 beta, never rounded to family (the recorded per-channel family contract)               |
## | cumulogdecay | (B·Hk, T, Dk) f32 per-channel cumulative log decay, the host prefix of g (the GateForm formula stays host-side)                              |
## | v, y         | (B·Hv, T, Dv) family dtype each, y gets one round-to-nearest-even per element                                                                |
## | family dtype | one compile-time element type, fp16 primary, bf16 the range-robust fallback (`kdaPrefillChunkScan`'s `Fam` generic)                          |
## | head mapping | value head bh reads key head `(bh mod Hv) div hkRatio + (bh div Hv)·Hk`, hkRatio = Hv div Hk                                                 |
## | chunk axis   | tokens are walked in chunks of ChunkC, the u solve sequential in t inside a chunk, chunks sequential on the register state                   |
## | decay / q̃   | exp2(cumulogdecay·log2e) per channel in-device, log2e is the shared `math_consts.Log2e`, q̃ = per-element division by the runtime f32 qScale |
##
## | contract                  | value                                                                                                                             |
## | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
## | cumulogdecay precondition | finite and monotone non-increasing per key per chunk (host prefix of g, terms ≤ 0), a rising cumulogdecay overflows the f32 state |

##
## | provenance | source                                                                                                                                                                |
## | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | schedule   | the naive WY/UT reference `kdaPrefillChunked` in workspace/positron/tests/naive/naive_kda.nim, the same per-channel cumulogdecay, pairdecay, solve and carry formulas |
## | tiles      | the GDN chunk-scan tile schedule of attn_ssm/gated_delta_net_prefill.nim, applied to the KDA per-channel decay chain                                                  |

##
## Implementation shape:
## - each lane computes its own state row's scalars (the solve, y, the u contributions),
##   so the u vectors live in a per-lane local array, no inter-threadgroup data movement needed
## - the per-token decay dT = exp2(cumulogdecay_t·log2e) folds into the carry reads once
##   per token, over the same broadcast Tile ops as the per-channel factors
## - the pair decay exp2((cumulogdecay_t − cumulogdecay_s)·log2e) folds into the A/B dots per
##   (t, s) pair, the chunk-end decay into the carry
##
## Pair decay's difference form removes the dT·invd_s overflow:
##
## - exp2(x)·exp2(y) = exp2(x + y)
## - cumulogdecay decreases along t, so the argument is ≤ 0 and no intermediate exceeds 1
## - the factorized spelling dT·invd_s overflows exp2 once |cumulogdecay_s| ≳ 88.7,
##   the resulting Inf × dT → 0 product NaNs the carry and the persistent state
##
## - the k·k and q̃·k dot products run as one broadcast-tile pass per (t, s) pair,
##   every lane reads the same row-identical row sum, lanes agree bit-exactly
## - the pair (t, s) passes reload the key and cumulogdecay tiles per pair, chunks of 64 stay inside
##   the register budget (per-lane state and u arrays are the only residents, no (ChunkC, Dk) working tile)
##
##   per chunk:   cumulogdecay tiles ─→ per token t: dT, G_t read ─→ u_t solve ─→ y_t store
##                (t in token order)                           └─────────┐
##            └──→ S ← dEnd ⊙ S_carry + Σ_s (dEnd·invd_s ⊙ k_s) [x] u_s
##
##
## Entries are consumer-side:
## - a `metal:` block wraps the grid-driven proc with concrete static (Dk, Dv, ChunkC),
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

from ../../../math_consts import Log2e
import workspace/crucible
import workspace/ceramic

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Core tile procs (inline-tile property) ──────────────────────────

proc kdaPrefillChunkScanAt*[Fam: bfloat16|float16](
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[Fam],           # (B·Hv, T, Dv) family dtype
    k: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    q: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    cumulogdecay: ptr UncheckedArray[float32],    # (B·Hk, T, Dk) f32 per-channel cumulative log decay
    v: ptr UncheckedArray[Fam],           # (B·Hv, T, Dv) family dtype
    beta: ptr UncheckedArray[float32],    # (B·Hv, T) f32 beta
    qScale: float32,                      # √Dk, host-computed f64→f32 cast
    Hv, Hk, hkRatio, T: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## One (bh, TileR-row) state tile of the chunked KDA prefill scan, the whole
  ## token sequence walked over the register state at the caller's coordinates
  ## (the family dtype a compile-time generic over bf16/fp16)
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
  ## - the family-dtype y round per element is the only dtype-dependent step,
  ##   the tile walk is dtype-mechanical
  ##
  ## - precondition, Hk > 0, Hv an exact multiple of Hk and hkRatio = Hv div Hk
  ##
  ##   per token:   dT ─→ G_t, H_t reads ─→ u_t solve ─→ y_t store
  ##   chunk end   S ← dEnd ⊙ S_carry + Σ_s (dEnd·invd_s ⊙ k_s) [x] u_s
  ##
  ## - `bh` is the (sequence, value head) row block, `dvBlock` the Dv/TileR row block
  ## - grid-driven wrapper, receiving the threadgroup coordinates from the grid
  ## - generic only over the family dtype and the static shape, every (Dk, Dv, ChunkC)
  ##   binding needs its own call-site line
  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store covers one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
    doAssert ChunkC <= 64, "the per-lane u local array is sized by ChunkC"
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
  let glCumulogdecay = cumulogdecay.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))

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
      var cumulogdecayT: rt_l(float32, TileR, Dk)
      cumulogdecayT.loadTile(glCumulogdecay, (kLinT, 0, 0, 0))

      # Per-token decay dT = exp2(cumulogdecay_t·log2e), one factor per key channel,
      # the exp2 form (see the module doc).
      var dT: rt_l(float32, TileR, Dk)
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            dT.frags[n][m].frag[f] = cumulogdecayT.frags[n][m].frag[f]
      dT.mul(dT, Log2e)
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
        var cumulogdecayS: rt_l(float32, TileR, Dk)
        cumulogdecayS.loadTile(glCumulogdecay, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        # pairdecay(t, s)[dk] = exp2((cumulogdecay_t[dk] − cumulogdecay_s[dk])·log2e) per key channel,
        # the difference form (dT·exp2(−cumulogdecay_s·log2e) in algebra).
        #
        # - the argument stays ≤ 0 (cumulogdecay decreases along t), no intermediate
        #   exceeds 1, exp2 cannot overflow
        # - the factorized spelling dT·exp2(−cumulogdecay_s·log2e) overflows exp2 once
        #   |cumulogdecay_s| ≳ 88.7, the resulting Inf × dT → 0 product NaNs the carry
        #   and the persistent state
        var pdT: rt_l(float32, TileR, Dk)
        for n in 0 ..< rowTiles:
          for m in 0 ..< colTiles:
            for f in 0 ..< vpt:
              pdT.frags[n][m].frag[f] = exp2(
                (cumulogdecayT.frags[n][m].frag[f] - cumulogdecayS.frags[n][m].frag[f]) * Log2e)
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
        var cumulogdecayS: rt_l(float32, TileR, Dk)
        cumulogdecayS.loadTile(glCumulogdecay, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
        var pdT: rt_l(float32, TileR, Dk)
        # pairdecay(t, s)[dk] = exp2((cumulogdecay_t[dk] − cumulogdecay_s[dk])·log2e) per key channel,
        # the difference form (dT·exp2(−cumulogdecay_s·log2e) in algebra).
        for n in 0 ..< rowTiles:
          for m in 0 ..< colTiles:
            for f in 0 ..< vpt:
              pdT.frags[n][m].frag[f] = exp2(
                (cumulogdecayT.frags[n][m].frag[f] - cumulogdecayS.frags[n][m].frag[f]) * Log2e)
        var qkProd: rt_l(float32, TileR, Dk)
        qkProd.mul(q32, ks32)
        qkProd.mul(qkProd, pdT)
        var qkVec: rv(float32, TileR, Dk)
        qkVec.row_sum(qkProd)
        yVal += qkVec.data[0] * uLoc[sIdx]

      if colIn == 0:
        when Fam is bfloat16:
          y[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)] = yVal.bfloat16
        else:
          y[seqLin + gt * Dv + dvBlock * TileR + int32(rowIn)] = yVal.float16

    # Carry out of the chunk, per key channel:
    # S = dEnd ⊙ S_carry + Σ_s (dEnd·invd_s ⊙ k_s) ⊗ u_s
    let gtEnd = c0 + int32(cLen - 1)
    var cumulogdecayEnd: rt_l(float32, TileR, Dk)
    cumulogdecayEnd.loadTile(glCumulogdecay, (kHeadLin + gtEnd * Dk, 0, 0, 0))
    var dEnd: rt_l(float32, TileR, Dk)
    dEnd.mul(cumulogdecayEnd, Log2e)
    exp2(dEnd, dEnd)
    s.mul(s, dEnd)
    for sIdx in 0 ..< cLen:
      var ks32: rt_l(float32, TileR, Dk)
      ks32.loadTile(glK, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
      var cumulogdecayS: rt_l(float32, TileR, Dk)
      cumulogdecayS.loadTile(glCumulogdecay, (kHeadLin + (c0 + int32(sIdx)) * Dk, 0, 0, 0))
      # pairdecay(end, s)[dk] = exp2((cumulogdecay_end[dk] − cumulogdecay_s[dk])·log2e) per key channel,
      # the difference form (the token pairdecay note carries the overflow bound)
      var pdEnd: rt_l(float32, TileR, Dk)
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            pdEnd.frags[n][m].frag[f] = exp2(
              (cumulogdecayEnd.frags[n][m].frag[f] - cumulogdecayS.frags[n][m].frag[f]) * Log2e)
      pdEnd.mul(pdEnd, ks32)
      let ws = uLoc[sIdx]
      for n in 0 ..< rowTiles:
        for m in 0 ..< colTiles:
          for f in 0 ..< vpt:
            s.frags[n][m].frag[f] = s.frags[n][m].frag[f] + pdEnd.frags[n][m].frag[f] * ws
    c0 += int32(cLen)

  glState.storeTile(s, (headLin, 0, dvBlock, 0))


proc kdaPrefillChunkScan*[Fam: bfloat16|float16](
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[Fam],           # (B·Hv, T, Dv) family dtype
    k: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    q: ptr UncheckedArray[float32],       # (B·Hk, T, Dk) f32, post-l2norm
    cumulogdecay: ptr UncheckedArray[float32],    # (B·Hk, T, Dk) f32 per-channel cumulative log decay
    v: ptr UncheckedArray[Fam],           # (B·Hv, T, Dv) family dtype
    beta: ptr UncheckedArray[float32],    # (B·Hv, T) f32 beta
    qScale: float32,                      # √Dk, host-computed f64→f32 cast
    Hv, Hk, hkRatio, T: int32,
    Dk, Dv, TileR, ChunkC: static int) {.device.} =
  ## Grid-driven form of `kdaPrefillChunkScanAt`, the caller's `metal:` entry wraps this proc.
  ## - grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  kdaPrefillChunkScanAt(state, y, k, q, cumulogdecay, v, beta, qScale, Hv, Hk, hkRatio, T,
    dvBlock, bh, Dk, Dv, TileR, ChunkC)
