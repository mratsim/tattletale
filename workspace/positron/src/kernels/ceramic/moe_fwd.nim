## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Mixture-of-experts forward (moe_fwd): Tile API port
#
# ############################################################

## Mixture-of-experts forward on the ceramic Tile API: the
## Glm4MoeLiteMoE routing + NaiveMoe experts + shared MLP for the
## GLM-4.7-Flash dims. Experimental: not a production kernel, known
## gaps below, not fixed.
##
## Per token t, fp32 arithmetic over fp16-rounded inputs:
##
##   logits = router_w @ x                     (64)
##   s      = sigmoid(logits)                  (in-kernel exp)
##   top4   = the 4 largest s, lowest-index tiebreak
##   w      = s[top4]
##   w      = w / (sum(w) + 1e-20) · 1.8
##   per slot e = top4[slot]:
##     gHalf, uHalf = gate_up_w[e] @ x         (gate cols 0:1536, up cols 1536:3072)
##     h            = silu(gHalf) · uHalf      (1536), fp16 round -> h_scratch[t, slot]
##   routed = Σ_slot w[slot] · (down_w[e] @ h_scratch[t, slot])
##   gs, us = shared_gate_up_w @ x
##   hs     = silu(gs) · us (fp16 round -> hs_scratch[t])
##   out_r[t] = fp16(routed + shared_down_w @ hs_scratch[t])
##
## The model's group gating (n_group = 1, topk_group = 1, zero bias)
## selects all 64 experts, so the group step is a no-op and folds into
## the plain top-4 over the sigmoid scores.
##
## Routing: the router GEMV fills a (32, 64) accumulator with only
## row 0 real (the exl3 MMODE-0 trick: the 32-row x tiles load
## row-bounded to 1). The 64 row-0 logits are redistributed into an
## (8, 8) score tile, 2 values per lane across the 32 lanes, then the
## top-4 runs as 4 passes over the tile: local max of the lane's 2
## scores, a 5-step simdShuffleDown max tree, the holding expert
## (lowest index on ties), and a −inf mask in the selection copy (the
## original sigmoid values stay for the weights).
##
## Grid (num_tokens, 1, 1), 32 lanes, ONE token per threadgroup: the
## expert sets differ per token, so a shared-B tile across rows is
## impossible. Every tile load/store carries the token in the origin's
## batch component (the tile's 32 plane rows have only row 0 real, and
## origin[2] steps in 32-row units).
##
## GEMMs: the exl3_gemv MMODE-0 trick (32-row tiles, `loadTileRows`
## rowLimit 1), 16-wide K-steps (HiddenDim div 16 = 128 over 2048)
## with mma_AB into fp32 accumulators. Register budget: at most ~2
## live 32×32 fp32 accumulators (the gHalf/uHalf pair) plus
## transients. The h intermediates round to fp16 and land in the
## working buffers (h_scratch (num_tokens, 4, 1536), hs_scratch
## (num_tokens, 1536)). They are NOT caller padding or a ragged
## strategy.
##
## Local device procs: siluMul16 (the expert activation), accScale
## (the weighted routed accumulation), addStore16 (the output round),
## gatherScores (the logit redistribution) and topk4 (the register
## selection) live in this module. The row-bounded load/store come
## from tile_io_rows (positron local extension).
##
## Known production gaps (documented, not fixed):
##   - one token per threadgroup: no expert-batched B tiles, no x
##     reuse across the per-slot projections (x re-read from global
##     per N-tile)
##   - the router weight is fp16 (the reference router is fp32)
##   - the top-4 is a fixed 4-pass register selection, no score
##     sorting output

import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# The real GLM-4.7-Flash dims, baked as module constants. The kernel
# is non-generic: its tile types cannot take the rt_l/rv default atoms
# (those call getTileConfig, which asserts a metal:/cuda: block context,
# and a non-generic proc body is typechecked on the host import). The
# explicit universal-atom enum members below are exactly what the
# defaults resolve to on the Metal and CUDA backends.

const
  HiddenDim = 2048          # the model hidden size
  NumRoutedExperts = 64     # n_routed_experts
  MoeIntermediate = 1536    # moe_intermediate_size, also the shared MLP width
  TopK = 4                  # num_experts_per_tok
  GateUpOut = 3072          # 2 · MoeIntermediate, the fused g|u width
  RoutedScaling = 1.8'f32   # routed_scaling_factor

# ═════════════════════════════════════════════════════════════════════
#  Local device extensions: the routing and activation arithmetic
#  ═════════════════════════════════════════════════════════════════════

proc siluMul16[A: static MmaAtom](
    dst: var RtLeft[float16, 32, 32, A],
    gHalf, uHalf: RtLeft[float32, 32, 32, A]) {.device.} =
  ## `dst[r][c] = fp16(silu(gHalf[r][c]) · uHalf[r][c])`: the expert
  ## activation with ONE fp16 RNE round (the h_scratch contract). The
  ## silu is fp32 from the fp32 gHalf operand (g / (1 + exp2(−g·log2e))),
  ## the product is fp32, one fp16 round at the end. The frag walk
  ## follows the loadTile lane→element mapping, so the operands agree
  ## elementwise.
  const rowTiles = 32 div A.getM()
  const colTiles = 32 div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let g = gHalf.frags[n][m].frag[v]
        let s = g / (1.0'f32 + exp2(-g * 1.4426950408889634'f32))
        dst.frags[n][m].frag[v] = (s * uHalf.frags[n][m].frag[v]).to(float16)

proc accScale[A: static MmaAtom](
    dst: var RtLeft[float32, 32, 32, A],
    src: RtLeft[float32, 32, 32, A],
    s: float32) {.device.} =
  ## `dst[r][c] += s · src[r][c]`: the weighted routed accumulation
  ## over the 4 slots.
  const rowTiles = 32 div A.getM()
  const colTiles = 32 div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] =
          dst.frags[n][m].frag[v] + s * src.frags[n][m].frag[v]

proc addStore16[A: static MmaAtom](
    dst: var RtLeft[float16, 32, 32, A],
    routed, shared: RtLeft[float32, 32, 32, A]) {.device.} =
  ## `dst[r][c] = fp16(routed[r][c] + shared[r][c])`: the output's one
  ## fp16 RNE round.
  const rowTiles = 32 div A.getM()
  const colTiles = 32 div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] =
          (routed.frags[n][m].frag[v] + shared.frags[n][m].frag[v]).to(float16)

proc gatherScores[AL, AS: static MmaAtom](
    scores: var RtLeft[float32, 8, 8, AS],
    logits: RtLeft[float32, 32, 64, AL]) {.device.} =
  ## Redistributes the 64 row-0 router logits of the (32, 64)
  ## accumulator into the (8, 8) score tile: element (r, c) holds
  ## sigmoid(logit of expert 8r + c), 2 values per lane across all 32
  ## lanes. The row-0 logits live in the accumulator's lanes {0, 1, 8,
  ## 9} (2 per col-frag). Destination lane d pulls its pair from the
  ## source lane `d and 9` (the row-0 owner of the same col pair).
  ## The lane→expert mapping follows the universal AC fragment layout
  ## of both tiles (a wrong mapping shows up as a routing mismatch,
  ## not a value error).
  static:
    doAssert AL.getM() == AS.getM() and AL.getN() == AS.getN() and
      AL.getVpt() == AS.getVpt(),
      "gatherScores: both atoms must share the lane→fragment cell mapping"
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(AL.getLayoutA(), (lane, 0)).toIntVal()
  let r = cell mod 8          # the destination row = expert div 8
  let srcLane = lane and 9    # the row-0 owner of the lane's col pair
  for m in 0 ..< 8:
    let g0 = simdShuffle(logits.frags[0][m].frag[0], uint32(srcLane))
    let g1 = simdShuffle(logits.frags[0][m].frag[1], uint32(srcLane))
    if m == r:
      scores.frags[0][0].frag[0] =
        1.0'f32 / (1.0'f32 + exp2(-g0 * 1.4426950408889634'f32))
      scores.frags[0][0].frag[1] =
        1.0'f32 / (1.0'f32 + exp2(-g1 * 1.4426950408889634'f32))

proc topk4[A: static MmaAtom](
    scores: RtLeft[float32, 8, 8, A],
    top4: var array[4, int32],
    w: var array[4, float32]) {.device.} =
  ## Selects the 4 largest router scores of the (8, 8) score tile, the
  ## top-4 of the sigmoid values with the lowest-index tiebreak. The
  ## weights come from the original tile. The selection runs on a
  ## masked copy. Per pass: the local max of the lane's 2 scores, a
  ## 5-step simdShuffleDown max tree (deltas 16, 8, 4, 2, 1)
  ## broadcast from lane 0, the candidate expert indices where
  ## score == max (no match = 64) reduced by a 5-step min tree
  ## broadcast from lane 0, and the found expert masked to −inf in the
  ## selection copy.
  var sel: RtLeft[float32, 8, 8, A]
  sel.frags[0][0].frag[0] = scores.frags[0][0].frag[0]
  sel.frags[0][0].frag[1] = scores.frags[0][0].frag[1]
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let r = cell mod 8
  let c0 = cell div 8
  let e0 = int32(8 * r + c0)
  let e1 = e0 + 1
  for slot in 0 ..< 4:
    var lm = max(sel.frags[0][0].frag[0], sel.frags[0][0].frag[1])
    lm = max(lm, simdShuffleDown(lm, 16'u32))
    lm = max(lm, simdShuffleDown(lm, 8'u32))
    lm = max(lm, simdShuffleDown(lm, 4'u32))
    lm = max(lm, simdShuffleDown(lm, 2'u32))
    lm = max(lm, simdShuffleDown(lm, 1'u32))
    lm = simdShuffle(lm, 0'u32)
    var cand = 64'i32
    if sel.frags[0][0].frag[0] == lm:
      cand = e0
    if sel.frags[0][0].frag[1] == lm:
      cand = min(cand, e1)
    cand = min(cand, simdShuffleDown(cand, 16'u32))
    cand = min(cand, simdShuffleDown(cand, 8'u32))
    cand = min(cand, simdShuffleDown(cand, 4'u32))
    cand = min(cand, simdShuffleDown(cand, 2'u32))
    cand = min(cand, simdShuffleDown(cand, 1'u32))
    cand = simdShuffle(cand, 0'u32)
    top4[slot] = cand
    let rw = int(cand div 8)
    let cw = int(cand mod 8)
    let own = (cw div 2 mod 2) + 2 * (rw mod 2) + 4 * ((rw div 2) mod 2) +
              8 * (cw div 4 mod 2) + 16 * ((rw div 4) mod 2)
    let w0 = simdShuffle(scores.frags[0][0].frag[0], uint32(own))
    let w1 = simdShuffle(scores.frags[0][0].frag[1], uint32(own))
    w[slot] = if (cw mod 2) == 0: w0 else: w1
    if e0 == cand:
      sel.frags[0][0].frag[0] = -3.402823466e38'f32
    if e1 == cand:
      sel.frags[0][0].frag[1] = -3.402823466e38'f32

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc moe_fwd*(
    out_r: ptr UncheckedArray[float16],   # (num_tokens, 2048) fp16 routed+shared output
    x: ptr UncheckedArray[float16],       # (num_tokens, 2048) fp16
    router_w: ptr UncheckedArray[float16],# (64, 2048) fp16
    gate_up_w: ptr UncheckedArray[float16],# (64, 3072, 2048) fp16 (expert, g|up)
    down_w: ptr UncheckedArray[float16],  # (64, 2048, 1536) fp16
    shared_gate_up_w: ptr UncheckedArray[float16], # (3072, 2048) fp16
    shared_down_w: ptr UncheckedArray[float16],    # (2048, 1536) fp16
    h_scratch: ptr UncheckedArray[float16],  # (num_tokens, 4, 1536) fp16 working buffer
    hs_scratch: ptr UncheckedArray[float16], # (num_tokens, 1536) fp16 working buffer
    num_tokens: int32) {.device.} =
  ## Computes the module doc's contract for one token: the router
  ## GEMV + in-register top-4, the 4 per-slot expert activations into
  ## h_scratch, the shared expert activation into hs_scratch, the
  ## weighted routed + shared down projections, and the fp16 output
  ## store. Every tile carries the token (or h_scratch row) in the
  ## origin's batch component.
  let t = int32(threadgroup_position_in_grid.x)

  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (2048, 0, 2048, 1))
  let glOut = out_r.gd(shape = (-1, -1, -1, -1), stride = (2048, 0, 2048, 1))
  let glRouter = router_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 2048, 1))
  let glGu = gate_up_w.gd(shape = (-1, -1, -1, -1), stride = (3072 * 2048, 0, 2048, 1))
  let glDown = down_w.gd(shape = (-1, -1, -1, -1), stride = (2048 * 1536, 0, 1536, 1))
  let glSgu = shared_gate_up_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 2048, 1))
  let glSd = shared_down_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1536, 1))
  let glH = h_scratch.gd(shape = (-1, -1, -1, -1), stride = (1536, 1536, 1536, 1))
  let glHs = hs_scratch.gd(shape = (-1, -1, -1, -1), stride = (1536, 0, 1536, 1))

  var dR: rt_l(float32, 32, 64, UNIVERSAL_8x8x8_F32F16F16F32)
  var a: rt_l(float16, 32, 16, UNIVERSAL_8x8x8_F32F16F16F32)
  var b16: rt_r(float16, 16, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var b64: rt_r(float16, 16, 64, UNIVERSAL_8x8x8_F32F16F16F32)
  var scores: rt_l(float32, 8, 8, UNIVERSAL_8x8x8_F32F32F32F32)
  var top4: array[4, int32]
  var w: array[4, float32]
  var gHalf: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var uHalf: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var h16: rt_l(float16, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var routed: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var d: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var sh: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var out16: rt_l(float16, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)

  # ── router GEMV (64 outputs, row 0 real) + sigmoid top-4 ──
  dR.zero()
  for kk in 0'i32 ..< HiddenDim div 16:
    a.loadTileRows(glX, (t, 0, 0, kk), 1)
    b64.loadTile(glRouter, (0, 0, 0, kk))
    dR.mma_AB(a, b64)
  scores.gatherScores(dR)
  scores.topk4(top4, w)
  var sumW = w[0] + w[1] + w[2] + w[3] + 1e-20'f32
  for slot in 0 ..< 4:
    w[slot] = w[slot] / sumW * RoutedScaling

  # ── shared expert activation: hs = silu(gs) · us -> hs_scratch ──
  for nt in 0'i32 ..< MoeIntermediate div 32:
    gHalf.zero()
    uHalf.zero()
    for kk in 0'i32 ..< HiddenDim div 16:
      a.loadTileRows(glX, (t, 0, 0, kk), 1)
      b16.loadTile(glSgu, (0, 0, nt, kk))
      gHalf.mma_AB(a, b16)
      b16.loadTile(glSgu, (0, 0, nt + MoeIntermediate div 32, kk))
      uHalf.mma_AB(a, b16)
    h16.siluMul16(gHalf, uHalf)
    glHs.storeTileRows(h16, (t, 0, 0, nt), 1)

  # ── the 4 routed expert activations -> h_scratch[t, slot] ──
  for slot in 0 ..< 4:
    for nt in 0'i32 ..< MoeIntermediate div 32:
      gHalf.zero()
      uHalf.zero()
      for kk in 0'i32 ..< HiddenDim div 16:
        a.loadTileRows(glX, (t, 0, 0, kk), 1)
        b16.loadTile(glGu, (top4[slot], 0, nt, kk))
        gHalf.mma_AB(a, b16)
        b16.loadTile(glGu, (top4[slot], 0, nt + MoeIntermediate div 32, kk))
        uHalf.mma_AB(a, b16)
      h16.siluMul16(gHalf, uHalf)
      glH.storeTileRows(h16, (t * 4 + slot, 0, 0, nt), 1)

  # ── output: routed = Σ w[slot]·down_w[e] @ h, + shared, fp16 store ──
  for nt in 0'i32 ..< HiddenDim div 32:
    routed.zero()
    for slot in 0 ..< 4:
      d.zero()
      for kk in 0'i32 ..< MoeIntermediate div 16:
        a.loadTileRows(glH, (t * 4 + slot, 0, 0, kk), 1)
        b16.loadTile(glDown, (top4[slot], 0, nt, kk))
        d.mma_AB(a, b16)
      routed.accScale(d, w[slot])
    sh.zero()
    for kk in 0'i32 ..< MoeIntermediate div 16:
      a.loadTileRows(glHs, (t, 0, 0, kk), 1)
      b16.loadTile(glSd, (0, 0, nt, kk))
      sh.mma_AB(a, b16)
    out16.addStore16(routed, sh)
    glOut.storeTileRows(out16, (t, 0, 0, nt), 1)
