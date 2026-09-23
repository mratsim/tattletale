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

## Mixture-of-experts forward on the ceramic Tile API.
## Implements the Glm4MoeLiteMoE routing, the NaiveMoe experts and the shared MLP for the GLM-4.7-Flash dims.
## Experimental: not a production kernel, known gaps below, not fixed.
##
## Dataflow per token (fp32 arithmetic over fp16-rounded inputs):
##
##   router:  x --> router_w @ --> logits (64) --> sigmoid --> s
##            top-4 of s (lowest-index tiebreak) --> w = s/(sum(w)+1e-20)·1.8
##   experts: per slot e = top4[slot]:
##            x --> gate_up_w[e] @ --> (gHalf, uHalf)
##                  --> h = silu(gHalf)·uHalf (1536) --> fp16 round
##                  --> h_scratch[t, slot]
##            routed = Σ_slot w[slot] · (down_w[e] @ h_scratch[t, slot])
##   shared:  x --> shared_gate_up_w @ --> (gs, us)
##                  --> hs = silu(gs)·us (1536) --> fp16 round
##                  --> hs_scratch[t]
##            out_r[t] = fp16(routed + shared_down_w @ hs_scratch[t])
##
## The model's group selection (n_group = 1, topk_group = 1, zero bias)
## selects all 64 experts, so the group step is a no-op.
## The top-4 runs directly over the sigmoid scores.
##
## `moe_fwd_generic` below drives the same chain from runtime model-config
## values. Ragged tails run through `loadTileBounded`/`storeTileMasked`.
##
## Buffers:
##   - out_r: (num_tokens, 2048) fp16 routed + shared output
##   - x: (num_tokens, 2048) fp16
##   - router_w: (64, 2048) fp16
##   - gate_up_w: (64, 3072, 2048) fp16 fused g|up weight (g 0:1536, up 1536:3072)
##   - down_w: (64, 2048, 1536) fp16
##   - shared_gate_up_w: (3072, 2048) fp16
##   - shared_down_w: (2048, 1536) fp16
##   - h_scratch: (num_tokens, 4, 1536) fp16 working buffer
##   - hs_scratch: (num_tokens, 1536) fp16 working buffer
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
import math_consts
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# The real GLM-4.7-Flash dims, baked as module constants.
#
# - the kernel is non-generic, so tile types spell the atoms explicitly
# - default atoms (rt_l/rv without an atom argument) need a backend tag
#   for getTileConfig, which crucibleSetBackend makes resolvable
# - on Metal those defaults resolve to the Apple simdgroup atoms,
#   whose codegen (simdgroup_multiply_accumulate)
#   differs from the universal arithmetic atoms spelled below
#
# Spelled members stay so generated kernels remain byte-identical.


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
  ## activation with one fp16 RNE round (the h_scratch contract).
  ## The silu is fp32 from the fp32 gHalf operand, g / (1 + exp2(−g·log2e)),
  ## and the product is fp32 with one fp16 round at the end.
  ## The frag walk follows the loadTile lane→element mapping, so the operands agree elementwise.
  const rowTiles = 32 div A.getM()
  const colTiles = 32 div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let g = gHalf.frags[n][m].frag[v]
        let s = g / (1.0'f32 + exp2(-g * Log2e))
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
  ## `dst[r][c] = fp16(routed[r][c] + shared[r][c])`: the output's one fp16 RNE round.
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
  ## lanes. The row-0 logits live in the accumulator's lanes
  ## {0, 1, 8, 9} (2 per col-frag). Destination lane d pulls its pair
  ## from the source lane `d and 9` (the row-0 owner of the same col
  ## pair). The lane→expert mapping follows the universal AC fragment
  ## layout of both tiles (a wrong mapping shows up as a routing
  ## mismatch, not a value error).
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
        1.0'f32 / (1.0'f32 + exp2(-g0 * Log2e))
      scores.frags[0][0].frag[1] =
        1.0'f32 / (1.0'f32 + exp2(-g1 * Log2e))

proc topk4[A: static MmaAtom](
    scores: RtLeft[float32, 8, 8, A],
    top4: var array[4, int32],
    w: var array[4, float32]) {.device.} =
  ## Selects the 4 largest router scores of the (8, 8) score tile:
  ## the top-4 of the sigmoid values with the lowest-index tiebreak.
  ## The weights come from the original tile. The selection runs on
  ## a masked copy. Per pass: the local max of the lane's 2 scores,
  ## a 5-step simdShuffleDown max tree (deltas 16, 8, 4, 2, 1)
  ## broadcast from lane 0, the candidate expert indices where
  ## score == max (no match = 64) reduced by a 5-step min tree
  ## broadcast from lane 0, and the found expert masked to −inf in
  ## the selection copy.
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
  ## Computes the module doc's contract for one token:
  ##   - the router GEMV + in-register top-4
  ##   - the 4 per-slot expert activations into h_scratch
  ##   - the shared expert activation into hs_scratch
  ##   - the weighted routed + shared down projections
  ##   - the fp16 output store
  ##
  ## Grid (num_tokens, 1, 1), 32 lanes, one token per threadgroup.
  ## The expert sets differ per token, so a shared-B tile across rows
  ## is impossible. Every tile load/store carries the token
  ## or the h_scratch row in the origin's batch component. The tile's
  ## 32 plane rows have only row 0 real, and origin[2] steps in 32-row
  ## units.
  ##
  ## GEMMs: 32-row tiles loaded row-bounded to 1 (only row 0 real),
  ## 16-wide K-steps (128 over 2048) with mma_AB into fp32
  ## accumulators. The router GEMV fills a (32, 64) accumulator.
  ## The 64 row-0 logits redistribute into an (8, 8) score tile,
  ## 2 values per lane. The top-4 runs as 4 passes over the tile
  ## (the gatherScores and topk4 device procs).
  ## Register budget: ~2 live 32×32 fp32 accumulators (the gHalf/uHalf pair) plus transients.
  ## The h intermediates round to fp16 and land in the working buffers, which are not caller padding.
  let t = int32(threadgroup_position_in_grid.x)

  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (2048, 0, 2048, 1))
  let glOut = out_r.gd(shape = (-1, -1, -1, -1), stride = (2048, 0, 2048, 1))
  let glRouter = router_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 2048, 1))
  let glGu = gate_up_w.gd(shape = (-1, -1, -1, -1), stride = (GateUpOut * 2048, 0, 2048, 1))
  let glDown = down_w.gd(shape = (-1, -1, -1, -1), stride = (2048 * 1536, 0, 1536, 1))
  let glSgu = shared_gate_up_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 2048, 1))
  let glSd = shared_down_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1536, 1))
  let glH = h_scratch.gd(shape = (-1, -1, -1, -1), stride = (1536, 1536, 1536, 1))
  let glHs = hs_scratch.gd(shape = (-1, -1, -1, -1), stride = (1536, 0, 1536, 1))

  var dR: rt_l(float32, 32, NumRoutedExperts, UNIVERSAL_8x8x8_F32F16F16F32)
  var a: rt_l(float16, 32, 16, UNIVERSAL_8x8x8_F32F16F16F32)
  var b16: rt_r(float16, 16, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var b64: rt_r(float16, 16, NumRoutedExperts, UNIVERSAL_8x8x8_F32F16F16F32)
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
  for slot in 0 ..< TopK:
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
    for slot in 0 ..< TopK:
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

# ═════════════════════════════════════════════════════════════════════
#  The runtime-dims generic entry
#  ═════════════════════════════════════════════════════════════════════

## | aspect         | contract                                                 |
## | ---------------- |
## | shape regime   | runtime tile counts over fixed tile shapes, masked tails |
## | kernel config  | the model config's values as runtime arguments           |
## | register tiles | the compiled-in fixed max, ragged tails bounded/masked   |
## | tile-internal  | frag walks over the fixed register tiles, static bounds  |
##
## Compiled-in fixed maxima, a config beyond one stops and reports:
##   - 64 experts per score chunk, at most 8 chunks, n_routed_experts <= 512
##   - at most MaxTopK routing slots
##
## GLM-4.7-Flash's baked `moe_fwd` constants above are one config point of this function.
## With those values the two entries produce the same output,
## verified bit-exact by the moe suite.

const
  ScoreChunk* = 64         # experts per router score chunk, the (32, 64) accumulator's width
  ScoreChunks* = 8         # compiled-in chunk max of the (8, 64) score tile
  MaxTopK* = 8             # compiled-in routing-slot max
  ActSilu* = 0'i32         # activation = silu(g) · u
  ActGeluTanh* = 1'i32     # activation = gelu_pytorch_tanh(g) · u

const
  InvSqrt2Pi = 0.7978845608028654'f32   # 1/sqrt(2·pi), the gelu_pytorch_tanh factor
  GeluCoef = 0.044715'f32               # the gelu_pytorch_tanh cubic coefficient

proc actMul16[A: static MmaAtom](
    dst: var RtLeft[float16, 32, 32, A],
    gHalf, uHalf: RtLeft[float32, 32, 32, A],
    activation: int32) {.device.} =
  ## `dst[r][c] = fp16(act(gHalf[r][c]) · uHalf[r][c])`, the expert
  ## activation with one fp16 RNE round (the h_scratch contract),
  ## the activation picked at runtime.
  ##
  ## - ActSilu, g / (1 + exp2(−g·log2e)), the baked `siluMul16` form
  ## - ActGeluTanh, 0.5·g·(1 + tanh(s)), s = InvSqrt2Pi·(g + GeluCoef·g³)
  ##
  ## The gelu tanh evaluates through exp2, tanh(s) = 1 − 2/(e²ˢ+1),
  ## stable at both saturation ends, fp32 end to end like the silu variant.
  ##
  ## The frag walk follows the loadTile lane→element mapping, so the operands agree elementwise.
  ## Tile-internal, the walk bounds stay static.
  const rowTiles = 32 div A.getM()
  const colTiles = 32 div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let g = gHalf.frags[n][m].frag[v]
        var a: float32
        if activation == ActSilu:
          a = g / (1.0'f32 + exp2(-g * Log2e))
        else:
          let s = InvSqrt2Pi * (g + GeluCoef * g * g * g)
          let th = 1.0'f32 - 2.0'f32 / (exp2(2.0'f32 * s * Log2e) + 1.0'f32)
          a = 0.5'f32 * g * (1.0'f32 + th)
        dst.frags[n][m].frag[v] = (a * uHalf.frags[n][m].frag[v]).to(float16)

proc gatherSigmoidScores[A, AL: static MmaAtom](
    scores: var RtLeft[float32, 8, ScoreChunk * ScoreChunks, A],
    chunk: RtLeft[float32, 32, ScoreChunk, AL],
    chunkIndex, eCount: int32) {.device.} =
  ## Gathers one 64-expert chunk's row-0 router logits into the score
  ## tile's col-frag `chunkIndex`, the sigmoid of each logit.
  ##
  ## - element (r, 8·chunkIndex + c) holds the sigmoid of expert
  ##   ScoreChunk·chunkIndex + 8r + c, 2 values per lane across all 32 lanes
  ## - expert indices at or beyond `eCount`, the last chunk's tail, are
  ##   set to −float32 max so they can never win the top-K
  ##
  ## The row-0 logits live in the accumulator's lanes {0, 1, 8, 9},
  ## 2 per col-frag, the destination lane pulls its pair from source lane
  ##
  ## `d and 9`, the row-0 owner of the same col pair under the universal
  ## AC fragment layout.
  static:
    doAssert AL.getM() == A.getM() and AL.getN() == A.getN() and
      AL.getVpt() == A.getVpt(),
      "gatherSigmoidScores: both atoms must share the lane→fragment cell mapping"
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(AL.getLayoutA(), (lane, 0)).toIntVal()
  let r = cell mod 8          # the destination row = expert div 8
  let c0 = cell div 8         # the lane's col pair base
  let e0 = int32(8 * r + c0)  # the lane's experts inside the chunk
  let srcLane = lane and 9    # the row-0 owner of the lane's col pair
  for m in 0 ..< ScoreChunk div 8:
    let g0 = simdShuffle(chunk.frags[0][m].frag[0], uint32(srcLane))
    let g1 = simdShuffle(chunk.frags[0][m].frag[1], uint32(srcLane))
    if m == r:
      scores.frags[0][chunkIndex].frag[0] =
        if e0 < eCount: 1.0'f32 / (1.0'f32 + exp2(-g0 * Log2e))
        else: -3.402823466e38'f32
      scores.frags[0][chunkIndex].frag[1] =
        if e0 + 1 < eCount: 1.0'f32 / (1.0'f32 + exp2(-g1 * Log2e))
        else: -3.402823466e38'f32

proc topkRouted[A: static MmaAtom](
    scores: RtLeft[float32, 8, ScoreChunk * ScoreChunks, A],
    topK, eCount: int32,
    ids: var array[8, int32],   # MaxTopK slots, literal per the array-length resolver
    w: var array[8, float32]) {.device.} =
  ## Selects the `topK` largest scores of the (8, ScoreChunk·ScoreChunks)
  ## score tile, the lowest-index tiebreak top-K, the weights read back
  ## from the original tile. The selection runs on a masked copy.
  ##
  ## | pass step | mechanism                                                         |
  ## | ----------- |
  ## | max       | the tile-wide max, a 5-step simdShuffleDown tree (16, 8, 4, 2, 1) |
  ## | candidate | expert indices where score == max, 5-step min tree, lowest index  |
  ## | mask      | the found expert set to −float32 max in the selection copy        |
  ## | cell      | element (r, 8m + c) holds expert ScoreChunk·m + 8r + c            |
  ##
  ## An unmatched pass, a NaN/Inf-poisoned score comparing false against
  ## everything so no score equals the max, routes the slot to expert
  ## eCount − 1 with zero weight, the ids stay in [0, eCount).
  ##
  ## The downstream expert-row reads stay in bounds.
  var sel: RtLeft[float32, 8, ScoreChunk * ScoreChunks, A]
  for m in 0 ..< ScoreChunks:
    sel.frags[0][m].frag[0] = scores.frags[0][m].frag[0]
    sel.frags[0][m].frag[1] = scores.frags[0][m].frag[1]
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let r = cell mod 8
  let c0 = cell div 8
  for slot in 0'i32 ..< topK:
    var lm = max(sel.frags[0][0].frag[0], sel.frags[0][0].frag[1])
    for m in 1 ..< ScoreChunks:
      lm = max(lm, sel.frags[0][m].frag[0])
      lm = max(lm, sel.frags[0][m].frag[1])
    lm = max(lm, simdShuffleDown(lm, 16'u32))
    lm = max(lm, simdShuffleDown(lm, 8'u32))
    lm = max(lm, simdShuffleDown(lm, 4'u32))
    lm = max(lm, simdShuffleDown(lm, 2'u32))
    lm = max(lm, simdShuffleDown(lm, 1'u32))
    lm = simdShuffle(lm, 0'u32)
    var localCand = int32(1 shl 30)
    for m in 0 ..< ScoreChunks:
      let e0 = int32(ScoreChunk * m + 8 * r + c0)
      if sel.frags[0][m].frag[0] == lm:
        localCand = min(localCand, e0)
      if sel.frags[0][m].frag[1] == lm:
        localCand = min(localCand, e0 + 1)
    var cand = min(localCand, simdShuffleDown(localCand, 16'u32))
    cand = min(cand, simdShuffleDown(cand, 8'u32))
    cand = min(cand, simdShuffleDown(cand, 4'u32))
    cand = min(cand, simdShuffleDown(cand, 2'u32))
    cand = min(cand, simdShuffleDown(cand, 1'u32))
    cand = simdShuffle(cand, 0'u32)
    if cand >= ScoreChunk * ScoreChunks:
      ids[slot] = eCount - 1
      w[slot] = 0.0'f32
    else:
      ids[slot] = cand
    let mSel = cand div ScoreChunk
    let rest = cand mod ScoreChunk
    let rw = rest div 8
    let cw = rest mod 8
    let own = (cw div 2 mod 2) + 2 * (rw mod 2) + 4 * ((rw div 2) mod 2) +
              8 * (cw div 4 mod 2) + 16 * ((rw div 4) mod 2)
    if cand < ScoreChunk * ScoreChunks:
      let w0 = simdShuffle(scores.frags[0][mSel].frag[0], uint32(own))
      let w1 = simdShuffle(scores.frags[0][mSel].frag[1], uint32(own))
      w[slot] = if (cw mod 2) == 0: w0 else: w1
      for m in 0 ..< ScoreChunks:
        let e0 = int32(ScoreChunk * m + 8 * r + c0)
        if e0 == cand:
          sel.frags[0][m].frag[0] = -3.402823466e38'f32
        if e0 + 1 == cand:
          sel.frags[0][m].frag[1] = -3.402823466e38'f32

proc moe_fwd_generic*(
    out_r: ptr UncheckedArray[float16],              # (num_tokens, hidden) fp16 output
    x: ptr UncheckedArray[float16],                  # (num_tokens, hidden) fp16 activations
    router_w: ptr UncheckedArray[float16],           # (n_routed_experts, hidden) fp16 router weight
    gate_up_w: ptr UncheckedArray[float16],          # (n_routed_experts, 2·moe_intermediate, hidden) fp16
    down_w: ptr UncheckedArray[float16],             # (n_routed_experts, hidden, moe_intermediate) fp16
    shared_gate_up_w: ptr UncheckedArray[float16],   # (n_shared_experts, 2·moe_intermediate, hidden) fp16
    shared_down_w: ptr UncheckedArray[float16],      # (n_shared_experts, hidden, moe_intermediate) fp16
    h_scratch: ptr UncheckedArray[float16],          # (num_tokens, top_k, moe_intermediate) fp16 buffer
    hs_scratch: ptr UncheckedArray[float16],         # (num_tokens, n_shared_experts, moe_intermediate) fp16 buffer
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts: int32,
    routed_scaling: float32,
    activation: int32) {.device.} =
  ## Runtime model-config values drive the module doc's chain through this entry.
  ##
  ## Routing, the GLM sigmoid skeleton.
  ##
  ##   logits → sigmoid → top-K (lowest-index tiebreak)
  ##          → w = s/(sum(w)+1e-20)·routed_scaling
  ##
  ## Expected input, per model config row:
  ##
  ## | tensor fp16      | shape                                                                                |
  ## | ------------------ |
  ## | x                | (num_tokens, hidden), one token per threadgroup                                      |
  ## | router_w         | (n_routed_experts, hidden)                                                           |
  ## | gate_up_w        | (n_routed_experts, 2·moe_intermediate, hidden) fused g/up, g half 0:moe_intermediate |
  ## | down_w           | (n_routed_experts, hidden, moe_intermediate)                                         |
  ## | shared_gate_up_w | (n_shared_experts, 2·moe_intermediate, hidden) fused g/up                            |
  ## | shared_down_w    | (n_shared_experts, hidden, moe_intermediate)                                         |
  ## | h_scratch        | (num_tokens, top_k, moe_intermediate) working buffer                                 |
  ## | hs_scratch       | (num_tokens, n_shared_experts, moe_intermediate) working buffer                      |
  ##
  ## The scalars num_tokens, hidden, n_routed_experts, moe_intermediate,
  ## top_k, n_shared_experts are int32, routed_scaling is float32,
  ## activation is ActSilu or ActGeluTanh.
  ##
  ## A config beyond the compiled-in fixed maxima, n_routed_experts >
  ## ScoreChunk·ScoreChunks (512) or top_k > MaxTopK, stops before launch.
  ##
  ## Output:
  ##
  ## | output    | value                                                                                                              |
  ## | --------- | ------------------------------------------------------------------------------------------------------------------ |
  ## | h_scratch | h_scratch[t, slot] = fp16(act(g)·u), hs_scratch[t, s] for shared expert s                                          |
  ## | out_r     | out_r[t] = fp16(Σ_slot w[slot]·(down_w[ids[slot]] @ h_scratch[t, slot]) + Σ_s shared_down_w[s] @ hs_scratch[t, s]) |
  ##
  ## Ragged-native over the runtime dims:
  ##   - the K walks run ceil(dim / tileK) steps, each load bounded
  ##     to the raw dim, out-of-range lanes hold the zero fill, a zero
  ##     operand leaves the mma accumulator untouched
  ##   - the N walks over moe_intermediate and hidden run ceil(dim / 32) tiles,
  ##     the stores masked at the real extent
  ##   - the router score chunks run ceil(n_routed_experts / ScoreChunk) passes,
  ##     the last chunk's tail experts masked to −float32 max
  ##
  ## Grid (num_tokens, 1, 1) at 32 lanes, one token per threadgroup,
  ## the baked `moe_fwd`'s geometry. Register budget near 2 live 32×32
  ## fp32 accumulators (the gHalf/uHalf pair) plus transients.
  let t = int32(threadgroup_position_in_grid.x)

  let gateUpOut = 2 * moe_intermediate
  let kSteps = (hidden + 15) div 16
  let iHalfTiles = (moe_intermediate + 31) div 32
  let hTiles = (hidden + 31) div 32
  let iSteps = (moe_intermediate + 15) div 16
  let scoreChunks = (n_routed_experts + ScoreChunk - 1) div ScoreChunk

  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (hidden, 0, hidden, 1))
  let glOut = out_r.gd(shape = (-1, -1, -1, -1), stride = (hidden, 0, hidden, 1))
  let glRouter = router_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, hidden, 1))
  let glGu = gate_up_w.gd(shape = (-1, -1, -1, -1), stride = (gateUpOut * hidden, moe_intermediate * hidden, hidden, 1))
  let glDown = down_w.gd(shape = (-1, -1, -1, -1), stride = (hidden * moe_intermediate, 0, moe_intermediate, 1))
  let glSgu = shared_gate_up_w.gd(shape = (-1, -1, -1, -1), stride = (gateUpOut * hidden, moe_intermediate * hidden, hidden, 1))
  let glSd = shared_down_w.gd(shape = (-1, -1, -1, -1), stride = (hidden * moe_intermediate, 0, moe_intermediate, 1))
  let glH = h_scratch.gd(shape = (-1, -1, -1, -1), stride = (moe_intermediate, moe_intermediate, moe_intermediate, 1))
  let glHs = hs_scratch.gd(shape = (-1, -1, -1, -1), stride = (n_shared_experts * moe_intermediate, moe_intermediate, moe_intermediate, 1))

  var dR: rt_l(float32, 32, ScoreChunk, UNIVERSAL_8x8x8_F32F16F16F32)
  var a: rt_l(float16, 32, 16, UNIVERSAL_8x8x8_F32F16F16F32)
  var b16: rt_r(float16, 16, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var b64: rt_r(float16, 16, ScoreChunk, UNIVERSAL_8x8x8_F32F16F16F32)
  var scores: rt_l(float32, 8, ScoreChunk * ScoreChunks, UNIVERSAL_8x8x8_F32F32F32F32)
  var ids: array[8, int32]      # MaxTopK slots, literal per the array-length resolver
  var w: array[8, float32]
  var gHalf: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var uHalf: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var h16: rt_l(float16, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var routed: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var d: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var sh: rt_l(float32, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var out16: rt_l(float16, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)

  # ── router GEMV per 64-expert chunk + sigmoid top-K ──
  for m in 0 ..< ScoreChunks:
    scores.frags[0][m].frag[0] = -3.402823466e38'f32
    scores.frags[0][m].frag[1] = -3.402823466e38'f32
  for cs in 0'i32 ..< scoreChunks:
    dR.zero()
    for kk in 0'i32 ..< kSteps:
      a.loadTileBounded(glX, (t, 0, 0, kk), 1, hidden)
      b64.loadTileBounded(glRouter, (0, 0, cs, kk), n_routed_experts, hidden)
      dR.mma_AB(a, b64)
    scores.gatherSigmoidScores(dR, cs, n_routed_experts)
  scores.topkRouted(top_k, n_routed_experts, ids, w)
  var sumW = 0.0'f32
  for slot in 0'i32 ..< top_k:
    sumW = sumW + w[slot]
  sumW = sumW + 1e-20'f32
  for slot in 0'i32 ..< top_k:
    w[slot] = w[slot] / sumW * routed_scaling

  # ── shared expert activations: hs[s] = act(gs)·us -> hs_scratch[t, s] ──
  for s in 0'i32 ..< n_shared_experts:
    for nt in 0'i32 ..< iHalfTiles:
      gHalf.zero()
      uHalf.zero()
      for kk in 0'i32 ..< kSteps:
        a.loadTileBounded(glX, (t, 0, 0, kk), 1, hidden)
        b16.loadTileBounded(glSgu, (s, 0, nt, kk), moe_intermediate, hidden)
        gHalf.mma_AB(a, b16)
        b16.loadTileBounded(glSgu, (s, 1, nt, kk), moe_intermediate, hidden)
        uHalf.mma_AB(a, b16)
      h16.actMul16(gHalf, uHalf, activation)
      glHs.storeTileMasked(h16, (t, s, 0, nt), 1,
        min(moe_intermediate - nt * 32, 32'i32))

  # ── the top_k routed expert activations -> h_scratch[t, slot] ──
  for slot in 0'i32 ..< top_k:
    let e = ids[slot]
    for nt in 0'i32 ..< iHalfTiles:
      gHalf.zero()
      uHalf.zero()
      for kk in 0'i32 ..< kSteps:
        a.loadTileBounded(glX, (t, 0, 0, kk), 1, hidden)
        b16.loadTileBounded(glGu, (e, 0, nt, kk), moe_intermediate, hidden)
        gHalf.mma_AB(a, b16)
        b16.loadTileBounded(glGu, (e, 1, nt, kk), moe_intermediate, hidden)
        uHalf.mma_AB(a, b16)
      h16.actMul16(gHalf, uHalf, activation)
      glH.storeTileMasked(h16, (t * top_k + slot, 0, 0, nt), 1,
        min(moe_intermediate - nt * 32, 32'i32))

  # ── output: routed = Σ w[slot]·down_w[e] @ h, + shared, fp16 store ──
  for nt in 0'i32 ..< hTiles:
    routed.zero()
    for slot in 0'i32 ..< top_k:
      d.zero()
      for kk in 0'i32 ..< iSteps:
        a.loadTileBounded(glH, (t * top_k + slot, 0, 0, kk), 1, moe_intermediate)
        b16.loadTileBounded(glDown, (ids[slot], 0, nt, kk), hidden, moe_intermediate)
        d.mma_AB(a, b16)
      routed.accScale(d, w[slot])
    sh.zero()
    for s in 0'i32 ..< n_shared_experts:
      for kk in 0'i32 ..< iSteps:
        a.loadTileBounded(glHs, (t, s, 0, kk), 1, moe_intermediate)
        b16.loadTileBounded(glSd, (s, 0, nt, kk), hidden, moe_intermediate)
        sh.mma_AB(a, b16)
    out16.addStore16(routed, sh)
    glOut.storeTileMasked(out16, (t, 0, 0, nt), 1,
      min(hidden - nt * 32, 32'i32))
