# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ──────────────────  moe_router (the Qwen softmax routing op)  ──────────────────

## Qwen3.5/3.6 MoE router on the ceramic Tile API, the softmax routing
## form of the qwen35_moe mega decode kernel. The GLM sigmoid form lives
## in `ffn_moe.nim`.
##
## Per token t, one threadgroup for the router, one (token, 32-col block)
## for the merge, El storage, fp32 score math, E = experts, K = top-K:
##
##   x (H) ──► [ router_w @ x, 64-expert chunks ] ──► logits (E, fp32)
##           ──► El round ──► [ softmax (fp32) ] ──► p (E)
##           ──► top-K, lowest-index tiebreak ──► ids (K), w (K)
##           ──► w = p / Σp · Scale ──► El round
##
import ../math_consts
import workspace/crucible
import workspace/ceramic
import ../tile_io_rows

export layout_algebra, tensors, tile_algebra, ptr_arithmetic

# The router writes ids and weights elementwise, so it needs no bounded store

# ─── Local device extensions: the score passes ───────────────────────

const
  ScoreChunk = 64
    ## Experts per router score chunk, the (32, 64) chunk accumulator's width,
    ## the shape `ffn_moe.nim` names `ScoreChunk` too.
  Lanes = 32
  ## Simdgroup lane count, the selection's expert-slice stride.

# tiles-allow gatherScores is a simdgroup lane machine, it needs a lane-permute gather primitive (simdShuffle over fragment pairs)
proc gatherScores[A, AL: static MmaAtom; F: static int](
    scores: var RtLeft[float32, 8, F, A],
    chunk: RtLeft[float32, 32, 64, AL],
    destFrag: int32) {.device.} =
  ## - Gathers one 64-expert chunk's row-0 logits into the score tile:
  ##   element (r, 8·destFrag + c) receives the raw logit of expert
  ##   64·destFrag + 8r + c, 2 values per lane across all 32 lanes
  ## - the row-0 logits live in the accumulator's lanes {0, 1, 8, 9}, 2 per
  ##   col-frag, destination lane d pulls its pair from source lane `d and 9`
  ## - `d and 9` is the row-0 owner of the same col pair under the universal AC
  ##   layout (row = b1+2b2+4b4, col = 2b0+4b3), a wrong mapping shows up
  ##   as a routing mismatch, never a value error
  static:
    doAssert AL.getM() == A.getM() and AL.getN() == A.getN() and
      AL.getVpt() == A.getVpt(),
      "gatherScores: both atoms must share the lane→fragment cell mapping"
  let lane = thread_index_in_threadgroup
  let cell = crd2idx(AL.getLayoutA(), (lane, 0)).toIntVal()
  let r = cell mod AL.getM()  # the destination row = expert div 8
  let srcLane = lane and 9    # the row-0 owner of the lane's col pair
  for m in 0 ..< 8:
    let g0 = simdShuffle(chunk.frags[0][m].frag[0], srcLane)
    let g1 = simdShuffle(chunk.frags[0][m].frag[1], srcLane)
    if m == r:
      scores.frags[0][destFrag].frag[0] = g0
      scores.frags[0][destFrag].frag[1] = g1

# tiles-allow softmaxScores needs tile-wide max/sum reductions that read the fragment lanes
# directly (a full-tile reduce over RtLeft fragments)
proc softmaxScores[A: static MmaAtom; F: static int](
    scores: var RtLeft[float32, 8, F, A]) {.device.} =
  ## - In-place fp32 softmax over the score tile's El-rounded logits:
  ## - a tile-wide max, the shifted exponentials' sum, then a division
  ##   per element, mirroring the eager softmax's opmath
  ## - the exp is the backend's native exp2, a 1-ulp-class exponential form
  const colFrags = F div 8
  const vpt = A.getVpt()
  var lm = max(scores.frags[0][0].frag[0], scores.frags[0][0].frag[1])
  for m in 1 ..< colFrags:
    lm = max(lm, scores.frags[0][m].frag[0])
    lm = max(lm, scores.frags[0][m].frag[1])
  lm = warpReduce(lm, max)
  var ls = 0.0'f32
  for m in 0 ..< colFrags:
    for v in 0 ..< vpt:
      ls += exp2((scores.frags[0][m].frag[v] - lm) * Log2e)
  ls = warpReduce(ls, `+`)
  scores.map(scores, exp2((x - lm) * Log2e) / ls)

# tiles-allow topkScores is the fragment staging machine, it needs a tile-level
# scatter primitive (fragment cell → the expert-indexed score row)
proc topkScores[A: static MmaAtom; F, K: static int](
    scores: RtLeft[float32, 8, F, A],
    scratch: ptr UncheckedArray[float32],
    ids: var array[K, int32],
    w: var array[K, float32]) {.device.} =
  ## Selects the K largest scores of the (8, F) score tile, the lowest-index
  ## tiebreak top-K, the weights read back from the staged scores.
  ##
  ##   scores (8, F) fragment tile ──► staged scratch row (E, fp32) ──► K passes ──► ids (K), w (K)
  ##
  ## The tile stages first, each lane writes its fragment cells to `scratch`
  ## at the element's expert index, every one of the 8·F experts exactly one
  ## writer (the AC layout's lane → cell mapping below). Each selection pass
  ## scans the linear expert vector with lane==expert-slice ownership
  ## (`e = lane + m·32`), the reductions are `simdShuffle` trees.
  ##
  ## Per-slot walk:
  ## - the slice's max reduced by the 5-step `simdShuffleDown` tree,
  ##   the maximum broadcast on every lane
  ## - the lowest expert with score == max reduced by a 5-step min tree,
  ##   the found expert masked to −float32 max in `scratch`
  ## - the weight is the unmasked staged score, which the owning lane
  ##   preloads before its mask store and broadcasts with `simdShuffle`,
  ##   a threadgroup barrier per slot then orders the next pass's reads
  ##
  ## Expert layout:
  ## - element (r, 8m + c) holds expert 64m + 8r + c, the per-chunk (8, 8)
  ##   mapping at chunk m
  ## - the lane's fragment cells decode through the layout's lane bits
  ##   b0..b4 (row = b1+2b2+4b4, col = 2b0+4b3), the mapping
  ##   the `Apple8x8_AC_Layout` doc in `hardware/h_registry.nim` documents
  ##
  ## Poisoned pass (NaN/Inf-poisoned scores):
  ## - the group max is NaN, `==` never matches NaN, no score equals the max,
  ##   the reduction keeps the sentinel
  ## - the slot routes to the last expert (8·F − 1) with zero weight, ids
  ##   stay inside [0, 8·F), the downstream expert-row reads stay in bounds,
  ##   a fully poisoned score set routes every K slot down the unmatched
  ##   branch, the normalized weights then sum to zero
  const E = 8 * F
  const colFrags = F div 8
  const maskScore = fp32Lowest
  static:
    doAssert E mod Lanes == 0,
      "topkScores: the lane-slice walk needs E a multiple of the 32 lanes"
  let lane = int32(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let r = int32(cell mod A.getM())
  let c0 = int32(cell div A.getM())
  for m in 0 ..< colFrags:
    let e0 = int32(ScoreChunk * m + A.getM() * r + c0)
    scratch[e0] = scores.frags[0][m].frag[0]
    scratch[e0 + 1] = scores.frags[0][m].frag[1]
  # the selection's reads are cross-lane, the barrier orders them after the stores
  threadgroup_barrier_device()
  for slot in 0 ..< K:
    var lm = scratch[lane]
    for m in 1'i32 ..< int32(E div Lanes):
      lm = max(lm, scratch[lane + m * Lanes])
    lm = warpReduce(lm, max)
    var localCand = int32(1 shl 30)
    for m in 0'i32 ..< int32(E div Lanes):
      let e = lane + m * Lanes
      if scratch[e] == lm:
        localCand = min(localCand, e)
    let cand = warpReduce(localCand, min)
    if cand >= int32(E):
      ids[slot] = int32(E - 1)
      w[slot] = 0.0'f32
    else:
      ids[slot] = cand
      let own = cand mod Lanes
      # the weight reads before the mask store, the owning lane preloads the unmasked staged score and the shuffle broadcasts it
      w[slot] = simdShuffle(if own == lane: scratch[cand] else: 0.0'f32,
        own)
      if own == lane:
        scratch[cand] = maskScore
    threadgroup_barrier_device()

# ─── The router core ─────────────────────────────────────────────────

proc moeRoute*[El; H, E, K: static int; Scale: static float32](
    x, router_w: ptr UncheckedArray[El],
    t: int32,
    scores_scratch: ptr UncheckedArray[float32],
    ids: var array[K, int32],
    w: var array[K, float32]) {.device.} =
  ## One token's top-K expert ids and routing weights, the chunked router
  ## GEMV, the softmax form's score pass, the scratch-staged top-K selection
  ## by lowest index.
  ##
  ## Expected input:
  ##   - x: (num_tokens, H) El, row-major, token `t`'s activation row,
  ##     the router GEMV input
  ##   - router_w: (E, H) El, row-major, the router weight matrix,
  ##     the checkpoint's routed-expert weights
  ##   - t: the token index, the caller's launch coordinate
  ##   - scores_scratch: E fp32 elements, the selection's staged score
  ##     row, the content may be uninitialized
  ##   - ids: K-element int32 register array, filled with the top-K
  ##     expert ids in score order, lowest index on ties, each in [0, E)
  ##   - w: K-element fp32 register array, filled with the normalized
  ##     weights w[slot] = p[ids[slot]] / sum·Scale, fp32 carriers
  ##     holding El-rounded values
  ##   - H a multiple of the 16-wide K step, E a multiple of the
  ##     64-expert chunk, K/Scale static compile-time
  ##
  ##   x (H) ──► [ router_w @ x, E div 64 chunks ] ──► logits (E, fp32)
  ##           ──► El round ──► [ softmax (fp32) ] ──► p (E)
  ##           ──► top-K, lowest-index tiebreak ──► ids (K), w (K)
  ##           ──► w = p / Σp · Scale ──► El round
  ##
  ## The (8, E div 8) score tile assembles chunk by chunk, chunk cs's 64
  ## row-0 logits land in col-frag cs. Composed in-group per slot group by
  ## the mega kernel.
  ##
  ## 
  ## - the logits round to El once, the fp32 softmax runs over the widened
  ##   values, the normalized weights round to El (the eager routing cast)
  ## - a poisoned score pass (NaN/Inf logits) leaves no candidate matching
  ##   the group max, the slot routes to expert E−1 with zero weight, ids
  ##   in [0, E), the downstream expert-row reads stay in bounds, and the
  ##   renorm guard keeps the 0/0 weight sum from NaNing
  const F = E div 8
  static:
    doAssert E mod 64 == 0, "moeRoute: E must be a multiple of the 64-expert chunk"
    doAssert H mod 16 == 0, "moeRoute: H must be a multiple of the 16-wide K step"
    doAssert K <= E, "moeRoute: the K slots need K distinct experts"
  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (H, 0, H, 1))
  let glRouter = router_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, H, 1))

  var scores: rt_l(float32, 8, F)
  var dR: rt_l(float32, 32, 64, getTileConfig(float32, El))
  var a: rt_l(El, 32, 16)
  var b64: rt_r(El, 16, 64)

  for cs in 0'i32 ..< E div 64:
    dR.zero()
    for kk in 0'i32 ..< H div 16:
      a.loadTileRows(glX, (t, 0, 0, kk), 1)
      b64.loadTile(glRouter, (0, 0, cs, kk))
      dR.mma_AB(a, b64)
    scores.gatherScores(dR, cs)
  scores.map(scores, roundToNearestEven[El](x).float32)
  scores.softmaxScores()
  scores.topkScores(scores_scratch, ids, w)
  var sumW = 0.0'f32
  for slot in 0 ..< K:
    sumW += w[slot]
  for slot in 0 ..< K:
    if sumW > 0.0'f32:
      w[slot] = w[slot] / sumW * Scale
    else:
      # every weight is zero (the poisoned pass) or the sum underflowed,
      # a 0/0 store would NaN the weight and everything downstream
      w[slot] = 0.0'f32
    w[slot] = roundToNearestEven[El](w[slot]).float32

# ─── The router-only entry ───────────────────────────────────────────

proc moe_route_fwd*[El; H, E, K: static int; Scale: static float32](
    ids: ptr UncheckedArray[int32],        # (num_tokens, K) expert ids
    rout_w: ptr UncheckedArray[El],        # (num_tokens, K) routing weights
    x: ptr UncheckedArray[El],             # (num_tokens, H) activations
    router_w: ptr UncheckedArray[El],      # (E, H) router weight
    scores_scratch: ptr UncheckedArray[float32], # (num_tokens, E) selection scratch
    num_tokens: int32) {.device.} =
  ## Router-only pass, grid (num_tokens, 1, 1) at 32 lanes, one token per
  ## threadgroup, `t` from the grid. Stores the top-K expert ids and the
  ## El-rounded routing weights under the `moeRoute` contract:
  ##
  ## Expected input:
  ##   - x, router_w: the same contract as `moeRoute`
  ##   - scores_scratch: (num_tokens, E) fp32, row-major, one E-slice
  ##     per threadgroup, the content may be uninitialized
  ##   - ids: (num_tokens, K) int32, row-major, filled with the top-K
  ##     expert ids, each in [0, E)
  ##   - rout_w: (num_tokens, K) El, row-major, filled with the
  ##     El-rounded routing weights
  ##   - num_tokens: the token count, grid.x bound
  ##   - H/E/K/Scale static compile-time, the `moeRoute` constraints
  ##
  ## Dataflow:
  ##
  ##   x (H) ──► [ moeRoute ] ──► ids[t] (K), rout_w[t] (K)
  let t = int32(threadgroup_position_in_grid.x)
  var idsReg: array[K, int32]
  var wReg: array[K, float32]
  moeRoute[El, H, E, K, Scale](x, router_w, t,
    scores_scratch +% t * int32(E), idsReg, wReg)
  for slot in 0 ..< K:
    ids[t * K + slot] = idsReg[slot]
    rout_w[t * K + slot] = roundToNearestEven[El](wReg[slot])

# ─── The shared-expert gate logit ────────────────────────────────────

proc sharedGateLogit*[El; H: static int](
    x, sgw: ptr UncheckedArray[El], t: int32): float32 {.device.} =
  ## Returns the raw fp32 shared-expert scalar logit of token t.
  ##
  ## Expected input:
  ##   - x: (num_tokens, H) El, row-major, token `t`'s activation row
  ##   - sgw: (1, H) El, row-major, the shared-gate weight row,
  ##     the checkpoint's shared-expert weights
  ##   - t: the token index, the caller's launch coordinate
  ##   - H: the hidden width, static compile-time, a multiple of the
  ##     16-wide K step
  ##
  ## Dataflow:
  ##
  ##   x[t] (H) ──► [ shared_gate_vec_w[0] @ x[t], 16-wide K steps ] ──► logit (fp32)
  ##
  ## The (1, H) row weight as a one-output GEMV, element (0, 0) of the
  ## (32, 8) accumulator carried on lane 0, the caller broadcasting with
  ## simdShuffle from lane 0. An El round, if any, happens after this load.
  static:
    doAssert H mod 16 == 0, "sharedGateLogit: H must be a multiple of 16"
  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (H, 0, H, 1))
  let glSgw = sgw.gd(shape = (-1, -1, -1, -1), stride = (1, 0, H, 1))
  var sg: rt_l(float32, 32, 8, getTileConfig(float32, El))
  var a: rt_l(El, 32, 16)
  var b: rt_r(El, 16, 8)
  sg.zero()
  for kk in 0'i32 ..< H div 16:
    a.loadTileRows(glX, (t, 0, 0, kk), 1)
    b.loadTile(glSgw, (0, 0, 0, kk))
    sg.mma_AB(a, b)
  result = simdShuffle(sg.laneScalar(), 0)

# ─── The fp32-partial merge (decode regime) ──────────────────────────

proc moe_decode_merge_at*[El; H, K: static int](
    out_r: ptr UncheckedArray[El],         # (num_tokens, H) routed+shared output
    partial: ptr UncheckedArray[float32],  # (num_tokens, K+1, H) fp32 partials
    t, nt: int32) {.device.} =
  ## Merge walk for one (token, column block) pair at caller coordinates,
  ## `t` the token, `nt` the H div 32 column block.
  ##
  ## Expected input:
  ##   - out_r: (num_tokens, H) El, row-major, the 32-wide column block
  ##     [nt·32, nt·32 + 32) written per call, El RNE rounded
  ##   - partial: (num_tokens, K+1, H) fp32, row-major, the routed
  ##     experts' partial sums plus the shared expert's row last
  ##   - t: the token, nt: the column block, caller coordinates
  ##   - H a multiple of the 32-wide lane tile, K static compile-time
  ##
  ## Dataflow:
  ##
  ##   partial[t, 0..K] (H, fp32 rows) ──► [ Σ slot order, shared last ] ──► El RNE ──► out_r[t] (H)
  ##
  ## The mega kernel composes this core inline, one instantiation contract:
  ## each static binding set of this core needs a distinct call-site line,
  ## calls sharing one call-site line collapse into a single body under the
  ## engine's monomorphization key.
  static:
    doAssert H mod 32 == 0,
      "moe_decode_merge_at: H must be a multiple of the 32-wide lane tile"
  let lane = int32(thread_index_in_threadgroup)
  let col = nt * 32 + lane
  var acc = 0.0'f32
  for y in 0'i32 ..< K + 1:
    acc += partial[int(t * (K + 1) + y) * H + int(col)]
  out_r[t * H + col] = roundToNearestEven[El](acc)

