# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ──────────────────  moe_router (MoE routing on the tile algebra)  ──────────────────

## MoE routing on the tile algebra: the tile-stack port of the v1
## ceramic `ffn_moe/moe_router.nim` router semantics.
## Element dtype El, fp32 score math, one threadgroup of 32 lanes per token.
##
## Forms:
##
## - Qwen softmax form (`moeRoute`): router GEMV → El round → fp32 softmax
##   → lowest-index top-K → w = p / Σp · Scale → El round.
## - GLM sigmoid form (`moeRouteSigmoid`): raw fp32 logits → sigmoid scores
##   → lowest-index top-K → w = s / (Σs + RouteSumEps) · routed_scaling,
##   with a runtime-dim variant `moeRouteSigmoidRT` whose hidden,
##   expert count and slot count are runtime values over ragged tails.
## - shared-expert gate weight logit (`sharedGateLogit`): one raw fp32
##   scalar output.
## - fp32-partial decode merge (`moe_decode_merge_at`):
##   a slot-order fp32 sum with one El round at the store.

import workspace/crucible
import workspace/math
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/kernels/k_layout_gemm

const
  ScoreChunk* = 64
    ## Experts per router score chunk: the (32, 64) accumulator width.
    ## Exported for the fused forward's expert max and scratch stride.
  Lanes = 32
    ## Simdgroup lane count: the selection walk's expert-slice stride.
  MaxTopK* = 8
    ## Compiled-in routing-slot max of the sigmoid form's register
    ## id array. Value copied from the v1 ceramic ffn_moe.nim.

const RouteSumEps* = 1e-20'f32
  ## Zero guard of the sigmoid routing sum's weight normalization.
  ## Divides by sum(w) + RouteSumEps, so an all-underflow weight
  ## vector still divides. Value copied from the v1 ceramic ffn_moe.nim.

template elRound*(x: float32, El: typedesc): float32 =
  ## One round-to-nearest-even round into El, carried back in fp32.
  x.to(El).to(float32)

# ─── The router GEMV and the score staging ───────────────────────────

template stageRouterLogitsCore[El: untyped, Sigmoid: static bool](
    x, router_w: distinct TensorView, t: int32, scratch: distinct TensorView,
    E, eCount, storeSpan: int, mmaConst, hidden, kSteps: untyped) =
  ## One token's router logits: the chunked router GEMV over El operands,
  ## with the E logits staged to `scratch` through the m = 0 fragment
  ## cells of the accumulator, one transform per staged logit.
  ##
  ## Shared core of `stageRouterLogits` and `stageRouterLogitsRT`, parameterized over:
  ##
  ## - mmaConst: the TiledMma constant, the generic schedule for the static forms, the explicit fp16 atom for the runtime form
  ## - hidden, kSteps: the hidden-dim expression and the K-step count,
  ##   `H` with `H div 16` static, runtime shapes over ragged tails
  ##
  ## Data flow:
  ##
  ## - the GEMV is a (8, 64, 16) TiledMma walk: token t's x values
  ##   as a (1, hidden) view, with the m > 0 tile positions zero-pad through
  ##   the padded loads, and with router_w (E, H) walked in ceil(E / 64)
  ##   chunks of kSteps 16-wide K steps.
  ## - each chunk contributes 16 m = 0 logits per lane on lanes
  ##   {0, 1, 8, 9} under the Apple AC fragment map, with m =
  ##   t1+2·t2+4·t4 and n = 2·t0+4·t3+v, so each lane stages the cells
  ##   whose m decodes to 0 and every expert gets exactly one writer.
  ## - With `Sigmoid = false` the staging writes the El-rounded logit:
  ##   the Qwen form rounds each logit once, before the softmax.
  ## - With `Sigmoid = true` the staging writes the raw logit's sigmoid,
  ##   1/(1 + exp2(−g·log2e)) with no El round, covering [0, storeSpan):
  ##   real experts (e < eCount) stage their score, cells at or beyond
  ##   eCount stage the fp32 finite floor so they cannot win the top-K.
  ##
  ## storeSpan separates the span regimes: the static-shape scratch is
  ## exactly E wide, the runtime-dim scratch spans the whole chunk tail,
  ## staging the fp32 finite floor over its stale cells
  ##
  ## The ceil chunk count stages every real expert when the runtime E is
  ## not a multiple of 64: the walk reads zero for router weights
  ## past E through the padded loads.
  const mma = mmaConst
  const
    tvmA = mma.getThreadValueMap(opA)
    tvmB = mma.getThreadValueMap(opB)
    tvmC = mma.getThreadValueMap(opC)
    cLayout = mma.getMmaLayout(opC)
  let
    thrID = getLocalThreadID()
    xOp = x(int(t) ..< int(t) + 1, _)
    thrCrd = idx2crd(tvmC.getThreadPartitioning(), thrID)
    regCrd = tvmC.layout.slice(((thrCrd, _), _))
    crdBase = tvmC.layout.crd2idx(((thrCrd, _), _))
  var
    dR: tile(float32, cLayout)
    a: tile(El, mma.getMmaLayout(opA))
    b64: tile(El, mma.getMmaLayout(opB))
  for cs in 0 ..< E.ceil_div(ScoreChunk):
    dR.zeros()
    for kk in 0 ..< kSteps:
      a.load(xOp, tvmA, thrID, (0, kk))
      b64.load(router_w, tvmB, thrID, (cs, kk))
      mma.atom.gemm_tile(dR, a, b64, dR)
    for i in 0 ..< cLayout.size():
      let (m, n) = crdBase + regCrd.crd2idx(i)
      if m == 0:
        let e = ScoreChunk * cs + n
        when Sigmoid:
          # a runtime E below the chunk boundary leaves the chunk's
          # tail cells beyond the store span, whose stores would bleed
          # into the next token's scratch slice
          if e < storeSpan:
            scratch[e] = if e < eCount:
              1.0'f32 / (1.0'f32 + exp2(-dR.data[i] * Log2e))
            else:
              finiteFloor(float32)
        else:
          scratch[e] = dR.data[i].elRound(El)

proc stageRouterLogits[El: untyped, H: static int, Sigmoid: static bool](
    x, router_w: distinct TensorView, t: int32,
    scratch: distinct TensorView, E, eCount: int) =
  ## One token's router logits at static dims via `stageRouterLogitsCore`:
  ## the generic `tiledMma(float32, El, El, 8, 64, 16)` schedule at static
  ## hidden dim H, the `H div 16` K-step loop bound a compile-time constant.
  stageRouterLogitsCore[El, Sigmoid](x, router_w, t, scratch, E, eCount, E, tiledMma(float32, El, El, 8, 64, 16), H, H div 16)

# ─── The score passes over the staged scratch ────────────────────────

proc softmaxScores(scratch: distinct TensorView, E: int) =
  ## In-place fp32 softmax over the staged El-rounded logits.
  ## Computes p[e] = exp2((logit[e] − lm) · log2e) / Σ_e exp2((logit[e] − lm) · log2e).
  ##
  ## - The pass subtracts the max before the exponentials, then folds
  ##   in the full-scratch sum.
  ## - The exp is the backend's native exp2, a 1-ulp-class exponential form.
  ## - The pass reads and writes the per-token scratch gmem slice.
  ##   Every pass reads what the sibling lanes wrote.
  ##   The caller's barrier between passes orders those reads.
  let lane = getLocalThreadID()
  var lm = scratch[lane]
  for m in 1 ..< E div Lanes:
    lm = max(lm, scratch[lane + m * Lanes])
  lm = reduceMax(lm)
  var ls = 0.0'f32
  for m in 0 ..< E div Lanes:
    let e = lane + m * Lanes
    scratch[e] = exp2((scratch[e] - lm) * Log2e)
    ls += scratch[e]
  ls = reduceAdd(ls)
  for m in 0 ..< E div Lanes:
    let e = lane + m * Lanes
    scratch[e] = scratch[e] / ls

proc topkScores[K: static int](
    scratch: distinct TensorView, E: int, ids: var array[K, int32],
    w: var array[K, float32], unmatchedId: int32) =
  ## Selects the K largest staged scores under the lowest-index
  ## tiebreak rule and reads the weights back from the scratch.
  ##
  ## The walk covers ceil(E / 32) slices, so a runtime E that is not
  ## a multiple of 32 is scanned fully.
  ## The padded load reads a slice cell past E as the fp32 finite floor, so it can neither win the max nor match it.
  ##
  ## Per slot walk, every lane owning the expert slice `e = lane + m·32`:
  ##
  ## - the subgroup butterfly reduces the slice's max through
  ##   `reduceMax` (every lane calls it unconditionally),
  ##   giving the maximum on every lane.
  ## - the min butterfly over `simd_shuffle_xor` masks 16/8/4/2/1
  ##   reduces the lowest expert with score == max, all masks uniform.
  ## - the slot's weight is the unmasked staged score. The owning lane
  ##   (`cand mod 32`) preloads it before the mask store.
  ##   `simd_shuffle` broadcasts the weight to the slot.
  ## - the found expert is masked to the fp32 finite floor in `scratch`.
  ##   A device-scope threadgroup barrier per slot then orders the next
  ##   pass's cross-lane reads.
  ##
  ## Poisoned scores (NaN or Inf planted in the staged scratch):
  ##
  ## - the group max is NaN, `==` never matches NaN, and no score
  ##   equals the max, so the reduction leaves the sentinel standing.
  ## - the slot routes to `unmatchedId` with zero weight, ids stay
  ##   inside [0, E), and the downstream expert reads stay in bounds.
  let lane = getLocalThreadID()
  for slot in 0 ..< K:
    var lm = scratch.paddedLoad(lane, finiteFloor(float32))
    for m in 1 ..< E.ceil_div(Lanes):
      lm = max(lm, scratch.paddedLoad(lane + m * Lanes, finiteFloor(float32)))
    lm = reduceMax(lm)
    # cand starts above every real expert index. The min reduction
    # leaves the sentinel standing only when no score matched the max.
    # So the `cand >= E` test below fires.
    var cand = 1 shl 30
    for m in 0 ..< E.ceil_div(Lanes):
      let e = lane + m * Lanes
      if scratch.paddedLoad(e, finiteFloor(float32)) == lm:
        cand = min(cand, e)
    cand = min(cand, cand.simd_shuffle_xor(16))
    cand = min(cand, cand.simd_shuffle_xor(8))
    cand = min(cand, cand.simd_shuffle_xor(4))
    cand = min(cand, cand.simd_shuffle_xor(2))
    cand = min(cand, cand.simd_shuffle_xor(1))
    if cand >= E:
      ids[slot] = unmatchedId
      w[slot] = 0.0'f32
    else:
      ids[slot] = int32(cand)
      let own = cand mod Lanes
      # weights read before the mask store: the owning lane preloads
      # its unmasked staged score, and the shuffle broadcasts it.
      w[slot] = simd_shuffle(if own == lane: scratch[cand] else: 0.0'f32, own)
      if own == lane:
        scratch[cand] = finiteFloor(float32)
    threadgroup_barrier_device()

# ─── The router core, the Qwen softmax form ──────────────────────────

proc moeRoute*[El: untyped, H, E, K: static int, Scale: static float32](
    x, router_w: distinct TensorView,
    t: int32,
    scores_scratch: distinct TensorView,
    ids: var array[K, int32],
    w: var array[K, float32]) =
  ## One token's top-K expert ids and routing weights: the chunked router
  ## GEMV, the softmax form's score passes, and the scratch-staged top-K
  ## selection by lowest index.
  ##
  ## Expected input:
  ##   - x: (num_tokens, H) El, token t's activations, the router GEMV input.
  ##   - router_w: (E, H) El, the routed-expert weight matrix.
  ##   - t: the token index, the caller's launch coordinate.
  ##   - scores_scratch: (E,) fp32, token t's staged score scratch,
  ##     with content that may be uninitialized.
  ##
  ## Results:
  ##
  ##   - ids: K-element int32 register array, the top-K expert ids,
  ##     in score order, lowest index on ties, each in [0, E).
  ##   - w: K-element fp32 register array, the normalized weights
  ##     w[slot] = p[ids[slot]] / sum · Scale, each an El-rounded value.
  ##
  ## Static bounds: H a multiple of the 16-wide K step, E a multiple
  ## of the 64-expert chunk, and K at most E.
  ##
  ##   x (H) ──► [ router_w @ x, E div 64 chunks ] ──► logits (E, fp32)
  ##           ──► El round ──► [ softmax (fp32) ] ──► p (E)
  ##           ──► top-K, lowest-index tiebreak ──► ids (K), w (K)
  ##           ──► w = p / Σp · Scale ──► El round.
  ##
  ## - The pass rounds each logit to El once, runs the fp32 softmax over
  ##   the widened values, and rounds each normalized weight to El under
  ##   the eager routing cast.
  ## - a poisoned score pass (NaN or Inf logits) leaves no candidate
  ##   matching the group max, so the slot routes to expert E−1,
  ##   stores zero weight, keeps ids inside [0, E), and keeps every
  ##   downstream expert read in bounds.
  ## - the renorm guard prevents NaN outputs from the 0/0 weight sum
  ##   of the poisoned pass or an underflowing weight sum.
  static:
    doAssert E mod ScoreChunk == 0,
      "moeRoute: E must be a multiple of the 64-expert chunk"
    doAssert H mod 16 == 0, "moeRoute: H must be a multiple of the 16-wide K step"
    doAssert K <= E, "moeRoute: the K slots need K distinct experts"
  stageRouterLogits[El, H, false](x, router_w, t, scores_scratch, E, 0)
  threadgroup_barrier_device()
  softmaxScores(scores_scratch, E)
  threadgroup_barrier_device()
  topkScores[K](scores_scratch, E, ids, w, int32(E - 1))
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
    w[slot] = w[slot].elRound(El)

# ─── The router-only entry ───────────────────────────────────────────

proc moe_route_fwd*[El: untyped, H, E, K: static int, Scale: static float32](
    ids: distinct TensorView,              # (num_tokens, K) expert ids
    rout_w: distinct TensorView,           # (num_tokens, K) routing weights
    x: distinct TensorView,                # (num_tokens, H) activations
    router_w: distinct TensorView,         # (E, H) router weight
    scores_scratch: distinct TensorView)   # (num_tokens, E) selection scratch
  {.inline.} =
  ## Router-only pass, one token per threadgroup on the grid
  ## (num_tokens, 1, 1) with 32 lanes, `t` taken from the grid.
  ## Stores the top-K expert ids and the El-rounded routing weights
  ## under the `moeRoute` contract:
  ##
  ##   - ids: (num_tokens, K) int32, the top-K expert ids, each
  ##     inside [0, E).
  ##   - rout_w: (num_tokens, K) El, the El-rounded routing weights.
  ##   - scores_scratch: (num_tokens, E) fp32 with one E-slice per
  ##     threadgroup and possibly uninitialized content.
  ##   - x, router_w: the same contract as `moeRoute`.
  ##
  ## Data flow:
  ##
  ##   x (H) ──► [ moeRoute ] ──► ids[t] (K), rout_w[t] (K).
  let t = int(threadgroup_position_in_grid.x)
  var idsReg: array[K, int32]
  var wReg: array[K, float32]
  moeRoute[El, H, E, K, Scale](x, router_w, int32(t),
    scores_scratch(t, _), idsReg, wReg)
  for slot in 0 ..< K:
    ids[t, slot] = idsReg[slot]
    rout_w[t, slot] = wReg[slot].to(El)

# ─── The router core, the GLM sigmoid form ───────────────────────────

proc moeRouteSigmoid*[El: untyped, H, K: static int](
    x, router_w: distinct TensorView,
    t: int32,
    scores_scratch: distinct TensorView,
    eCount: int32,
    routed_scaling: float32,
    ids: var array[K, int32],
    w: var array[K, float32]) =
  ## One token's top-K expert ids and routing weights in the GLM
  ## sigmoid form: raw fp32 logits, sigmoid scores,
  ## scratch-staged top-K selection by lowest index, and no El round
  ## anywhere on logits, scores, or weights.
  ##
  ## Expected input:
  ##   - x: (num_tokens, H) El, token t's activations, the router GEMV input.
  ##   - router_w: (E, H) El, the routed-expert weight matrix.
  ##     E is read from the view's shape at runtime, any value >= 1.
  ##   - t: the token index, the caller's launch coordinate.
  ##   - scores_scratch: (E,) fp32, token t's staged score scratch,
  ##     with content that may be uninitialized. After the call,
  ##     the scratch contains the masked selection state, not the scores.
  ##
  ## Options and results:
  ##
  ##   - eCount: the runtime expert count, the real experts, inside [1, E].
  ##     K at most eCount is the caller's obligation to guard before launch.
  ##     Experts at or beyond eCount stage the fp32 finite floor,
  ##     so they cannot win the top-K.
  ##   - routed_scaling: the weight scale, fp32.
  ##   - ids: K-element int32 register array, the top-K expert ids,
  ##     in score order, lowest index on ties, each in [0, eCount).
  ##   - w: K-element fp32 register array, holding
  ##     w[slot] = s[ids[slot]] / (sum + RouteSumEps) · routed_scaling.
  ##
  ##   x (H) ──► [ router_w @ x, ceil(E / 64) chunks ] ──► logits (E, fp32)
  ##           ──► sigmoid ──► scores (E)
  ##           ──► top-K, lowest-index tiebreak ──► ids (K), w (K)
  ##           ──► w = s / (Σs + RouteSumEps) · routed_scaling.
  ##
  ## Static bounds: K at most MaxTopK (the register id array bound),
  ## and H a multiple of the 16-wide K step.
  ##
  ## - a poisoned score pass (NaN or Inf logits) leaves no candidate
  ##   matching the group max, so the slot routes to expert eCount−1,
  ##   storing zero weight, and the RouteSumEps guards the all-zero
  ##   weight sum's division.
  static:
    doAssert K <= MaxTopK,
      "moeRouteSigmoid: K exceeds the register id array's compiled-in max"
    doAssert H mod 16 == 0, "moeRouteSigmoid: H must be a multiple of the 16-wide K step"
  let E = router_w.shape[0]
  stageRouterLogits[El, H, true](x, router_w, t, scores_scratch, E, eCount.int)
  threadgroup_barrier_device()
  topkScores[K](scores_scratch, E, ids, w, int32(eCount - 1))
  var sumW = 0.0'f32
  for slot in 0 ..< K:
    sumW += w[slot]
  sumW += RouteSumEps
  for slot in 0 ..< K:
    w[slot] = w[slot] / sumW * routed_scaling

# ─── The router core, the GLM sigmoid form at runtime dims ───────────

proc stageRouterLogitsRT[El: untyped](
    x, router_w: distinct TensorView, t: int32,
    scratch: distinct TensorView) =
  ## One token's router logits at runtime dims via `stageRouterLogitsCore`,
  ## the hidden dim, the expert count E and the K-step count
  ## `hidden.ceil_div(16)` all runtime values over ragged tails.
  ##
  ## Staging covers the sigmoid half with eCount = E, storeSpan 64·ceil(E / 64).
  ##
  ## Host-typecheck note:
  ##
  ## - a concrete proc body is typechecked on the host import too, outside
  ##   any metal: block, where backend atom resolution cannot run
  ## - the fp16 atom therefore binds as an explicit TiledMma constant,
  ##   TiledMma[float32, float16, float16, 8, 64, 16](atom: APPLE_8x8x8_F16),
  ##   never as a tiledMma(...) call
  static:
    doAssert El is float16,
      "stageRouterLogitsRT: the runtime-dim staging is fp16-only"
  let
    hidden = int(x.shape[1])
    E = int(router_w.shape[0])
  stageRouterLogitsCore[El, true](x, router_w, t, scratch, E, E,
    ScoreChunk * E.ceil_div(ScoreChunk),
    TiledMma[float32, float16, float16, 8, 64, 16](atom: APPLE_8x8x8_F16),
    hidden, hidden.ceil_div(16))

proc moeRouteSigmoidRT*[El: untyped](
    x, router_w: distinct TensorView,
    t: int32,
    scores_row: distinct TensorView,
    topK: int32,
    routed_scaling: float32,
    ids: var array[MaxTopK, int32],
    w: var array[MaxTopK, float32]) =
  ## One token's top-K expert ids and routing weights in the GLM sigmoid
  ## form at runtime dims, composed of:
  ##
  ## - the runtime-dim staging (`stageRouterLogitsRT`)
  ## - the shared `topkScores[MaxTopK]` selection core
  ## - the RouteSumEps weight normalization over the first `topK` slots
  ##
  ## Both the fused MoE forward (per token) and the routing pre-pass
  ## call this core.
  ##
  ## Expected input:
  ##
  ##   - x: (num_tokens, hidden) El, token t's activations, the router
  ##     GEMV input. hidden is read from the view at runtime, any value >= 1.
  ##   - router_w: (E, hidden) El, the routed-expert weight matrix, E
  ##     read from the view at runtime, any value >= 1.
  ##   - scores_row: the token's fp32 score slice, any content, covering
  ##     at least 32·ceil(E / 32) cells. After the call the slice
  ##     contains the masked selection state, not the scores.
  ##   - topK: the runtime routing-slot count, accepted range
  ##     1 ≤ topK ≤ min(MaxTopK, E). The caller guards it before launch
  ##     (`moeFwdConfigGuard` in `moe_fwd`).
  ##
  ## Scalar arguments:
  ##
  ##   - t: the token index, the caller's launch coordinate
  ##   - routed_scaling: the weight scale, fp32
  ##
  ## Results:
  ##
  ##   - ids: MaxTopK-element int32 register array. Slots [0, topK) hold
  ##     the top-K expert ids in score order, lowest index on ties, each
  ##     inside [0, E). The remaining slots carry no contract.
  ##   - w: MaxTopK-element fp32 register array. Slots [0, topK) hold
  ##     the normalized weights, each w[slot] =
  ##     s[ids[slot]] / (sum + RouteSumEps) * routed_scaling with no
  ##     El round. The remaining slots hold selection state.
  ##
  ##   x (hidden) ──► [ router_w @ x, ceil(E / 64) chunks ] ──► logits (E, fp32)
  ##              ──► sigmoid ──► scores (E)
  ##              ──► top-K, lowest-index tiebreak ──► ids (K), w (K)
  ##              ──► w = s / (Σs + RouteSumEps) · routed_scaling.
  ##
  ## A poisoned score pass (NaN or Inf logits) leaves no candidate
  ## matching the group max, so the unmatched slot routes to expert E−1
  ## with zero weight. The RouteSumEps guard preserves a finite
  ## division over an all-zero weight sum.
  let E = int(router_w.shape[0])
  stageRouterLogitsRT[El](x, router_w, t, scores_row)
  threadgroup_barrier_device()
  topkScores[MaxTopK](scores_row, E, ids, w, int32(E - 1))
  var sumW = 0.0'f32
  for slot in 0 ..< topK.int:
    sumW += w[slot]
  sumW += RouteSumEps
  for slot in 0 ..< topK.int:
    w[slot] = w[slot] / sumW * routed_scaling

# ─── The shared-expert gate weight logit ─────────────────────────────

proc sharedGateLogit*[El: untyped, H: static int](
    x: distinct TensorView, sgw: distinct TensorView, t: int32): float32 =
  ## Returns the raw fp32 shared-expert gate weight logit of token t.
  ##
  ## Expected input:
  ##   - x: (num_tokens, H) El, token t's activations.
  ##   - sgw: (1, H) El, the shared-expert gate weight vector, holding
  ##     the checkpoint's shared-expert weights.
  ##   - t: the token index, the caller's launch coordinate.
  ##
  ## Data flow:
  ##
  ##   x[t] (H) ──► [ shared_gate_vec_w[0] @ x[t], 16-wide K steps ] ──► logit (fp32).
  ##
  ## The (1, H) weight vector is a one-output GEMV through the smallest
  ## legal TiledMma(float32, El, El, 8, 8, 16) schedule, with the accumulator
  ## element (0, 0) on lane 0. `simd_shuffle` then broadcasts it.
  ## An El round, if any, happens at the caller.
  ##
  ## Static bounds: H a multiple of the 16-wide K step.
  static:
    doAssert H mod 16 == 0, "sharedGateLogit: H must be a multiple of 16"
  const mma = tiledMma(float32, El, El, 8, 8, 16)
  const
    tvmA = mma.getThreadValueMap(opA)
    tvmB = mma.getThreadValueMap(opB)
    cLayout = mma.getMmaLayout(opC)
  let
    thrID = getLocalThreadID()
    xOp = x(int(t) ..< int(t) + 1, _)
    sgwOp = sgw(0 ..< 1, _)
  var
    sg: tile(float32, cLayout)
    a: tile(El, mma.getMmaLayout(opA))
    b: tile(El, mma.getMmaLayout(opB))
  sg.zeros()
  for kk in 0 ..< H div 16:
    a.load(xOp, tvmA, thrID, (0, kk))
    b.load(sgwOp, tvmB, thrID, (0, kk))
    mma.atom.gemm_tile(sg, a, b, sg)
  result = simd_shuffle(sg.data[0], 0)

# ─── The fp32-partial merge (decode regime) ──────────────────────────

proc moe_decode_merge_at*[El: untyped, H, K: static int](
    out_r: distinct TensorView,            # (num_tokens, H) routed+shared output
    partial: distinct TensorView,          # (num_tokens, K+1, H) fp32 partials
    t, nt: int32) =
  ## Merge walk for one (token, column block) pair at caller coordinates:
  ## `t` is the token, `nt` is the H div 32 column block, and the lane
  ## is the column.
  ##
  ## Expected input:
  ##   - out_r: (num_tokens, H) El, with the 32-wide column block
  ##     [nt·32, nt·32 + 32) written per call, with one El RNE round
  ##     at the store.
  ##   - partial: (num_tokens, K+1, H) fp32, the routed experts' partial
  ##     sums plus the shared expert's slice last.
  ##   - t and nt: the token and the column block, caller coordinates.
  ##
  ## Data flow:
  ##
  ##   partial[t, 0..K] (H, fp32 slices) ──► [ Σ slot order, shared last ] ──► El RNE ──► out_r[t] (H).
  ##
  ## Scalar arithmetic only, with no tiles and no thread->value maps.
  ##
  ## The mega kernel inlines this core. Each static binding set needs
  ## a distinct call-site line, since calls sharing one call-site
  ## line collapse into a single body under the engine's monomorphization key.
  ##
  ## Static bounds: H a multiple of the 32-wide lane tile, K static.
  static:
    doAssert H mod 32 == 0,
      "moe_decode_merge_at: H must be a multiple of the 32-wide lane tile"
  let
    tok = int(t)
    col = int(nt) * 32 + getLocalThreadID()
  var acc = 0.0'f32
  for y in 0'i32 ..< K + 1:
    acc += partial[tok, int(y), col]
  out_r[tok, col] = acc.to(El)
