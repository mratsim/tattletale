# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ─────────────────────────────────────────────────────────────────────
# ─── Fused GDN decoder layer (qwen35_moe): one Metal launch ──────────
# ─────────────────────────────────────────────────────────────────────

## One-launch decode step of a Qwen3.5/3.6-35B-A3B GDN decoder layer on the ceramic Tile API.
## One token per launch, the 13 stages composed inline from the taxonomy kernels
## (see the inline-tile-function property in `src/kernels/README.md`), no exit to the host between stages.
##
## Stage order, one launch = one token's layer pass, every consumer's producers preceding it:
##
## | stage | role             | threadgroups | counter | stage block |
## | ----- | ---------------- | ------------ | ------- | ----------- |
## | 1     | add + norm1      | 1            | 0       | 0           |
## | 2     | qkv GEMV         | 128          | 1       | 1..128      |
## | 3     | z GEMV           | 64           | 2       | 129..192    |
## | 4     | a/b GEMV         | 2            | 3       | 193..194    |
## | 5     | conv + ring roll | 128          | 4       | 195..322    |
## | 6     | q/k l2norm       | 4            | 5       | 323..326    |
## | 7     | g + beta         | 1            | 6       | 327         |
## | 8     | GDN step         | 512          | 7       | 328..839    |
## | 9     | o_norm           | 4            | 8       | 840..843    |
## | 10    | out_proj         | 32           | 9       | 844..875    |
## | 11    | fold + norm2     | 1            | 10      | 876         |
## | 12    | MoE decode       | 9            | 11      | 877..885    |
## | 13    | merge            | 64           | 12      | 886..949    |
## | grid  | (950, 1, 1), 32-lane threadgroups, the stage role read from grid.x via static boundaries | | | |
##
## Two entries share one dispatcher, with the mixer serving mixer-internals parity
## (the conv, split, l2norm, g/beta, recurrence and out_proj anchors judged without the norm's reduction band compounding):
##
## | entry      | shape                                                                                                                                                                         |
## | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | full layer | all 13 stages                                                                                                                                                                 |
## | mixer      | static `HaveNorm = false` compiles the norm bookends and the MoE tail out, the projection stages read the host-preloaded normed row, the walk stopping after the out_proj row |
##
## Weights bind as flat pointers, row-major over the checkpoint layouts, in `F.linear`'s projection spelling:
##
## | group  | layout                                                                                                  |
## | ------ | ------------------------------------------------------------------------------------------------------- |
## | linear | the (N, K) projections, the (convDim, 1, K) conv weight, the (width) norms                              |
## | MoE    | fused gate_up (E, 2I, H), down (E, H, I), the separate shared projections, the (1, H) shared-expert row |
## | norms  | `A_log` f32 (the canonicalized spelling), `dt_bias` bf16                                                |
##
## | geometry | the Qwen bf16 class: hidden 2048, convDim 8192, Hv 32, Hk 16, Dk = Dv = 128, conv kernel 4, router softmax top-8 over 256 experts, intermediate 512 |
## | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |

import workspace/crucible
import workspace/ceramic
import ../../../kernels/ceramic/dense_linear
import ../../../kernels/ceramic/moe_fwd_decode
import ../../../kernels/ceramic/moe_router
import ../../../kernels/ceramic/o_norm_gated
import ../../../kernels/ceramic/sequence_mixers/state_space/gdn/gdn_decode_single

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Geometry + arena map ────────────────────────────────────────────

const
  Hidden* = 2048
    ## Layer width:
    ##   the norm rows, the MoE hidden and the out_proj rows.
  ConvDim* = 8192
    ## Fused qkv projection width:
    ##   2·Hk·Dk key channels then Hv·Dv value channels, the conv column order.
  NumVHeads* = 32
  NumKHeads* = 16
  HeadKDim* = 128
  HeadVDim* = 128
  ConvKernel* = 4
    ## Depthwise conv width. The decode ring carries the 3
    ## most recent history taps.
  RingWidth* = ConvKernel - 1
  HkRatio* = NumVHeads div NumKHeads
  TopK* = 8
  NumExperts* = 256
  Inter* = 512
  RowLaneSpan* = 64
    ## Norm-stage elements per lane:
    ##   2048 hidden / 32 lanes.

const
  sQkvCol* = 0
    ## bf16 arena:
    ##   the fused qkv projection column (the conv input).
  sZ* = 8192
    ## o_norm z row (Hv·Dv).
  sA* = 12288
    ## Decay projection row (Hv).
  sB* = 12320
    ## Beta projection row (Hv).
  sQN* = 12352
    ## l2-normalized q heads (Hk·Dk).
  sKN* = 14400
    ## l2-normalized k heads (Hk·Dk).
  sBeta* = 16448
    ## Beta values (Hv).
  sY* = 16480
    ## GDN core output rows (Hv·Dv).
  sNormed* = 20576
    ## o_norm output rows, the out_proj input (Hv·Dv).
  sConv* = 24672
    ## Conv output column (ConvDim), the split source.
  sH* = 32864
    ## MoE routed h scratch (TopK·Inter).
  sHs* = 36960
    ## MoE shared h scratch (Inter).
  sMoeOut* = 37472
    ## Merged MoE output row (Hidden).
  sH1* = 39520
    ## Folded residual row (Hidden), the next layer's residual add.
  sNormed2* = 41568
    ## Post-LN normed row (Hidden), the MoE input.
  sStream* = 43616
    ## Residual stream row (Hidden), the out_proj fold's addend.
  sNorm1* = 45664
    ## Norm1 output row (Hidden), the projection input.
    ## The mixer entry's host preloads this section with the recorded chain's normed rows.
  sBlockOut* = 47712
    ## Out_proj row before the fold (Hidden), the mixer
    ## entry's block-output readback section.
  BfArenaLen* = 49760
    ## bf16 arena extent in elements (≈ 97 KiB).

const
  sG* = 0
    ## f32 arena:
    ##   log-decay values (Hv).
  sPartial* = 32
    ## MoE fp32 partials, (TopK+1)·Hidden.
  F32ArenaLen* = 18464
    ## f32 arena extent in elements (≈ 72 KiB).

const WaveCounts*: array[13, uint32] = [1'u32, 128, 64, 2, 128, 4, 1, 512,
    4, 32, 1, 9, 64]
    ## Per-stage threadgroup totals, the stage counters' expected counts
    ## summing to the 950-threadgroup grid.

# ─── Wave sync (device-memory counters, seq_cst fences) ──────────────

proc waveAdd(counters: ptr UncheckedArray[uint32], idx: int32) {.device.} =
  ## One increment per threadgroup, lane 0 only, after the release
  ## fence that orders the stage's writes.
  ##
  ## The toolchain spells no acquire/release loads or stores, the fences
  ## carry the ordering and the increment stays relaxed.
  if thread_index_in_threadgroup != 0:
    return
  {.emit: """
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  atomic_fetch_add_explicit(reinterpret_cast<volatile device atomic_uint*>(&`counters`[`idx`]), 1U, memory_order_relaxed);
  """.}

proc waveWait(counters: ptr UncheckedArray[uint32], idx: int32, target: uint32) {.device.} =
  ## Spins until the stage counter reaches `target`, then the acquire
  ## fence orders the producers' writes before the consumer reads.
  {.emit: """
  uint32_t observed;
  do {
    observed = atomic_load_explicit(
      reinterpret_cast<const volatile device atomic_uint*>(&`counters`[`idx`]),
      memory_order_relaxed);
  } while (observed < `target`);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  """.}

# ─── Local device helpers ────────────────────────────────────────────

proc softplusDev(x: float32): float32 {.device.} =
  ## Softplus in f32, the ATen `softplus(x, 1, 20)` shape,
  ## linear past the threshold, `log(1 + exp(x))` under it.
  ##
  ## MSL has no `log1p`, the plain-log form's lost tail sits
  ## under the recurrence's f32 noise floor, the decay difference staying
  ## below 2e-9 relative for x < -16, well under the state's ulp budget.
  {.emit: """
  if (`x` > 20.0f) {
    `result` = `x`;
  } else {
    `result` = log(1.0f + exp(`x`));
  }
  """.}

proc normRow(x, y, normW, stream, outp: ptr UncheckedArray[bfloat16], eps: float32) {.device.} =
  ## One threadgroup's pass over the (Hidden) norm row in the bias-one RmsNormOne
  ## spelling:
  ##
## | aspect    | contract                                                                                         |
## | --------- | ------------------------------------------------------------------------------------------------ |
## | variance  | taken over the rounded row sums of squares, one bf16 round each                                  |
## | multiply  | rstd-first order `(x·rstd)·(1+w)`, one bf16 round at the store                                   |
## | lane walk | per lane the serial element walk of the RowLaneSpan block, the partial row-sum of squares in f32 |
## | reduction | a 5-step lane butterfly with the rstd broadcast, per-lane serial f32 sums plus a lane butterfly  |
## | barrier   | every lane re-reads only its own stored elements, no threadgroup barrier                         |
## | rounding  | the rstd lands inside the recorded chain's bf16 band, the reference rows pricing the difference  |
  let lane = int32(thread_index_in_threadgroup)
  let base = lane * RowLaneSpan
  var acc = 0.0'f32
  for e in base ..< base + RowLaneSpan:
    let s = (x[e].float32 + y[e].float32).bfloat16
    stream[e] = s
    acc += s.float32 * s.float32
  var total = acc
  total += simdShuffleDown(total, 16'u32)
  total += simdShuffleDown(total, 8'u32)
  total += simdShuffleDown(total, 4'u32)
  total += simdShuffleDown(total, 2'u32)
  total += simdShuffleDown(total, 1'u32)
  total = simdShuffle(total, 0'u32)
  let rstd = rsqrt(total / float32(Hidden) + eps)
  for e in base ..< base + RowLaneSpan:
    outp[e] =
      (stream[e].float32 * rstd * (normW[e].float32 + 1.0'f32)).bfloat16

proc convRingChannels(convW, ring, xCol, outCol: ptr UncheckedArray[bfloat16], chanBase: int32) {.device.} =
  ## One threadgroup's ConvDim div 64 channels of the decode conv step
  ## in the recorded two-round spelling:
  ## - the f32 tap dot over the ring's (RingWidth) history taps
  ##   then the step column, one bf16 round
  ## - the silu in f32 over the widened value, one bf16 round
  ## - the ring rolls in the same walk, the step's pre-conv column
  ##   landing in the newest slot, the history shifting down,
  ##   each channel independent of its neighbors
  let lane = int32(thread_index_in_threadgroup)
  var c = chanBase + lane
  let chanEnd = chanBase + ConvDim div 128
  while c < chanEnd:
    var acc = 0.0'f32
    for j in 0'i32 ..< RingWidth:
      acc += convW[c * ConvKernel + j].float32 * ring[c * RingWidth + j].float32
    acc += convW[c * ConvKernel + RingWidth].float32 * xCol[c].float32
    let tapped = acc.bfloat16
    let sig = tapped.float32 /
      (1.0'f32 + exp2((-tapped.float32) * 1.4426950408889634'f32))
    outCol[c] = sig.bfloat16
    ring[c * RingWidth + 0] = ring[c * RingWidth + 1]
    ring[c * RingWidth + 1] = ring[c * RingWidth + 2]
    ring[c * RingWidth + 2] = xCol[c]
    c += 32

proc f32InvSqrt(x: float32): float32 {.device.} =
  ## 1/sqrt in the correctly-rounded IEEE spelling (sqrt then divide),
  ## the l2norm chain's reciprocal form. Metal's approximate `rsqrt`
  ## builtin is a different rounding class.
  {.emit: """
  `result` = 1.0f / sqrt(`x`);
  """.}

proc l2normRow(x, outp: ptr UncheckedArray[bfloat16], cols: int32) {.device.} =
  ## One l2-normalized row in the recorded chain's rounding pipeline:
  ##
## | step                | rounding                                                             |
## | ------------------- | -------------------------------------------------------------------- |
## | elementwise squares | round to bf16                                                        |
## | row sum             | f32 serial accumulation, the sum rounds to bf16                      |
## | sum + eps, rsqrt    | the sum rounds to bf16, the rsqrt computes in f32 and rounds to bf16 |
## | normalization       | the multiply rounds to bf16 per element                              |
  ##
  ## Every lane walks the whole row serially, all lanes compute identical sums,
  ## the lanes then scatter the multiply.
  var acc = 0.0'f32
  for c in 0'i32 ..< cols:
    let xi = x[c].float32
    acc += (xi * xi).bfloat16.float32
  let sumBf = acc.bfloat16.float32
  let inv =
    f32InvSqrt((sumBf + 1.0e-6'f32).bfloat16.float32).bfloat16.float32
  var i = int32(thread_index_in_threadgroup)
  while i < cols:
    outp[i] = (x[i].float32 * inv).bfloat16
    i += 32

proc gateValues(aRow, bRow, dtBias: ptr UncheckedArray[bfloat16],
    aLog: ptr UncheckedArray[float32],
    g: ptr UncheckedArray[float32],
    beta: ptr UncheckedArray[bfloat16]) {.device.} =
  ## Recurrence values g and beta over the Hv heads, one head per lane,
  ## in the recorded spellings:
  ## - g stays f32 end to end with no round,
  ##   g = -exp(A_log)·softplus(a + dt_bias)
  ## - beta = sigmoid(b) with one bf16 round
  let h = int32(thread_index_in_threadgroup)
  let x = aRow[h].float32 + dtBias[h].float32
  g[h] = -exp2(aLog[h] * 1.4426950408889634'f32) * softplusDev(x)
  beta[h] = (1.0'f32 / (1.0'f32 +
      exp2((-bRow[h].float32) * 1.4426950408889634'f32))).bfloat16

# ─── The dispatcher ───────────────────────────────────────────────────

proc qwen35GdnLayerWalk*[HaveNorm: static bool](
    counters: ptr UncheckedArray[uint32],
    bfA: ptr UncheckedArray[bfloat16],
    f32A: ptr UncheckedArray[float32],
    xPrev, rPrev: ptr UncheckedArray[bfloat16],
    state: ptr UncheckedArray[float32],
    ring: ptr UncheckedArray[bfloat16],
    norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W,
    routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW:
      ptr UncheckedArray[bfloat16],
    aLog: ptr UncheckedArray[float32],
    dtBias: ptr UncheckedArray[bfloat16],
    eps: float32) {.device.} =
  ## Role dispatch contract over grid.x, one stage branch per threadgroup block:
  ##
## | rule     | behavior                                                                                                             |
## | -------- | -------------------------------------------------------------------------------------------------------------------- |
## | producer | each branch waits its producers' stage counters, runs its taxonomy core (or stage proc) inline, adds its own counter |
## | HaveNorm | `false` compiles the norm bookends and the MoE tail out                                                              |
## | stage 0  | runs empty, its counter still increments                                                                             |
## | GEMV     | the stages read the host-preloaded normed row, the walk stopping after the out_proj row                              |
  let tx = int32(threadgroup_position_in_grid.x)

  if tx == 0:
    when HaveNorm:
      # Stage 1:
      #   the residual add plus the bias-one norm1, one bf16
      # round each. The deferred-add decoder contract, the norm
      # consumes the rounded stream, the fold re-adds it downstream.
      normRow(xPrev, rPrev, norm1W, (bfA +% sStream), (bfA +% sNorm1), eps)
    waveAdd(counters, 0)
  elif tx <= 128:
    waveWait(counters, 0, 1)
    dense_linear_tile_fwd[bfloat16, 8192, 2048, 64](
      (bfA +% sQkvCol), (bfA +% sNorm1), qkvW, 1, tx - 1, 0)
    waveAdd(counters, 1)
  elif tx <= 192:
    waveWait(counters, 0, 1)
    dense_linear_tile_fwd[bfloat16, 4096, 2048, 64](
      (bfA +% sZ), (bfA +% sNorm1), zW, 1, tx - 129, 0)
    waveAdd(counters, 2)
  elif tx <= 194:
    waveWait(counters, 0, 1)
    if tx == 193:
      dense_linear_tile_fwd[bfloat16, 32, 2048, 32](
        (bfA +% sA), (bfA +% sNorm1), aW, 1, 0, 0)
    else:
      dense_linear_tile_fwd[bfloat16, 32, 2048, 32](
        (bfA +% sB), (bfA +% sNorm1), bW, 1, 0, 0)
    waveAdd(counters, 3)
  elif tx <= 322:
    waveWait(counters, 1, 128)
    convRingChannels(convW, ring, (bfA +% sQkvCol), (bfA +% sConv),
      (tx - 195) * 64)
    waveAdd(counters, 4)
  elif tx <= 326:
    # Stage 6:
    #   the q/k l2 normalization, 4 q rows then 4 k rows per threadgroup,
    # gathered straight from the conv column's head rows.
    waveWait(counters, 4, 128)
    let row0 = (tx - 323) * 512
    for r in 0'i32 ..< 4:
      l2normRow((bfA +% sConv +% (row0 + r * 128)),
        (bfA +% sQN +% (row0 + r * 128)), 128)
    for r in 0'i32 ..< 4:
      l2normRow((bfA +% sConv +% (2048 + row0 + r * 128)),
        (bfA +% sKN +% (row0 + r * 128)), 128)
    waveAdd(counters, 5)
  elif tx == 327:
    waveWait(counters, 3, 2)
    gateValues((bfA +% sA), (bfA +% sB), dtBias, aLog,
      (f32A +% sG), (bfA +% sBeta))
    waveAdd(counters, 6)
  elif tx <= 839:
    # Stage 8:
    #   the gated-delta-rule step, one (head, Dv block) state
    # tile per threadgroup, state updated in place. The value rows
    # read straight out of the conv column's value channels.
    let local = tx - 328
    waveWait(counters, 5, 4)
    waveWait(counters, 4, 128)
    waveWait(counters, 6, 1)
    gdnDecodeStepTileBf16At(state, (bfA +% sY), (bfA +% sKN), (bfA +% sQN),
      (bfA +% sConv +% (2 * NumKHeads * HeadKDim)), (f32A +% sG),
      (bfA +% sBeta), 32, 16, 2,
      local mod (HeadVDim div 8), local div (HeadVDim div 8),
      128, 128, 8)
    waveAdd(counters, 7)
  elif tx <= 843:
    waveWait(counters, 7, 512)
    waveWait(counters, 2, 64)
    rmsNormGatedTileAt((bfA +% sNormed), (bfA +% sY), (bfA +% sZ), onormW,
      32, eps, tx - 840, 128, 8)
    waveAdd(counters, 8)
  elif tx <= 875:
    waveWait(counters, 8, 4)
    dense_linear_tile_fwd[bfloat16, 2048, 4096, 64](
      (bfA +% sBlockOut), (bfA +% sNormed), outprojW, 1, tx - 844, 0)
    waveAdd(counters, 9)
  elif tx == 876:
    when HaveNorm:
      # Stage 11:
      #   the residual fold plus the bias-one post-LN norm,
      # one bf16 round each, the fold's sum becoming
      # the new residual of the deferred-add contract.
      waveWait(counters, 9, 32)
      normRow((bfA +% sStream), (bfA +% sBlockOut), norm2W,
        (bfA +% sH1), (bfA +% sNormed2), eps)
    waveAdd(counters, 10)
  elif tx <= 885:
    when HaveNorm:
      waveWait(counters, 10, 1)
      moe_fwd_decode_at[2048, 256, 8, 512, 1.0'f32, true]((f32A +% sPartial), (bfA +% sNormed2), routerW, gateUpW,
        downW, sharedGW, sharedUW, sharedDW, sharedGVW,
        (bfA +% sH), (bfA +% sHs), 0, tx - 877)
    waveAdd(counters, 11)
  else:
    when HaveNorm:
      waveWait(counters, 11, 9)
      moe_decode_merge_at[bfloat16, 2048, 8](
        (bfA +% sMoeOut), (f32A +% sPartial), 0, tx - 886)
    waveAdd(counters, 12)
