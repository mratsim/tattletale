# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Naive composition of one token's full GDN decoder layer pass, mirroring
## the `qwen35_moe` mega kernel over the landed naive ops, bf16 storage handoffs
## between stages, fp32 state, the comparison tier's reference side.
##
## Stage order, mirroring the mega module's stage table:
##
## | stage | role        | stage | role         |
## | ----- | ----------- | ----- | ------------ |
## | 1     | add + norm1 | 8     | GDN step     |
## | 2     | qkv GEMV    | 9     | o_norm       |
## | 3     | z GEMV      | 10    | out_proj     |
## | 4     | a/b GEMV    | 11    | fold + norm2 |
## | 5     | conv + ring | 12    | MoE decode   |
## | 6     | q/k l2norm  | 13    | merge        |
## | 7     | g/beta      |       |              |
##
## Rounding contract, where the mega rounds:
##
## | site      | rounding                                                                                          |
## | --------- | ------------------------------------------------------------------------------------------------- |
## | handoffs  | every stage handoff stores one bf16 round (the arena sections)                                    |
## | GDN state | fp32 through the step, the core output rounds once                                                |
## | MoE body  | fp32 projections (the mma accumulators), activation bf16(silu)·u, down walk fp32 into the partial |
## | merge     | the fp32 partials summed in slot order, one bf16 round                                            |

import std/math
import naive_tensors
import naive_layer_ops
import naive_gdn
import naive_grouped_mm

# ─── Geometry, the Qwen bf16 class (the mega module's constants) ──────

const
  H* = 2048
    ## Layer width, the norm rows, the MoE hidden and the out_proj rows.
  ConvDim* = 8192
    ## Fused qkv projection width.
  Hv* = 32
  Hk* = 16
  Dk* = 128
  Dv* = 128
  ConvKernel* = 4
  TopK* = 8
  NumExperts* = 256
  Inter* = 512
  HkRatio* = 2

type LayerOut* = object
  ## One launch's intermediate and output sections, all bf16 bit patterns
  ## except the fp32 fields, named after the mega module's arena sections:
  ##
## | field    | section                                        |
## | -------- | ---------------------------------------------- |
## | stream   | the residual add, the new residual stream (H)  |
## | norm1    | norm1 output row (H)                           |
## | qkvCol   | fused qkv projection column (ConvDim)          |
## | z        | o_norm z row (Hv·Dv)                           |
## | a        | decay projection row (Hv)                      |
## | b        | beta projection row (Hv)                       |
## | conv     | conv + silu output column (ConvDim)            |
## | qn       | l2-normalized q heads (Hk·Dk)                  |
## | kn       | l2-normalized k heads (Hk·Dk)                  |
## | g        | log-decay values (Hv), fp32 end to end         |
## | beta     | beta values (Hv)                               |
## | y        | GDN core output rows (Hv·Dv)                   |
## | normed   | o_norm output rows, the out_proj input (Hv·Dv) |
## | blockOut | out_proj row before the fold (H)               |
## | h1       | folded residual row (H), the next addend       |
## | normed2  | post-LN normed row, the MoE input (H)          |
## | moeOut   | merged MoE output row (H)                      |

  stream*: seq[uint16]
  norm1*: seq[uint16]
  qkvCol*: seq[uint16]
  z*: seq[uint16]
  a*: seq[uint16]
  b*: seq[uint16]
  conv*: seq[uint16]
  qn*: seq[uint16]
  kn*: seq[uint16]
  g*: seq[float32]
  beta*: seq[uint16]
  y*: seq[uint16]
  normed*: seq[uint16]
  blockOut*: seq[uint16]
  h1*: seq[uint16]
  normed2*: seq[uint16]
  moeOut*: seq[uint16]

proc moeDecodeBody*(x: seq[uint16];
    routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW: seq[uint16]):
    tuple[ids: seq[int32], w: seq[float32], partial: seq[float32], moeOut: seq[uint16]] =
  ## One token's MoE decode body, the mega's slot-group walk over the landed
  ## grouped-GEMM reference:
  ##
## | part          | contract                                                                          |
## | ------------- | --------------------------------------------------------------------------------- |
## | projections   | fp32 sums (the mma accumulator form), one-expert cubes over the weight rows       |
## | activation    | h = bf16(bf16(silu(g))·u) per element                                             |
## | down walk     | fp32 into the partial row, scaled by the routing weight                           |
## | shared expert | the scalar's gated contribution in the last partial row, the merge one bf16 round |
  ##
  ## Returns:
  ## - ids, w, the recomputed router's top-K expert ids and weights
  ## - partial, the (K+1, H) fp32 partial rows in the mega's partial contract
  ## - moeOut, the merged output row
  let (ids, w) = naiveSoftmaxTopKRouter(x, routerW, NumExperts, H, TopK, 1.0'f32)
  result.partial = newSeq[float32]((TopK + 1) * H)
  for slot in 0 ..< TopK:
    let id = ids[slot].int
    # gate/up walk, one-expert cube over the fused (2I, H) weight rows
    let guData = gateUpW[(id * 2 * Inter) * H ..< ((id + 1) * 2 * Inter) * H]
    let guCube = NaiveCube[uint16](planes: 1, rows: 2 * Inter, cols: H, data: guData)
    let gu = naiveGroupedMmSums(gmmBf16, NaiveMat[uint16](rows: 1, cols: H, data: x),
      guCube, @[1'i32])
    var hBits = newSeq[uint16](Inter)
    for i in 0 ..< Inter:
      hBits[i] = naiveSiluMulEl(gu.data[i], gu.data[Inter + i])
    # down walk, one-expert cube (H, I) over the expert's (H, I) weight rows
    let dnData = downW[id * H * Inter ..< (id + 1) * H * Inter]
    let dnCube = NaiveCube[uint16](planes: 1, rows: H, cols: Inter, data: dnData)
    let dn = naiveGroupedMmSums(gmmBf16, NaiveMat[uint16](rows: 1, cols: Inter, data: hBits),
      dnCube, @[1'i32])
    for e in 0 ..< H:
      result.partial[slot * H + e] = w[slot] * dn.data[e]
  # shared expert walk, the scalar then the separate projections
  let gateVal = naiveSharedGate(x, sharedGVW, H)
  let sg = naiveGroupedMmSums(gmmBf16, NaiveMat[uint16](rows: 1, cols: H, data: x),
    NaiveCube[uint16](planes: 1, rows: Inter, cols: H, data: sharedGW), @[1'i32])
  let su = naiveGroupedMmSums(gmmBf16, NaiveMat[uint16](rows: 1, cols: H, data: x),
    NaiveCube[uint16](planes: 1, rows: Inter, cols: H, data: sharedUW), @[1'i32])
  var hsBits = newSeq[uint16](Inter)
  for i in 0 ..< Inter:
    hsBits[i] = naiveSiluMulEl(sg.data[i], su.data[i])
  let sd = naiveGroupedMmSums(gmmBf16, NaiveMat[uint16](rows: 1, cols: Inter, data: hsBits),
    NaiveCube[uint16](planes: 1, rows: H, cols: Inter, data: sharedDW), @[1'i32])
  for e in 0 ..< H:
    result.partial[TopK * H + e] = gateVal * sd.data[e]
  result.ids = ids
  result.w = w
  result.moeOut = naiveMoeMerge(result.partial, TopK, H)

proc naiveQwen35GdnLayer*(state: var NaiveCube[float32], ring: var seq[uint16];
    x, r: seq[uint16];
    norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W: seq[uint16];
    routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW: seq[uint16];
    aLog: seq[float32], dtBias: seq[uint16]; eps: float32): LayerOut =
  ## One token's layer pass over the landed naive ops, the mega's 13-stage order,
  ## the state and ring updated in place.
  ##
  ## Returns the LayerOut record, every arena section the comparison tier
  ## reads per stage.
  doAssert x.len == H and r.len == H, "x/r width mismatch"
  doAssert qkvW.len == ConvDim * H, "qkv weight shape mismatch"
  doAssert convW.len == ConvDim * ConvKernel, "conv weight shape mismatch"

  # Stage 1, add + norm1
  let n1 = naiveRmsNormRes(x, r, norm1W, H, eps)
  result.stream = n1.stream
  result.norm1 = n1.normed

  # Stage 2, qkv GEMV (one 32-row block at M = 1)
  result.qkvCol = naiveDenseLinear(n1.normed, qkvW, ConvDim, H)

  # Stage 3, z GEMV
  result.z = naiveDenseLinear(n1.normed, zW, Hv * Dv, H)

  # Stage 4, a/b GEMVs
  result.a = naiveDenseLinear(n1.normed, aW, Hv, H)
  result.b = naiveDenseLinear(n1.normed, bW, Hv, H)

  # Stage 5, conv + ring roll, the conv input column the fused qkv projection column
  result.conv = naiveCausalConvSiluStep(convW, ring, result.qkvCol, ConvDim, ConvKernel)

  # Stage 6, q/k l2norm, 16 heads × 128 each, q from rows 0..2048, k from 2048..4096
  result.qn = newSeq[uint16](Hk * Dk)
  result.kn = newSeq[uint16](Hk * Dk)
  for h in 0 ..< Hk:
    result.qn[h * Dk ..< (h + 1) * Dk] =
      naiveL2NormRow(result.conv[h * Dk ..< (h + 1) * Dk], Dk)
    result.kn[h * Dk ..< (h + 1) * Dk] =
      naiveL2NormRow(result.conv[(Hk * Dk) + h * Dk ..< (Hk * Dk) + (h + 1) * Dk], Dk)

  # Stage 7, g/beta
  let gates = naiveGdnGates(result.a, result.b, dtBias, aLog, Hv)
  result.g = gates.g
  result.beta = gates.beta

  # Stage 8, GDN step, the value rows read from the conv column's value channels
  var qMat = NaiveMat[float32](rows: Hk, cols: Dk)
  qMat.data = newSeq[float32](Hk * Dk)
  var kMat = NaiveMat[float32](rows: Hk, cols: Dk)
  kMat.data = newSeq[float32](Hk * Dk)
  var vMat = NaiveMat[float32](rows: Hv, cols: Dv)
  vMat.data = newSeq[float32](Hv * Dv)
  var betaF = newSeq[float32](Hv)
  for i in 0 ..< Hk * Dk:
    qMat.data[i] = bf16ToF32(result.qn[i])
    kMat.data[i] = bf16ToF32(result.kn[i])
  for bh in 0 ..< Hv:
    betaF[bh] = bf16ToF32(result.beta[bh])
    for d in 0 ..< Dv:
      vMat.data[bh * Dv + d] =
        bf16ToF32(result.conv[(2 * Hk * Dk) + bh * Dv + d])
  var yMat = NaiveMat[float32](rows: Hv, cols: Dv)
  yMat.data = newSeq[float32](Hv * Dv)
  gdnDecodeStep(state, yMat, qMat, kMat, vMat, betaF, result.g, Hv, Hk, HkRatio)
  result.y = newSeq[uint16](Hv * Dv)
  for i in 0 ..< Hv * Dv:
    result.y[i] = f32ToBf16(yMat.data[i])

  # Stage 9, o_norm, one gated RMSNorm row per value head
  result.normed = newSeq[uint16](Hv * Dv)
  for bh in 0 ..< Hv:
    result.normed[bh * Dv ..< (bh + 1) * Dv] = naiveRmsNormGated(
      result.y[bh * Dv ..< (bh + 1) * Dv],
      result.z[bh * Dv ..< (bh + 1) * Dv],
      onormW[bh * Dv ..< (bh + 1) * Dv], Dv, eps)

  # Stage 10, out_proj
  result.blockOut = naiveDenseLinear(result.normed, outprojW, H, Hv * Dv)

  # Stage 11, fold + norm2, the fold's sum the new residual
  let n2 = naiveRmsNormRes(result.stream, result.blockOut, norm2W, H, eps)
  result.h1 = n2.stream
  result.normed2 = n2.normed

  # Stages 12 + 13, the MoE decode then the merge
  let moe = moeDecodeBody(result.normed2, routerW, gateUpW, downW,
    sharedGW, sharedUW, sharedDW, sharedGVW)
  result.moeOut = moe.moeOut
