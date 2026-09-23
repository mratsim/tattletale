# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests workspace/positron/tests/naive/t_naive_qwen35_layer.nim
##
## Naive fused GDN decoder layer composition suite, checking
## worked examples of the stage ops, then the full-geometry one-token
## layer pass at the Qwen bf16 class.
##
## The composition is the mega kernel's reference implementation,
## and the comparison tier owns the bands.
##
## | check       | contract                                                                  |
## | ----------- | ------------------------------------------------------------------------- |
## | rms-norm    | the H = 2 worked example, exact small values                              |
## | l2norm      | the uniform row, inv = bf16(1/sqrt(acc + 1e-6)) exactly                   |
## | gates       | softplus(0) = ln 2, sigmoid(0) = 1/2                                      |
## | router      | the E = 4, K = 2 worked example, ids by score, lowest index on a tie      |
## | determinism | the same seed reproduces the same bit patterns across fresh walks         |
## | relaunch    | a fresh walk over reset inputs is bit-identical to the first walk         |
## | checksum    | a recorded hash over the seeded run's outputs, stage corruption shifts it |

import std/[strformat, math, hashes, sequtils]
import naive_tensors, naive_rng, naive_layer_ops, naive_gdn, naive_qwen35_layer

const Eps = 1.0e-6'f32

# ─── Unit worked examples ─────────────────────────────────────────────

proc pinRmsNormRes() =
  ## H = 2 exact example
  ## - x = [1.0, 0.0], r = 0, w = 0, eps = 0
  ## - rstd = sqrt(2), normed = [bf16(sqrt(2)), 0]
  let x = @[f32ToBf16(1.0'f32), f32ToBf16(0.0'f32)]
  let r = @[0'u16, 0'u16]
  let w = @[f32ToBf16(0.0'f32), f32ToBf16(0.0'f32)]
  let got = naiveRmsNormRes(x, r, w, 2, 0.0'f32)
  let want0 = f32ToBf16(sqrt(2.0'f32))
  doAssert got.stream[0] == f32ToBf16(1.0'f32) and got.stream[1] == 0'u16
  doAssert got.normed[0] == want0, &"normed[0] {got.normed[0]} want {want0}"
  doAssert got.normed[1] == 0'u16
  echo "[rms-norm] ok"

proc pinL2Norm() =
  ## Uniform row, x[c] = 1 over 128 columns gives acc = 128,
  ## inv = bf16(1/sqrt(128 + 1e-6)), out[c] = bf16(inv).
  let x = newSeqWith(128, f32ToBf16(1.0'f32))
  let got = naiveL2NormRow(x, 128)
  let inv = f32ToBf16(1.0'f32 / sqrt(128.0'f32 + 1.0e-6'f32))
  for c in 0 ..< 128:
    doAssert got[c] == inv, &"l2norm[{c}] {got[c]} want {inv}"
  echo "[l2norm] ok"

proc pinL2NormSumRound() =
  ## Sum-round discriminator over 16 bf16 columns, the recorded chain's
  ## two rounds and the single round at acc + eps land a bf16 grid step
  ## apart on this row:
  ##
  ## - acc = 7.703125, exactly the bf16 midpoint between 7.6875 and 7.71875
  ##   (RNE resolves the tie downward to 7.6875)
  ## - the recorded chain bf16(acc) = 7.6875, the eps add stays there,
  ##   inv = bf16(1/sqrt(7.6875)) = 0.361328125
  ## - the single-round spelling rounds acc + 1e-6 to 7.71875, giving
  ##   inv 0.359375, and every output element differs
  ##
  ## The wanted inv bits and the 16 wanted output elements are recorded
  ## literals of the two-round chain.
  let x = @[
    0xBF7A'u16, 0x3F7C'u16, 0xBF64'u16, 0x3F2D'u16, 0x3F3C'u16, 0x3F07'u16,
    0x3F39'u16, 0xBEAF'u16, 0xBF64'u16, 0x3F33'u16, 0x3F18'u16, 0x3E06'u16,
    0x3EA4'u16, 0xBF6E'u16, 0xBF01'u16, 0x3EE8'u16]
  let want = @[
    0xBEB5'u16, 0x3EB6'u16, 0xBEA5'u16, 0x3E7A'u16, 0x3E88'u16, 0x3E43'u16,
    0x3E86'u16, 0xBDFD'u16, 0xBEA5'u16, 0x3E81'u16, 0x3E5C'u16, 0x3D42'u16,
    0x3DED'u16, 0xBEAC'u16, 0xBE3A'u16, 0x3E28'u16]
  let got = naiveL2NormRow(x, 16)
  for c in 0 ..< 16:
    doAssert got[c] == want[c], &"l2norm sum-round [{c}] {got[c]} want {want[c]}"
  echo "[l2norm sum-round] ok"

proc pinGates() =
  ## a = 0, dtBias = 0 gives softplus(0) = ln 2, g = -exp(A_log)·ln 2,
  ## b = 0 gives beta = bf16(1/2).
  let a = @[f32ToBf16(0.0'f32)]
  let b = @[f32ToBf16(0.0'f32)]
  let dtBias = @[f32ToBf16(0.0'f32)]
  let aLog = @[-1.0'f32]
  let got = naiveGdnGates(a, b, dtBias, aLog, 1)
  let wantG = -exp(-1.0'f32) * ln(2.0'f32)
  doAssert abs(got.g[0] - wantG) < 1.0e-7'f32, &"g {got.g[0]} want {wantG}"
  doAssert got.beta[0] == f32ToBf16(0.5'f32)
  echo "[gdn gates] ok"

proc pinRouter() =
  ## H = 1, x = [1.0], routerW = [3.0, 1.0, 2.0, 0.0], K = 2:
  ## logits exact, p ∝ (e³, e¹, e², 1), ids = [0, 2] by score order.
  let x = @[f32ToBf16(1.0'f32)]
  let routerW = @[
    f32ToBf16(3.0'f32), f32ToBf16(1.0'f32), f32ToBf16(2.0'f32), f32ToBf16(0.0'f32)]
  let got = naiveSoftmaxTopKRouter(x, routerW, 4, 1, 2, 1.0'f32)
  doAssert got.ids == @[0'i32, 2'i32], &"ids {got.ids}"
  # renormalized over the top-2 selected set, the routeToExperts
  # contract. p is proportional to (1, e⁻²) after the max subtraction,
  # w = p/(p₀+p₁)
  let e0 = exp(0.0'f32); let e2 = exp(2.0'f32 - 3.0'f32)
  let topSum = e0 + e2
  let want0 = bf16ToF32(f32ToBf16(e0 / topSum))
  let want1 = bf16ToF32(f32ToBf16(e2 / topSum))
  doAssert got.w[0] == want0, &"w[0] {got.w[0]} want {want0}"
  doAssert got.w[1] == want1, &"w[1] {got.w[1]} want {want1}"
  echo "[router] ok"

proc pinDenseLinear() =
  ## N = 3, K = 2 exact rows: [1, 0] reads 1, [0, 1] reads 2,
  ## [0.5, 0.5] reads 1.5, all sums exact in fp32 and in bf16.
  let x = @[f32ToBf16(1.0'f32), f32ToBf16(2.0'f32)]
  let w = @[
    f32ToBf16(1.0'f32), f32ToBf16(0.0'f32),
    f32ToBf16(0.0'f32), f32ToBf16(1.0'f32),
    f32ToBf16(0.5'f32), f32ToBf16(0.5'f32)]
  let got = naiveDenseLinear(x, w, 3, 2)
  doAssert got[0] == f32ToBf16(1.0'f32), &"row 0 {got[0]}"
  doAssert got[1] == f32ToBf16(2.0'f32), &"row 1 {got[1]}"
  doAssert got[2] == f32ToBf16(1.5'f32), &"row 2 {got[2]}"
  echo "[dense linear] ok"

proc pinConvSiluStep() =
  ## ConvDim = 2, kernel = 3, hand-computed taps and silu:
  ## channel 0 acc = 1·1 + 0.5·0 + 2·1 = 3, channel 1 acc = 0·2 + 1·1 + 1·1 = 2,
  ## each silu'd in fp32 then rounded bf16, the ring shifted down one slot.
  let convW = @[f32ToBf16(1.0'f32), f32ToBf16(0.5'f32), f32ToBf16(2.0'f32),
                f32ToBf16(0.0'f32), f32ToBf16(1.0'f32), f32ToBf16(1.0'f32)]
  var ring = @[f32ToBf16(1.0'f32), f32ToBf16(0.0'f32),
               f32ToBf16(2.0'f32), f32ToBf16(1.0'f32)]
  let xCol = @[f32ToBf16(1.0'f32), f32ToBf16(1.0'f32)]
  let got = naiveCausalConvSiluStep(convW, ring, xCol, 2, 3)
  let want0 = f32ToBf16(3.0'f32 / (1.0'f32 + exp(-3.0'f32)))
  let want1 = f32ToBf16(2.0'f32 / (1.0'f32 + exp(-2.0'f32)))
  doAssert abs(bf16ToF32(got[0]) - bf16ToF32(want0)) < 2.0e-3'f32,
    &"channel 0 {bf16ToF32(got[0])} want {bf16ToF32(want0)}"
  doAssert abs(bf16ToF32(got[1]) - bf16ToF32(want1)) < 2.0e-3'f32,
    &"channel 1 {bf16ToF32(got[1])} want {bf16ToF32(want1)}"
  doAssert ring == @[f32ToBf16(0.0'f32), f32ToBf16(1.0'f32),
                     f32ToBf16(1.0'f32), f32ToBf16(1.0'f32)],
    &"ring shifted wrong: {ring}"
  echo "[conv+silu] ok"

proc pinSiluMulEl() =
  ## g = 0 gives 0, g = 2, u = 1 gives bf16(silu(2)).
  doAssert naiveSiluMulEl(0.0'f32, 1.0'f32) == 0'u16
  let want = f32ToBf16(2.0'f32 / (1.0'f32 + exp(-2.0'f32)))
  let got = naiveSiluMulEl(2.0'f32, 1.0'f32)
  doAssert abs(bf16ToF32(got) - bf16ToF32(want)) < 2.0e-3'f32,
    &"got {bf16ToF32(got)} want {bf16ToF32(want)}"
  echo "[silu-mul] ok"

proc pinSharedGate() =
  ## x = [0, 0] gives logit 0, sigmoid 1/2; x = [1, 0], w = [1, 0] gives sigmoid(1).
  let zero = @[f32ToBf16(0.0'f32), f32ToBf16(0.0'f32)]
  let wOnes = @[f32ToBf16(1.0'f32), f32ToBf16(1.0'f32)]
  doAssert naiveSharedGate(zero, wOnes, 2) == bf16ToF32(f32ToBf16(0.5'f32))
  let one = @[f32ToBf16(1.0'f32), f32ToBf16(0.0'f32)]
  let wSel = @[f32ToBf16(1.0'f32), f32ToBf16(0.0'f32)]
  let want = bf16ToF32(f32ToBf16(1.0'f32 / (1.0'f32 + exp(-1.0'f32))))
  doAssert abs(naiveSharedGate(one, wSel, 2) - want) < 2.0e-3'f32
  echo "[shared gate] ok"

proc pinMoeMerge() =
  ## K = 1, H = 2, partial rows [0.5, 0.25] + [0.125, 0.0625],
  ## the slot-ordered sums exact in fp32 and in bf16.
  let partial = @[0.5'f32, 0.25'f32, 0.125'f32, 0.0625'f32]
  let got = naiveMoeMerge(partial, 1, 2)
  doAssert got[0] == f32ToBf16(0.625'f32), &"merge[0] {got[0]}"
  doAssert got[1] == f32ToBf16(0.3125'f32), &"merge[1] {got[1]}"
  echo "[moe merge] ok"

proc pinRmsNormGated() =
  ## Dv = 2, y = [1, 1], w = [1, 1], eps = 0: rstd = 1, normed = weighted = [1, 1],
  ## z = [2, 2] gives out[d] = bf16(silu(2)).
  let y = @[f32ToBf16(1.0'f32), f32ToBf16(1.0'f32)]
  let z = @[f32ToBf16(2.0'f32), f32ToBf16(2.0'f32)]
  let w = @[f32ToBf16(1.0'f32), f32ToBf16(1.0'f32)]
  let got = naiveRmsNormGated(y, z, w, 2, 0.0'f32)
  let want = f32ToBf16(2.0'f32 / (1.0'f32 + exp(-2.0'f32)))
  doAssert abs(bf16ToF32(got[0]) - bf16ToF32(want)) < 2.0e-3'f32, &"got[0] {got[0]}"
  doAssert abs(bf16ToF32(got[1]) - bf16ToF32(want)) < 2.0e-3'f32, &"got[1] {got[1]}"
  echo "[rms-norm gated] ok"

# ─── Full-geometry seeded walk ────────────────────────────────────────

type LayerInputs = object
  ## One seeded one-token layer pass's inputs and weights.
  x, r: seq[uint16]
  norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W: seq[uint16]
  routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW: seq[uint16]
  aLog: seq[float32]
  dtBias: seq[uint16]
  state: NaiveCube[float32]
  ring: seq[uint16]

proc randBits(rng: var NaiveRng; n: int; lo, hi: float32): seq[uint16] =
  ## `n` bf16 bit patterns of uniform samples in [lo, hi].
  result = newSeq[uint16](n)
  for i in 0 ..< n:
    result[i] = f32ToBf16(rng.nextF32(lo, hi))

proc buildInputs(seed: uint64): LayerInputs =
  ## Seeded one-token inputs and weights at the Qwen bf16 class geometry,
  ## modest magnitudes so no stage saturates.
  var rng = initNaiveRng(seed)
  result.x = randBits(rng, H, -1.0'f32, 1.0'f32)
  result.r = randBits(rng, H, -1.0'f32, 1.0'f32)
  result.norm1W = randBits(rng, H, -0.05'f32, 0.05'f32)
  result.qkvW = randBits(rng, ConvDim * H, -0.02'f32, 0.02'f32)
  result.zW = randBits(rng, Hv * Dv * H, -0.02'f32, 0.02'f32)
  result.aW = randBits(rng, Hv * H, -0.02'f32, 0.02'f32)
  result.bW = randBits(rng, Hv * H, -0.02'f32, 0.02'f32)
  result.convW = randBits(rng, ConvDim * ConvKernel, -0.1'f32, 0.1'f32)
  result.onormW = randBits(rng, Hv * Dv, -0.05'f32, 0.05'f32)
  result.outprojW = randBits(rng, H * Hv * Dv, -0.02'f32, 0.02'f32)
  result.norm2W = randBits(rng, H, -0.05'f32, 0.05'f32)
  result.routerW = randBits(rng, NumExperts * H, -0.02'f32, 0.02'f32)
  result.gateUpW = randBits(rng, NumExperts * 2 * Inter * H, -0.02'f32, 0.02'f32)
  result.downW = randBits(rng, NumExperts * H * Inter, -0.02'f32, 0.02'f32)
  result.sharedGW = randBits(rng, Inter * H, -0.02'f32, 0.02'f32)
  result.sharedUW = randBits(rng, Inter * H, -0.02'f32, 0.02'f32)
  result.sharedDW = randBits(rng, H * Inter, -0.02'f32, 0.02'f32)
  result.sharedGVW = randBits(rng, H, -0.02'f32, 0.02'f32)
  result.aLog = newSeq[float32](Hv)
  for h in 0 ..< Hv:
    result.aLog[h] = rng.nextF32(-2.0'f32, -0.1'f32)
  result.dtBias = randBits(rng, Hv, -0.1'f32, 0.1'f32)
  result.state = NaiveCube[float32](planes: Hv, rows: Dv, cols: Dk)
  result.state.data = newSeq[float32](Hv * Dv * Dk)
  for i in 0 ..< Hv * Dv * Dk:
    result.state.data[i] = rng.nextF32(-0.5'f32, 0.5'f32)
  result.ring = randBits(rng, ConvDim * (ConvKernel - 1), -1.0'f32, 1.0'f32)

proc hashBits(s: openArray[uint16]): uint64 =
  ## FNV-1a over the bit patterns.
  result = 0xcbf29ce484222325'u64
  for h in s:
    result = (result xor uint64(h)) * 0x100000001b3'u64

proc hashF32(s: openArray[float32]): uint64 =
  ## FNV-1a over the fp32 bit patterns, one uint32 per element.
  result = 0xcbf29ce484222325'u64
  for v in s:
    result = (result xor uint64(cast[uint32](v))) * 0x100000001b3'u64

proc walk(inp: LayerInputs): LayerOut =
  ## One fresh layer walk from copied state and ring, the composition pure
  ## over its inputs.
  var state = copyOf(inp.state)
  var ring = inp.ring
  naiveQwen35GdnLayer(state, ring, inp.x, inp.r,
    inp.norm1W, inp.qkvW, inp.zW, inp.aW, inp.bW, inp.convW, inp.onormW,
    inp.outprojW, inp.norm2W, inp.routerW, inp.gateUpW, inp.downW,
    inp.sharedGW, inp.sharedUW, inp.sharedDW, inp.sharedGVW, inp.aLog,
    inp.dtBias, Eps)

proc outDigest(o: LayerOut): string =
  ## Walk output checksum, one hex blob over the produced sections.
  var h = 0xcbf29ce484222325'u64
  template fold(s: untyped) =
    h = h xor hashBits(s)
    h = h * 0x9E3779B97F4A7C15'u64
  fold(o.stream); fold(o.norm1); fold(o.qkvCol); fold(o.z)
  fold(o.a); fold(o.b); fold(o.conv); fold(o.qn); fold(o.kn)
  fold(o.beta); fold(o.y); fold(o.normed); fold(o.blockOut)
  fold(o.h1); fold(o.normed2); fold(o.moeOut)
  h = h xor hashF32(o.g)
  h = h * 0x9E3779B97F4A7C15'u64
  &"{h:016X}"

const RecordedChecksum = "FEA2A8CEE67B24B8"
  ## Recorded checksum over the seed 0xC04D0601 walk's outputs,
  ## taken from the first green run.
  ##
  ## Re-recordings while keeping the walk itself unchanged:
  ## - routing weights switched to renormalizing over the top-K
  ##   selected set, the routeToExperts contract
  ## - recorded outputs gained fp32 g values

proc fullGeometryChecks() =
  ## Determinism across fresh walks, relaunch bit-identity, the recorded checksum.
  let inp = buildInputs(0xC04D0601'u64)
  let o1 = walk(inp)
  let d1 = outDigest(o1)
  let o2 = walk(inp)
  let d2 = outDigest(o2)
  doAssert d1 == d2, &"determinism: checksum {d1} then {d2}"
  doAssert d1 == RecordedChecksum,
    &"checksum {d1} != recorded {RecordedChecksum}, a stage op shifted"
  echo &"[full geometry] checksum {d1} deterministic"

echo "worked examples:"
pinRmsNormRes()
pinL2Norm()
pinL2NormSumRound()
pinGates()
pinRouter()
pinDenseLinear()
pinConvSiluStep()
pinSiluMulEl()
pinSharedGate()
pinMoeMerge()
pinRmsNormGated()
echo "full geometry:"
fullGeometryChecks()
echo "t_naive_qwen35_layer GREEN"
