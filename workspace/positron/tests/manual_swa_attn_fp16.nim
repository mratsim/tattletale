## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip \
##   --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_swa_attn_fp16.nim

import std/[strformat, math, options]
import workspace/crucible
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../src/kernels/ceramic/swa_attn
import ./attn_test_utils

const swaMsl = metal:
  proc swaD64(
      o, q, k, v: ptr UncheckedArray[float16],
      num_qo, num_kv, q_offset, H, Nkv, window: int32) {.global.} =
    swa_attn_fwd(o, q, k, v, num_qo, num_kv, q_offset, H, Nkv, window, 64)
  proc swaD128(
      o, q, k, v: ptr UncheckedArray[float16],
      num_qo, num_kv, q_offset, H, Nkv, window: int32) {.global.} =
    swa_attn_fwd(o, q, k, v, num_qo, num_kv, q_offset, H, Nkv, window, 128)

proc swaAttnKernel(qf, kf, vf: seq[float32], numQo, numKv, qOffset, H, Nkv, window, D: int): F.Tensor =
  ## Runs the D-specialized wrapper on the fp16-rounded inputs, returns the fp16 output widened to fp32.
  var engine = bkMetal.init()
  engine.ingest(swaMsl)
  let args = (fp32sToFp16(qf), fp32sToFp16(kf), fp32sToFp16(vf),
              int32(numQo), int32(numKv), int32(qOffset), int32(H), int32(Nkv), int32(window))
  var outO = newSeq[uint16](numQo * H * D)
  engine.run << (grid: ((numQo + 7) div 8, H, 1), blk: (32, 1)) >> ("swaD" & $D, outO, args)
  result = toTensor(fp16sToF32(outO)).reshape(numQo, H, D)

proc swaAttnReference(qf, kf, vf: seq[float32], numQo, numKv, qOffset, H, Nkv, window, D: int): F.Tensor =
  ## torch SDPA over the same fp16-rounded values: q (1, H, num_qo, D),
  ## k/v (1, Nkv, num_kv, D), scale 1.0, the banded window mask as an
  ## additive bias, GQA on when H > Nkv.
  let qT = toTensor(qf).reshape(numQo, H, D).to(kFloat16).to(kFloat32)
  let kT = toTensor(kf).reshape(numKv, Nkv, D).to(kFloat16).to(kFloat32)
  let vT = toTensor(vf).reshape(numKv, Nkv, D).to(kFloat16).to(kFloat32)
  let q4 = qT.reshape(1, numQo, H, D).transpose(1, 2)
  let k4 = kT.reshape(1, numKv, Nkv, D).transpose(1, 2)
  let v4 = vT.reshape(1, numKv, Nkv, D).transpose(1, 2)
  var maskF = newSeq[float32](numQo * numKv)
  for i in 0 ..< numQo:
    for j in 0 ..< numKv:
      maskF[i * numKv + j] = if j <= qOffset + i and j >= qOffset + i - window + 1: 0.0'f32 else: float32(NegInf)
  let mt = toTensor(maskF).reshape(1, numQo, numKv)
  let o4 = scaled_dot_product_attention(q4, k4, v4, attn_mask = some(mt),
             scale = some(1.0'f64), enable_gqa = H > Nkv)
  result = o4.transpose(1, 2).reshape(numQo, H, D)

proc checkSwaAttn(): bool =
  ## Five cases vs torch SDPA: window 512 (causal only), window 16
  ## (band binds), GQA with D = 64, a decode query at the tail, and a
  ## short decode window at the causal edge. The 5e-3 bound sits ~7x
  ## above the observed 7e-4 accumulation-order noise between the
  ## torch matmuls and the Metal mma, so a wrong band edge, scale or
  ## GQA mapping shows up as O(1) errors.
  Torch.manual_seed(0x5EED'u64)
  let cases = [(8, 1, 128, 64, 64, 0, 512), (8, 1, 128, 128, 128, 0, 16),
               (4, 2, 64, 40, 40, 0, 12), (8, 1, 128, 1, 33, 32, 8),
               (4, 2, 64, 4, 28, 24, 8)]
  var worstAll = 0.0'f32
  for d in 0 ..< cases.len:
    let (H, Nkv, D, numQo, numKv, qOffset, window) = cases[d]
    let qf = scaledRand(numQo, H * D, 2.0'f32)
    let kf = scaledRand(numKv, Nkv * D, 2.0'f32)
    let vf = scaledRand(numKv, Nkv * D, 2.0'f32)
    let actual = swaAttnKernel(qf, kf, vf, numQo, numKv, qOffset, H, Nkv, window, D)
    let expected = swaAttnReference(qf, kf, vf, numQo, numKv, qOffset, H, Nkv, window, D)
    let w = worstAbsDiff(actual, expected)
    if w > worstAll: worstAll = w
    echo &"  H={H} Nkv={Nkv} D={D} num_qo={numQo} num_kv={numKv} q_offset={qOffset} window={window}: worst |Δ| = {w}"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  echo &"  worst |Δ| across {cases.len} cases = {worstAll} (tolerance 5e-3)"
  result = true

when isMainModule:
  runCppTest("swa_attn_fwd vs torch SDPA (window mask)", checkSwaAttn)
