## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_mla_latent_fp16.nim
import std/strformat
import workspace/[crucible, libtorch, libtorch_testutils]
import workspace/libtorch as F
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../src/kernels/ceramic/mla_latent_fwd
import ./attn_test_utils
const mlaMsl = metal:
  proc mlaLatent(combined: ptr UncheckedArray[float16], x, qa_w, qa_g, qb_w, kva_w, kva_g, kvb_w: ptr UncheckedArray[float16],
      cos_t, sin_t: ptr UncheckedArray[float32], num_tokens, num_heads: int32) {.global.} =
    let n = num_tokens * num_heads * 256
    mla_latent_fwd(combined, combined +% int(n), combined +% (2 * int(n)),
      x, qa_w, qa_g, qb_w, kva_w, kva_g, kvb_w, cos_t, sin_t, num_tokens, num_heads)
proc w32(fs: seq[float32], shape: varargs[int]): F.Tensor =
  toTensor(fp16sToF32(fp32sToFp16(fs))).reshape(shape)
proc dup64(a: seq[float32], T: int): seq[float32] =
  ## The model's emb = cat((freqs, freqs)): entry c = freq (c mod 32).
  for r in 0 ..< T:
    for half in 0 ..< 2:
      for c in 0 ..< 32:
        result.add a[r * 32 + c]
proc gammaVals(n: int): seq[float32] =
  for c in 0 ..< n:
    result.add 0.5'f32 + float32(((7 * c + 55 + 13 * (c div 32)) mod 16)) / 16.0'f32
proc ropePairs(r, cos32, sin32: F.Tensor): F.Tensor =
  let rb = r.reshape(r.size(0), r.size(1), 32, 2)
  let c32 = cos32.unsqueeze(1)
  let s32 = sin32.unsqueeze(1)
  let a = rb.narrow(3, 0, 1).squeeze(3)
  let b = rb.narrow(3, 1, 1).squeeze(3)
  F.cat([(a * c32 - b * s32).unsqueeze(3), (b * c32 + a * s32).unsqueeze(3)], 3).reshape(r.size(0), r.size(1), 64)
proc genIn(T, H: int): tuple[xf, qaWF, qaGF, qbWF, kvaWF, kvaGF, kvbWF: seq[float32]] =
  (scaledRand(T, 2048, 0.125'f32), scaledRand(768, 2048, 0.125'f32), gammaVals(768),
   scaledRand(H * 256, 768, 0.125'f32), scaledRand(576, 2048, 0.125'f32), gammaVals(512),
   scaledRand(H * 448, 512, 0.125'f32))
proc mlaKernel(g: tuple[xf, qaWF, qaGF, qbWF, kvaWF, kvaGF, kvbWF: seq[float32]],
               cosF, sinF: seq[float32], T, H: int): F.Tensor =
  var engine = bkMetal.init()
  engine.ingest(mlaMsl)
  var outAll = newSeq[uint16](3 * T * H * 256)
  engine.run << (grid: (T div 8, H div 2, 1), blk: (32, 1)) >> ("mlaLatent", outAll,
    (fp32sToFp16(g.xf), fp32sToFp16(g.qaWF), fp32sToFp16(g.qaGF), fp32sToFp16(g.qbWF),
     fp32sToFp16(g.kvaWF), fp32sToFp16(g.kvaGF), fp32sToFp16(g.kvbWF), cosF, sinF,
     int32(T), int32(H)))
  let n = T * H * 256
  let q = toTensor(fp16sToF32(outAll[0 ..< n])).reshape(T, H, 256)
  F.cat([q, toTensor(fp16sToF32(outAll[n ..< 2 * n])).reshape(T, H, 256),
         toTensor(fp16sToF32(outAll[2 * n ..< 3 * n])).reshape(T, H, 256)], 0).reshape(-1)
proc mlaReference(g: tuple[xf, qaWF, qaGF, qbWF, kvaWF, kvaGF, kvbWF: seq[float32]],
                  cos32, sin32: seq[float32], T, H: int): F.Tensor =
  let x32 = w32(g.xf, T, 2048)
  let qn = F.rms_norm(F.linear(x32, w32(g.qaWF, 768, 2048)), 768, w32(g.qaGF, 768), 1e-5).to(kFloat16).to(kFloat32)
  let q = F.linear(qn, w32(g.qbWF, H * 256, 768)).reshape(T, H, 256)
  let kv = F.linear(x32, w32(g.kvaWF, 576, 2048))
  let kp = F.rms_norm(kv.narrow(1, 0, 512), 512, w32(g.kvaGF, 512), 1e-5).to(kFloat16).to(kFloat32)
  let kb = F.linear(kp, w32(g.kvbWF, H * 448, 512)).reshape(T, H, 448)
  let c32 = toTensor(cos32).reshape(T, 32)
  let s32 = toTensor(sin32).reshape(T, 32)
  let qRotR = ropePairs(q.narrow(2, 192, 64), c32, s32)
  let kRotR = ropePairs(kv.narrow(1, 512, 64).unsqueeze(1), c32, s32).expand(T, H, 64, implicit = false)
  F.cat([F.cat([q.narrow(2, 0, 192), qRotR], 2).to(kFloat16).to(kFloat32),
         F.cat([kb.narrow(2, 0, 192), kRotR], 2).to(kFloat16).to(kFloat32),
         kb.narrow(2, 192, 256).to(kFloat16).to(kFloat32)], 0).reshape(-1)
proc checkMlaLatent(): bool =
  ## T=8/H=4 and T=16/H=20 (the real head count). The 1e-2 bound
  ## covers 4 chained matmuls, 2 norms and the rope at the 0.125 scale
  ## (observed worst ~1e-3).
  Torch.manual_seed(0x5EED'u64)
  for (T, H) in [(8, 4), (16, 20)]:
    let g = genIn(T, H)
    let (cos32, sin32) = buildRopeCosSin(T, 64, 1e6'f32)
    let actual = mlaKernel(g, dup64(cos32, T), dup64(sin32, T), T, H)
    let expected = mlaReference(g, cos32, sin32, T, H)
    echo &"  T={T} H={H}: worst |Δ| = {worstAbsDiff(actual, expected)}"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 1e-2'f64)
  result = true
when isMainModule:
  runCppTest("mla_latent_fwd vs the torch reference", checkMlaLatent)