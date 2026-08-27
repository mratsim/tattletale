## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_moe_fp16.nim

import std/strformat
import workspace/[crucible, libtorch, libtorch_testutils]
import workspace/libtorch as F
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../src/kernels/ceramic/moe_fwd
import ./attn_test_utils

const moeMsl = metal:
  proc moeRun(out_r, x, router_w, gate_up_w, down_w, shared_gate_up_w, shared_down_w,
      h_scratch, hs_scratch: ptr UncheckedArray[float16], num_tokens: int32) {.global.} =
    moe_fwd(out_r, x, router_w, gate_up_w, down_w, shared_gate_up_w, shared_down_w,
      h_scratch, hs_scratch, num_tokens)

type MoEG = tuple[xf, rwf, guf, dnf, sguf, sdwf: seq[float32]]

proc genMoE(T: int): MoEG =
  (scaledRand(T, 2048, 0.3'f32), scaledRand(64, 2048, 0.125'f32),
   scaledRand(64 * 3072, 2048, 0.125'f32), scaledRand(64 * 2048, 1536, 0.125'f32),
   scaledRand(3072, 2048, 0.125'f32), scaledRand(2048, 1536, 0.125'f32))

proc moeKernel(g: MoEG, T: int): F.Tensor =
  var engine = bkMetal.init()
  engine.ingest(moeMsl)
  var outO = newSeq[uint16](T * 2048)
  var hScr = newSeq[uint16](T * 4 * 1536)
  var hsScr = newSeq[uint16](T * 1536)
  engine.run << (grid: (T, 1, 1), blk: (32, 1)) >> ("moeRun", outO,
    (fp32sToFp16(g.xf), fp32sToFp16(g.rwf), fp32sToFp16(g.guf), fp32sToFp16(g.dnf),
     fp32sToFp16(g.sguf), fp32sToFp16(g.sdwf), hScr, hsScr, int32(T)))
  toTensor(fp16sToF32(outO)).reshape(T, 2048)

proc moeReference(g: MoEG, T: int): F.Tensor =
  let x32 = w32(g.xf, T, 2048)
  let logits = F.linear(x32, w32(g.rwf, 64, 2048))
  let s = (1.0'f32 + (-logits).exp()).reciprocal()
  let (_, idx) = F.sort(s, axis = 1, descending = true)
  let top4 = idx.narrow(1, 0, 4)
  var w = s.gather(1, top4)
  w = w / (w.sum(1, keepdim = true) + 1e-20'f32)
  w = w * 1.8'f32
  let xg = x32.unsqueeze(1).expand(T, 4, 2048, implicit = false).reshape(T * 4, 2048)
  let gu4 = toTensor(fp16sToF32(fp32sToFp16(g.guf))).reshape(64, 3072, 2048)
    .index_select(0, top4.reshape(-1))
  let gu = F.bmm(xg.unsqueeze(1), gu4.transpose(1, 2)).squeeze(1).chunk(2, dim = 1)
  let h16 = (F.silu(gu[0]) * gu[1]).to(kFloat16).to(kFloat32)
  let dn4 = toTensor(fp16sToF32(fp32sToFp16(g.dnf))).reshape(64, 2048, 1536)
    .index_select(0, top4.reshape(-1))
  let oe = F.bmm(h16.unsqueeze(1), dn4.transpose(1, 2)).squeeze(1).reshape(T, 4, 2048)
  let routed = (oe * w.unsqueeze(2)).sum(1)
  let sgu = F.linear(x32, w32(g.sguf, 3072, 2048)).chunk(2, dim = 1)
  let hs = F.silu(sgu[0]) * sgu[1]
  let shared = F.linear(hs.to(kFloat16).to(kFloat32), w32(g.sdwf, 2048, 1536))
  (routed + shared).to(kFloat16).to(kFloat32)

proc checkMoe(): bool =
  ## T=8 and T=4 against the torch reference.
  ## The 1e-2 bound covers the 4-slot routed chain: gate_up, silu,
  ## one fp16 h round, and down. The weighted sum and the shared MLP
  ## fit within the bound at the 0.125 weight scale. A wrong top-4
  ## set, expert index mapping or scale shows up as O(1) errors.
  Torch.manual_seed(0x5EED'u64)
  for T in [8, 4]:
    let g = genMoE(T)
    let actual = moeKernel(g, T)
    let expected = moeReference(g, T)
    echo &"  T={T}: worst |Δ| = {worstAbsDiff(actual, expected)}"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 1e-2'f64)
  result = true

when isMainModule:
  runCppTest("moe_fwd vs the torch reference", checkMoe)
