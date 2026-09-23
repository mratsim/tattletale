## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run command, from the repo root:
## - nim cpp -r -d:release --hints:off --warnings:off --passC:-std=c++20 \
##   --outdir:build/wip --nimcache:nimcache/wip workspace/positron/tests/manual_ffn_moe_fp16.nim
import std/[strformat, strutils]
import workspace/[crucible, libtorch, libtorch_testutils]
import workspace/libtorch as F
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../src/kernels/ceramic/ffn_moe
import ./ceramic/moe_fwd_glm_baked_receipts
import ./attn_test_utils

const moeMsl = metal:
  proc moeRun(out_r, x, router_w, gate_up_w, down_w,
      shared_gate_up_w, shared_down_w, h_scratch,
      hs_scratch: ptr UncheckedArray[float16],
      num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
      n_shared_experts: int32, routed_scaling: float32,
      activation: int32) {.global.} =
    moe_fwd(out_r, x, router_w, gate_up_w, down_w, shared_gate_up_w,
      shared_down_w, h_scratch, hs_scratch, num_tokens, hidden,
      n_routed_experts, moe_intermediate, top_k, n_shared_experts,
      routed_scaling, activation)

type MoEG = tuple[xf, rwf, guf, dnf, sguf, sdwf: seq[float32]]

type MoeRow = object
  name: string
  tokens, hidden, nExperts, inter, topK, nShared: int
  scale: float32
  act: int32
  seed: uint64

const
  glm47Row = MoeRow(name: "glm47-flash", tokens: 8, hidden: 2048, nExperts: 64,
    inter: 1536, topK: 4, nShared: 1, scale: 1.8'f32, act: ActSilu,
    seed: 0x5EED)
  qwen36Row = MoeRow(name: "qwen36-35b-a3b", tokens: 2, hidden: 2048,
    nExperts: 256, inter: 512, topK: 8, nShared: 1, scale: 1.0'f32,
    act: ActSilu, seed: 0x5EED)
  raggedRow = MoeRow(name: "ragged-gelu", tokens: 2, hidden: 2064,
    nExperts: 70, inter: 200, topK: 4, nShared: 2, scale: 1.0'f32,
    act: ActGeluTanh, seed: 0x5EED)

proc genMoERow(r: MoeRow): MoEG =
  let gu = 2 * r.inter
  Torch.manual_seed(r.seed)
  (scaledRand(r.tokens, r.hidden, 0.3'f32),
   scaledRand(r.nExperts, r.hidden, 0.125'f32),
   scaledRand(r.nExperts * gu, r.hidden, 0.125'f32),
   scaledRand(r.nExperts * r.hidden, r.inter, 0.125'f32),
   scaledRand(r.nShared * gu, r.hidden, 0.125'f32),
   scaledRand(r.nShared * r.hidden, r.inter, 0.125'f32))

proc moeKernelBits(g: MoEG, r: MoeRow): seq[uint16] =
  var engine = bkMetal.init()
  engine.ingest(moeMsl)
  let gu = 2 * r.inter
  var outO = newSeq[uint16](r.tokens * r.hidden)
  var hScr = newSeq[uint16](r.tokens * r.topK * r.inter)
  var hsScr = newSeq[uint16](r.tokens * r.nShared * r.inter)
  engine.run << (grid: (r.tokens, 1, 1), blk: (32, 1)) >> ("moeRun",
    outO,
    (fp32sToFp16(g.xf), fp32sToFp16(g.rwf), fp32sToFp16(g.guf),
     fp32sToFp16(g.dnf), fp32sToFp16(g.sguf), fp32sToFp16(g.sdwf),
     hScr, hsScr, int32(r.tokens), int32(r.hidden), int32(r.nExperts),
     int32(r.inter), int32(r.topK), int32(r.nShared), r.scale, r.act))
  result = outO

proc moeBakedReceiptBits(seed: uint64, tokens: int): seq[uint16] =
  ## Output bits recorded from the deleted baked GLM-4.7-Flash proc,
  ## from the committed fixture, the delta-zero baseline.
  for r in moeGlmBakedReceipts:
    if r.seed.uint64 == seed and r.tokens == tokens:
      for chunk in split(r.bits, ","):
        result.add(parseHexInt(chunk).uint16)
      return
  doAssert false, "no receipt for the seed/tokens pair"

proc subseq(s: seq[float32], a, b: int): seq[float32] =
  result = newSeq[float32](b - a)
  for i in 0 ..< b - a:
    result[i] = s[a + i]

proc actMul(gate, up: F.Tensor, act: int32): F.Tensor =
  ## Activation variant the row selects, silu or the tanh-approximate
  ## gelu (the fused op the reference runtimes call).
  if act == ActGeluTanh:
    F.gelu(gate, "tanh") * up
  else:
    F.silu(gate) * up

proc moeReferenceRow(g: MoEG, r: MoeRow): F.Tensor =
  ## Torch chain of the generic contract at the row's dims.
  ## Sigmoid routing skeleton, activation variant, shared experts.
  let H = r.hidden
  let E = r.nExperts
  let I = r.inter
  let gu = 2 * I
  let x32 = w32(g.xf, r.tokens, H)
  let logits = F.linear(x32, w32(g.rwf, E, H))
  let s = (1.0'f32 + (-logits).exp()).reciprocal()
  let (_, idx) = F.sort(s, axis = 1, descending = true)
  let topK = idx.narrow(1, 0, r.topK)
  var w = s.gather(1, topK)
  w = w / (w.sum(1, keepdim = true) + 1e-20'f32)
  w = w * r.scale
  let xg = x32.unsqueeze(1).expand(r.tokens, r.topK, H, implicit = false)
    .reshape(r.tokens * r.topK, H)
  let gu4 = toTensor(fp16sToF32(fp32sToFp16(g.guf))).reshape(E, gu, H).index_select(0, topK.reshape(-1))
  let guX = F.bmm(xg.unsqueeze(1), gu4.transpose(1, 2)).squeeze(1).chunk(2, dim = 1)
  let h16 = actMul(guX[0], guX[1], r.act).to(kFloat16).to(kFloat32)
  let dn4 = toTensor(fp16sToF32(fp32sToFp16(g.dnf))).reshape(E, H, I)
    .index_select(0, topK.reshape(-1))
  let oe = F.bmm(h16.unsqueeze(1), dn4.transpose(1, 2)).squeeze(1)
    .reshape(r.tokens, r.topK, H)
  let routed = (oe * w.unsqueeze(2)).sum(1)
  var shared = F.zeros(r.tokens, H)
  for s in 0 ..< r.nShared:
    let sgu = F.linear(x32, w32(subseq(g.sguf, s * gu * H, (s + 1) * gu * H), gu, H)).chunk(2, dim = 1)
    let hs = actMul(sgu[0], sgu[1], r.act).to(kFloat16).to(kFloat32)
    shared = shared + F.linear(hs, w32(subseq(g.sdwf, s * H * I, (s + 1) * H * I),
        H, I))
  (routed + shared).to(kFloat16).to(kFloat32)

proc checkMoe(): bool =
  ## GLM-4.7-Flash row through the runtime-config entry, T=8 and T=4
  ## against the torch reference.
  ##
  ## - the 1e-2 bound covers the 4-slot routed chain, gate_up, silu,
  ##   one fp16 h round, and down
  ## - the weighted sum and the shared MLP fit within the 0.125-weight scale bound
  ## - a wrong top-4 set, expert index mapping or scale shows up as O(1) order errors
  for tokens in [8, 4]:
    let r = MoeRow(name: "glm47-flash", tokens: tokens, hidden: 2048,
      nExperts: 64, inter: 1536, topK: 4, nShared: 1, scale: 1.8'f32,
      act: ActSilu, seed: 0x5EED)
    let g = genMoERow(r)
    let actual = toTensor(fp16sToF32(moeKernelBits(g, r)))
      .reshape(tokens, r.hidden)
    let expected = moeReferenceRow(g, r)
    echo &"  T={tokens}: worst |Δ| = {worstAbsDiff(actual, expected)}"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 1e-2'f64)
  result = true

proc checkGenericDeltaZero(): bool =
  ## - the runtime-config entry's output bits equal the recorded baked
  ##   output bits at the GLM-4.7-Flash shape
  ## - three seeds over the T=8 and T=4 token counts
  for seed in [0x5EED'u64, 7'u64, 13'u64]:
    for tokens in [8, 4]:
      let r = MoeRow(name: "glm47-flash", tokens: tokens, hidden: 2048,
        nExperts: 64, inter: 1536, topK: 4, nShared: 1, scale: 1.8'f32,
        act: ActSilu, seed: seed)
      let g = genMoERow(r)
      let baked = moeBakedReceiptBits(seed, tokens)
      let generic = moeKernelBits(g, r)
      var mismatches = 0
      for i in 0 ..< baked.len:
        if baked[i] != generic[i]:
          mismatches += 1
      echo &"  seed={seed:x} T={tokens}: bit mismatches = {mismatches}"
      if mismatches != 0:
        return false
  result = true

proc checkGenericRows(): bool =
  ## Kernel vs torch chain of the same contract, at the Qwen3.6-35B-A3B
  ## shape and the ragged gelu shape.
  ##
  ## The 1e-2 bound covers the routed chain, the exponential forms'
  ## 1-ulp spread and the fp32 accumulation orders.
  for r in [qwen36Row, raggedRow]:
    let g = genMoERow(r)
    let actual = toTensor(fp16sToF32(moeKernelBits(g, r)))
      .reshape(r.tokens, r.hidden)
    let expected = moeReferenceRow(g, r)
    echo &"  {r.name}: worst |Δ| = {worstAbsDiff(actual, expected)}"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 1e-2'f64)
  result = true

proc checkConfigGuard(): bool =
  ## Runtime-entry compiled-in maxima guard.
  ##
  ## The host companion raises with the offending dimension named.
  ## The device entry drops the launch for an over-max config.
  ## The output bits stay at the prefill value.
  for bad in [("n_routed_experts", 8'i32, 2048'i32, 1024'i32, 1536'i32, 4'i32, 1'i32),
              ("top_k", 8, 2048, 64, 1536, 12, 1),
              ("num_tokens", 0, 2048, 64, 1536, 4, 1)]:
    let (dimName, nt, h, e, i, k, ns) = bad
    try:
      moeFwdConfigGuard(nt, h, e, i, k, ns)
      echo "  the guard accepted the over-max ", dimName, " config"
      return false
    except AssertionDefect as ex:
      echo "  ", dimName, " rejected: ", ex.msg
  moeFwdConfigGuard(8, 2048, 64, 1536, 4, 1)
  moeFwdConfigGuard(2, 2048, 256, 512, 8, 1)
  echo "  accepted the GLM and Qwen rows"
  # the device half drops the launch, the output stays the prefill
  var engine = bkMetal.init()
  engine.ingest(moeMsl)
  var outO = newSeq[uint16](8 * 2048)
  for i in 0 ..< outO.len:
    outO[i] = 0x7BFF'u16
  engine.run << (grid: (8, 1, 1), blk: (32, 1)) >> ("moeRun", outO,
    (newSeq[uint16](8 * 2048), newSeq[uint16](1024 * 2048),
     newSeq[uint16](1024 * 3072), newSeq[uint16](1024 * 2048),
     newSeq[uint16](3072), newSeq[uint16](2048),
     newSeq[uint16](8 * 4 * 1536), newSeq[uint16](8 * 1536),
     8'i32, 2048'i32, 1024'i32, 1536'i32, 4'i32, 1'i32, 1.8'f32, ActSilu))
  var untouched = true
  for i in 0 ..< outO.len:
    if outO[i] != 0x7BFF'u16:
      untouched = false
  echo "  over-max n_routed_experts=1024 launch dropped: ", untouched
  if not untouched:
    return false
  result = true

when isMainModule:
  runCppTest("moe_fwd vs the torch reference", checkMoe)
  runCppTest("the compiled-in maxima guard", checkConfigGuard)
  runCppTest("moe_fwd GLM delta-zero vs the recorded baked receipts",
    checkGenericDeltaZero)
  runCppTest("moe_fwd rows vs the torch reference", checkGenericRows)
