# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run command, from the repo root:
## - nim cpp -r -d:release --hints:off --warnings:off --passC:-std=c++20 \
##   --outdir:build/wip --nimcache:nimcache/wip workspace/positron/tests/manual_ffn_moe_fp16.nim
import std/[strformat, math]
import workspace/crucible
import ../src/kernels/ceramic/ffn_moe
import properties/properties

# One launcher per MoeAct member, the gguf launchers' shape.
# The activation is a static binding of the entry, each launcher
# instantiates its member and engine.run addresses the instantiation
# by the launcher's name.
const moeMsl = metal:
  proc moeRunSilu(out_r, x, router_w, gate_up_w, down_w,
      shared_gate_up_w, shared_down_w, h_scratch,
      hs_scratch: ptr UncheckedArray[float16],
      num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
      n_shared_experts: int32, routed_scaling: float32) {.global.} =
    moe_fwd(out_r, x, router_w, gate_up_w, down_w, shared_gate_up_w,
      shared_down_w, h_scratch, hs_scratch, num_tokens, hidden,
      n_routed_experts, moe_intermediate, top_k, n_shared_experts,
      routed_scaling, maSilu)

  proc moeRunGeluTanh(out_r, x, router_w, gate_up_w, down_w,
      shared_gate_up_w, shared_down_w, h_scratch,
      hs_scratch: ptr UncheckedArray[float16],
      num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
      n_shared_experts: int32, routed_scaling: float32) {.global.} =
    moe_fwd(out_r, x, router_w, gate_up_w, down_w, shared_gate_up_w,
      shared_down_w, h_scratch, hs_scratch, num_tokens, hidden,
      n_routed_experts, moe_intermediate, top_k, n_shared_experts,
      routed_scaling, maGeluTanh)

func moeRunName(act: MoeAct): string =
  ## Metal launcher name for the activation member.
  case act
  of maSilu: "moeRunSilu"
  of maGeluTanh: "moeRunGeluTanh"

type MoEG = tuple[xf, rwf, guf, dnf, sguf, sdwf: seq[float32]]

type MoeRow = object
  name: string
  tokens, hidden, nExperts, inter, topK, nShared: int
  scale: float32
  act: MoeAct
  seed: uint64

const
  glm47Row = MoeRow(name: "glm47-flash", tokens: 8, hidden: 2048, nExperts: 64,
    inter: 1536, topK: 4, nShared: 1, scale: 1.8'f32, act: maSilu,
    seed: 0x5EED)
  qwen36Row = MoeRow(name: "qwen36-35b-a3b", tokens: 2, hidden: 2048,
    nExperts: 256, inter: 512, topK: 8, nShared: 1, scale: 1.0'f32,
    act: maSilu, seed: 0x5EED)
  raggedRow = MoeRow(name: "ragged-gelu", tokens: 2, hidden: 2064,
    nExperts: 70, inter: 200, topK: 4, nShared: 2, scale: 1.0'f32,
    act: maGeluTanh, seed: 0x5EED)

proc genMoERow(r: MoeRow): MoEG =
  ## One seeded rng read per buffer, the buffers in the record's order
  ## (x, router, gate_up, down, shared gate_up, shared down), each element
  ## uniform in [-scaleF, scaleF).
  let gu = 2 * r.inter
  var rng = initPropRng(r.seed)
  proc fill(dst: var seq[float32], rows, cols: int, scaleF: float32) =
    dst = newSeq[float32](rows * cols)
    for i in 0 ..< rows * cols:
      dst[i] = rng.nextF32(-scaleF, scaleF)
  fill(result.xf, r.tokens, r.hidden, 0.3'f32)
  fill(result.rwf, r.nExperts, r.hidden, 0.125'f32)
  fill(result.guf, r.nExperts * gu, r.hidden, 0.125'f32)
  fill(result.dnf, r.nExperts * r.hidden, r.inter, 0.125'f32)
  fill(result.sguf, r.nShared * gu, r.hidden, 0.125'f32)
  fill(result.sdwf, r.nShared * r.hidden, r.inter, 0.125'f32)


proc fp16sToF32(hs: seq[uint16]): seq[float32] =
  ## Widens an fp16 bit-pattern buffer to fp32 values.
  result = newSeq[float32](hs.len)
  for i in 0 ..< hs.len:
    result[i] = fp16ToFp32(hs[i])

proc fp32sToFp16*(fs: seq[float32]): seq[uint16] =
  ## Rounds an fp32 value buffer down to fp16 bit patterns (RNE).
  result = newSeq[uint16](fs.len)
  for i in 0 ..< fs.len:
    result[i] = fp32ToFp16(fs[i])

proc moeKernelBits(g: MoEG, r: MoeRow): seq[uint16] =
  var engine = bkMetal.init()
  engine.ingest(moeMsl)
  let gu = 2 * r.inter
  var outO = newSeq[uint16](r.tokens * r.hidden)
  var hScr = newSeq[uint16](r.tokens * r.topK * r.inter)
  var hsScr = newSeq[uint16](r.tokens * r.nShared * r.inter)
  engine.run << (grid: (r.tokens, 1, 1), blk: (32, 1)) >> (moeRunName(r.act),
    outO,
    (fp32sToFp16(g.xf), fp32sToFp16(g.rwf), fp32sToFp16(g.guf),
     fp32sToFp16(g.dnf), fp32sToFp16(g.sguf), fp32sToFp16(g.sdwf),
     hScr, hsScr, int32(r.tokens), int32(r.hidden), int32(r.nExperts),
     int32(r.inter), int32(r.topK), int32(r.nShared), r.scale))
  result = outO

# ─── The closed-form reference walk ───────────────────────────────────

proc refDot(x, w: seq[float32]; xOff, wOff, K: int): float32 =
  ## Sequential fp32 dot of two fp16-rounded operand rows widened to fp32:
  ## Σ_k widen(x[xOff + k])·widen(w[wOff + k]), the naive accumulation order
  ## the reassociation bound is judged against.
  result = 0.0'f32
  for k in 0 ..< K:
    result += fp16ToFp32(fp32ToFp16(x[xOff + k])) *
              fp16ToFp32(fp32ToFp16(w[wOff + k]))

proc refAct(x: float32, act: MoeAct): float32 =
  ## Activation variant the row selects, silu or the tanh-approximate gelu
  ## (the fused op the reference runtimes call).
  case act
  of maSilu:
    x / (1.0'f32 + exp(-x))
  of maGeluTanh:
    let inner = sqrt(2.0'f32 / PI) * (x + 0.044715'f32 * x * x * x)
    0.5'f32 * x * (1.0'f32 + tanh(inner))

proc moeReferenceRow(g: MoEG, r: MoeRow): seq[float32] =
  ## Closed-form fp32 chain of the generic contract at the row's dims,
  ## the same math as the kernel in the reference accumulation order.
  ##
  ## - every weight and activation row enters as its fp16-rounded widening
  ## - the fp16 h/hs store rounds re-enter widened, the fp16 output
  ##   store rounds at the final sum
  ## - the top-K selection is the kernel's own rule, the largest score
  ##   with the lowest expert id on ties, so a tie cannot hide in the band
  ##
  ## | stage   | accumulation site                       |
  ## | ------- | --------------------------------------- |
  ## | router  | one E-wide dot per token                |
  ## | gate/up | two I-wide dots per slot                |
  ## | down    | one H-wide dot per slot, h fp16-rounded |
  ## | shared  | the same three dots per shared expert   |
  ## | merge   | Σ w[slot]·down + Σ shared, fp16 round   |
  ##
  ## 1e-2 output band, fp32-reference derivation:
  ##
  ## - each dot's reassociation against the kernel's mma chunk walks stays
  ##   inside 2·K·u32·Σk abs(x·w), u32 = 2⁻²⁴, order 2.4e-3 per dot at H = 2048
  ## - the h/hs contributions carry one fp16 store round each,
  ##   u_step·abs ≈ 4.9e-4 relative
  ## - the merge sums 1 to 9 such terms in fp32 and the fp16 output
  ##   store rounds at u_step·abs(out)
  ##
  ## Every term sits two orders below the band. A wrong top-K set,
  ## expert index mapping or scale shows up as O(1) order errors.
  let H = r.hidden
  let E = r.nExperts
  let I = r.inter
  let gu = 2 * I
  # ── router: logits → sigmoid → top-K (lowest id on ties) → weights ──
  result = newSeq[float32](r.tokens * H)
  for t in 0 ..< r.tokens:
    let xOff = t * H
    var logits = newSeq[float32](E)
    for e in 0 ..< E:
      logits[e] = refDot(g.xf, g.rwf, xOff, e * H, H)
    var s = newSeq[float32](E)
    for e in 0 ..< E:
      s[e] = 1.0'f32 / (1.0'f32 + exp(-logits[e]))
    var topIds = newSeq[int](r.topK)
    var topW = newSeq[float32](r.topK)
    var chosen = newSeq[bool](E)
    for slot in 0 ..< r.topK:
      var best = -1
      var bestS = -3.402823466e38'f32
      for e in 0 ..< E:
        if not chosen[e] and s[e] > bestS:
          best = e
          bestS = s[e]
      chosen[best] = true
      topIds[slot] = best
      topW[slot] = bestS
    var sumW = 0.0'f32
    for slot in 0 ..< r.topK:
      sumW += topW[slot]
    for slot in 0 ..< r.topK:
      topW[slot] = topW[slot] / (sumW + 1.0e-20'f32) * r.scale
    # ── routed experts: gate/up → act → fp16 h → down → weighted sum ──
    var routed = newSeq[float32](H)
    for slot in 0 ..< r.topK:
      let e = topIds[slot]
      var h16 = newSeq[uint16](I)
      for i in 0 ..< I:
        let gV = refDot(g.xf, g.guf, xOff, e * gu * H + i * H, H)
        let uV = refDot(g.xf, g.guf, xOff, e * gu * H + (I + i) * H, H)
        h16[i] = fp32ToFp16(refAct(gV, r.act) * uV)
      for h in 0 ..< H:
        var down = 0.0'f32
        for i in 0 ..< I:
          down += fp16ToFp32(h16[i]) *
                  fp16ToFp32(fp32ToFp16(g.dnf[e * H * I + h * I + i]))
        routed[h] += topW[slot] * down
    # ── shared experts: the same three dots per shared expert ──
    for s in 0 ..< r.nShared:
      let sguOff = s * gu * H
      let sdwOff = s * H * I
      var hs16 = newSeq[uint16](I)
      for i in 0 ..< I:
        let gV = refDot(g.xf, g.sguf, xOff, sguOff + i * H, H)
        let uV = refDot(g.xf, g.sguf, xOff, sguOff + (I + i) * H, H)
        hs16[i] = fp32ToFp16(refAct(gV, r.act) * uV)
      for h in 0 ..< H:
        var down = 0.0'f32
        for i in 0 ..< I:
          down += fp16ToFp32(hs16[i]) *
                  fp16ToFp32(fp32ToFp16(g.sdwf[sdwOff + h * I + i]))
        result[t * H + h] += down
    # ── merge: routed + shared, one fp16 output round ──
    for h in 0 ..< H:
      result[t * H + h] = fp16ToFp32(fp32ToFp16(routed[h] + result[t * H + h]))

proc worstAbsDiff(a, b: seq[float32]): float32 =
  ## Returns the largest |a[i] - b[i]| over all elements.
  ## `a` and `b` must have the same element count.
  doAssert a.len == b.len,
    "worstAbsDiff: element count mismatch, got " & $a.len & " vs " & $b.len
  for i in 0 ..< a.len:
    result = max(result, abs(a[i] - b[i]))

proc checkMoe() =
  ## GLM-4.7-Flash row through the runtime-config entry, T=8 and T=4,
  ## against the closed-form reference.
  ##
  ## - the 1e-2 bound covers the 4-slot routed chain, gate_up, silu,
  ##   one fp16 h round, and down
  ## - the weighted sum and the shared MLP fit within the 0.125-weight scale bound
  ## - a wrong top-4 set, expert index mapping or scale shows up as O(1) order errors
  for tokens in [8, 4]:
    let r = MoeRow(name: "glm47-flash", tokens: tokens, hidden: 2048,
      nExperts: 64, inter: 1536, topK: 4, nShared: 1, scale: 1.8'f32,
      act: maSilu, seed: 0x5EED)
    let g = genMoERow(r)
    let actual = fp16sToF32(moeKernelBits(g, r))
    let expected = moeReferenceRow(g, r)
    echo &"  T={tokens}: worst |Δ| = {worstAbsDiff(actual, expected)}"
    doAssert worstAbsDiff(actual, expected) <= 1.0e-2'f32,
      &"glm47-flash T={tokens} left the 1e-2 band"

proc checkGenericRows() =
  ## Kernel vs closed-form chain of the same contract, at the Qwen3.6-35B-A3B
  ## shape and the ragged gelu shape.
  ##
  ## The 1e-2 bound covers the routed chain, the exponential forms'
  ## 1-ulp spread and the fp32 accumulation orders.
  for r in [qwen36Row, raggedRow]:
    let g = genMoERow(r)
    let actual = fp16sToF32(moeKernelBits(g, r))
    let expected = moeReferenceRow(g, r)
    echo &"  {r.name}: worst |Δ| = {worstAbsDiff(actual, expected)}"
    doAssert worstAbsDiff(actual, expected) <= 1.0e-2'f32,
      &"{r.name} left the 1e-2 band"

proc checkRaggedTail() =
  ## Ragged gelu shape, 70 experts, a router whose every real logit is negative.
  ## The chunk tail's zero-filled rows must lose the top-4.
  ##
  ## - experts 70..127 are chunk 1's tail
  ## - their router rows load zero-filled, sigmoid(0) = 0.5
  ## - unmasked, that 0.5 beats every real score, the top-4 picks expert
  ##   ids at or beyond 70 and reads past the 70-row expert weights
  ##
  ## The kernel and the reference chain read the same bits, a top-4 tail
  ## expert shows up as an O(1) order deviation.
  var g = genMoERow(raggedRow)
  for i in 0 ..< g.xf.len:
    g.xf[i] = abs(g.xf[i])
  for i in 0 ..< g.rwf.len:
    g.rwf[i] = -abs(g.rwf[i])
  let actual = fp16sToF32(moeKernelBits(g, raggedRow))
  let expected = moeReferenceRow(g, raggedRow)
  echo &"  {raggedRow.name} tail: worst |Δ| = {worstAbsDiff(actual, expected)}"
  doAssert worstAbsDiff(actual, expected) <= 1.0e-2'f32,
    "the ragged tail left the 1e-2 band"

proc checkConfigGuard() =
  ## Runtime-entry compiled-in maxima guard.
  ##
  ## The host companion raises with the offending dimension named.
  ## The device entry drops the launch for an over-max config.
  ## The output bits stay at the prefill value.
  for bad in [("n_routed_experts", 8'i32, 2048'i32, 1024'i32, 1536'i32, 4'i32, 1'i32),
              ("top_k", 8, 2048, 64, 1536, 12, 1),
              ("num_tokens", 0, 2048, 64, 1536, 4, 1)]:
    let (dimName, nt, h, e, i, k, ns) = bad
    var rejected = false
    try:
      moeFwdConfigGuard(nt, h, e, i, k, ns)
    except AssertionDefect as ex:
      rejected = true
      echo "  ", dimName, " rejected: ", ex.msg
    doAssert rejected, "the guard accepted the over-max " & dimName & " config"
  moeFwdConfigGuard(8, 2048, 64, 1536, 4, 1)
  moeFwdConfigGuard(2, 2048, 256, 512, 8, 1)
  echo "  accepted the GLM and Qwen rows"
  # the device half drops the launch, the output stays the prefill
  var engine = bkMetal.init()
  engine.ingest(moeMsl)
  var outO = newSeq[uint16](8 * 2048)
  for i in 0 ..< outO.len:
    outO[i] = 0x7BFF'u16
  engine.run << (grid: (8, 1, 1), blk: (32, 1)) >> ("moeRunSilu", outO,
    (newSeq[uint16](8 * 2048), newSeq[uint16](1024 * 2048),
     newSeq[uint16](1024 * 3072), newSeq[uint16](1024 * 2048),
     newSeq[uint16](3072), newSeq[uint16](2048),
     newSeq[uint16](8 * 4 * 1536), newSeq[uint16](8 * 1536),
     8'i32, 2048'i32, 1024'i32, 1536'i32, 4'i32, 1'i32, 1.8'f32))
  var untouched = true
  for i in 0 ..< outO.len:
    if outO[i] != 0x7BFF'u16:
      untouched = false
  echo "  over-max n_routed_experts=1024 launch dropped: ", untouched
  doAssert untouched, "the over-max launch wrote the output"

proc main =
  checkMoe()
  checkConfigGuard()
  checkGenericRows()
  checkRaggedTail()
  echo "manual_ffn_moe_fp16: all checks passed"

main()
