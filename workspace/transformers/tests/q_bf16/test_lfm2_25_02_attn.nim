# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/lfm2-attn \
##   --nimcache:nimcache/tests/lfm2-attn \
##   workspace/transformers/tests/q_bf16/test_lfm2_25_02_attn.nim

import
  std/options,
  std/memfiles,
  std/os,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/lfm2 {.all.},
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(RopeGQAttention)
privateAccess(Lfm2DecoderLayer)

const
  # Layer 2 is the first full_attention layer of the checkpoint, layer_types
  # starts conv, conv, full_attention.
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "LFM2.5-230M-layer-2"
  # Weights load from the real checkpoint through the git-ignored
  # hf_models/LFM2.5-230M symlink, the layout the Qwen3.5 suites use.
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "LFM2.5-230M"
  NormEps = 1e-5
  LayerIdx = 2
  NumLayers = 14
  NumQHeads = 16
  NumKvHeads = 8
  HeadDim = 64
  RopeTheta = 1e6
  MaxPositions = 128000
  SeqLen = 5

proc openFixture(dir: string, name: string): (MemFile, Safetensor) =
  let memFile = memFiles.open(dir / name, mode = fmRead)
  result = (memFile, safetensors.load(memFile))

proc newCtx(): (InferenceContext, PagePool) =
  ## Context with one page borrowed. 256 token slots per page cover the 5-token
  ## prefill.
  ##
  ## Ownership: the PagePool ref keeps the borrowed pages alive for the test.
  var ctx = InferenceContext.init(NumLayers, 1, NumKvHeads, 512, HeadDim)
  let pool = PagePool.init(
    64, num_layers = NumLayers, kv_heads = NumKvHeads, head_dim = HeadDim,
    dtype = F.kBFloat16, device = F.kCPU)
  let numPages = ceilDiv(SeqLen, TokensPerPage)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())
  (ctx, pool)

proc checkAttn(): bool =
  var (memFile, wSt) = openFixture(ModelDir, "model.safetensors")
  defer: close(memFile)
  let cfgJson = parseFile(ModelDir / "config.json")
  let lp = "model.layers." & $LayerIdx & "."
  let opNorm = RmsNorm.load(wSt, cfgJson, lp & "operator_norm", eps = some(NormEps))
  let ffnNorm = RmsNorm.load(wSt, cfgJson, lp & "ffn_norm", eps = some(NormEps))
  let w1 = Linear.load(wSt, cfgJson, lp & "feed_forward.w1")
  let w2 = Linear.load(wSt, cfgJson, lp & "feed_forward.w2")
  let w3 = Linear.load(wSt, cfgJson, lp & "feed_forward.w3")
  let mlp = GatedMLP.init(w1, w3, w2)
  let qProj = Linear.load(wSt, cfgJson, lp & "self_attn.q_proj")
  let kProj = Linear.load(wSt, cfgJson, lp & "self_attn.k_proj")
  let vProj = Linear.load(wSt, cfgJson, lp & "self_attn.v_proj")
  let oProj = Linear.load(wSt, cfgJson, lp & "self_attn.out_proj")
  let qNorm = RmsNorm.load(wSt, cfgJson, lp & "self_attn.q_layernorm", eps = some(NormEps))
  let kNorm = RmsNorm.load(wSt, cfgJson, lp & "self_attn.k_layernorm", eps = some(NormEps))
  let rotary = RotaryPositionEmbeddingRef.new(
    HeadDim, MaxPositions, RopeTheta, F.kBFloat16, F.kCPU)
  let attn = RopeGQAttention.init(
    LayerIdx, lp & "self_attn", qProj, kProj, vProj, oProj, qNorm, kNorm,
    NumQHeads, NumKvHeads, HeadDim, rotary)
  let layer2 = Lfm2DecoderLayer.init("full_attention", opNorm, ffnNorm, attn, nil, mlp)

  var (pMem, pSt) = openFixture(FixtureDir, "attn-prefill.safetensor")
  defer: close(pMem)
  let x = pSt.getTensorOwned("x")
  var (ctx, pool) = newCtx()
  ctx.position_ids = F.arange(SeqLen, F.tensorOptions(F.kInt64, F.kCPU))
  ctx.setRopeForPositions(rotary)

  let output = layer2(ctx, x)
  assertAllClose(output, pSt.getTensorOwned("block_out"),
    rtol = 0.0, abstol = 5e-3, msg = "attention block output mismatch")
  let hNorm = layer2.operator_norm(x)
  assertAllClose(hNorm, pSt.getTensorOwned("operator_norm_out"),
    rtol = 0.0, abstol = 0.0, msg = "operator_norm mismatch")

  # RoPE tables for positions 0..4, sliced from the same cache the layer reads.
  let (cosTab, sinTab) = rotary.ropeByPositions(ctx.position_ids)
  assertAllClose(cosTab, pSt.getTensorOwned("cos").squeeze(0),
    rtol = 0.0, abstol = 0.0, msg = "rope cos mismatch")
  assertAllClose(sinTab, pSt.getTensorOwned("sin").squeeze(0),
    rtol = 0.0, abstol = 0.0, msg = "rope sin mismatch")

  # Deterministic pre-SDPA intermediates: projections + per-head QK norms.
  let q = layer2.self_attn.q_proj.forward(hNorm).reshape([1, SeqLen, NumQHeads, HeadDim])
  let k = layer2.self_attn.k_proj.forward(hNorm).reshape([1, SeqLen, NumKvHeads, HeadDim])
  let v = layer2.self_attn.v_proj.forward(hNorm).reshape([1, SeqLen, NumKvHeads, HeadDim])
  assertAllClose(q.transpose(1, 2), pSt.getTensorOwned("q_proj"),
    rtol = 0.0, abstol = 0.0, msg = "q_proj mismatch")
  assertAllClose(k.transpose(1, 2), pSt.getTensorOwned("k_proj"),
    rtol = 0.0, abstol = 0.0, msg = "k_proj mismatch")
  assertAllClose(v.transpose(1, 2), pSt.getTensorOwned("v_proj"),
    rtol = 0.0, abstol = 0.0, msg = "v_proj mismatch")
  let qNormed = layer2.self_attn.q_norm.get().forward(q)
  let kNormed = layer2.self_attn.k_norm.get().forward(k)
  assertAllClose(qNormed.transpose(1, 2), pSt.getTensorOwned("q_normed"),
    rtol = 0.0, abstol = 0.0, msg = "q_layernorm mismatch")
  assertAllClose(kNormed.transpose(1, 2), pSt.getTensorOwned("k_normed"),
    rtol = 0.0, abstol = 0.0, msg = "k_layernorm mismatch")

  # Rotated q/k: same elementwise NEOX rotation on the normed heads.
  let (qRot, kRot) = rotary.applyRope(qNormed, kNormed, ctx.cos, ctx.sin)
  assertAllClose(qRot.transpose(1, 2), pSt.getTensorOwned("q_rot"),
    rtol = 0.0, abstol = 0.0, msg = "rotated q mismatch")
  assertAllClose(kRot.transpose(1, 2), pSt.getTensorOwned("k_rot"),
    rtol = 0.0, abstol = 0.0, msg = "rotated k mismatch")

  # Causal GQA over the rotated heads, then the output projection.
  let attnOut = layer2.self_attn.gqa_attn(qRot, kRot, v)
  assertAllClose(attnOut, pSt.getTensorOwned("attn_out"),
    rtol = 0.0, abstol = 5e-3, msg = "attention output mismatch")
  let oProjOut = layer2.self_attn.o_proj.forward(attnOut)
  assertAllClose(oProjOut, pSt.getTensorOwned("out_proj_out"),
    rtol = 0.0, abstol = 5e-3, msg = "out_proj mismatch")
  result = true

when isMainModule:
  runCppTest("LFM2.5-230M attention layer vs fixture", checkAttn)
