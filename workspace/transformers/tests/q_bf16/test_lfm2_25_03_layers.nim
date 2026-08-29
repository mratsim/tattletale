# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/lfm2-chain \
##   --nimcache:nimcache/tests/lfm2-chain \
##   workspace/transformers/tests/q_bf16/test_lfm2_25_03_layers.nim

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
  ChainDir = currentSourcePath().parentDir() / ".." / "fixtures" / "chain" / "LFM2.5-230M"
  # Weights load from the real checkpoint through the git-ignored
  # hf_models/LFM2.5-230M symlink, the layout the Qwen3.5 suites use.
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "LFM2.5-230M"
  NormEps = 1e-5
  NumLayers = 14
  NumQHeads = 16
  NumKvHeads = 8
  HeadDim = 64
  RopeTheta = 1e6
  MaxPositions = 128000
  Hidden = 1024
  ConvKernel = 3
  ChainLen = 3      # real layers 0..2, conv, conv, full_attention
  SeqLen = 4

proc openFixture(dir: string, name: string): (MemFile, Safetensor) =
  let memFile = memFiles.open(dir / name, mode = fmRead)
  result = (memFile, safetensors.load(memFile))

proc loadLayer(st: Safetensor, cfgJson: JsonNode, i: int,
               rotary: RotaryPositionEmbeddingRef): Lfm2DecoderLayer =
  ## Layer `i` weights from the real checkpoint, for i in 0..2:
  ## layer 0 conv, layer 1 conv, layer 2 full_attention.
  let lp = "model.layers." & $i & "."
  let opNorm = RmsNorm.load(st, cfgJson, lp & "operator_norm", eps = some(NormEps))
  let ffnNorm = RmsNorm.load(st, cfgJson, lp & "ffn_norm", eps = some(NormEps))
  let w1 = Linear.load(st, cfgJson, lp & "feed_forward.w1")
  let w2 = Linear.load(st, cfgJson, lp & "feed_forward.w2")
  let w3 = Linear.load(st, cfgJson, lp & "feed_forward.w3")
  let mlp = GatedMLP.init(w1, w3, w2)
  if i == 2:
    let qProj = Linear.load(st, cfgJson, lp & "self_attn.q_proj")
    let kProj = Linear.load(st, cfgJson, lp & "self_attn.k_proj")
    let vProj = Linear.load(st, cfgJson, lp & "self_attn.v_proj")
    let oProj = Linear.load(st, cfgJson, lp & "self_attn.out_proj")
    let qNorm = RmsNorm.load(st, cfgJson, lp & "self_attn.q_layernorm", eps = some(NormEps))
    let kNorm = RmsNorm.load(st, cfgJson, lp & "self_attn.k_layernorm", eps = some(NormEps))
    let attn = RopeGQAttention.init(
      i, lp & "self_attn", qProj, kProj, vProj, oProj, qNorm, kNorm,
      NumQHeads, NumKvHeads, HeadDim, rotary)
    result = Lfm2DecoderLayer.init("full_attention", opNorm, ffnNorm, attn, nil, mlp)
  else:
    let inProj = Linear.load(st, cfgJson, lp & "conv.in_proj")
    let outProj = Linear.load(st, cfgJson, lp & "conv.out_proj")
    let convW = st.getTensorOwned(lp & "conv.conv.weight")
    let conv = Lfm2ShortConv.init(i, lp & "conv", inProj, convW, outProj, ConvKernel, Hidden)
    result = Lfm2DecoderLayer.init("conv", opNorm, ffnNorm, nil, conv, mlp)

proc checkChain(): bool =
  var (memFile, wSt) = openFixture(ModelDir, "model.safetensors")
  defer: close(memFile)
  let cfgJson = parseFile(ModelDir / "config.json")
  let rotary = RotaryPositionEmbeddingRef.new(
    HeadDim, MaxPositions, RopeTheta, F.kBFloat16, F.kCPU)
  let layers = [loadLayer(wSt, cfgJson, 0, rotary),
                loadLayer(wSt, cfgJson, 1, rotary),
                loadLayer(wSt, cfgJson, 2, rotary)]

  var (cMem, cSt) = openFixture(ChainDir, "chain.safetensor")
  defer: close(cMem)
  let x = cSt.getTensorOwned("layer0_in")

  var ctx = InferenceContext.init(NumLayers, 1, NumKvHeads, 512, HeadDim)
  let pool = PagePool.init(
    64, num_layers = NumLayers, kv_heads = NumKvHeads, head_dim = HeadDim,
    dtype = F.kBFloat16, device = F.kCPU)
  let numPages = ceilDiv(SeqLen, TokensPerPage)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())
  ctx.position_ids = F.arange(SeqLen, F.tensorOptions(F.kInt64, F.kCPU))
  ctx.setRopeForPositions(rotary)

  var h = x
  for i in 0 ..< ChainLen:
    let expectedIn = cSt.getTensorOwned("layer" & $i & "_in")
    assertAllClose(h, expectedIn,
      rtol = 0.0, abstol = 5e-3, msg = "chain layer " & $i & " input mismatch")
    let layerOut = layers[i](ctx, h)
    let expected = cSt.getTensorOwned("layer" & $i & "_out")
    assertAllClose(layerOut, expected,
      rtol = 0.0, abstol = 5e-3, msg = "chain layer " & $i & " output mismatch")
    h = layerOut

  # Final norm runs on the fixture's layer2_out, not on the chained activation.
  # Feeding the chained output would only re-check the 5e-3 band accumulated
  # across the 3 layers and prove nothing about the norm.
  let norm = RmsNorm.load(wSt, cfgJson, "model.embedding_norm", eps = some(NormEps))
  assertAllClose(norm.forward(cSt.getTensorOwned("layer2_out")),
    cSt.getTensorOwned("embedding_norm_out"),
    rtol = 0.0, abstol = 0.0, msg = "embedding_norm output mismatch")
  result = true

when isMainModule:
  runCppTest("LFM2.5-230M 3-layer chain vs fixture", checkChain)
