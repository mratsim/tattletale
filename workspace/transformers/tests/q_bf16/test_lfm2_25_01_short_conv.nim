# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/lfm2-shortconv \
##   --nimcache:nimcache/tests/lfm2-shortconv \
##   workspace/transformers/tests/q_bf16/test_lfm2_25_01_short_conv.nim

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
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/models/lfm2 {.all.},
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Lfm2ShortConv)
privateAccess(Lfm2DecoderLayer)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "LFM2.5-230M-layer-0"
  # Weights load from the real checkpoint through the git-ignored
  # hf_models/LFM2.5-230M symlink, the layout the Qwen3.5 suites use.
  # Fixture layer: real layer index 0, a conv layer.
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "LFM2.5-230M"
  NormEps = 1e-5
  Hidden = 1024         # conv_dim == hidden_size on LFM2.5-230M
  ConvKernel = 3        # conv_L_cache
  NumLayers = 14

proc openFixture(dir: string, name: string): (MemFile, Safetensor) =
  let memFile = memFiles.open(dir / name, mode = fmRead)
  result = (memFile, safetensors.load(memFile))

proc loadLayer0(st: Safetensor, cfgJson: JsonNode): Lfm2DecoderLayer =
  ## Load real layer-0 (conv) weights into a hybrid decoder layer.
  let lp = "model.layers.0."
  let opNorm = RmsNorm.load(st, cfgJson, lp & "operator_norm", eps = some(NormEps))
  let ffnNorm = RmsNorm.load(st, cfgJson, lp & "ffn_norm", eps = some(NormEps))
  let w1 = Linear.load(st, cfgJson, lp & "feed_forward.w1")
  let w2 = Linear.load(st, cfgJson, lp & "feed_forward.w2")
  let w3 = Linear.load(st, cfgJson, lp & "feed_forward.w3")
  let mlp = GatedMLP.init(w1, w3, w2)
  let inProj = Linear.load(st, cfgJson, lp & "conv.in_proj")
  let outProj = Linear.load(st, cfgJson, lp & "conv.out_proj")
  let convW = st.getTensorOwned(lp & "conv.conv.weight")
  let conv = Lfm2ShortConv.init(0, lp & "conv", inProj, convW, outProj, ConvKernel, Hidden)
  result = Lfm2DecoderLayer.init("conv", opNorm, ffnNorm, nil, conv, mlp)

proc newCtx(): InferenceContext =
  result = InferenceContext.init(NumLayers, 1, 8, 512, 64)

proc checkShortConv(): bool =
  var (memFile, wSt) = openFixture(ModelDir, "model.safetensors")
  defer: close(memFile)
  let cfgJson = parseFile(ModelDir / "config.json")
  let layer0 = loadLayer0(wSt, cfgJson)

  # ── Prefill T=5, intermediates + block output ──
  var (pMem, pSt) = openFixture(FixtureDir, "conv-prefill.safetensor")
  defer: close(pMem)
  let x = pSt.getTensorOwned("x")
  var ctx = newCtx()
  let output = layer0(ctx, x)
  assertAllClose(output, pSt.getTensorOwned("block_out"),
    rtol = 0.0, abstol = 5e-3, msg = "conv block output mismatch")
  let hNorm = layer0.operator_norm(x)
  assertAllClose(hNorm, pSt.getTensorOwned("operator_norm_out"),
    rtol = 0.0, abstol = 0.0, msg = "operator_norm mismatch")
  # Conv-block intermediates (deterministic elementwise/conv path)
  let bcx = layer0.conv.in_proj.forward(hNorm).transpose(1, 2)
  assertAllClose(bcx, pSt.getTensorOwned("in_proj_out"),
    rtol = 0.0, abstol = 0.0, msg = "in_proj mismatch")
  let seqLen = x.size(1)
  let split = F.chunk(bcx, 3, -2)
  assertAllClose(split[0], pSt.getTensorOwned("branch_b"),
    rtol = 0.0, abstol = 0.0, msg = "B branch mismatch")
  assertAllClose(split[1], pSt.getTensorOwned("branch_c"),
    rtol = 0.0, abstol = 0.0, msg = "C branch mismatch")
  assertAllClose(split[2], pSt.getTensorOwned("branch_x"),
    rtol = 0.0, abstol = 0.0, msg = "x branch mismatch")
  let mixed = split[0] * split[2]
  assertAllClose(mixed, pSt.getTensorOwned("mixed"),
    rtol = 0.0, abstol = 0.0, msg = "B·x mismatch")
  let convFull = F.conv1d(mixed, layer0.conv.conv_weight, padding = [2], groups = Hidden)
  let convOut = convFull.narrow(2, 0, seqLen)
  assertAllClose(convOut, pSt.getTensorOwned("conv_out"),
    rtol = 0.0, abstol = 0.0, msg = "causal conv1d mismatch")
  let y = split[1] * convOut
  assertAllClose(y, pSt.getTensorOwned("post_conv_y"),
    rtol = 0.0, abstol = 0.0, msg = "C·conv mismatch")
  let oProj = layer0.conv.out_proj.forward(y.transpose(1, 2))
  assertAllClose(oProj, pSt.getTensorOwned("out_proj_out"),
    rtol = 0.0, abstol = 0.0, msg = "out_proj mismatch")

  # ── Decode trajectory: 3-token prefill + 2 decode steps, state carry ──
  var (dMem, dSt) = openFixture(FixtureDir, "conv-decode.safetensor")
  defer: close(dMem)
  var ctxD = newCtx()
  let x3 = dSt.getTensorOwned("x_prefill3")
  let out3 = layer0(ctxD, x3)
  assertAllClose(out3, dSt.getTensorOwned("out_prefill3"),
    rtol = 0.0, abstol = 5e-3, msg = "decode-prefill output mismatch")
  # History after prefill: the K-1 newest pre-conv columns of the 3-token input.
  let nimStateP = ctxD.convState[0]
  let fixtureStateP = dSt.getTensorOwned("conv_state_prefill")
  assertAllClose(nimStateP,
    fixtureStateP.narrow(0, 0, 1).narrow(2, 1, ConvKernel - 1).reshape(Hidden, ConvKernel - 1),
    rtol = 0.0, abstol = 0.0, msg = "prefill state tail mismatch")
  let xd1 = dSt.getTensorOwned("x_decode1")
  let outD1 = layer0(ctxD, xd1)
  assertAllClose(outD1, dSt.getTensorOwned("out_decode1"),
    rtol = 0.0, abstol = 5e-3, msg = "decode step 1 mismatch")
  # Each decode step must match the one-shot forward at the same position.
  let oneShot = dSt.getTensorOwned("out_oneshot5")
  assertAllClose(outD1, oneShot.narrow(1, 3, 1),
    rtol = 0.0, abstol = 5e-3, msg = "decode step 1 vs one-shot mismatch")
  let xd2 = dSt.getTensorOwned("x_decode2")
  let outD2 = layer0(ctxD, xd2)
  assertAllClose(outD2, dSt.getTensorOwned("out_decode2"),
    rtol = 0.0, abstol = 5e-3, msg = "decode step 2 mismatch")
  assertAllClose(outD2, oneShot.narrow(1, 4, 1),
    rtol = 0.0, abstol = 5e-3, msg = "decode step 2 vs one-shot mismatch")
  # Nim keeps a (H, K-1) state while the vendored cache stores the K-wide
  # window, so the comparison drops the fixture window's oldest column.
  let nimState = ctxD.convState[0]
  let fixtureState = dSt.getTensorOwned("conv_state_d2")
  assertAllClose(nimState,
    fixtureState.narrow(0, 0, 1).narrow(2, 1, ConvKernel - 1).reshape(Hidden, ConvKernel - 1),
    rtol = 0.0, abstol = 0.0, msg = "decode state tail mismatch")
  result = true

when isMainModule:
  runCppTest("LFM2.5-230M conv layer (short-conv) vs fixture", checkShortConv)
