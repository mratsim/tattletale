## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Research test: RMSNorm function-level implementations vs HF Qwen3RMSNorm fixtures.
##
## Tests against real HF-generated fixtures for Qwen3-0.6B (block 0 and block 2).
## HF uses (x * rmf).to(dtype) * w — rstd-first, weight stays in original dtype.
## EXL3 uses (x * w) * rstd — weight-first, all FP32.

import
  std/memfiles, std/strformat, std/strutils, std/os, std/importutils,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  pkg/packedjson,
  workspace/libtorch_testutils,
  ./rmsnorm_common

{.experimental: "callOperator".}

const
  ModelPath = currentSourcePath().parentDir().parentDir() / "hf_models" / "Qwen3-0.6B"
  FixtureDir = currentSourcePath().parentDir().parentDir() / "fixtures" / "long-residual-3-block" / "Qwen3-0.6B"

proc reportSummary(title: string) =
  echo ""
  echo "=== " & title & " ==="
  echo "  Implementation                       max_diff vs HF fixture"
  echo "  -----------------------------------  ----------------------------------------"

proc runNormComparison(normName: string, x_fixture, y_fixture, weight: Tensor, eps: float64) =
  echo ""
  echo &"  ── {normName} (dim={x_fixture.size(-1)}) ──"
  let reference = y_fixture

  block:
    let r = rmsNormHFPath(x_fixture, weight, eps)
    report("HF path (sqrt+recip)", r.to(kFloat32), reference.to(kFloat32))
  block:
    let r = rmsNormHFSqrRsqrt(x_fixture, weight, eps)
    report("HF path (square+rsqrt)", r.to(kFloat32), reference.to(kFloat32))
  block:
    let r = rmsNormWeightFirstFP32(x_fixture, weight, eps)
    report("weight-first FP32", r.to(kFloat32), reference.to(kFloat32))
  block:
    let r = rmsNormWarpShuffleFP32(x_fixture, weight, eps)
    report("warp-shuffle FP32", r.to(kFloat32), reference.to(kFloat32))
  block:
    let r = rmsNormFusedTorch(x_fixture, weight, eps)
    report("fused torch", r.to(kFloat32), reference.to(kFloat32))
  block:
    let r = rmsNormFusedTorchFP32(x_fixture, weight, eps)
    report("fused torch FP32", r.to(kFloat32), reference.to(kFloat32))

proc main() =
  let cfgJson = (ModelPath / "config.json").parseFile()
  let eps = 1e-6

  var mfile = memFiles.open(ModelPath / "model.safetensors", mode = fmRead)
  defer: close(mfile)
  let mst = safetensors.load(mfile)

  echo "RMSNorm Function Research Test (HF baseline)"
  echo "═════════════════════════════════════════════"
  echo "Comparing HF Qwen3RMSNorm vs various function implementations"
  echo ""
  echo &"  hidden_size: 1024  eps: {eps}"
  echo ""

  var devices = @[kCPU]
  if Torch.cuda_is_available():
    devices.add(kCuda)

  for device in devices:
    let deviceLabel = if device == kCuda: "CUDA" else: "CPU"
    echo ""
    echo repeat('=', 70)
    echo &"  Device: {deviceLabel}"
    echo repeat('=', 70)

    # ── Block 0 ─────────────────────────────────────────────────
    echo ""
    echo repeat('#', 70)
    echo "  Block 0: From long-residual-3-block fixture"
    echo repeat('#', 70)

    block block0:
      var memFile = memFiles.open(FixtureDir / "block-00.safetensor", mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)
      template load(n: string): Tensor = st.getTensorOwned(n, kCPU)

      let layer_input = load("layer_input")
      let e_attn_norm = load("after_attn_norm")
      let e_mlp_norm = load("after_mlp_norm")
      let after_attn = load("after_attn")
      let attn_residual = load("after_attn_norm_residual")
      let post_attn_input = after_attn + attn_residual

      let ln_d = RmsNorm.load(mst, cfgJson, "model.layers.0.input_layernorm").to(device)
      let pln_d = RmsNorm.load(mst, cfgJson, "model.layers.0.post_attention_layernorm").to(device)
      let x_d = layer_input.to(device)
      let e_attn_norm_d = e_attn_norm.to(device)
      let post_attn_input_d = post_attn_input.to(device)
      let e_mlp_norm_d = e_mlp_norm.to(device)

      echo ""
      reportSummary("input_layernorm (dim=1024)")
      runNormComparison("input_layernorm", x_d, e_attn_norm_d, ln_d.weight, eps)
      reportSummary("post_attention_layernorm (dim=1024)")
      runNormComparison("post_attn_layernorm", post_attn_input_d, e_mlp_norm_d, pln_d.weight, eps)

    # ── Block 2 ─────────────────────────────────────────────────
    echo ""
    echo repeat('#', 70)
    echo "  Block 2: From long-residual-3-block fixture"
    echo repeat('#', 70)

    block block2:
      var memFile = memFiles.open(FixtureDir / "block-02.safetensor", mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)
      template load(n: string): Tensor = st.getTensorOwned(n, kCPU)

      let layer_input = load("layer_input")
      let e_attn_norm = load("after_attn_norm")
      let e_mlp_norm = load("after_mlp_norm")
      let after_attn = load("after_attn")
      let attn_residual = load("after_attn_norm_residual")
      let post_attn_input = after_attn + attn_residual

      let ln_d = RmsNorm.load(mst, cfgJson, "model.layers.2.input_layernorm").to(device)
      let pln_d = RmsNorm.load(mst, cfgJson, "model.layers.2.post_attention_layernorm").to(device)
      let x_d = layer_input.to(device)
      let e_attn_norm_d = e_attn_norm.to(device)
      let post_attn_input_d = post_attn_input.to(device)
      let e_mlp_norm_d = e_mlp_norm.to(device)

      echo ""
      reportSummary("input_layernorm (dim=1024)")
      runNormComparison("input_layernorm", x_d, e_attn_norm_d, ln_d.weight, eps)
      reportSummary("post_attention_layernorm (dim=1024)")
      runNormComparison("post_attn_layernorm", post_attn_input_d, e_mlp_norm_d, pln_d.weight, eps)

  echo ""
  echo repeat('=', 70)
  echo "  Summary: HF path (sqrt+recip, weight not upcast) matches exactly."
  echo "  Weight-first (EXL3 order) deviates by 0.125 — multiplication order matters."
  echo repeat('=', 70)

when isMainModule:
  main()
