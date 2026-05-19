## Test HF Qwen3RMSNorm against fixtures.
##
## Regresses the HF Qwen3RMSNorm behavior:
##   - HF computes (x * rmf).to(dtype) * w (rms-factor-first)
##
## This test uses real HF model weights from Qwen3-0.6B.
##
## This test highlights floating point associativity and rounding issues.

import
  std/memfiles, std/strformat, std/os, std/importutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/norm {.all.},
  workspace/transformers/src/deserialization,
  pkg/packedjson,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

const
  ModelPath = currentSourcePath().parentDir() / "hf_models" / "Qwen3-0.6B"
  FixtureDir = currentSourcePath().parentDir() / "fixtures" / "long-residual-3-block" / "Qwen3-0.6B"
  MaxDiffTol = 1e-4

proc d(name: string, a, b: Tensor): float =
  result = (a.to(kFloat32) - b.to(kFloat32)).abs().max().item(float)
  let s = if result < 1e-6: "✅" elif result < MaxDiffTol: "⚠️" else: "❌"
  echo &"  {s} {name}: max={result:.8f}"

proc main() =
  let cfgJson = (ModelPath / "config.json").parseFile()

  # Load HF long residual fixtures
  var memFile = memFiles.open(FixtureDir / "block-00.safetensor", mode = fmRead)
  defer: close(memFile)
  let st = safetensors.load(memFile)
  template load(n: string): Tensor = st.getTensorOwned(n, kCPU)

  let x = load("layer_input")
  let e_attn_norm = load("after_attn_norm")
  let e_mlp_norm = load("after_mlp_norm")
  let e_after_attn = load("after_attn")
  let e_after_attn_residual = load("after_attn_norm_residual")

  # Load norm layers from model
  var mfile = memFiles.open(ModelPath / "model.safetensors", mode = fmRead)
  defer: close(mfile)
  let mst = safetensors.load(mfile)

  let ln = RmsNorm.load(mst, cfgJson, "model.layers.0.input_layernorm")
  let pln = RmsNorm.load(mst, cfgJson, "model.layers.0.post_attention_layernorm")

  for device in [kCuda, kCPU]:
    let deviceLabel = if device == kCuda: "CUDA" else: "CPU"
    echo &"\n=== Device: {deviceLabel} ===\n"

    let ln_d = ln.to(device)
    let pln_d = pln.to(device)

    let x_d = x.to(device)
    let e_attn_norm_d = e_attn_norm.to(device)
    let e_mlp_norm_d = e_mlp_norm.to(device)
    let e_after_attn_d = e_after_attn.to(device)
    let e_after_attn_residual_d = e_after_attn_residual.to(device)
    let res = e_after_attn_d + e_after_attn_residual_d

    echo "=== HF Qwen3RMSNorm ==="
    echo ""

    discard d("input_layernorm", ln_d(x_d), e_attn_norm_d)

    echo ""
    echo "  post_attention_layernorm:"
    discard d("post_layernorm", pln_d(res), e_mlp_norm_d)

    echo "  All HF RMSNorm calls should match HF Qwen3RMSNorm fixtures."

when isMainModule:
  main()
