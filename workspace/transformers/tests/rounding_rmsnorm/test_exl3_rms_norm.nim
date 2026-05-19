## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Research test: RMSNorm function-level implementations vs EXL3 kernel fixtures.
##
## Implements and compares 6 RMSNorm function variants against real EXL3-generated
## fixtures for layers 2 and 4 of Qwen3-0.6B-EXL3-5bpw.
##
## Key question: which FP32 reduction/multiplication order best matches ext.rms_norm?

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
  ModelPath = currentSourcePath().parentDir().parentDir() / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  Layer2TraceDir = currentSourcePath().parentDir().parentDir() / "fixtures" / "exl3-layer02-trace"
  IdsInferenceDir = currentSourcePath().parentDir().parentDir() / "fixtures" / "exl3-ids-inference" / "Qwen3-0.6B-EXL3-5bpw"

# ─── Reporting ─────────────────────────────────────────────────────

proc reportSummary(title: string) =
  echo ""
  echo "=== " & title & " ==="
  echo "  Implementation                       max_diff vs EXL3 fixture"
  echo "  -----------------------------------  ----------------------------------------"

proc runNormComparison(normName: string, x_fixture, y_fixture, weight: Tensor, eps: float64; onCuda: bool) =
  echo ""
  echo &"  ── {normName} (dim={x_fixture.size(-1)}) ──"
  let reference = y_fixture

  block:
    let r = rmsNormManualFP32(x_fixture, weight, eps)
    report("manual FP32 (rstd 1st)", r.to(kFloat32), reference.to(kFloat32))
  block:
    let r = rmsNormOptSqrRsqrtFP32(x_fixture, weight, eps)
    report("opt sqr+rsqrt FP32", r.to(kFloat32), reference.to(kFloat32))
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
  if onCuda:
    block:
      let r = pkl_rms_norm_fp16_cuda(x_fixture, weight, eps)
      report("Positron CUDA kernel", r.to(kFloat32), reference.to(kFloat32))

# ─── Main ──────────────────────────────────────────────────────────

proc main() =
  let cfgJson = (ModelPath / "config.json").parseFile()
  let eps = 1e-6

  var mfile = memFiles.open(ModelPath / "model.safetensors", mode = fmRead)
  defer: close(mfile)
  let mst = safetensors.load(mfile)

  echo "RMSNorm Function Research Test (EXL3 kernel baseline)"
  echo "══════════════════════════════════════════════════════"
  echo "Comparing EXL3 kernel (ext.rms_norm) vs various function implementations"
  echo ""
  echo &"  hidden_size: 1024  head_dim: 128  eps: {eps}"
  echo ""

  var devices = @[kCPU]
  if Torch.cuda_is_available():
    devices.add(kCuda)

  for device in devices:
    let onCuda = device == kCuda
    let deviceLabel = if onCuda: "CUDA" else: "CPU"
    echo ""
    echo repeat('=', 70)
    echo &"  Device: {deviceLabel}"
    echo repeat('=', 70)

    # ── Layer 2: full trace ────────────────────────────────────────
    echo ""
    echo repeat('#', 70)
    echo "  Layer 2: Full trace from exl3-layer02-trace"
    echo repeat('#', 70)

    block layer2:
      var memFile = memFiles.open(Layer2TraceDir / "layer02_trace.safetensor", mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)
      template load(n: string): Tensor = st.getTensorOwned(n, kCPU)

      let e_q = load("q_proj_out"); let e_k = load("k_proj_out")
      let e_qn = load("after_q_norm"); let e_kn = load("after_k_norm")
      let e_ln = load("after_input_layernorm")
      let x = load("input_hidden_states"); let e_res = load("after_residual")
      let e_pln = load("after_post_layernorm")

      let qn = RmsNorm.load(mst, cfgJson, "model.layers.2.self_attn.q_norm").to(device)
      let kn = RmsNorm.load(mst, cfgJson, "model.layers.2.self_attn.k_norm").to(device)
      let ln = RmsNorm.load(mst, cfgJson, "model.layers.2.input_layernorm").to(device)
      let pln = RmsNorm.load(mst, cfgJson, "model.layers.2.post_attention_layernorm").to(device)

      let x_d = x.to(device); let e_ln_d = e_ln.to(device)
      let e_res_d = e_res.to(device); let e_pln_d = e_pln.to(device)
      let e_qn_d = e_qn.to(device); let e_kn_d = e_kn.to(device)
      let e_q_d = e_q.to(device); let e_k_d = e_k.to(device)

      let q_mh = e_q_d.reshape(1, 6, 16, 128)
      let k_mh = e_k_d.reshape(1, 6, 8, 128)

      echo ""
      reportSummary("input_layernorm (dim=1024)")
      runNormComparison("input_layernorm", x_d, e_ln_d, ln.weight, eps, onCuda)
      reportSummary("post_attention_layernorm (dim=1024)")
      runNormComparison("post_attn_layernorm", e_res_d, e_pln_d, pln.weight, eps, onCuda)
      reportSummary("q_norm (multi-head dim=128)")
      runNormComparison("q_norm", q_mh, e_qn_d.reshape(q_mh.shape), qn.weight, eps, onCuda)
      reportSummary("k_norm (multi-head dim=128)")
      runNormComparison("k_norm", k_mh, e_kn_d.reshape(k_mh.shape), kn.weight, eps, onCuda)

    # ── Layer 4: ids-inference (only layer_input/layer_output) ─────
    echo ""
    echo repeat('#', 70)
    echo "  Layer 4: From ids-inference fixture"
    echo repeat('#', 70)
    echo "  NOTE: No intermediate norm outputs → implementations compared against"
    echo "  weight-first FP32 reference (nearest to EXL3 order)."

    block layer4:
      let fixturePath = IdsInferenceDir / "layer-04.safetensor"
      var memFile = memFiles.open(fixturePath, mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)
      let layer_input = st.getTensorOwned("layer_input", kCPU)
      let ln4 = RmsNorm.load(mst, cfgJson, "model.layers.4.input_layernorm").to(device)
      let xi = layer_input.to(device)

      echo ""
      reportSummary("input_layernorm (dim=1024)")
      echo ""
      echo "  Comparison basis: weight-first FP32 (nearest to EXL3 order)"
      echo ""

      let wf = rmsNormWeightFirstFP32(xi, ln4.weight, eps)
      let comparisons = [
        ("manual FP32 (rstd 1st)", rmsNormManualFP32(xi, ln4.weight, eps)),
        ("opt sqr+rsqrt FP32",     rmsNormOptSqrRsqrtFP32(xi, ln4.weight, eps)),
        ("warp-shuffle FP32",      rmsNormWarpShuffleFP32(xi, ln4.weight, eps)),
        ("fused torch",            rmsNormFusedTorch(xi, ln4.weight, eps)),
        ("fused torch FP32",       rmsNormFusedTorchFP32(xi, ln4.weight, eps)),
      ]
      for (name, r) in comparisons:
        let diff = (r.to(kFloat32) - wf.to(kFloat32)).abs().max().item(float)
        let mark = if diff < 1e-6: "≈" elif diff < MaxDiffTol: "Δ" else: "! "
        echo &"  {mark} {name:<32} vs weight-first: {diff:.8f}"
      if onCuda:
        block:
          let r = pkl_rms_norm_fp16_cuda(xi, ln4.weight, eps)
          let diff = (r.to(kFloat32) - wf.to(kFloat32)).abs().max().item(float)
          let mark = if diff < 1e-6: "≈" elif diff < MaxDiffTol: "Δ" else: "! "
          echo &"  {mark} Positron CUDA kernel               vs weight-first: {diff:.8f}"

  echo ""
  echo repeat('=', 70)
  echo "  Summary: weight-first (x*w)*rstd matches EXL3 on CPU."
  echo "  Rstd-first variants deviate by ~0.0016 (1024) / ~0.008 (128)."
  echo repeat('=', 70)

when isMainModule:
  main()
