## Test EXL3 RMSNorm against fixtures.
##
## Regresses the findings from the 4.0 blowup investigation:
##   - ext.rms_norm computes (x*w)*rmf (weight-first, all FP32)
##
## This test uses real EXL3 model weights from Qwen3-0.6B-EXL3-5bpw.
##
## This test highlights floating point associativity and rounding issues.
## Also it seems imposssible to replicate EXL3 in a bit-perfect manner
## unless we implement the same warp-shuffle reduction.
##
## q_norm CPU has a difference of 0.000244 with EXL3 that doesn't seem to be fixable in an easy way

import
  std/memfiles, std/strformat, std/os, std/importutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/norm {.all.},
  workspace/transformers/src/quantizations/datatypes {.all.},
  workspace/transformers/src/deserialization,
  pkg/packedjson,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

const
  ModelPath = currentSourcePath().parentDir() / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  FixtureDir = currentSourcePath().parentDir() / "fixtures" / "exl3-layer02-trace"
  MaxDiffTol = 1e-4

proc d(name: string, a, b: Tensor): float =
  result = (a.to(kFloat32) - b.to(kFloat32)).abs().max().item(float)
  let s = if result < 1e-6: "✓" elif result < MaxDiffTol: "⚠" else: "✗"
  echo &"  {s} {name}: max={result:.8f}"

proc main() =
  # Load config for deserialization
  let cfgJson = (ModelPath / "config.json").parseFile()

  # Load EXL3 layer 02 trace fixtures
  var memFile = memFiles.open(FixtureDir / "layer02_trace.safetensor", mode = fmRead)
  defer: close(memFile)
  let st = safetensors.load(memFile)
  template load(n: string): Tensor = st.getTensorOwned(n, kCPU)

  let e_q = load("q_proj_out")
  let e_k = load("k_proj_out")
  let e_qn = load("after_q_norm")
  let e_kn = load("after_k_norm")
  let e_ln = load("after_input_layernorm")
  let x = load("input_hidden_states")
  let e_res = load("after_residual")
  let e_pln = load("after_post_layernorm")

  # Load norm weights from model
  var mfile = memFiles.open(ModelPath / "model.safetensors", mode = fmRead)
  defer: close(mfile)
  let mst = safetensors.load(mfile)

  let qn = RmsNorm.load(mst, cfgJson, "model.layers.2.self_attn.q_norm")
  let kn = RmsNorm.load(mst, cfgJson, "model.layers.2.self_attn.k_norm")
  let ln = RmsNorm.load(mst, cfgJson, "model.layers.2.input_layernorm")
  let pln = RmsNorm.load(mst, cfgJson, "model.layers.2.post_attention_layernorm")

  let hd = 128

  for device in [kCuda, kCPU]:
    let deviceLabel = if device == kCuda: "CUDA" else: "CPU"
    echo &"\n=== Device: {deviceLabel} ===\n"

    let qn_d = qn.to(device)
    let kn_d = kn.to(device)
    let ln_d = ln.to(device)
    let pln_d = pln.to(device)

    let e_q_d = e_q.to(device)
    let e_k_d = e_k.to(device)
    let e_qn_d = e_qn.to(device)
    let e_kn_d = e_kn.to(device)
    let e_ln_d = e_ln.to(device)
    let x_d = x.to(device)
    let e_res_d = e_res.to(device)
    let e_pln_d = e_pln.to(device)

    echo "=== EXL3 RMSNorm: QK norms ==="
    echo "  (x*w)*rmf matches ext.rms_norm multiplication order"
    echo ""

    let q_mh = e_q_d.reshape(1, 6, 16, hd)
    let k_mh = e_k_d.reshape(1, 6, 8, hd)

    discard d("q_norm", qn_d(q_mh).reshape(e_q_d.shape), e_qn_d)
    discard d("k_norm", kn_d(k_mh).reshape(e_k_d.shape), e_kn_d)

    echo ""
    echo "=== EXL3 RMSNorm: input_layernorm ==="

    discard d("input_layernorm", ln_d(x_d), e_ln_d)

    echo ""
    echo "=== EXL3 RMSNorm: post_attention_layernorm ==="

    discard d("post_layernorm", pln_d(e_res_d), e_pln_d)

    echo "  All EXL3 RMSNorm calls should be near-bit-exact with ext.rms_norm fixtures."

when isMainModule:
  main()
