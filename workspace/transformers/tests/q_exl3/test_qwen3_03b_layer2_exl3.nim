## Test Qwen3 EXL3: layer 02 step-by-step trace on CUDA.
## Pinpoints exactly where the blowup occurs.

import
  std/memfiles, std/strformat, std/tables, std/os, std/options, std/importutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(TransformerBlock)
privateAccess(RopeGQAttention)
privateAccess(GatedMLP)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-layer02-trace"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"

proc check(name: string, actual, expected: Tensor, maxDiff: var float) =
  let d = (actual.to(kFloat32) - expected.to(kFloat32)).abs().max().item(float)
  maxDiff = max(maxDiff, d)
  let symbol = if d < 1e-4: "✅" elif d < 0.01: "⚠️" else: "❌"
  echo &"  {symbol} {name}: max|Δ|={d:.6f}"

proc main() =
  let model = loadQwen3ModelRaw(ModelPath, kCuda)
  let layer = model.layers[2]

  # Load fixture tensors on CUDA
  var memFile = memFiles.open(FixtureDir / "layer02_trace.safetensor", mode = fmRead)
  defer: close(memFile)
  let st = safetensors.load(memFile)

  template load(name: string): Tensor =
    # Due to floating point associativity issue, rounding and
    # warp-shuffle reduction, the tests cannot match on CPU
    # and tests against EXL3 fixtures MUST be done with Cuda backend.
    st.getTensorOwned(name, kCPU).to(kCuda)

  let x = load("input_hidden_states")
  let e_ln = load("after_input_layernorm")
  let e_q = load("q_proj_out")
  let e_k = load("k_proj_out")
  let e_v = load("v_proj_out")
  let e_qn = load("after_q_norm")
  let e_kn = load("after_k_norm")
  let e_rq = load("after_rope_q")
  let e_rk = load("after_rope_k")
  let e_attn = load("attn_output")
  let e_o = load("after_o_proj")
  let e_res = load("after_residual")
  let e_pln = load("after_post_layernorm")
  let e_gate = load("mlp_gate_out")
  let e_up = load("mlp_up_out")
  let e_act = load("mlp_activation")
  let e_down = load("mlp_down_out")
  let e_out = load("output")

  let batch = x.size(0)
  let S = x.size(1)
  let hd = 128
  var maxDiff = 0.0

  # Step 1: input_layernorm
  debugEcho "x cuda: ", x.is_cuda()
  let h = layer.input_layernorm(x)
  check("input_layernorm", h, e_ln, maxDiff)

  # Step 2: QKV projections
  let q = layer.self_attn.q_proj(h)
  let k = layer.self_attn.k_proj(h)
  let v = layer.self_attn.v_proj(h)
  check("q_proj", q, e_q, maxDiff)
  check("k_proj", k, e_k, maxDiff)
  check("v_proj", v, e_v, maxDiff)

  # Step 3: QK norms (per-head)
  let q_mh = q.reshape(batch, S, 16, hd)
  let k_mh = k.reshape(batch, S, 8, hd)
  let v_mh = v.reshape(batch, S, 8, hd)
  let q_normed = layer.self_attn.q_norm.get()(q_mh).reshape(q.shape)
  let k_normed = layer.self_attn.k_norm.get()(k_mh).reshape(k.shape)
  check("q_norm", q_normed, e_qn, maxDiff)
  check("k_norm", k_normed, e_kn, maxDiff)

  # Step 4: RoPE
  var ctx = InferenceContext.init(
    num_layers = 1, batch_size = 1, kv_heads = 8,
    max_seq = 4096, head_dim = hd,
    dtype = F.kFloat16, device = F.kCuda)
  ctx.reset()
  ctx.position_ids = arange(S).unsqueeze(0).to(kInt64).to(kCuda)
  ctx.setRopeForPositions(layer.self_attn.rotary)
  let qn_mh = q_normed.reshape(batch, S, 16, hd)
  let kn_mh = k_normed.reshape(batch, S, 8, hd)
  let (q_rot, k_rot) = layer.self_attn.rotary.applyRope(qn_mh, kn_mh, ctx.cos, ctx.sin)
  check("rope_q", q_rot.reshape(batch, S, -1), e_rq, maxDiff)
  check("rope_k", k_rot.reshape(batch, S, -1), e_rk, maxDiff)

  # Step 5: SDPA
  let qs = q_rot.transpose(1, 2)
  let ks = k_rot.transpose(1, 2)
  let vs = v_mh.transpose(1, 2)
  let attn = F.scaled_dot_product_attention(
    qs, ks, vs, attn_mask = none(Tensor), dropout_p = 0.0'f64,
    is_causal = true, scale = some(0.088388347648'f64), enable_gqa = true)
  let attn_out = attn.transpose(1, 2).reshape(batch, S, -1)
  check("attn (sdpa)", attn_out, e_attn, maxDiff)

  # Step 6: O projection
  let o_out = layer.self_attn.o_proj(attn_out)
  check("o_proj", o_out, e_o, maxDiff)

  # Step 7: Residual
  let res = x + o_out; check("residual", res, e_res, maxDiff)

  # Step 8: Post-attention RMSNorm
  let h2 = layer.post_attention_layernorm(res)
  check("post_layernorm", h2, e_pln, maxDiff)

  # Step 9: MLP
  let gate = layer.mlp.gate_proj(h2); check("mlp_gate", gate, e_gate, maxDiff)
  let up = layer.mlp.up_proj(h2); check("mlp_up", up, e_up, maxDiff)
  let act = F.silu(gate) * up; check("mlp_activation", act, e_act, maxDiff)
  let down = layer.mlp.down_proj(act); check("mlp_down", down, e_down, maxDiff)

  # Step 10: Output
  check("output", down, e_out, maxDiff)

  echo ""
  echo &"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if maxDiff < 5e-2: echo &"  Overall: max|Δ|={maxDiff:.6f}"
  else: echo &"  ❌ Overall: max|Δ|={maxDiff:.6f}"; quit 1

when isMainModule:
  main()
