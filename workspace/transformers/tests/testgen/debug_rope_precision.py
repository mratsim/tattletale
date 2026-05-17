#!/usr/bin/env python3
"""
Prove: RoPE arithmetic in float16 vs float32 causes attention fixture divergence.

Hypothesis: The Python generator computes RoPE in float32 (cos/sin are float32),
but Nim applies RoPE in float16 (cos/sin are float16 from rotary cache).
This ~1-2 ULP difference accumulates in attention output.

Test: Compute q_rot using both float16 and float32 cos/sin, compare.
"""
import os, sys, json, torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Add testgen dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q_exl3_common import linear_forward_reimpl_exl3, get_exl3_tensors, reconstruct_reimpl_exl3, load_config, get_in_features_out_features, derive_K, derive_cb, had_r_128_reimpl_exl3

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hf_models", "Qwen3-0.6B-EXL3-5bpw")
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
config = load_config()
tensors = get_exl3_tensors(MODEL_PATH)

hidden_size = config["hidden_size"]
num_heads = config["num_attention_heads"]
num_kv_heads = config["num_key_value_heads"]
head_dim = config["head_dim"]
rms_eps = config.get("rms_norm_eps", 1e-6)
rope_theta = config.get("rope_theta", 1000000.0)
max_seq_len = config.get("max_position_embeddings", 40960)
DTYPE = torch.float16

prefix = "model.layers.0"
norms = {}
for k, v in tensors["_norms"].items():
    norms[k] = v.to(device)

def rms_norm(x, weight, eps=rms_eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True, dtype=torch.float32) + eps) * weight

def precompute_freqs_cis(dim, max_position, theta=rope_theta):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

def build_linear(layer_key, entry):
    trellis = entry["trellis"].to(device)
    suh = entry["suh"].to(device)
    svh = entry["svh"].to(device)
    K = derive_K(trellis)
    cb = derive_cb(entry)
    in_f, out_f = get_in_features_out_features(layer_key, trellis, config)
    w = reconstruct_reimpl_exl3(trellis, K, cb, (in_f, out_f))
    weight = w.t().contiguous()
    return {"weight": weight, "suh": suh, "svh": svh}

# Build linear layers
lin = {}
for name, key in [("q", f"{prefix}.self_attn.q_proj"), ("k", f"{prefix}.self_attn.k_proj"),
                   ("v", f"{prefix}.self_attn.v_proj"), ("o", f"{prefix}.self_attn.o_proj")]:
    lin[name] = build_linear(key, tensors[key])

q_norm_w = norms.get(f"{prefix}.self_attn.q_norm.weight").to(device)
k_norm_w = norms.get(f"{prefix}.self_attn.k_norm.weight").to(device)

# --- Test: compare RoPE in f16 vs f32 ---
batch, seq = 2, 8
hidden_states = torch.randn(batch, seq, hidden_size, dtype=DTYPE, device=device)
position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1).contiguous()

# Q, K, V projections
q = linear_forward_reimpl_exl3(hidden_states, lin["q"]["weight"], lin["q"]["suh"], lin["q"]["svh"], device=device)
k = linear_forward_reimpl_exl3(hidden_states, lin["k"]["weight"], lin["k"]["suh"], lin["k"]["svh"], device=device)
v = linear_forward_reimpl_exl3(hidden_states, lin["v"]["weight"], lin["v"]["suh"], lin["v"]["svh"], device=device)

q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)
k = k.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
v = v.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

# QK norm
if q_norm_w is not None:
    q = rms_norm(q.float(), q_norm_w.float(), rms_eps).to(DTYPE)
if k_norm_w is not None:
    k = rms_norm(k.float(), k_norm_w.float(), rms_eps).to(DTYPE)

# Precompute cos/sin
cos_f32, sin_f32 = precompute_freqs_cis(head_dim, max_seq_len, theta=rope_theta)
cos_f32 = cos_f32.to(device)  # float32
sin_f32 = sin_f32.to(device)  # float32
cos_f16 = cos_f32.to(DTYPE)   # float16
sin_f16 = sin_f32.to(DTYPE)   # float16

# Apply RoPE with float32 cos/sin (current generator behavior)
q_f32, k_f32 = apply_rope(q, k, cos_f32, sin_f32, position_ids)
q_f32 = q_f32.to(DTYPE)
k_f32 = k_f32.to(DTYPE)

# Apply RoPE with float16 cos/sin (Nim behavior)
q_f16, k_f16 = apply_rope(q, k, cos_f16, sin_f16, position_ids)
# No cast needed if result is float16; but cos is f16, q is f16 => result is f16

# Compare RoPE outputs
rope_diff = (q_f32.float() - q_f16.float()).abs().max().item()
print(f"\nRoPE q_rot max diff (f32 vs f16): {rope_diff:.6e}")

# Full attention with f32 RoPE
if num_kv_heads < num_heads:
    n_repeat = num_heads // num_kv_heads
    k_f32 = k_f32.repeat_interleave(n_repeat, dim=1)
    v_rep = v.repeat_interleave(n_repeat, dim=1)
else:
    v_rep = v

attn_f32 = torch.nn.functional.scaled_dot_product_attention(
    q_f32, k_f32, v_rep, attn_mask=None, dropout_p=0.0, is_causal=True, scale=head_dim ** -0.5)
attn_f32 = attn_f32.transpose(1, 2).contiguous().view(batch, seq, num_heads * head_dim)
out_f32 = linear_forward_reimpl_exl3(attn_f32, lin["o"]["weight"], lin["o"]["suh"], lin["o"]["svh"], device=device)

# Full attention with f16 RoPE
if num_kv_heads < num_heads:
    k_f16 = k_f16.repeat_interleave(n_repeat, dim=1)
attn_f16 = torch.nn.functional.scaled_dot_product_attention(
    q_f16, k_f16, v_rep, attn_mask=None, dropout_p=0.0, is_causal=True, scale=head_dim ** -0.5)
attn_f16 = attn_f16.transpose(1, 2).contiguous().view(batch, seq, num_heads * head_dim)
out_f16 = linear_forward_reimpl_exl3(attn_f16, lin["o"]["weight"], lin["o"]["suh"], lin["o"]["svh"], device=device)

out_diff = (out_f32.float() - out_f16.float()).abs().max().item()
print(f"Attention output max diff (f32 vs f16 RoPE): {out_diff:.6e}")

# Show a few values
print(f"\nSample output values (first batch, first 5 dims):")
print(f"  f32 RoPE: {out_f32[0, 0, :5]}")
print(f"  f16 RoPE: {out_f16[0, 0, :5]}")

# Compare the RoPE q values directly
print(f"\nSample q after RoPE (first head, pos 0, first 5 dims):")
print(f"  f32 RoPE: {q_f32[0, 0, 0, :5]}")
print(f"  f16 RoPE: {q_f16[0, 0, 0, :5]}")
print(f"  q before: {q[0, 0, 0, :5]}")

print(f"\nSample q after RoPE (first head, pos 1, first 5 dims):")
print(f"  f32 RoPE: {q_f32[0, 0, 1, :5]}")
print(f"  f16 RoPE: {q_f16[0, 0, 1, :5]}")
print(f"  q before: {q[0, 0, 1, :5]}")

# Also check: does QK norm output differ between f32/f16 accumulation?
print(f"\nQK norm dtype check:")
print(f"  q after norm dtype: {q.dtype}")
print(f"  cos_f32 dtype: {cos_f32.dtype}")
print(f"  cos_f16 dtype: {cos_f16.dtype}")

if out_diff > 1e-4:
    print(f"\n*** VERIFIED: RoPE float16 vs float32 causes {out_diff:.6e} diff in attention output ***")
    print(f"*** Fix: generate fixtures with float16 cos/sin to match Nim's behavior ***")
else:
    print(f"\n*** RoPE precision is NOT the cause of the mismatch ***")
