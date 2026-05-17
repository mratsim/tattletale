#!/usr/bin/env python3
"""Debug attention precision: find exact discrepancy source."""
from __future__ import annotations
import os, sys, json, math, torch
from safetensors.torch import load_file as st_load

_venv_python = os.path.dirname(sys.executable)
_venv_bin = os.path.dirname(_venv_python)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + ':' + os.environ.get("PATH", "")
if "CUDA_HOME" not in os.environ:
    import glob
    sp_base = os.path.join(os.path.dirname(_venv_python), '..', 'lib')
    for d in glob.glob(os.path.join(sp_base, 'python*', 'site-packages', 'nvidia', 'cu*')):
        if os.path.exists(os.path.join(d, 'bin', 'nvcc')):
            os.environ["CUDA_HOME"] = os.path.abspath(d)
            break

DEVICE = "cuda:0"
DTYPE = torch.float16
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "..", "fixtures", "exl3-layers", "Qwen3-0.6B-EXL3-5bpw-layer-0")
MODEL_DIR = os.path.join(HERE, "..", "hf_models", "Qwen3-0.6B-EXL3-5bpw")
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")

# Load fixture
fixture = st_load(os.path.join(FIXTURE_DIR, "attn-Qwen3-0.6B-EXL3-5bpw-01.safetensor"), device=DEVICE)
hidden_states = fixture["hidden_states"].to(DTYPE)
expected_output = fixture["output"]
expected_cos = fixture["cos"]
expected_sin = fixture["sin"]
position_ids = fixture["position_ids"].to(DEVICE)
batch, seq, hidden_size = hidden_states.shape

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    cfg = json.load(f)
num_qo_heads = cfg["num_attention_heads"]
num_kv_heads = cfg["num_key_value_heads"]
head_dim = cfg.get("head_dim", hidden_size // num_qo_heads)
rms_eps = cfg.get("rms_norm_eps", 1e-6)
rope_theta = cfg.get("rope_theta", 1000000.0)
n_rep = num_qo_heads // num_kv_heads
print(f"Config: heads={num_qo_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, seq={seq}")

# Load raw model tensors
print("Loading model tensors...")
raw = st_load(MODEL_PATH, device=DEVICE)

# Group tensors by EXL3 layer
def group_exl3_tensors(raw):
    """Replicate get_exl3_tensors grouping manually."""
    groups = {}
    for suffix in (".trellis", ".suh", ".svh", ".mcg", ".mul1", ".bias", ".K"):
        for k in raw:
            if k.endswith(suffix):
                group = k[: -len(suffix)]
                if group not in groups:
                    groups[group] = {}
                name = suffix[1:]  # remove leading dot
                groups[group][name] = raw[k]
    # Norms & embeddings
    norms = {}
    for k in raw:
        if "layernorm" in k:
            norms[k] = raw[k]
    groups["_norms"] = norms
    return groups

tensors = group_exl3_tensors(raw)

P = "model.layers.0.self_attn"
from q_exl3_common import (
    derive_K, derive_cb, reconstruct_orig_exl3, linear_forward_reimpl_exl3,
    rms_norm_orig_exl3, precompute_freqs_cis_reimpl_exl3, apply_rotary_pos_emb_reimpl_exl3,
)

def build_linear(proj):
    key = f"{P}.{proj}"
    g = tensors[key]
    trellis = g["trellis"]
    K = derive_K(trellis)
    mcg = "mcg" in g
    mul1 = "mul1" in g
    k_tiles, n_tiles, _ = trellis.shape
    in_f, out_f = k_tiles * 16, n_tiles * 16
    w = reconstruct_orig_exl3(trellis, K, mcg, mul1, (in_f, out_f)).contiguous()
    w_t = w.t().contiguous().to(DEVICE)
    return w_t, g["suh"].to(DEVICE), g["svh"].to(DEVICE)

print("Building layers...")
q_w, q_suh, q_svh = build_linear("q_proj")
k_w, k_suh, k_svh = build_linear("k_proj")
v_w, v_suh, v_svh = build_linear("v_proj")
o_w, o_suh, o_svh = build_linear("o_proj")

norms = tensors.get("_norms", {})
q_norm_w = norms.get(f"{P}.q_norm.weight", None)
k_norm_w = norms.get(f"{P}.k_norm.weight", None)

cos_tbl, sin_tbl = precompute_freqs_cis_reimpl_exl3(head_dim, 40960, theta=rope_theta)
cos_tbl = cos_tbl.to(DTYPE).to(DEVICE)
sin_tbl = sin_tbl.to(DTYPE).to(DEVICE)

x = hidden_states.to(DTYPE)

# ══════════════════════════════════════════════════════════════════
# STEP 1: QKV projection
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 1: QKV projection ===")
q = linear_forward_reimpl_exl3(x, q_w, q_suh, q_svh, device=DEVICE)
k = linear_forward_reimpl_exl3(x, k_w, k_suh, k_svh, device=DEVICE)
v = linear_forward_reimpl_exl3(x, v_w, v_suh, v_svh, device=DEVICE)
q_4d = q.view(batch, seq, num_qo_heads, head_dim).transpose(1, 2)
k_4d = k.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
v_4d = v.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

if q_norm_w is not None:
    q_4d = rms_norm_orig_exl3(q_4d, q_norm_w.to(DEVICE), rms_eps)
if k_norm_w is not None:
    k_4d = rms_norm_orig_exl3(k_4d, k_norm_w.to(DEVICE), rms_eps)

# ══════════════════════════════════════════════════════════════════
# STEP 2: RoPE — compare 2D cos/sin approach vs 3D
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 2: RoPE application ===")
# Approach A: Python generator (apply_rotary_pos_emb_reimpl_exl3)
#   slices cos[position_ids] -> [batch, seq, head_dim], unsqueeze -> [batch, 1, seq, head_dim]
q_a, k_a = apply_rotary_pos_emb_reimpl_exl3(q_4d, k_4d, cos_tbl, sin_tbl, position_ids)
q_a = q_a.to(DTYPE); k_a = k_a.to(DTYPE)

# Approach B: Nim-style (per-batch, unsqueeze(0).unsqueeze(0))
q_b_list, k_b_list = [], []
for b in range(batch):
    pos_b = position_ids[b:b+1]
    cos_b = cos_tbl[pos_b].unsqueeze(1)  # [1, 1, seq, head_dim]
    sin_b = sin_tbl[pos_b].unsqueeze(1)
    qb = q_4d[b:b+1]; kb = k_4d[b:b+1]
    half = head_dim // 2
    q_rot = qb * cos_b + torch.cat((-qb[:,:,:,half:], qb[:,:,:,:half]), dim=-1) * sin_b
    k_rot = kb * cos_b + torch.cat((-kb[:,:,:,half:], kb[:,:,:,:half]), dim=-1) * sin_b
    q_b_list.append(q_rot); k_b_list.append(k_rot)
q_b = torch.cat(q_b_list, dim=0).to(DTYPE)
k_b = torch.cat(k_b_list, dim=0).to(DTYPE)

diff_q = (q_a - q_b).abs().max().item()
diff_k = (k_a - k_b).abs().max().item()
print(f"  q_rot A vs B: diff={diff_q:.10f}")
print(f"  k_rot A vs B: diff={diff_k:.10f}")
if diff_q > 0:
    print(f"  q_rot_py[0,0,:5]={q_a[0,0,:5]}")
    print(f"  q_rot_nim[0,0,:5]={q_b[0,0,:5]}")

# ══════════════════════════════════════════════════════════════════
# STEP 3: SDPA
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 3: SDPA backends ===")
k_rpt = k_a.repeat_interleave(n_rep, dim=1)
v_rpt = v_4d.repeat_interleave(n_rep, dim=1)

def sdpa_with_backend(q, k, v, kwargs):
    with torch.backends.cuda.sdp_kernel(**kwargs):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0,
            is_causal=True, scale=head_dim ** -0.5)

backends = [
    ("flash", dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)),
    ("math", dict(enable_flash=False, enable_math=True, enable_mem_efficient=False)),
    ("mem_eff", dict(enable_flash=False, enable_math=False, enable_mem_efficient=True)),
]
for name, kwargs in backends:
    a = sdpa_with_backend(q_a, k_rpt, v_rpt, kwargs)
    a_reshaped = a.transpose(1, 2).reshape(batch, seq, -1)
    o = linear_forward_reimpl_exl3(a_reshaped, o_w, o_suh, o_svh, device=DEVICE)
    d = (o - expected_output).abs().max().item()
    print(f"  {name:8s}: vs expected={d:.6f}")

# ══════════════════════════════════════════════════════════════════
# STEP 4: Naive manual attention
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 4: Naive manual attention ===")
scale_f = head_dim ** -0.5
scores = torch.matmul(q_a.float(), k_rpt.float().transpose(-2, -1)) * scale_f
if seq > 1:
    mask = torch.triu(torch.full((seq,seq), float('-inf'), device=DEVICE, dtype=torch.float32), diagonal=1)
    scores = scores + mask
aw = torch.softmax(scores.float(), dim=-1).to(DTYPE)
on = torch.matmul(aw, v_rpt)
on_r = on.transpose(1,2).reshape(batch, seq, -1)
o_n = linear_forward_reimpl_exl3(on_r, o_w, o_suh, o_svh, device=DEVICE)
print(f"  naive vs expected: {((o_n - expected_output).abs().max().item()):.6f}")

# ══════════════════════════════════════════════════════════════════
# STEP 5: Cos/sin shape check
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 5: Cos/sin dimensions ===")
print(f"  position_ids: {position_ids}")
print(f"  cos_tbl: {cos_tbl.shape}")
print(f"  fixture cos: {expected_cos.shape}, sin: {expected_sin.shape}")
# Nim gives 2D cos per batch; fixture might be 2D or 3D
cos_nim = cos_tbl[position_ids[0]]  # [seq, head_dim]
print(f"  cos_nim_2d: {cos_nim.shape}")
cos_fix_b0 = expected_cos[0] if expected_cos.dim() == 3 else expected_cos
print(f"  cos_fixture_b0: {cos_fix_b0.shape}")
d_cos = (cos_nim - cos_fix_b0).abs().max().item()
print(f"  cos_nim vs fixture: diff={d_cos:.10f}")

print("\nDone.")
