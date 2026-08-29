#!/usr/bin/env python3
"""Generate the Qwen3.6-35B-A3B gated full-attention layer (layer 3)
fixtures from the real checkpoint shards, using the vendored reference
modeling on CPU torch bf16.

What is generated (under tests/fixtures/qwen36-attn/):

  rope-*   partial rotary: q/k (bf16) + positions + cos/sin (bf16, 64 wide)
           -> q_rot/k_rot via the reference apply_rotary_pos_emb
  norm-*   GemmaRMSNorm (1+w): x (bf16) -> norm(x), real q_norm weight
  attn-*   gated full attention layer 3: x -> output (post o_proj), plus
           intermediates (q_normed, k_normed, q_rot, k_rot, gate,
           attn_output_gated)

The attention forward is replayed step by step with the reference ops so the
intermediates can be captured. The replay output is asserted bit-identical to
the module's own forward before saving.

Run (twice; cmp proves byte determinism):
  cd <worktree root> && uv run --no-project --python 3.12 \
    --with "transformers @ file://<root>/_references_prod/transformers" \
    --with torch \
    workspace/transformers/tests/testgen/gen_qwen36_attn_fixtures.py

RAM: the invoking shell runs `vm_stat` and `pgrep -f "python.*(torch|hf)"`
before this script. The script re-runs both checks itself and refuses to
load weights when free memory is low or another python/torch process is
running (its own process chain is excluded).
"""

import json
from collections import OrderedDict
import os
import subprocess
import sys
import torch
from safetensors import safe_open
from safetensors import torch as st

# The reference transformers checkout is the source of truth, found by walking
# up from this script until a _references_prod/transformers/src directory
# appears (a worktree sits one level deeper than the repo root).
# QWEN36_VENDORED_SRC overrides the walk.
_here = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _here
VENDORED_SRC = None
for _ in range(8):
    _candidate = os.path.join(_REPO_ROOT, "_references_prod", "transformers", "src")
    if os.path.isdir(_candidate):
        VENDORED_SRC = _candidate
        break
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
if VENDORED_SRC is None:
    VENDORED_SRC = os.environ.get("QWEN36_VENDORED_SRC")
if not VENDORED_SRC or not os.path.isdir(VENDORED_SRC):
    raise SystemExit(
        f"[gen_qwen36_attn_fixtures] vendored modeling not found above {_here}. "
        "Set QWEN36_VENDORED_SRC to the _references_prod/transformers/src directory")
sys.path.insert(0, VENDORED_SRC)

import transformers  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeAttention,
    Qwen3_5MoeTextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)

# Determinism: single intra-op thread, deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config.
MODEL_NAME = "Qwen3.6-35B-A3B"
LAYER_IDX = 3
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(GRANDPARENT_DIR, "fixtures", "qwen36-attn")
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests/hf_models", MODEL_NAME
)
SHARD_3 = os.path.join(MODEL_DIR, "model-00003-of-00026.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# The vendored tree sha this fixture was generated against.
VENDORED_SHA = "36deb0b53ed0863f4b4dfdea23dcaec7f3df3701"
NUM_THREADS = 1

# Per-generator seeds, independent and order-agnostic.
SEED_NORM = 81
SEED_ATTN = 82
SEED_ROPE = 83

# Attention geometry of the checkpoint.
NUM_QO_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
HIDDEN = 2048
ROTARY_DIM = 64

PREFIX = f"model.language_model.layers.{LAYER_IDX}.self_attn."
MIN_FREE_BYTES = 8 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_qwen36_attn_fixtures] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_qwen36_attn_fixtures] vm_stat gave no 'Pages free' line")


def ancestor_pids() -> set:
    """PIDs of this process and its ancestors, up to init."""
    chain = set()
    pid = os.getpid()
    for _ in range(16):
        if pid <= 1:
            break
        chain.add(pid)
        out = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True, text=True)
        try:
            pid = int(out.stdout.strip())
        except ValueError:
            break
    return chain


def check_ram() -> None:
    """Refuse to load weights when memory is low or another python/torch
    process holds RAM (this process chain is excluded from the pgrep match,
    whose command line spells the torch dependency of this run)."""
    free = free_bytes()
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"[gen_qwen36_attn_fixtures] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_qwen36_attn_fixtures] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def check_vendored_sha() -> str:
    """Record the vendored tree sha and fail loudly on drift."""
    out = subprocess.run(
        ["git", "-C", VENDORED_SRC, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True)
    sha = out.stdout.strip()
    if sha != VENDORED_SHA:
        raise SystemExit(
            f"[gen_qwen36_attn_fixtures] vendored tree sha {sha} differs from "
            f"{VENDORED_SHA}; the fixture truth moved")
    return sha


def load_text_config() -> Qwen3_5MoeTextConfig:
    """Load the nested text_config from the checkpoint config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5MoeTextConfig.from_dict(wrapper["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


def load_layer3_weights() -> dict:
    """Load the six layer-3 self_attn tensors from their shard, via
    safe_open (memory-mapped, only these tensors are copied)."""
    weights = {}
    with safe_open(SHARD_3, framework="pt") as f:
        for key in f.keys():
            if key.startswith(PREFIX):
                weights[key[len(PREFIX):]] = f.get_tensor(key).clone()
    return weights


def build_attention(weights: dict, cfg: Qwen3_5MoeTextConfig) -> Qwen3_5MoeAttention:
    """Qwen3_5MoeAttention with real layer-3 weights loaded."""
    attn = Qwen3_5MoeAttention(cfg, layer_idx=LAYER_IDX)
    with torch.no_grad():
        attn.q_proj.weight.data = weights["q_proj.weight"]
        attn.k_proj.weight.data = weights["k_proj.weight"]
        attn.v_proj.weight.data = weights["v_proj.weight"]
        attn.o_proj.weight.data = weights["o_proj.weight"]
        attn.q_norm.weight.data = weights["q_norm.weight"]
        attn.k_norm.weight.data = weights["k_norm.weight"]
    attn.eval()
    return attn


# ── Partial rotary ─────────────────────────────────────────────────────────

def generate_rope_fixtures(rotary: Qwen3_5MoeTextRotaryEmbedding) -> None:
    """Partial-rope apply fixtures: only the first 64 of 256 dims rotate."""
    torch.manual_seed(SEED_ROPE)
    layer_name = "rope"

    def run_case(case_num, metadata, batch, seq_len, position_ids):
        position_ids = torch.tensor(position_ids).reshape(batch, seq_len).contiguous()
        dummy = torch.randn(batch, seq_len, HIDDEN, dtype=torch.bfloat16)
        cos, sin = rotary(dummy, position_ids)  # (batch, seq, 64) bf16
        cos_2d = cos[0].contiguous()  # (seq, 64)
        sin_2d = sin[0].contiguous()
        q = torch.randn(batch, seq_len, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        k = torch.randn(batch, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        save_fixture(
            layer_name, case_num, metadata,
            {
                "q": q, "k": k,
                "cos": cos_2d, "sin": sin_2d,
                "q_rot": q_rot, "k_rot": k_rot,
                "position_ids": position_ids,
            },
        )

    # Case 00: prefill, batch 2, seq 8, positions 0..7.
    run_case(
        0,
        {
            "model": MODEL_NAME,
            "layer": PREFIX,
            "case": "prefill_batch2_seq8",
            "rotary_dim": ROTARY_DIM,
            "theta": 10000000,
        },
        2, 8, [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
    )

    # Case 01: decode, single token at nonzero position.
    run_case(
        1,
        {
            "model": MODEL_NAME,
            "layer": PREFIX,
            "case": "decode_single_token_pos5",
            "rotary_dim": ROTARY_DIM,
            "theta": 10000000,
        },
        1, 1, [[5]],
    )

    # Case 02: scattered positions (index_select path, large angles).
    run_case(
        2,
        {
            "model": MODEL_NAME,
            "layer": PREFIX,
            "case": "scattered_positions",
            "rotary_dim": ROTARY_DIM,
            "theta": 10000000,
        },
        1, 4, [[3, 17, 255, 4096]],
    )

    # Case 03: rotate_half on a 64-wide slice (pair split over the partial dim).
    torch.manual_seed(SEED_ROPE)
    x = torch.randn(2, 8, NUM_QO_HEADS, ROTARY_DIM, dtype=torch.bfloat16)
    rotated = rotate_half(x)
    save_fixture(
        layer_name, 3,
        {
            "model": MODEL_NAME,
            "layer": PREFIX,
            "case": "rotate_half_64wide",
        },
        {"input": x, "output": rotated},
    )
    print(f"Generated {layer_name} fixtures")


# ── GemmaRMSNorm ───────────────────────────────────────────────────────────

def generate_norm_fixtures(q_norm) -> None:
    """GemmaRMSNorm (1+w) fixtures using the real q_norm weight."""
    torch.manual_seed(SEED_NORM)
    layer_name = "norm"

    cases = [
        (0, "head_dim_forward", torch.randn(2, 8, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16)),
        (1, "single_token", torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16)),
        (2, "zeros_input", torch.zeros(2, 4, HEAD_DIM, dtype=torch.bfloat16)),
    ]
    for case_num, case, x in cases:
        output = q_norm(x)
        save_fixture(
            layer_name, case_num,
            {
                "model": MODEL_NAME,
                "layer": PREFIX + "q_norm",
                "case": case,
                "eps": q_norm.eps,
            },
            {"input": x, "output": output, "weight": q_norm.weight.data},
        )
    print(f"Generated {layer_name} fixtures")


# ── Gated full attention ───────────────────────────────────────────────────

def attention_forward_capture(attn, hidden_states, position_embeddings):
    """Replay of the reference Qwen3_5MoeAttention.forward with intermediate
    capture.

    The replay copies the reference forward body op for op (sdpa interface):
    q|gate chunk, Gemma qk-norm, partial rope, repeat_kv, torch SDPA,
    sigmoid gate, o_proj. The caller asserts the replayed output equals the
    module's own forward output.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states, gate = torch.chunk(
        attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    q_normed = attn.q_norm(query_states.view(hidden_shape))  # (b, s, heads, dim)
    query_states = q_normed.transpose(1, 2)
    k_normed = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape))  # (b, s, kv, dim)
    key_states = k_normed.transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    q_rot = query_states
    k_rot = key_states

    key_states_r = repeat_kv(key_states, attn.num_key_value_groups)
    value_states_r = repeat_kv(value_states, attn.num_key_value_groups)

    is_causal = query_states.shape[2] > 1 and attn.is_causal
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states_r, value_states_r,
        attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=attn.scaling,
    )

    # sdpa_attention_forward transposes back to (batch, seq, heads, dim)
    # and makes the result contiguous before the model reshapes to (batch, seq, -1).
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output_gated = attn_output * torch.sigmoid(gate)
    output = attn.o_proj(attn_output_gated)
    return output, attn_output_gated, gate, q_normed, k_normed, q_rot, k_rot


def generate_attn_fixtures(attn: Qwen3_5MoeAttention, rotary: Qwen3_5MoeTextRotaryEmbedding) -> None:
    """Gated full-attention fixtures with real weights (layer 3)."""
    torch.manual_seed(SEED_ATTN)
    layer_name = "attn"

    cases = [
        (0, "prefill_seq8", (1, 8), [[0, 1, 2, 3, 4, 5, 6, 7]]),
        (1, "decode_single_token_pos5", (1, 1), [[5]]),
    ]
    for case_num, case, shape, pos_rows in cases:
        batch, seq_len = shape
        hidden_states = torch.randn(batch, seq_len, HIDDEN, dtype=torch.bfloat16)
        position_ids = torch.tensor(pos_rows).reshape(batch, seq_len).contiguous()
        cos, sin = rotary(hidden_states, position_ids)

        # Real forward (ground truth).
        output_real, _ = attn(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )

        # Replay with capture. Must be bit-identical to the real forward.
        output_cap, attn_output_gated, gate, q_normed, k_normed, q_rot, k_rot = (
            attention_forward_capture(attn, hidden_states, (cos, sin))
        )
        assert torch.equal(output_real, output_cap), (
            f"replay diverged from real forward for case {case_num}"
        )

        save_fixture(
            layer_name, case_num,
            {
                "model": MODEL_NAME,
                "layer": PREFIX,
                "case": case,
                "num_qo_heads": NUM_QO_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM,
                "rotary_dim": ROTARY_DIM,
                "hidden_size": HIDDEN,
                "seed": SEED_ATTN,
                "num_threads": NUM_THREADS,
                "dtype": "bfloat16",
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            },
            {
                "hidden_states": hidden_states,
                "position_ids": position_ids,
                "cos": cos, "sin": sin,
                "q_normed": q_normed, "k_normed": k_normed,
                "q_rot": q_rot, "k_rot": k_rot,
                "gate": gate,
                "attn_output_gated": attn_output_gated,
                "output": output_real,
            },
        )
    print(f"Generated {layer_name} fixtures")


def save_fixture(layer_name: str, case_num: int, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata file."""
    filename = f"{layer_name}-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(FIXTURE_DIR, filename)

    sorted_tensors = OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in sorted(tensors.items())
        if tensor is not None
    )
    serialized = st.save(sorted_tensors, metadata=None)
    with open(filepath, "wb") as f:
        f.write(serialized)

    metadata_path = filepath + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")
    return filepath


def main() -> None:
    check_ram()
    sha = check_vendored_sha()

    cfg = load_text_config()
    weights = load_layer3_weights()
    attn = build_attention(weights, cfg)
    q_norm = attn.q_norm
    rotary = Qwen3_5MoeTextRotaryEmbedding(cfg)

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate_rope_fixtures(rotary)
    generate_norm_fixtures(q_norm)
    generate_attn_fixtures(attn, rotary)

    print(f"[gen_qwen36_attn_fixtures] torch {torch.__version__}, vendored sha {sha[:12]}")
    print(f"[gen_qwen36_attn_fixtures] wrote {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
