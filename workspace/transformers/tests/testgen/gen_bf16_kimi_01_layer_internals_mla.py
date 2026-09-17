#!/usr/bin/env python3
"""Kimi-Linear-48B-A3B-Instruct fixture generator over the real checkpoint shards, resolved through the gitignored tests/hf_models symlink.

MLA layer-3 records (norm-*, attn-*, attnout-*) through the reference modeling
(KimiLinearAttention, eager torch fallback spelling) on torch bf16:
- fixture dir tests/fixtures/bf16-01-layer-internals/Kimi-Linear-48B-A3B-Instruct-layer-3/
- consumer tests/q_bf16/t_bf16_kimi_01_layer_internals_mla.nim

- <name>-Kimi-Linear-48B-A3B-Instruct-<case>.safetensor, the recorded boundary slices of the case
- .metadata.json.zst, the case identity, the recorded geometry and margins
- .stats.json.zst, the ttt-tf-004-uniform-stats frame over the floating-point payload tensors

Recorded content:
- op-for-op attention replay slabs sliced to heads 0 and 1, score rows and the composed o_proj output full
- the replay asserted bit-identical to the module's own forward
- no rotary exists, the q rope channels and the kpe plane cache and score unrotated, the counterfactual rows reject a rope-adding spelling

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root, PYTHONPATH pointing at the kimi-linear reference
worktree src carrying the torch fallback kernels:
  PYTHONPATH=<kimi-linear-reference-worktree>/src uv run python workspace/transformers/tests/testgen/gen_bf16_kimi_01_layer_internals_mla.py
"""
import argparse
import json
import compression.zstd

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}


def write_json_zst(path, obj, ensure_ascii=True):
    """Write obj as one zstd frame, level 19, content size and checksum recorded in the frame header."""
    payload = json.dumps(
        obj, sort_keys=True, indent=2, ensure_ascii=ensure_ascii
    ).encode("utf-8") + b"\n"
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload, options=ZSTD_WRITE_OPTIONS))

from collections import OrderedDict
import os
import subprocess
import sys


import torch  # noqa: E402
from safetensors import safe_open
from safetensors import torch as st


import transformers  # noqa: E402
import transformers.models.kimi_linear.modeling_kimi_linear as kimi_ref  # noqa: E402
from transformers.models.kimi_linear.configuration_kimi_linear import (  # noqa: E402
    KimiLinearConfig,
)
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import


# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# The checkpoint resolves through the gitignored tests/hf_models symlink.
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference module lock, the kimi-linear reference worktree, never an installed package. PYTHONPATH decides the resolution.
BRANCH_MARKER = "kimi-linear-branch"
REFERENCE_COMMIT = "eea35a8513"
if BRANCH_MARKER not in kimi_ref.__file__.replace("\\", "/"):
    raise SystemExit(
        f"[gen_bf16_kimi_01_layer_internals_mla] KimiLinearAttention resolved to "
        f"{kimi_ref.__file__}, not inside the {BRANCH_MARKER} worktree. Set "
        "PYTHONPATH to the branch worktree src, never to site-packages")


def branch_head() -> str:
    """HEAD of the reference worktree the module resolved from, asserted onto the recorded reference commit."""
    worktree = kimi_ref.__file__.split("/src/")[0]
    out = subprocess.run(
        ["git", "-C", worktree, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True)
    head = out.stdout.strip()
    if not head.startswith(REFERENCE_COMMIT):
        raise SystemExit(
            f"[gen_bf16_kimi_01_layer_internals_mla] branch worktree HEAD {head} does not "
            f"match the recorded reference commit {REFERENCE_COMMIT}")
    return head


MODEL_NAME = "Kimi-Linear-48B-A3B-Instruct"
LAYER_IDX = 3
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals",
    f"{MODEL_NAME}-layer-{LAYER_IDX}"
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests", "hf_models", MODEL_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

with open(CONFIG_PATH) as f:
    _CKPT_CONFIG = json.load(f)

NUM_THREADS = 1

DEVICE = "cpu"
    # Recording device, the --device argument overrides the default, the metadata and env device rows record the value the run used.

# Per-generator seeds, independent and order-agnostic.
SEED_NORM = 261
SEED_ATTN = 262
SEED_ATTN_STEP = 263

# MLA geometry of the checkpoint, config is king, the parsed values equal
# the 32/512/128/64/128/2304 tuple the checkpoint carries.
NUM_HEADS = _CKPT_CONFIG["num_attention_heads"]
KV_LORA_RANK = _CKPT_CONFIG["kv_lora_rank"]
QK_NOPE_HEAD_DIM = _CKPT_CONFIG["qk_nope_head_dim"]
QK_ROPE_HEAD_DIM = _CKPT_CONFIG["qk_rope_head_dim"]
V_HEAD_DIM = _CKPT_CONFIG["v_head_dim"]
HIDDEN = _CKPT_CONFIG["hidden_size"]
ROPE_THETA = 10000.0
MAX_SEQ = 1048576
MLA_LAYERS_0BASED = [3, 7, 11, 15, 19, 23, 26]

PREFIX = f"model.layers.{LAYER_IDX}.self_attn."
ATTN_TENSORS = (
    "q_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
)
def shard_of(key: str) -> str:
    """Shard filename holding one checkpoint tensor, read off the weight map, shard numbering on this checkpoint stays 1-indexed."""
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    if key not in weight_map:
        raise SystemExit(
            f"[gen_bf16_kimi_01_layer_internals_mla] key {key} missing from the weight map")
    return weight_map[key]


def load_config() -> KimiLinearConfig:
    """KimiLinearConfig built from the checkpoint fields the attention consumes plus the schedule block:
    - the strict dataclass rejects the checkpoint's extra top-level keys,
      the checkpoint head_dim 72 spelling, the dtype row and the router
      fields never reach the attention path
    - the loaded shapes below assert the real geometry"""
    with open(CONFIG_PATH) as f:
        raw = json.load(f)
    cfg = KimiLinearConfig.from_dict({
        "model_type": raw["model_type"],
        "vocab_size": raw["vocab_size"],
        "hidden_size": raw["hidden_size"],
        "num_hidden_layers": raw["num_hidden_layers"],
        "num_attention_heads": raw["num_attention_heads"],
        "num_key_value_heads": raw["num_key_value_heads"],
        "kv_lora_rank": raw["kv_lora_rank"],
        "q_lora_rank": raw["q_lora_rank"],
        "qk_rope_head_dim": raw["qk_rope_head_dim"],
        "v_head_dim": raw["v_head_dim"],
        "qk_nope_head_dim": raw["qk_nope_head_dim"],
        "rms_norm_eps": raw["rms_norm_eps"],
        "max_position_embeddings": raw["model_max_length"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "linear_attn_config": raw["linear_attn_config"],
    })
    cfg._attn_implementation = "eager"
    return cfg


def load_layer_weights() -> dict:
    """Load the five layer-3 self_attn tensors from the shards that hold them, memory-mapped safe_open reads."""
    weights = {}
    shards = {name: shard_of(PREFIX + name) for name in ATTN_TENSORS}
    for name, shard in shards.items():
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
            weights[name] = f.get_tensor(PREFIX + name).clone()
    return weights


def build_attention(weights: dict, cfg: KimiLinearConfig) -> kimi_ref.KimiLinearAttention:
    """KimiLinearAttention with real layer-3 weights loaded:
    - the RMSNorm weights land bf16, the checkpoint stores bf16 and a real
      from_pretrained load keeps them bf16, the norm forward multiplies
      the bf16 weight against the re-rounded bf16 activations"""
    attn = kimi_ref.KimiLinearAttention(cfg, layer_idx=LAYER_IDX)
    with torch.no_grad():
        for name in ATTN_TENSORS:
            module = {
                "q_proj.weight": attn.q_proj,
                "kv_a_proj_with_mqa.weight": attn.kv_a_proj_with_mqa,
                "kv_a_layernorm.weight": attn.kv_a_layernorm,
                "kv_b_proj.weight": attn.kv_b_proj,
                "o_proj.weight": attn.o_proj,
            }[name]
            module.weight.data = weights[name]
    attn.eval()
    assert attn.kv_a_layernorm.weight.dtype == torch.bfloat16
    return attn


def expand_kv_capture(attn, kv_nope: torch.Tensor, k_rot: torch.Tensor):
    """Replay of the reference expand_kv plus the rotation-off assert:
    - the kv_b decompress splits k_nope from value, the raw plane
      broadcasts per head
    - the key rope channels copy the plane bit-exactly, pure data movement
      with no rotation anywhere"""
    batch_size, _, seq_length, _ = kv_nope.shape
    key_shape = (batch_size, seq_length, -1, QK_NOPE_HEAD_DIM + V_HEAD_DIM)
    kv = attn.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)
    k_nope, value_states = torch.split(
        kv, [QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
    k_rot_expanded = k_rot.expand(-1, k_nope.shape[1], -1, -1)
    key_states = kv.new_empty(*kv.shape[:-1], QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    key_states[..., :QK_NOPE_HEAD_DIM].copy_(k_nope)
    key_states[..., QK_NOPE_HEAD_DIM:].copy_(k_rot_expanded)
    assert torch.equal(key_states[..., QK_NOPE_HEAD_DIM:],
                       k_rot.expand(-1, k_nope.shape[1], -1, -1)), (
        "key rope channels moved off the raw plane, the unrotated spelling broke")
    return key_states, value_states


def eager_scores(query_states, key_states, scaling):
    """Eager score spelling, bf16 GEMM times the scaling, pre-softmax, no mask,
    the additive mask joins after."""
    return torch.matmul(query_states, key_states.transpose(2, 3)) * scaling


def eager_attention(module, query, key, value, mask):
    """Reference eager_attention_forward spelling on the MLA geometry:
    - repeat_kv with one group is the identity, the score GEMM runs bf16
    - the additive mask joins, the softmax runs f32 and the result casts
      back to the query dtype, the value matmul runs at the query dtype
    - output transposed into the plane layout (b, s, h, v), the layout
      the module reshape and the recorded rows carry"""
    attn_weights = eager_scores(query, key, module.scaling)
    if mask is not None:
        attn_weights = attn_weights + mask
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = attn_weights.to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous()


def sdpa_plane(query, key, value, causal, scaling):
    """SDPA counterpart at the plane layout (b, s, h, v), the sdpa-native output
    transposed so the drift diff compares plane to plane."""
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=causal, scale=scaling).transpose(1, 2)


def rotated_counterfactual(q_rope, k_plane, positions):
    """NoPE-ignoring spelling a wrong port would run,
    plain-theta rotation on the q rope channels and the k plane, rotate_half
    form:
    - cos/sin tables built f64 to f32 and cast bf16 the way the reference
      rope spellings round
    - any fixed invertible rotation demonstrates the lock, the plain-theta
      one is the spelling the rope_theta config field would suggest to a template-blind port"""
    d = q_rope.shape[-1]
    inv = ROPE_THETA ** (-torch.arange(0, d, 2, dtype=torch.float64, device=DEVICE) / d)
    ang = torch.outer(torch.tensor(positions, dtype=torch.float64, device=DEVICE), inv)
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1).to(torch.bfloat16)
    cos = cos.to(q_rope.device)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1).to(torch.bfloat16)
    sin = sin.to(q_rope.device)

    def rot_half(t):
        half = t.shape[-1] // 2
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat((-t2, t1), dim=-1) * cos + torch.cat((t1, t2), dim=-1) * sin

    return rot_half(q_rope), rot_half(k_plane)


# Mask names excluded from the 004 stats sweep:
# - the masks are deterministic constants recorded in the payload,
#   their spelling documented in the metadata, not value instruments
# - the finfo(bfloat16).min fill above the diagonal sits in binade 127,
#   outside the ttt-tf-004 histogram's [-64, 63] binade range
STATS_EXCLUDED_NAMES = ("causal_mask", "decode_mask")


def save_fixture(layer_name: str, case_num: int, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata file plus the 004 stats sidecar."""
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

    metadata_path = filepath + ".metadata.json.zst"
    write_json_zst(metadata_path, metadata)
    write_stats_file(filepath + ".stats.json.zst", os.path.basename(filepath), [
        (name, tensor) for name, tensor in sorted_tensors.items()
        if tensor.is_floating_point() and name not in STATS_EXCLUDED_NAMES
    ])
    return filepath


def slice_heads_tagged(tensors: dict, names: tuple, head_dim_axis: int,
                       tag: str) -> dict:
    """slice_heads with a step tag on the name, X_h00 becomes X_<tag>_h00, keeping the step rows off the prefill row names."""
    out = {}
    for name in names:
        tensor = tensors[name]
        for h in (0, 1):
            out[f"{name}_{tag}_h{h:02d}"] = tensor.narrow(head_dim_axis, h, 1).contiguous()
    return out


def slice_heads(tensors: dict, names: tuple, head_dim_axis: int) -> dict:
    """Heads 0 and 1 slices of the named 4D slabs, key suffix _h00/_h01:
    - heads are independent in attention, a two-head slice is a valid
      per-op comparison at the 32-head geometry
    - the composed o_proj output collapses the head axis and stays full
    - head_dim_axis selects the head axis, 1 on the sdpa-native
      (b, h, s, d) slabs, 2 on the plane-layout (b, s, h, d) outputs"""
    out = {}
    for name in names:
        tensor = tensors[name]
        for h in (0, 1):
            out[f"{name}_h{h:02d}"] = tensor.narrow(head_dim_axis, h, 1).contiguous()
    return out


COMMON_META = {
    "model": MODEL_NAME,
    "layer": PREFIX,
    "num_heads": NUM_HEADS,
    "kv_lora_rank": KV_LORA_RANK,
    "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
    "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
    "v_head_dim": V_HEAD_DIM,
    "hidden_size": HIDDEN,
    "rope_theta": ROPE_THETA,
    "nope": True,
    "num_threads": NUM_THREADS,
    "dtype": "bfloat16",
    "torch_version": torch.__version__,
    "transformers_version": f"origin/kimi-linear {REFERENCE_COMMIT} "
                            "(branch worktree, torch fallback kernels)",
    "mla_layers_0based": MLA_LAYERS_0BASED,
    "full_attn_layers_1indexed": [m + 1 for m in MLA_LAYERS_0BASED],
    "last_layer_is_mla": True,
    "head_slice": [0, 1],
    "attention_spelling": "eager_attention_forward, bf16 score GEMM, "
                          "softmax f32 cast back to the query dtype, "
                          "value matmul at the query dtype",
    "causal_mask_spelling": "additive bf16 mask from create_causal_mask, "
                            "finfo(bfloat16).min above the diagonal, the "
                            "decode-step mask is the (1, 1, 1, 1) zero "
                            "no-op the model-level cache view produces "
                            "(the DynamicCache sequence length reads layer "
                            "0, which never writes on this hybrid template)",
    "unrotated_plane_proof": "the gathered plane equals the raw kv_a slice "
                             "bit-exactly at every capture point, the "
                             "recorded key rope channels equal the cached "
                             "plane broadcast bit-exactly, and the recorded "
                             "eager scores match the unrotated recompute "
                             "under the attention budget while the rotated "
                             "counterfactual is rejected under the same "
                             "budget",
}


# ── RMSNorm ────────────────────────────────────────────────────────────────

def generate_norm_fixtures(kv_a_layernorm) -> None:
    """kv_a_layernorm fixtures using the real weight, the module default eps 1e-6 as the recorded spelling."""
    torch.manual_seed(SEED_NORM)
    layer_name = "norm"
    cases = [
        (0, "prefill_latent", torch.randn(1, 8, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)),
        (1, "decode_latent", torch.randn(1, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)),
        (2, "zeros_input", torch.zeros(1, 4, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)),
    ]
    for case_num, case, x in cases:
        output = kv_a_layernorm(x)
        save_fixture(
            layer_name, case_num,
            {
                **COMMON_META,
                "case": case,
                "eps": kv_a_layernorm.variance_epsilon,
                "eps_note": "module default eps 1e-6, the reference attention "
                    "constructs the latent norm without config.rms_norm_eps",
            },
            {"input": x, "output": output, "weight": kv_a_layernorm.weight.data},
        )
    print(f"Generated {layer_name} fixtures")


# ── MLA layer forward with capture ────────────────────────────────────────

def attention_forward_capture(attn, hidden_states, cache, mask):
    """Replay of the reference KimiLinearAttention.forward with intermediate capture, the reference forward body op for op on the eager
    interface:
    - q_proj split into q_pass and the raw rope channels
    - kv_a_proj_with_mqa split into the latent and the raw plane
    - latent norm at the module default eps
    - the compressed cache update, the plane cached unrotated
    - expand_kv, the eager score GEMM, softmax f32 cast bf16, the value matmul, the o_proj projection
    The caller asserts the replayed output equals the module's own forward before saving. No rotation exists anywhere, the rope channels
    of q and the plane of k stay the raw projection slices."""
    batch_size, seq_length = hidden_states.shape[:-1]
    q = attn.q_proj(hidden_states)
    q = q.view(batch_size, seq_length, -1, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    q = q.transpose(1, 2)
    q_pass, q_rot_raw = torch.split(
        q, [QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)

    compressed_kv = attn.kv_a_proj_with_mqa(hidden_states)
    kv_nope, k_rot_raw = torch.split(
        compressed_kv, [KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)
    latent_normed = attn.kv_a_layernorm(kv_nope).view(
        batch_size, 1, seq_length, KV_LORA_RANK)
    k_rot = k_rot_raw.view(batch_size, 1, seq_length, QK_ROPE_HEAD_DIM)

    # Cache read / write is performed while latent KV is still compressed,
    # the plane cached unrotated beside it.
    if cache is not None:
        latent_cached, kpe_cached = cache.update(latent_normed, k_rot, attn.layer_idx)
    else:
        latent_cached, kpe_cached = latent_normed, k_rot

    query_states = torch.cat((q_pass, q_rot_raw), dim=-1)
    key_states, value_states = expand_kv_capture(attn, latent_cached, kpe_cached)

    attn_out_4d = eager_attention(
        attn, query_states, key_states, value_states, mask)
    output = attn.o_proj(
        attn_out_4d.reshape(batch_size, seq_length, -1).contiguous())
    return {
        "q_pe": q_rot_raw,
        "query_states": query_states,
        "latent_normed": latent_normed,
        "latent_cached": latent_cached, "kpe_cached": kpe_cached,
        "k_rot_raw": k_rot_raw.view(batch_size, 1, seq_length, QK_ROPE_HEAD_DIM),
        "key_states": key_states, "value_states": value_states,
        "attn_output": attn_out_4d, "output": output,
    }


def generate_attn_fixtures(attn: kimi_ref.KimiLinearAttention) -> None:
    """MLA layer fixtures with real weights (layer 3), latent-cache path, eager
    reference spelling plus the SDPA counterpart and the unrotated lock."""
    layer_name = "attn"
    out_layer_name = "attnout"

    def fresh_cache(cache_layers):
        cache = DynamicCache()
        for i, (lat, kpe) in enumerate(cache_layers):
            cache.update(lat, kpe, LAYER_IDX)
        return cache

    def run_pass(x, positions, cache_layers, mask):
        # Replay and module forward each get a fresh cache preloaded
        # with the prior passes' cached tensors, so neither run advances
        # the state the other reads.
        cap = attention_forward_capture(attn, x, fresh_cache(cache_layers), mask)
        output_real, attn_weights = attn(
            hidden_states=x,
            attention_mask=mask,
            past_key_values=fresh_cache(cache_layers),
        )
        assert torch.equal(output_real, cap["output"]), (
            "replay diverged from real forward")
        cap["attn_weights"] = attn_weights
        cap["sdpa_twin"] = sdpa_plane(
            cap["query_states"], cap["key_states"], cap["value_states"],
            causal=(x.shape[1] > 1), scaling=attn.scaling)
        cap["eager_sdpa_drift"] = (
            (cap["attn_output"].float() - cap["sdpa_twin"].float())
            .abs().max().item())
        return cap

    def state_meta(case, positions, drift):
        return {
            **COMMON_META,
            "case": case,
            "positions": positions,
            "eager_vs_sdpa_maxdiff": drift,
        }

    # Case 00 prefill seq 8, positions 0..7, cache write of the full pass,
    # the unrotated check runs over the full score rows.
    torch.manual_seed(SEED_ATTN)
    x = torch.randn(1, 8, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    mask = create_causal_mask(config=attn.config, inputs_embeds=x,
                              attention_mask=None, past_key_values=None,
                              position_ids=None)
    cap = run_pass(x, list(range(8)), [], mask)
    scores = eager_scores(cap["query_states"], cap["key_states"], attn.scaling)
    q_rope_cf, k_plane_cf = rotated_counterfactual(
        cap["q_pe"], cap["k_rot_raw"], list(range(8)))
    key_cf = torch.cat([cap["key_states"][..., :QK_NOPE_HEAD_DIM],
                        k_plane_cf.expand(-1, NUM_HEADS, -1, -1)], dim=-1)
    q_cf = torch.cat([cap["query_states"][..., :QK_NOPE_HEAD_DIM], q_rope_cf], dim=-1)
    scores_cf = eager_scores(q_cf, key_cf, attn.scaling)
    score_drift = (scores.float() - scores_cf.float()).abs().max().item()
    assert score_drift > 0.0, "the rotated counterfactual must move the scores"
    out_cf = eager_attention(
        attn, q_cf, key_cf, cap["value_states"], mask)
    out_drift = (cap["attn_output"].float() - out_cf.float()).abs().max().item()
    assert out_drift > 0.0, "the rotated counterfactual must move the output"
    save_fixture(
        layer_name, 0,
        {
            **state_meta("prefill_seq8_unrotated_pin", list(range(8)),
                         cap["eager_sdpa_drift"]),
            "softmax_scaling": attn.scaling,
            "unrotated_plane_score_maxdiff": score_drift,
            "unrotated_plane_output_maxdiff": out_drift,
            "counterfactual_spelling": "plain-theta rotate_half on the q rope "
                "channels and the k plane at rope_theta 10000, the "
                "NoPE-ignoring spelling a template-blind port would run",
        },
        {
            "hidden_states": x,
            "positions": torch.arange(8, dtype=torch.long, device=DEVICE),
            "causal_mask": mask,
            "latent_normed": cap["latent_normed"],
            "latent_cached": cap["latent_cached"],
            "kpe_cached": cap["kpe_cached"],
            "scores": scores,
            "scores_rotated_cf": scores_cf,
            **slice_heads(cap, ("query_states", "key_states", "value_states"), 1),
        },
    )
    save_fixture(
        out_layer_name, 0,
        {
            **state_meta("prefill_seq8_outputs", list(range(8)),
                         cap["eager_sdpa_drift"]),
            "softmax_scaling": attn.scaling,
            "sdpa_twin_layout": "plane (b, s, h, v), the sdpa-native output "
                "transposed so the drift diff and the fixture rows compare "
                "plane to plane",
            "unrotated_plane_output_maxdiff": out_drift,
        },
        {
            **slice_heads(cap, ("attn_output",), 2),
            "sdpa_twin_h00": cap["sdpa_twin"].narrow(2, 0, 1).contiguous(),
            "sdpa_twin_h01": cap["sdpa_twin"].narrow(2, 1, 1).contiguous(),
            "output": cap["output"],
            "attn_output_rotated_cf_h00": out_cf.narrow(2, 0, 1).contiguous(),
            "attn_output_rotated_cf_h01": out_cf.narrow(2, 1, 1).contiguous(),
        },
    )
    print(f"Generated {layer_name} case 0")

    # Case 01 prefill 3 then one decode step at position 3.
    torch.manual_seed(SEED_ATTN + 1)
    x = torch.randn(1, 3, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    x_step = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    mask3 = create_causal_mask(config=attn.config, inputs_embeds=x,
                               attention_mask=None, past_key_values=None,
                               position_ids=None)
    cap = run_pass(x, [0, 1, 2], [], mask3)
    cache_layers = [(cap["latent_cached"], cap["kpe_cached"])]
    dmask = create_causal_mask(config=attn.config, inputs_embeds=x_step,
                               attention_mask=None, past_key_values=None,
                               position_ids=None)
    cap_step = run_pass(x_step, [3], cache_layers, dmask)
    save_fixture(
        layer_name, 1,
        {
            **state_meta("prefill3_decode1_pos3", [0, 1, 2], cap["eager_sdpa_drift"]),
            "eager_vs_sdpa_maxdiff_step": cap_step["eager_sdpa_drift"],
            "softmax_scaling": attn.scaling,
        },
        {
            "hidden_states": x, "x_step": x_step,
            "positions": torch.tensor([0, 1, 2], dtype=torch.long, device=DEVICE),
            "positions_step": torch.tensor([3], dtype=torch.long, device=DEVICE),
            "causal_mask": mask3, "decode_mask": dmask,
            "latent_normed": cap["latent_normed"],
            "latent_normed_step": cap_step["latent_normed"],
            "latent_cached": cap_step["latent_cached"],
            "kpe_cached": cap_step["kpe_cached"],
            "k_rot_raw_step": cap_step["k_rot_raw"],
            "scores_step": eager_scores(
                cap_step["query_states"], cap_step["key_states"], attn.scaling),
            **slice_heads(cap, ("query_states", "key_states", "value_states"), 1),
            **slice_heads_tagged(cap_step,
                ("query_states", "key_states", "value_states"), 1, "step"),
        },
    )
    save_fixture(
        out_layer_name, 1,
        {
            **state_meta("prefill3_decode1_outputs", [0, 1, 2],
                         cap["eager_sdpa_drift"]),
            "eager_vs_sdpa_maxdiff_step": cap_step["eager_sdpa_drift"],
            "softmax_scaling": attn.scaling,
            "sdpa_twin_layout": "plane (b, s, h, v), the sdpa-native output "
                "transposed so the drift diff and the fixture rows compare "
                "plane to plane",
        },
        {
            **slice_heads(cap, ("attn_output",), 2),
            **slice_heads_tagged(cap_step, ("attn_output",), 2, "step"),
            "sdpa_twin_h00": cap["sdpa_twin"].narrow(2, 0, 1).contiguous(),
            "sdpa_twin_h01": cap["sdpa_twin"].narrow(2, 1, 1).contiguous(),
            "sdpa_twin_step_h00": cap_step["sdpa_twin"].narrow(2, 0, 1).contiguous(),
            "sdpa_twin_step_h01": cap_step["sdpa_twin"].narrow(2, 1, 1).contiguous(),
            "output": cap["output"],
            "output_step": cap_step["output"],
        },
    )
    print(f"Generated {layer_name} case 1")

    # Case 02 prefill 3 then a 3-step decode sequence, positions 3, 4, 5.
    torch.manual_seed(SEED_ATTN + 2)
    x = torch.randn(1, 3, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    x_steps = [torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device=DEVICE) for _ in range(3)]
    mask3 = create_causal_mask(config=attn.config, inputs_embeds=x,
                               attention_mask=None, past_key_values=None,
                               position_ids=None)
    cap = run_pass(x, [0, 1, 2], [], mask3)
    cache_layers = [(cap["latent_cached"], cap["kpe_cached"])]
    dmask = create_causal_mask(config=attn.config, inputs_embeds=x_steps[0],
                               attention_mask=None, past_key_values=None,
                               position_ids=None)
    state_tensors = {
        "hidden_states": x,
        "positions": torch.tensor([0, 1, 2], dtype=torch.long, device=DEVICE),
        "causal_mask": mask3, "decode_mask": dmask,
        "latent_normed": cap["latent_normed"],
        "latent_cached": cap["latent_cached"],
        "kpe_cached": cap["kpe_cached"],
        **slice_heads(cap, ("query_states", "key_states", "value_states"), 1),
    }
    out_tensors = {
        **slice_heads(cap, ("attn_output",), 2),
        "sdpa_twin_h00": cap["sdpa_twin"].narrow(2, 0, 1).contiguous(),
        "sdpa_twin_h01": cap["sdpa_twin"].narrow(2, 1, 1).contiguous(),
        "output": cap["output"],
    }
    step_meta_positions = []
    max_step_drift = cap["eager_sdpa_drift"]
    for i, x_step in enumerate(x_steps):
        position = 3 + i
        step_meta_positions.append(position)
        cap_step = run_pass(x_step, [position], cache_layers, dmask)
        cache_layers = [(cap_step["latent_cached"], cap_step["kpe_cached"])]
        max_step_drift = max(max_step_drift, cap_step["eager_sdpa_drift"])
        state_tensors[f"x_step{i}"] = x_step
        state_tensors[f"latent_normed_step{i}"] = cap_step["latent_normed"]
        state_tensors[f"k_rot_raw_step{i}"] = cap_step["k_rot_raw"]
        state_tensors[f"scores_step{i}"] = eager_scores(
            cap_step["query_states"], cap_step["key_states"], attn.scaling)
        for name, tensor in slice_heads_tagged(
                cap_step, ("query_states", "key_states", "value_states"),
                1, f"step{i}").items():
            state_tensors[name] = tensor
        out_tensors[f"output_step{i}"] = cap_step["output"]
        for name, tensor in slice_heads_tagged(
                cap_step, ("attn_output",), 2, f"step{i}").items():
            out_tensors[name] = tensor
        out_tensors[f"sdpa_twin_step{i}_h00"] = cap_step["sdpa_twin"].narrow(2, 0, 1).contiguous()
        out_tensors[f"sdpa_twin_step{i}_h01"] = cap_step["sdpa_twin"].narrow(2, 1, 1).contiguous()
    state_tensors["latent_cached"] = cap_step["latent_cached"]
    state_tensors["kpe_cached"] = cap_step["kpe_cached"]
    state_tensors["positions_steps"] = torch.tensor(step_meta_positions, dtype=torch.long, device=DEVICE)
    save_fixture(
        layer_name, 2,
        {
            **state_meta("prefill3_decode3_steps", [0, 1, 2],
                         cap["eager_sdpa_drift"]),
            "positions_steps": step_meta_positions,
            "eager_vs_sdpa_maxdiff_step": max_step_drift,
            "softmax_scaling": attn.scaling,
        },
        state_tensors,
    )
    save_fixture(
        out_layer_name, 2,
        {
            **state_meta("prefill3_decode3_outputs", [0, 1, 2],
                         cap["eager_sdpa_drift"]),
            "positions_steps": step_meta_positions,
            "eager_vs_sdpa_maxdiff_step": max_step_drift,
            "softmax_scaling": attn.scaling,
            "sdpa_twin_layout": "plane (b, s, h, v), the sdpa-native output "
                "transposed so the drift diff and the fixture rows compare "
                "plane to plane",
        },
        out_tensors,
    )
    print(f"Generated {layer_name} case 2")


def main() -> None:
    """Records the layer-3 MLA fixtures of the checkpoint."""
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="recording device, the metadata device rows record it")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    COMMON_META["recorded_from"] = os.environ.get("TTT_RECORD_FROM", "m4max-cpu")
    COMMON_META["device"] = DEVICE
    branch_head()
    cfg = load_config()
    # Schedule assert, the checkpoint 1-indexed list converts
    # to the 0-based MLA set and the LAST layer stays full attention
    # (the audit invariant the parse enforces).
    full0 = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    assert full0 == MLA_LAYERS_0BASED, f"layer_types gave {full0}"
    assert cfg.layer_types[-1] == "full_attention"
    assert cfg.q_lora_rank is None, "direct-Q is the recorded spelling"

    weights = load_layer_weights()
    attn = build_attention(weights, cfg)
    attn = attn.to(DEVICE)
    assert attn.kv_a_layernorm.variance_epsilon == 1e-6, (
        "the reference latent norm runs the module default eps 1e-6, "
        "config.rms_norm_eps never reaches it")
    assert attn.scaling == 0.07216878364870322
    assert attn.num_key_value_groups == 1

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate_norm_fixtures(attn.kv_a_layernorm)
    generate_attn_fixtures(attn)



    print(f"[gen_bf16_kimi_01_layer_internals_mla] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_kimi_01_layer_internals_mla] wrote {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
