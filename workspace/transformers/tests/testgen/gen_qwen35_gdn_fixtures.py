"""
Generate fixtures for the Qwen3.5-0.8B Gated DeltaNet (GDN) layer 0, full
decoder layers 0 and 3, and a 3-block chain of layers 0..2, using the
reference transformers modeling on CPU torch bf16.

Reference: gen_qwen35_attn_fixtures.py conventions.

What is generated:

  tests/fixtures/layers/Qwen3.5-0.8B-layer-0/
    gdn-Qwen3.5-0.8B-00.safetensor
      GDN block prefill T=5 with real layer-0 weights: conv output, q/k/v
      post-split, z, g, beta, sequential core output + per-step SSM states,
      gated RMSNorm output, sequential block output (0.00 reference) and the
      chunked module output (5e-3 reference).
    gdn-Qwen3.5-0.8B-01.safetensor
      State trajectory: a sequential one-shot over 5 tokens (per-step conv
      output, core output, SSM states, block output) plus a 2-step decode
      after a 3-token prefill through the reference module with a cache
      (conv states, SSM states, per-step conv inputs and outputs, decode
      outputs). The generator asserts the two paths agree bit for bit.
    layer-Qwen3.5-0.8B-00.safetensor
      Full decoder layer 0 forward on T=5: the real chunked layer output and
      the sequential-replay layer output (0.00 reference) with intermediates.

  tests/fixtures/layers/Qwen3.5-0.8B-layer-3/
    layer-Qwen3.5-0.8B-03.safetensor
      Full decoder layer 3 (full attention) forward on T=5 with real rotary
      embeddings, plus the attn intermediates.

  tests/fixtures/long-residual-3-block/Qwen3.5-0.8B/
    block-00..02.safetensor
      Layers 0, 1, 2 run in sequence on a seeded T=4 input. Each block saves
      the reference chunked chain (layer_input, layer_output) and the
      sequential-replay chain (layer_input_seq, layer_output_seq).

  tests/fixtures/layers/Qwen3.5-0.8B-layer-0/
    gdn-Qwen3.5-0.8B-02.safetensor
      Multi-chunk prefill (T=70, two FLA chunks): the reference chunked block
      output, the sequential-replay output and the f32 core/state values.
      The generator asserts the sequential-vs-chunked max diff lies in
      (0, 1e-3), locking the documented ~1.5e-5 divergence band.
"""

import json
from collections import OrderedDict
import os
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors import torch as st

# The reference transformers checkout is the source of truth.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
VENDORED_SRC = os.environ.get(
    "QWEN35_VENDORED_SRC",
    os.path.join(_REPO_ROOT, "_references_prod", "transformers", "src"))
if not os.path.isdir(VENDORED_SRC):
    raise SystemExit(
        f"[gen_qwen35_gdn_fixtures] vendored modeling not found at {VENDORED_SRC}. "
        "Set QWEN35_VENDORED_SRC to the _references_prod/transformers/src directory")
sys.path.insert(0, VENDORED_SRC)

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextRotaryEmbedding,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

# Determinism (called once at import time).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config.
MODEL_NAME = "Qwen3.5-0.8B"
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
LAYER0_FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "layers", f"{MODEL_NAME}-layer-0"
)
LAYER3_FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "layers", f"{MODEL_NAME}-layer-3"
)
CHAIN_FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "long-residual-3-block", MODEL_NAME
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), f"tests/hf_models/{MODEL_NAME}"
)
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Per-generator seeds, independent and order-agnostic.
SEED_GDN_PREFILL = 61
SEED_STATE = 62
SEED_LAYER0 = 63
SEED_LAYER3 = 64
SEED_CHAIN = 65
SEED_MULTICHUNK = 66

HIDDEN = 1024
PREFILL_SEQ = 5
CHAIN_SEQ = 4
PREFILL_STATE_TOKENS = 3
MULTICHUNK_SEQ = 70


def load_text_config() -> Qwen3_5TextConfig:
    """Load the nested text_config from the wrapper config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5TextConfig.from_dict(wrapper["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


def ensure_fixture_dirs() -> None:
    for d in (LAYER0_FIXTURE_DIR, LAYER3_FIXTURE_DIR, CHAIN_FIXTURE_DIR):
        os.makedirs(d, exist_ok=True)


def save_fixture(fixture_dir: str, layer_name: str, case_num: int, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata file."""
    filename = f"{layer_name}-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(fixture_dir, filename)

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


def load_shard_tensors(prefix: str) -> dict:
    """Load every tensor under a shard key prefix, with the prefix stripped."""
    weights = {}
    with safe_open(MODEL_PATH, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                weights[key[len(prefix):]] = f.get_tensor(key).clone()
    return weights


def build_gdn_layer0(config: Qwen3_5TextConfig) -> Qwen3_5GatedDeltaNet:
    """Qwen3_5GatedDeltaNet with real layer-0 weights loaded."""
    block = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    w = load_shard_tensors("model.language_model.layers.0.linear_attn.")
    block.in_proj_qkv.weight.data = w["in_proj_qkv.weight"]
    block.in_proj_z.weight.data = w["in_proj_z.weight"]
    block.in_proj_a.weight.data = w["in_proj_a.weight"]
    block.in_proj_b.weight.data = w["in_proj_b.weight"]
    block.conv1d.weight.data = w["conv1d.weight"]
    block.A_log.data = w["A_log"].to(torch.bfloat16)
    block.dt_bias.data = w["dt_bias"]
    block.norm.weight.data = w["norm.weight"].to(torch.bfloat16)
    block.out_proj.weight.data = w["out_proj.weight"]
    return block


def build_decoder_layer(config: Qwen3_5TextConfig, layer_idx: int) -> Qwen3_5DecoderLayer:
    """Qwen3_5DecoderLayer with real weights for one layer."""
    layer = Qwen3_5DecoderLayer(config, layer_idx=layer_idx)
    w = load_shard_tensors(f"model.language_model.layers.{layer_idx}.")
    layer.input_layernorm.weight.data = w["input_layernorm.weight"]
    layer.post_attention_layernorm.weight.data = w["post_attention_layernorm.weight"]
    layer.mlp.gate_proj.weight.data = w["mlp.gate_proj.weight"]
    layer.mlp.up_proj.weight.data = w["mlp.up_proj.weight"]
    layer.mlp.down_proj.weight.data = w["mlp.down_proj.weight"]
    if layer.layer_type == "linear_attention":
        a = w["linear_attn.in_proj_qkv.weight"]
        layer.linear_attn.in_proj_qkv.weight.data = a
        layer.linear_attn.in_proj_z.weight.data = w["linear_attn.in_proj_z.weight"]
        layer.linear_attn.in_proj_a.weight.data = w["linear_attn.in_proj_a.weight"]
        layer.linear_attn.in_proj_b.weight.data = w["linear_attn.in_proj_b.weight"]
        layer.linear_attn.conv1d.weight.data = w["linear_attn.conv1d.weight"]
        layer.linear_attn.A_log.data = w["linear_attn.A_log"].to(torch.bfloat16)
        layer.linear_attn.dt_bias.data = w["linear_attn.dt_bias"]
        layer.linear_attn.norm.weight.data = w["linear_attn.norm.weight"].to(torch.bfloat16)
        layer.linear_attn.out_proj.weight.data = w["linear_attn.out_proj.weight"]
    else:
        attn = layer.self_attn
        attn.q_proj.weight.data = w["self_attn.q_proj.weight"]
        attn.k_proj.weight.data = w["self_attn.k_proj.weight"]
        attn.v_proj.weight.data = w["self_attn.v_proj.weight"]
        attn.o_proj.weight.data = w["self_attn.o_proj.weight"]
        attn.q_norm.weight.data = w["self_attn.q_norm.weight"]
        attn.k_norm.weight.data = w["self_attn.k_norm.weight"]
    return layer


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize over one dim in the input dtype (FLA alignment)."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class _LayerCache:
    """Minimal per-layer cache for the reference GDN decode path.

    Implements the surface the module forward touches: conv_states,
    recurrent_states, has_previous_state and the two update procs, which
    copy in place like the reference LinearAttentionLayer cache.
    """

    def __init__(self, conv_states: torch.Tensor, recurrent_states: torch.Tensor):
        self.conv_states = conv_states
        self.recurrent_states = recurrent_states
        self.has_previous_state = True

    def update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
        self.conv_states.copy_(conv_states)
        return self.conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
        self.recurrent_states.copy_(recurrent_states)
        return self.recurrent_states


class _GdnCache:
    """Cache facade for one GDN layer (decode path only)."""

    def __init__(self, layer_idx: int, conv_states: torch.Tensor, recurrent_states: torch.Tensor):
        self.layers = {layer_idx: _LayerCache(conv_states, recurrent_states)}

    def has_previous_state(self, layer_idx: int) -> bool:
        return self.layers[layer_idx].has_previous_state

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        return self.layers[layer_idx].update_conv_state(conv_states, **kwargs)

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        return self.layers[layer_idx].update_recurrent_state(recurrent_states, **kwargs)


def gdn_projections(block: Qwen3_5GatedDeltaNet, hidden_states: torch.Tensor):
    """in_proj_qkv/z/a/b of the GDN block, matching the reference forward."""
    mixed_qkv = block.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = block.in_proj_z(hidden_states)
    z = z.reshape(hidden_states.shape[0], hidden_states.shape[1], -1, block.head_v_dim)
    b = block.in_proj_b(hidden_states)
    a = block.in_proj_a(hidden_states)
    return mixed_qkv, z, b, a


def gdn_forward_replay(block: Qwen3_5GatedDeltaNet, hidden_states: torch.Tensor, use_recurrent: bool) -> dict:
    """Replay of the reference Qwen3_5GatedDeltaNet.forward with a selectable
    core rule, capturing every intermediate.

    The chunked replay must be bit-identical to the module's own forward
    (asserted by the caller). The sequential replay is the 0.00 reference for
    the Nim implementation.
    """
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv, z, b, a = gdn_projections(block, hidden_states)

    conv_output = F.silu(block.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
    mixed = conv_output.transpose(1, 2)
    query, key, value = torch.split(mixed, [block.key_dim, block.key_dim, block.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, block.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, block.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, block.head_v_dim)

    beta = b.sigmoid()
    g = -block.A_log.float().exp() * F.softplus(a.float() + block.dt_bias)

    rule = torch_recurrent_gated_delta_rule if use_recurrent else torch_chunk_gated_delta_rule
    core_attn_out, ssm_state = rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    core_2d = core_attn_out.reshape(-1, block.head_v_dim)
    z_2d = z.reshape(-1, block.head_v_dim)
    normed = block.norm(core_2d, z_2d).reshape(batch_size, seq_len, -1)
    output = block.out_proj(normed)

    return {
        "conv_output": conv_output,
        "query": query, "key": key, "value": value,
        "z": z, "g": g, "beta": beta,
        "core_attn_out": core_attn_out,
        "ssm_state": ssm_state,
        "normed": normed,
        "output": output,
    }


def recurrent_rule_with_trajectory(query, key, value, g, beta, eps: float = 1e-6):
    """torch_recurrent_gated_delta_rule with per-step state capture.

    Replicates the reference loop op for op and asserts the final output and
    state match the function's own results, so the captured per-step
    trajectory is trustworthy.
    """
    initial_dtype = query.dtype
    q_n = l2norm(query, dim=-1, eps=eps)
    k_n = l2norm(key, dim=-1, eps=eps)
    q32, k32, v32, beta32, g32 = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (q_n, k_n, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = k32.shape
    v_head_dim = v32.shape[-1]
    scale = 1 / (q32.shape[-1] ** 0.5)
    q_scaled = q32 * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim,
        dtype=v32.dtype, device=v32.device,
    )
    s = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=v32.dtype, device=v32.device)
    states = [s.clone()]
    for i in range(sequence_length):
        q_t = q_scaled[:, :, i]
        k_t = k32[:, :, i]
        v_t = v32[:, :, i]
        g_t = g32[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta32[:, :, i].unsqueeze(-1)
        s = s * g_t
        kv_mem = (s * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        s = s + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (s * q_t.unsqueeze(-1)).sum(dim=-2)
        states.append(s.clone())

    out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    ref_out, ref_state = torch_recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert torch.equal(out, ref_out), "trajectory output diverged from the recurrent rule"
    assert torch.equal(s, ref_state), "trajectory state diverged from the recurrent rule"
    return out, s, torch.stack(states)[:, 0]


def decoder_layer_forward_seq(layer: Qwen3_5DecoderLayer, x: torch.Tensor) -> torch.Tensor:
    """Qwen3_5DecoderLayer.forward with the GDN block on the sequential rule.

    The reference forward runs the chunked rule for prefill. This replay
    substitutes the sequential rule so the Nim implementation (sequential
    always) has a 0.00 reference at the layer level.
    """
    residual = x
    h = layer.input_layernorm(x)
    if layer.layer_type == "linear_attention":
        gdn_out = gdn_forward_replay(layer.linear_attn, h, use_recurrent=True)["output"]
    else:
        raise ValueError("decoder_layer_forward_seq supports linear_attention layers only")
    h = residual + gdn_out
    residual = h
    h = layer.post_attention_layernorm(h)
    h = layer.mlp(h)
    return residual + h


def generate_gdn_prefill_fixture(block: Qwen3_5GatedDeltaNet) -> None:
    """GDN block prefill T=5: sequential reference + chunked module output."""
    torch.manual_seed(SEED_GDN_PREFILL)
    x = torch.randn(1, PREFILL_SEQ, HIDDEN, dtype=torch.bfloat16)

    module_output = block(x)  # reference forward, chunked rule
    chunk_replay = gdn_forward_replay(block, x, use_recurrent=False)
    assert torch.equal(module_output, chunk_replay["output"]), (
        "chunked replay diverged from the module forward"
    )

    core_out, _, states = recurrent_rule_with_trajectory(
        chunk_replay["query"], chunk_replay["key"], chunk_replay["value"],
        chunk_replay["g"], chunk_replay["beta"],
    )
    seq_replay = gdn_forward_replay(block, x, use_recurrent=True)
    assert torch.equal(seq_replay["core_attn_out"], core_out), (
        "sequential replay core diverged from the trajectory"
    )

    save_fixture(
        LAYER0_FIXTURE_DIR, "gdn", 0,
        {
            "model": MODEL_NAME,
            "layer": "model.language_model.layers.0.linear_attn",
            "case": "prefill_seq5",
            "seq_len": PREFILL_SEQ,
            "head_k_dim": block.head_k_dim,
            "head_v_dim": block.head_v_dim,
            "num_heads": block.num_k_heads,
        },
        {
            "input": x,
            "conv_output": chunk_replay["conv_output"],
            "q": chunk_replay["query"], "k": chunk_replay["key"], "v": chunk_replay["value"],
            "z": chunk_replay["z"],
            "g": chunk_replay["g"], "beta": chunk_replay["beta"],
            "core_attn_out_seq": seq_replay["core_attn_out"],
            "ssm_states": states,
            "rmsnorm_gated_output": seq_replay["normed"],
            "output_seq": seq_replay["output"],
            "output_chunked": module_output,
        },
    )
    print(f"Generated gdn prefill fixtures")



def generate_multichunk_fixture(block: Qwen3_5GatedDeltaNet) -> None:
    """Multi-chunk GDN prefill T=70: lock the sequential-vs-chunked band.

    Two FLA chunks (64 + 6) exercise the cross-chunk state handoff that a
    single-chunk prefill never touches. The chunked replay is asserted
    bit-identical to the module forward, and the sequential-vs-chunked max
    diff is asserted inside (0, 1e-3): the documented multi-chunk divergence
    (~1.5e-5 in the f32 core, a few bf16 ULPs at the block output). The band
    is a property of the chunked and sequential rules, so the Nim test
    asserts it from the fixture tensors directly.
    """
    torch.manual_seed(SEED_MULTICHUNK)
    x = torch.randn(1, MULTICHUNK_SEQ, HIDDEN, dtype=torch.bfloat16)

    module_output = block(x)  # reference forward, chunked rule, two chunks
    chunk_replay = gdn_forward_replay(block, x, use_recurrent=False)
    assert torch.equal(module_output, chunk_replay["output"]), (
        "chunked replay diverged from the module forward"
    )
    seq_replay = gdn_forward_replay(block, x, use_recurrent=True)

    out_diff = (seq_replay["output"].float() - module_output.float()).abs().max().item()
    assert 0.0 < out_diff < 1e-3, f"sequential-vs-chunked diff outside (0, 1e-3): {out_diff}"

    save_fixture(
        LAYER0_FIXTURE_DIR, "gdn", 2,
        {
            "model": MODEL_NAME,
            "layer": "model.language_model.layers.0.linear_attn",
            "case": "multichunk_prefill_seq70",
            "seq_len": MULTICHUNK_SEQ,
            "chunk_size": 64,
            "head_k_dim": block.head_k_dim,
            "head_v_dim": block.head_v_dim,
            "num_heads": block.num_k_heads,
            "note": "output_seq is the sequential replay. output_chunked is "
                    "the chunked forward. Max diff is inside (0, 1e-3)",
        },
        {
            "input": x,
            "output_seq": seq_replay["output"],
            "output_chunked": module_output,
            "core_attn_out_seq": seq_replay["core_attn_out"],
            "core_attn_out_chunked": chunk_replay["core_attn_out"],
            "ssm_state_seq": seq_replay["ssm_state"],
            "ssm_state_chunked": chunk_replay["ssm_state"],
        },
    )
    print(f"Generated multi-chunk (T={MULTICHUNK_SEQ}) fixtures "
          f"(sequential-vs-chunked max diff {out_diff:.2e})")


def generate_state_fixture(block: Qwen3_5GatedDeltaNet) -> None:
    """State trajectory: 5-token sequential one-shot + 2-step decode."""
    torch.manual_seed(SEED_STATE)
    prefill_x = torch.randn(1, PREFILL_STATE_TOKENS, HIDDEN, dtype=torch.bfloat16)
    decode_x_d = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16)
    decode_x_e = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16)
    one_shot_input = torch.cat([prefill_x, decode_x_d, decode_x_e], dim=1)

    # Sequential one-shot over the 5 tokens, the 0.00 reference.
    oneshot = gdn_forward_replay(block, one_shot_input, use_recurrent=True)
    core_out, _, states = recurrent_rule_with_trajectory(
        oneshot["query"], oneshot["key"], oneshot["value"],
        oneshot["g"], oneshot["beta"],
    )

    # Two-step decode through the reference module with a cache. The cache
    # starts from the sequential state over the 3-token prefill, so every
    # decode output must match the one-shot positions 3 and 4 bit for bit.
    mixed_prefill, _, _, _ = gdn_projections(block, prefill_x)
    conv_state_prefill = F.pad(mixed_prefill, (block.conv_kernel_size - PREFILL_STATE_TOKENS, 0))
    prefill_seq = gdn_forward_replay(block, prefill_x, use_recurrent=True)
    assert torch.equal(prefill_seq["ssm_state"], states[PREFILL_STATE_TOKENS].unsqueeze(0)), (
        "prefill state diverged from the one-shot state at the same step"
    )
    # The cache must own clones: the decode steps mutate the conv and
    # recurrent states in place, and the pre-decode values are saved too.
    cache = _GdnCache(0, conv_state_prefill.clone(), prefill_seq["ssm_state"].clone())

    decode_tensors = {}
    for name, tok in (("d", decode_x_d), ("e", decode_x_e)):
        # Snapshot the conv state before the forward: the module's decode
        # updates the cache state in place inside causal_conv1d_update.
        state_before = cache.layers[0].conv_states.clone()
        out = block(tok, cache_params=cache)
        mixed_tok, _, _, _ = gdn_projections(block, tok)
        conv_input = torch.cat([state_before, mixed_tok], dim=-1).to(block.conv1d.weight.dtype)
        conv_out = torch_causal_conv1d_update(
            mixed_tok, state_before, block.conv1d.weight.squeeze(1),
            block.conv1d.bias, block.activation,
        )
        decode_tensors[name] = {
            "output": out,
            "conv_input": conv_input,
            "conv_output": conv_out,
            "conv_state": cache.layers[0].conv_states.clone(),
            "ssm_state": cache.layers[0].recurrent_states.clone(),
        }

    step_d = PREFILL_STATE_TOKENS
    step_e = PREFILL_STATE_TOKENS + 1
    assert torch.equal(decode_tensors["d"]["output"], oneshot["output"][:, step_d:step_d + 1]), (
        "decode d output diverged from the one-shot reference"
    )
    assert torch.equal(decode_tensors["e"]["output"], oneshot["output"][:, step_e:step_e + 1]), (
        "decode e output diverged from the one-shot reference"
    )
    assert torch.equal(decode_tensors["d"]["conv_output"], oneshot["conv_output"][:, :, step_d:step_d + 1]), (
        "decode d conv output diverged from the one-shot reference"
    )
    assert torch.equal(decode_tensors["e"]["conv_output"], oneshot["conv_output"][:, :, step_e:step_e + 1]), (
        "decode e conv output diverged from the one-shot reference"
    )
    assert torch.equal(decode_tensors["d"]["ssm_state"], states[step_d + 1].unsqueeze(0)), (
        "decode d ssm state diverged from the one-shot reference"
    )
    assert torch.equal(decode_tensors["e"]["ssm_state"], states[step_e + 1].unsqueeze(0)), (
        "decode e ssm state diverged from the one-shot reference"
    )

    # The reference cache holds a 4-wide conv state (the decode conv sees
    # [state(4), x(1)] and the state never shrinks). The Nim layer keeps the
    # equivalent 3-wide tail: the dropped oldest column never enters a conv
    # output, so every decode conv output is identical (asserted above). The
    # tail slices are saved so the Nim test can compare its own state shape.
    conv_state_tail3 = {
        "conv_state_after_prefill_tail3": conv_state_prefill[..., 1:],
        "conv_state_after_d_tail3": decode_tensors["d"]["conv_state"][..., 1:],
        "conv_state_after_e_tail3": decode_tensors["e"]["conv_state"][..., 1:],
    }

    save_fixture(
        LAYER0_FIXTURE_DIR, "gdn", 1,
        {
            "model": MODEL_NAME,
            "layer": "model.language_model.layers.0.linear_attn",
            "case": "state_trajectory_3prefill_2decode",
            "prefill_tokens": PREFILL_STATE_TOKENS,
            "one_shot_tokens": PREFILL_STATE_TOKENS + 2,
            "head_k_dim": block.head_k_dim,
            "head_v_dim": block.head_v_dim,
            "num_heads": block.num_k_heads,
            "conv_kernel": block.conv_kernel_size,
            "note": "conv_state_after_prefill/d/e are the vendored 4-wide "
                    "cache states. The _tail3 variants are the 3-wide tails",
        },
        {
            "one_shot_input": one_shot_input,
            "one_shot_block_output": oneshot["output"],
            "one_shot_conv_output": oneshot["conv_output"],
            "one_shot_core_attn_out": core_out,
            "one_shot_ssm_states": states,
            "prefill_x": prefill_x,
            "decode_x_d": decode_x_d,
            "decode_x_e": decode_x_e,
            "conv_state_after_prefill": conv_state_prefill,
            "conv_state_after_d": decode_tensors["d"]["conv_state"],
            "conv_state_after_e": decode_tensors["e"]["conv_state"],
            "ssm_state_after_prefill": prefill_seq["ssm_state"][0],
            "ssm_state_after_d": decode_tensors["d"]["ssm_state"][0],
            "ssm_state_after_e": decode_tensors["e"]["ssm_state"][0],
            "decode_conv_input_d": decode_tensors["d"]["conv_input"],
            "decode_conv_input_e": decode_tensors["e"]["conv_input"],
            "decode_conv_output_d": decode_tensors["d"]["conv_output"],
            "decode_conv_output_e": decode_tensors["e"]["conv_output"],
            "decode_output_d": decode_tensors["d"]["output"],
            "decode_output_e": decode_tensors["e"]["output"],
            **conv_state_tail3,
        },
    )
    print(f"Generated gdn state fixtures")


def generate_layer0_fixture(layer0: Qwen3_5DecoderLayer) -> None:
    """Full decoder layer 0 forward, T=5: chunked real + sequential replay."""
    torch.manual_seed(SEED_LAYER0)
    x = torch.randn(1, PREFILL_SEQ, HIDDEN, dtype=torch.bfloat16)

    input_ln = layer0.input_layernorm(x)
    layer_output = layer0(x, (None, None))

    gdn_chunk = gdn_forward_replay(layer0.linear_attn, input_ln, use_recurrent=False)
    h_chunk = x + gdn_chunk["output"]
    layer_output_chunk = h_chunk + layer0.mlp(layer0.post_attention_layernorm(h_chunk))
    assert torch.equal(layer_output_chunk, layer_output), (
        "layer 0 chunked replay diverged from the module forward"
    )

    layer_output_seq = decoder_layer_forward_seq(layer0, x)
    diff = (layer_output_seq.float() - layer_output.float()).abs().max().item()
    assert diff < 1e-3, f"sequential vs chunked layer 0 diff too large: {diff}"

    gdn_seq = gdn_forward_replay(layer0.linear_attn, input_ln, use_recurrent=True)
    h_seq = x + gdn_seq["output"]
    post_ln_seq = layer0.post_attention_layernorm(h_seq)
    mlp_out_seq = layer0.mlp(post_ln_seq)

    save_fixture(
        LAYER0_FIXTURE_DIR, "layer", 0,
        {
            "model": MODEL_NAME,
            "layer": "model.language_model.layers.0",
            "case": "full_layer_prefill_seq5",
            "seq_len": PREFILL_SEQ,
            "layer_type": layer0.layer_type,
            "note": "layer_output is the vendored chunked forward (5e-3). "
                    "layer_output_seq is the sequential replay (0.00)",
        },
        {
            "layer_input": x,
            "input_layernorm_output": input_ln,
            "gdn_block_output_seq": gdn_seq["output"],
            "post_attention_layernorm_output_seq": post_ln_seq,
            "mlp_output_seq": mlp_out_seq,
            "layer_output_seq": layer_output_seq,
            "layer_output": layer_output,
        },
    )
    print(f"Generated layer 0 fixtures")


def generate_layer3_fixture(layer3: Qwen3_5DecoderLayer, config: Qwen3_5TextConfig) -> None:
    """Full decoder layer 3 (full attention) forward, T=5."""
    torch.manual_seed(SEED_LAYER3)
    x = torch.randn(1, PREFILL_SEQ, HIDDEN, dtype=torch.bfloat16)
    position_ids = torch.arange(PREFILL_SEQ).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    cos, sin = rotary(x, position_ids)

    input_ln = layer3.input_layernorm(x)
    layer_output = layer3(x, position_embeddings=(cos, sin), attention_mask=None)

    attn = layer3.self_attn
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    # The attention sees the input_layernorm output, not the raw input.
    query_states, gate = torch.chunk(
        attn.q_proj(input_ln).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)
    q_normed = attn.q_norm(query_states.view(hidden_shape))
    k_normed = attn.k_norm(attn.k_proj(input_ln).view(hidden_shape))

    save_fixture(
        LAYER3_FIXTURE_DIR, "layer", 3,
        {
            "model": MODEL_NAME,
            "layer": "model.language_model.layers.3",
            "case": "full_layer_prefill_seq5",
            "seq_len": PREFILL_SEQ,
            "layer_type": layer3.layer_type,
            "num_qo_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": attn.head_dim,
        },
        {
            "layer_input": x,
            "position_ids": position_ids,
            "cos": cos, "sin": sin,
            "input_layernorm_output": input_ln,
            "q_normed": q_normed, "k_normed": k_normed,
            "gate": gate,
            "layer_output": layer_output,
        },
    )
    print(f"Generated layer 3 fixtures")


def generate_chain_fixture(layers, config: Qwen3_5TextConfig) -> None:
    """Layers 0, 1, 2 run in sequence on a seeded T=4 input."""
    torch.manual_seed(SEED_CHAIN)
    x = torch.randn(1, CHAIN_SEQ, HIDDEN, dtype=torch.bfloat16)

    hidden = x
    hidden_seq = x
    for i, layer in enumerate(layers):
        layer_input = hidden.clone()
        layer_output = layer(layer_input, (None, None))
        layer_input_seq = hidden_seq.clone()
        layer_output_seq = decoder_layer_forward_seq(layer, layer_input_seq)

        diff = (layer_output_seq.float() - layer_output.float()).abs().max().item()
        assert diff < 1e-3, f"sequential vs chunked chain layer {i} diff too large: {diff}"

        filename = f"block-{i:02d}.safetensor"
        filepath = os.path.join(CHAIN_FIXTURE_DIR, filename)
        sorted_tensors = OrderedDict(
            (name, tensor.detach().cpu().contiguous())
            for name, tensor in sorted({
                "layer_input": layer_input,
                "layer_output": layer_output,
                "layer_input_seq": layer_input_seq,
                "layer_output_seq": layer_output_seq,
            }.items())
        )
        serialized = st.save(sorted_tensors, metadata=None)
        with open(filepath, "wb") as f:
            f.write(serialized)
        metadata = {
            "model": MODEL_NAME,
            "layer": f"model.language_model.layers.{i}",
            "case": f"chain_block_{i}_seq{CHAIN_SEQ}",
            "seq_len": CHAIN_SEQ,
            "layer_type": layer.layer_type,
            "note": "layer_output is the vendored chunked chain (5e-3). "
                    "layer_output_seq is the sequential chain (0.00)",
        }
        metadata_path = filepath + ".metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)
            f.write("\n")
        print(f"  Saved: {filepath}")

        hidden = layer_output
        hidden_seq = layer_output_seq
    print(f"Generated chain fixtures")


def main() -> None:
    print(f"Generating {MODEL_NAME} GDN / layer / chain fixtures")
    print("=" * 60)
    ensure_fixture_dirs()

    config = load_text_config()
    block = build_gdn_layer0(config)
    layer0 = build_decoder_layer(config, 0)
    layer3 = build_decoder_layer(config, 3)
    chain_layers = [build_decoder_layer(config, i) for i in (0, 1, 2)]

    generate_gdn_prefill_fixture(block)
    generate_state_fixture(block)
    generate_multichunk_fixture(block)
    generate_layer0_fixture(layer0)
    generate_layer3_fixture(layer3, config)
    generate_chain_fixture(chain_layers, config)

    print("=" * 60)
    print(f"Fixture generation complete: {LAYER0_FIXTURE_DIR}")
    print(f"                          : {LAYER3_FIXTURE_DIR}")
    print(f"                          : {CHAIN_FIXTURE_DIR}")


if __name__ == "__main__":
    main()
