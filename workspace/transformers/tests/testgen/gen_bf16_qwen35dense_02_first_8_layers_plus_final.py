"""
Generates the Qwen3.5-0.8B fixtures for the Gated DeltaNet (GDN) layer 0,
the 8-block chain and the final pre-norm tail checkpoint, CPU torch bf16,
reference transformers modeling.

The conventions follow gen_bf16_qwen35dense_01_layer_internals.py.

Consumed by tests/q_bf16/t_bf16_qwen35dense_02_first_8_layers_plus_final.nim,
the gdn-* state boundary references and the 8-block chain checkpoints.

  - no Qwen3 analog exists, GatedDeltaNet is the SSM (linear-attention) layer Qwen3.5 interleaves with gated attention, Qwen3 has none
  - no Qwen3-era fixture records a recurrent-versus-chunked state-space contract or SSM state trajectories

Generated fixtures:

| file                                                                                          | contents                                              |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| tests/fixtures/bf16-01-layer-internals/Qwen3.5-0.8B-layer-0/layer0-Qwen3.5-0.8B-00.safetensor | the two GDN mixtures of layer 0, details below        |
| tests/fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/block-00..07.safetensor         | the 8 prefix chain checkpoints, details below         |
| tests/fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/tail.safetensor                 | the 24-layer chain tail pre-final-norm, details below |
"""

import json
from collections import OrderedDict
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import (  # noqa, the path insert precedes the import
    assert_path_equivalent,
    write_stats_file,
    write_text_zst,
)

import torch  # noqa, the path insert precedes the import
import torch.nn.functional as F
from safetensors import safe_open
from safetensors import torch as st


from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextRotaryEmbedding,
    causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

# Determinism (called once at import time).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# config constants:
MODEL_NAME = "Qwen3.5-0.8B"
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
LAYER0_FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals", f"{MODEL_NAME}-layer-0"
)
LAYER0_FIXTURE_STEM = f"layer0-{MODEL_NAME}-00"
LAYER0_FIXTURE_PATH = os.path.join(LAYER0_FIXTURE_DIR, LAYER0_FIXTURE_STEM + ".safetensor")
CHAIN_FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-02-first-8-layers-plus-final", MODEL_NAME
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
SEED_CHAIN = 65

PREFILL_SEQ = 5
CHAIN_SEQ = 4
CHAIN_PREFIX_BLOCKS = 8
PREFILL_STATE_TOKENS = 3
MULTICHUNK_SEQ = 70


def load_text_config() -> Qwen3_5TextConfig:
    """Load the nested text_config from the wrapper config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5TextConfig.from_dict(wrapper["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


# The hidden size reads from the parsed config (config is king), the value
# equals the 1024 the checkpoint carries.
HIDDEN = load_text_config().hidden_size


def ensure_fixture_dirs() -> None:
    """Creates the fixture directories."""
    for d in (LAYER0_FIXTURE_DIR, CHAIN_FIXTURE_DIR):
        os.makedirs(d, exist_ok=True)


def load_file_tensors(prefix: str) -> dict:
    """Load every tensor of the checkpoint under a key prefix, prefix stripped."""
    weights = {}
    with safe_open(MODEL_PATH, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                weights[key[len(prefix):]] = f.get_tensor(key).clone()
    return weights


def build_gdn_layer0(config: Qwen3_5TextConfig) -> Qwen3_5GatedDeltaNet:
    """Qwen3_5GatedDeltaNet with real layer-0 weights loaded."""
    block = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    w = load_file_tensors("model.language_model.layers.0.linear_attn.")
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
    w = load_file_tensors(f"model.language_model.layers.{layer_idx}.")
    layer.input_layernorm.weight.data = w["input_layernorm.weight"]
    layer.post_attention_layernorm.weight.data = w["post_attention_layernorm.weight"]
    layer.mlp.gate_proj.weight.data = w["mlp.gate_proj.weight"]
    layer.mlp.up_proj.weight.data = w["mlp.up_proj.weight"]
    layer.mlp.down_proj.weight.data = w["mlp.down_proj.weight"]
    if layer.block_type == "linear_attention":
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

    The contract covers the surface the module forward touches, conv_states,
    recurrent_states, has_previous_state and the two update procs, which
    copy in place like the reference LinearAttentionLayer cache.
    """

    def __init__(self, conv_states: torch.Tensor, recurrent_states: torch.Tensor):
        self.conv_states = conv_states
        self.recurrent_states = recurrent_states
        self.has_previous_state = True
        # The installed forward consults this on the single-token
        # decode path. The shim always carries a usable previous state
        # and updates in place, so there is nothing extra to record.
        self.record_past = False

    def update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
        self.conv_states.copy_(conv_states)
        return self.conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
        self.recurrent_states.copy_(recurrent_states)
        return self.recurrent_states


class _GdnCache:
    """Cache facade for one GDN layer (decode path only)."""

    def __init__(self, layer_idx: int, conv_states: torch.Tensor, recurrent_states: torch.Tensor):
        # the installed forward indexes conv_states[0] / recurrent_states[0]
        # (state-slot dim leading)
        #
        # - the shim stores the tensors under one extra leading dim
        # - the fixture saves recover the Nim contract shape by indexing
        #   the extra dim away
        self.layers = {layer_idx: _LayerCache(conv_states.unsqueeze(0), recurrent_states.unsqueeze(0))}

    def has_previous_state(self, layer_idx: int, state_idx: int = 0) -> bool:
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

    Returns the intermediate dict.

    - the chunked replay must match the module's own forward through
      the ulp-band instrument, the caller asserts it
    - the sequential replay is the 0.00 reference for the Nim implementation
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

    Returns the per-step trajectory. Replicates the reference loop op for op
    and asserts the final output and state match the function's own results,
    so the captured per-step trajectory is trustworthy.
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
    assert_path_equivalent(out, ref_out, "trajectory output vs the recurrent rule")
    assert_path_equivalent(s, ref_state, "trajectory state vs the recurrent rule")
    return out, s, torch.stack(states)[:, 0]


def decoder_layer_forward_seq(
    layer: Qwen3_5DecoderLayer, x: torch.Tensor,
    position_embeddings=None,
) -> torch.Tensor:
    """Qwen3_5DecoderLayer.forward with the GDN block on the sequential rule.

    Returns the layer output. The reference forward runs the chunked rule
    for prefill, this replay substitutes the sequential rule so the Nim
    implementation (sequential always) has a 0.00 reference at the layer level.

    Full-attention layers carry no chunked versus sequential split,
    the module forward is the sequential reference.
    """
    if layer.block_type != "linear_attention":
        return layer(
            x, position_embeddings=position_embeddings, attention_mask=None
        )
    residual = x
    h = layer.input_layernorm(x)
    gdn_out = gdn_forward_replay(layer.linear_attn, h, use_recurrent=True)["output"]
    h = residual + gdn_out
    residual = h
    h = layer.post_attention_layernorm(h)
    h = layer.mlp(h)
    return residual + h


def generate_gdn_prefill_fixture(block: Qwen3_5GatedDeltaNet) -> tuple:
    """GDN block prefill T=5, the sequential reference plus the chunked module output.

    Args:
    - block, the weighted layer-0 GatedDeltaNet module

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the captured prefill intermediates for the stats frame

    The chunked replay must match the module's own forward through the ulp-band instrument,
    the caller asserts it. The sequential replay is the 0.00 reference for the Nim implementation.
    """
    torch.manual_seed(SEED_GDN_PREFILL)
    x = torch.randn(1, PREFILL_SEQ, HIDDEN, dtype=torch.bfloat16)

    module_output = block(x)  # reference forward, chunked rule
    chunk_replay = gdn_forward_replay(block, x, use_recurrent=False)
    assert_path_equivalent(module_output, chunk_replay["output"],
                           "chunked replay vs the module forward")

    core_out, _, states = recurrent_rule_with_trajectory(
        chunk_replay["query"], chunk_replay["key"], chunk_replay["value"],
        chunk_replay["g"], chunk_replay["beta"],
    )
    seq_replay = gdn_forward_replay(block, x, use_recurrent=True)
    assert_path_equivalent(seq_replay["core_attn_out"], core_out,
                           "sequential replay core vs the trajectory")

    meta = {
        "layer": "model.language_model.layers.0.linear_attn",
        "case": "prefill_seq5",
        "seq_len": PREFILL_SEQ,
        "seed": SEED_GDN_PREFILL,
        "head_k_dim": block.head_k_dim,
        "head_v_dim": block.head_v_dim,
        "num_heads": block.num_k_heads,
    }
    payload = {"gdn_prefill.input": x}
    captured = {
        "gdn_prefill.q": chunk_replay["query"],
        "gdn_prefill.k": chunk_replay["key"],
        "gdn_prefill.v": chunk_replay["value"],
        "gdn_prefill.z": chunk_replay["z"],
        "gdn_prefill.g": chunk_replay["g"],
        "gdn_prefill.beta": chunk_replay["beta"],
        "gdn_prefill.conv_output": chunk_replay["conv_output"],
        "gdn_prefill.core_attn_out_seq": core_out,
        "gdn_prefill.rmsnorm_gated_output": chunk_replay["normed"],
        "gdn_prefill.output_seq": seq_replay["output"],
        "gdn_prefill.output_chunked": module_output,
    }
    print("Generated gdn prefill mixture")
    return meta, payload, captured


def generate_state_fixture(block: Qwen3_5GatedDeltaNet) -> tuple:
    """State trajectory mixture, a 5-token sequential one-shot plus a 2-step decode.

    Args:
    - block, the weighted layer-0 GatedDeltaNet module

    Returns:
    - meta, the mixture metadata
    - payload, the suite-read driving tensors under their file names
    - captured, the per-step trajectory tensors for the stats frame

    The cache starts from the sequential state over the 3-token prefill, every decode output
    matching one-shot positions 3 and 4 through the ulp-band instrument.
    """
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

    # two-step decode through the reference module with a cache:
    # - the cache starts from the sequential state over the 3-token prefill
    # - every decode output matches one-shot positions 3 and 4 within two
    #   ulps of the dtype, the path-equivalence instrument
    mixed_prefill, _, _, _ = gdn_projections(block, prefill_x)
    conv_state_prefill = F.pad(mixed_prefill, (block.conv_kernel_size - PREFILL_STATE_TOKENS, 0))
    prefill_seq = gdn_forward_replay(block, prefill_x, use_recurrent=True)
    assert torch.equal(prefill_seq["ssm_state"],
                       states[PREFILL_STATE_TOKENS].unsqueeze(0)), (
        "prefill state diverged from the one-shot state at the same step"
    )
    # the cache owns clones, decode steps mutate conv and recurrent states
    # in place, pre-decode values are saved too
    cache = _GdnCache(0, conv_state_prefill.clone(), prefill_seq["ssm_state"].clone())

    decode_tensors = {}
    for name, tok in (("d", decode_x_d), ("e", decode_x_e)):
        # snapshot the conv state before the forward, the module's decode
        # updates the cache state in place inside causal_conv1d_update
        state_before = cache.layers[0].conv_states[0].clone()
        out = block(tok, cache_params=cache)
        mixed_tok, _, _, _ = gdn_projections(block, tok)
        conv_input = torch.cat([state_before, mixed_tok], dim=-1).to(block.conv1d.weight.dtype)
        conv_out = causal_conv1d_update(
            mixed_tok, state_before, block.conv1d.weight.squeeze(1),
            block.conv1d.bias, block.activation,
        )
        decode_tensors[name] = {
            "output": out,
            "conv_input": conv_input,
            "conv_output": conv_out,
            "conv_state": cache.layers[0].conv_states[0].clone(),
            "ssm_state": cache.layers[0].recurrent_states[0].clone(),
        }

    step_d = PREFILL_STATE_TOKENS
    step_e = PREFILL_STATE_TOKENS + 1
    assert_path_equivalent(decode_tensors["d"]["output"],
                           oneshot["output"][:, step_d:step_d + 1],
        "decode d output diverged from the one-shot reference"
    )
    assert_path_equivalent(decode_tensors["e"]["output"],
                           oneshot["output"][:, step_e:step_e + 1],
        "decode e output diverged from the one-shot reference"
    )
    assert_path_equivalent(decode_tensors["d"]["conv_output"],
                           oneshot["conv_output"][:, :, step_d:step_d + 1],
        "decode d conv output diverged from the one-shot reference"
    )
    assert_path_equivalent(decode_tensors["e"]["conv_output"],
                           oneshot["conv_output"][:, :, step_e:step_e + 1],
        "decode e conv output diverged from the one-shot reference"
    )
    assert_path_equivalent(decode_tensors["d"]["ssm_state"],
                           states[step_d + 1].unsqueeze(0),
        "decode d ssm state diverged from the one-shot reference"
    )
    assert_path_equivalent(decode_tensors["e"]["ssm_state"],
                           states[step_e + 1].unsqueeze(0),
        "decode e ssm state diverged from the one-shot reference"
    )

    meta = {
        "layer": "model.language_model.layers.0.linear_attn",
        "case": "state_trajectory_3prefill_2decode",
        "seed": SEED_STATE,
        "prefill_tokens": PREFILL_STATE_TOKENS,
        "one_shot_tokens": PREFILL_STATE_TOKENS + 2,
        "head_k_dim": block.head_k_dim,
        "head_v_dim": block.head_v_dim,
        "num_heads": block.num_k_heads,
        "conv_kernel": block.conv_kernel_size,
    }
    payload = {
        "gdn_state.prefill_x": prefill_x,
        "gdn_state.decode_x_d": decode_x_d,
        "gdn_state.decode_x_e": decode_x_e,
        "gdn_state.one_shot_block_output": oneshot["output"],
    }
    captured = {
        "gdn_state.one_shot_input": one_shot_input,
        "gdn_state.one_shot_block_output_steps0to2": oneshot["output"][:, 0:PREFILL_STATE_TOKENS],
        "gdn_state.one_shot_block_output_step3": oneshot["output"][:, step_d:step_d + 1],
        "gdn_state.one_shot_block_output_step4": oneshot["output"][:, step_e:step_e + 1],
        "gdn_state.decode_output_d": decode_tensors["d"]["output"],
        "gdn_state.decode_conv_output_d": decode_tensors["d"]["conv_output"],
        "gdn_state.decode_output_e": decode_tensors["e"]["output"],
        "gdn_state.decode_conv_output_e": decode_tensors["e"]["conv_output"],
    }
    print("Generated gdn state mixture")
    return meta, payload, captured


def save_layer0_fixture(metadata: dict, mixtures: list) -> None:
    """Writes the layer-0 fixture file set, one safetensors payload carrying
    every mixture as a named tensor group, one metadata sidecar and one
    stats sidecar with keys namespaced by mixture.

    Args:
    - metadata, the merged metadata frame
    - mixtures, one (payload, captured) pair per mixture, the payload
      tensors carrying their file names, the captured tensors already
      carrying their namespaced stats keys
    """
    file_tensors = OrderedDict()
    stats_entries = []
    for payload, captured in mixtures:
        for name, tensor in payload.items():
            file_tensors[name] = tensor.detach().cpu().contiguous()
            stats_entries.append((name, file_tensors[name]))
        for name, tensor in captured.items():
            stats_entries.append((name, tensor.detach().cpu().contiguous()))

    serialized = st.save(file_tensors, metadata=None)
    with open(LAYER0_FIXTURE_PATH, "wb") as f:
        f.write(serialized)

    write_text_zst(LAYER0_FIXTURE_PATH + ".metadata.json.zst",
                   json.dumps(metadata, sort_keys=True, indent=2)
                   .encode("utf-8") + b"\n")
    write_stats_file(LAYER0_FIXTURE_PATH + ".stats.json.zst",
                     LAYER0_FIXTURE_STEM + ".safetensor", stats_entries)


def generate_chain_fixture(layers, config: Qwen3_5TextConfig) -> None:
    """Records the 8+1 chain checkpoints on a seeded T=4 input.

    The prefix checkpoints cover two full periods of the period-4 pattern, the tail validates the depth extrapolation.
    """
    torch.manual_seed(SEED_CHAIN)
    x = torch.randn(1, CHAIN_SEQ, HIDDEN, dtype=torch.bfloat16)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_ids = torch.arange(CHAIN_SEQ).unsqueeze(0)

    hidden = x       # the chunked chain
    hidden_seq = x   # the sequential chain
    fixtures = []
    for i, layer in enumerate(layers):
        layer_input = hidden.clone()
        if layer.block_type == "linear_attention":
            layer_output = layer(layer_input, (None, None))
            pe = None
        else:
            pe = rotary(layer_input, position_ids)
            layer_output = layer(
                layer_input, position_embeddings=pe, attention_mask=None
            )
        layer_input_seq = hidden_seq.clone()
        layer_output_seq = decoder_layer_forward_seq(layer, layer_input_seq, pe)

        diff = (layer_output_seq.float() - layer_output.float()).abs().max().item()
        # coarse gross-breakage guard only
        #
        # - the sequential versus chunked divergence accumulates through the chain,
        #   so the guard scales with the depth
        # - the calibrated band lives in the suites
        assert diff < 0.02 * (i + 1), (
            f"sequential vs chunked chain layer {i} diff too large: {diff}")

        if i < CHAIN_PREFIX_BLOCKS:
            fixtures.append((f"block-{i:02d}.safetensor", {
                "layer_input": layer_input,
                "layer_output": layer_output,
                "layer_input_seq": layer_input_seq,
                "layer_output_seq": layer_output_seq,
            }, {
                "model": MODEL_NAME,
                "layer": f"model.language_model.layers.{i}",
                "case": f"chain_block_{i}_seq{CHAIN_SEQ}",
                "seq_len": CHAIN_SEQ,
                "layer_type": layer.block_type,
                "note": "layer_output is the chunked chain (5e-3). "
                        "layer_output_seq is the sequential chain (0.00)",
            }))

        hidden = layer_output
        hidden_seq = layer_output_seq

    fixtures.append(("tail.safetensor", {
        "pre_final_norm": hidden_seq,
        "pre_final_norm_chunked": hidden,
    }, {
        "model": MODEL_NAME,
        "layer": "model.norm (input side)",
        "case": f"chain_tail_seq{CHAIN_SEQ}",
        "seq_len": CHAIN_SEQ,
        "num_hidden_layers": len(layers),
        "note": "pre_final_norm is the sequential chain output taken "
                "pre-final-norm. pre_final_norm_chunked is the chunked chain",
    }))

    for filename, tensors, metadata in fixtures:
        filepath = os.path.join(CHAIN_FIXTURE_DIR, filename)
        sorted_tensors = OrderedDict(
            (name, tensor.detach().cpu().contiguous())
            for name, tensor in sorted(tensors.items())
        )
        serialized = st.save(sorted_tensors, metadata=None)
        with open(filepath, "wb") as f:
            f.write(serialized)
        metadata_path = filepath + ".metadata.json.zst"
        write_text_zst(metadata_path,
                       json.dumps(metadata, sort_keys=True, indent=2)
                       .encode("utf-8") + b"\n")
        print(f"  Saved: {filepath}")
    print(f"Generated chain fixtures (8 prefix checkpoints + tail over "
          f"{len(layers)} layers)")


def main() -> None:
    """Generates the layer-0 fixture file and the chain fixtures."""
    print(f"Generating {MODEL_NAME} layer-0 / chain fixtures")
    print("=" * 60)
    ensure_fixture_dirs()

    config = load_text_config()
    block = build_gdn_layer0(config)
    chain_layers = [
        build_decoder_layer(config, i) for i in range(config.num_hidden_layers)
    ]

    prefill_meta, prefill_payload, prefill_captured = \
        generate_gdn_prefill_fixture(block)
    state_meta, state_payload, state_captured = generate_state_fixture(block)
    metadata = {
        "model": MODEL_NAME,
        "file": LAYER0_FIXTURE_STEM + ".safetensor",
        "dtype": "bfloat16",
        "hidden_size": HIDDEN,
        "mixtures": {
            "gdn_prefill": prefill_meta,
            "gdn_state": state_meta,
        },
    }
    save_layer0_fixture(metadata, [
        (prefill_payload, prefill_captured),
        (state_payload, state_captured),
    ])
    print(f"Saved: {LAYER0_FIXTURE_PATH}")

    generate_chain_fixture(chain_layers, config)

    print("=" * 60)
    print(f"Fixture generation complete: {LAYER0_FIXTURE_DIR}")
    print(f"                          : {CHAIN_FIXTURE_DIR}")


if __name__ == "__main__":
    main()
