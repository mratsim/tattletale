"""
Trace through Qwen3 reference implementation step by step.
Compare reference HF vs custom SDPA with enable_gqa implementation.
"""

import os
import sys

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from safetensors import safe_open
from safetensors import torch as st

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention as Qwen3AttentionRef,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    repeat_kv,
    Qwen3RMSNorm,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


MODEL_NAME = "Qwen3-0.6B"
LAYER_IDX = 8
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "tests", "fixtures", "layers", f"{MODEL_NAME}-layer-{LAYER_IDX}"
)
WEIGHTS_FILE = f"{FIXTURE_DIR}/Weights-{MODEL_NAME}-layer-{LAYER_IDX}.safetensor"
FIXED_SEED = 42


def set_seed(seed: int = FIXED_SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_layer_weights():
    with safe_open(WEIGHTS_FILE, framework="pt") as f:
        weight = {key: f.get_tensor(key).clone() for key in f.keys()}
    return weight


class Qwen3AttentionCustomGQA(nn.Module):
    """Custom Qwen3 attention using torch.scaled_dot_product_attention with enable_gqa=True.

    This mimics the Qwen3 attention but uses PyTorch's SDPA with GQA support directly.
    Key differences from reference:
    - Uses enable_gqa=True instead of manually expanding KV with repeat_kv
    - Same Q/K norm, RoPE, and output projection logic
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_head
        )
        self.num_qo_head = config.num_attention_head
        self.num_kv_head = config.num_key_value_head
        self.num_key_value_groups = self.num_qo_head // self.num_kv_head
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = (
            config.attention_dropout if config.attention_dropout is not None else 0.0
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_qo_head * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_head * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_head * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_qo_head * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values=None,
        cache_position=None,
    ):
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_len = input_shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        # 1. QKV projection and reshape
        query_state = self.q_proj(hidden_states).view(hidden_shape)
        key_state = self.k_proj(hidden_states).view(hidden_shape)
        value_state = self.v_proj(hidden_states).view(hidden_shape)

        # 2. Apply Q/K norm (per-head), then transpose to (batch, head, seq, head_dim)
        query_state = self.q_norm(query_state).transpose(1, 2)
        key_state = self.k_norm(key_state).transpose(1, 2)
        value_state = value_state.transpose(1, 2)

        # 3. Apply RoPE
        cos, sin = position_embeddings
        query_state, key_state = apply_rotary_pos_emb(query_state, key_state, cos, sin)

        # 4. Use SDPA with enable_gqa=True (handles KV expansion internally)
        # Shape: (batch, num_qo_head, seq_len, head_dim)
        attn_output = scaled_dot_product_attention(
            query_state,
            key_state,
            value_state,
            is_causal=True,
            scale=self.scaling,
            enable_gqa=True,  # Key: let SDPA handle GQA expansion
        )

        # 5. Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_qo_head * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, None


def main():
    print("=" * 70)
    print("COMPARING REFERENCE HF vs CUSTOM SDPA with enable_gqa")
    print("=" * 70)

    set_seed(FIXED_SEED)
    weight = load_layer_weights()

    # Load fixture
    fixture_path = os.path.join(FIXTURE_DIR, f"attn-{MODEL_NAME}-00.safetensor")
    with safe_open(fixture_path, framework="pt") as f:
        keys = list(f.keys())
        hidden_key = [k for k in keys if "hidden" in k][0]
        fixture_hidden = f.get_tensor(hidden_key)
        fixture_output = f.get_tensor("output")
        fixture_cos = f.get_tensor("cos")
        fixture_sin = f.get_tensor("sin")

    print(f"\nFIXTURE INPUT:")
    print(f"  hidden_states shape: {fixture_hidden.shape}")
    print(f"  cos shape: {fixture_cos.shape}")
    print(f"  sin shape: {fixture_sin.shape}")

    # Config
    config = Qwen3Config(
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_head=16,
        num_key_value_head=8,
        head_dim=128,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    config._attn_implementation = "sdpa"

    # Reference HF attention
    attn_ref = Qwen3AttentionRef(config, layer_idx=LAYER_IDX)
    attn_ref.q_proj.weight.data = weight["self_attn.q_proj.weight"]
    attn_ref.k_proj.weight.data = weight["self_attn.k_proj.weight"]
    attn_ref.v_proj.weight.data = weight["self_attn.v_proj.weight"]
    attn_ref.o_proj.weight.data = weight["self_attn.o_proj.weight"]
    attn_ref.q_norm.weight.data = weight["self_attn.q_norm.weight"]
    attn_ref.k_norm.weight.data = weight["self_attn.k_norm.weight"]

    # Custom GQA attention
    attn_custom = Qwen3AttentionCustomGQA(config, layer_idx=LAYER_IDX)
    attn_custom.q_proj.weight.data = weight["self_attn.q_proj.weight"]
    attn_custom.k_proj.weight.data = weight["self_attn.k_proj.weight"]
    attn_custom.v_proj.weight.data = weight["self_attn.v_proj.weight"]
    attn_custom.o_proj.weight.data = weight["self_attn.o_proj.weight"]
    attn_custom.q_norm.weight.data = weight["self_attn.q_norm.weight"]
    attn_custom.k_norm.weight.data = weight["self_attn.k_norm.weight"]

    # Same input
    hidden_states = fixture_hidden.clone()

    # Reference output
    print(f"\n--- Running Reference HF Qwen3Attention (SDPA) ---")
    output_ref, _ = attn_ref(
        hidden_states,
        position_embeddings=(fixture_cos, fixture_sin),
        attention_mask=None,
        past_key_values=None,
    )
    print(f"  Reference output[0,0,:5]: {output_ref[0, 0, :5]}")
    print(f"  Fixture output[0,0,:5]: {fixture_output[0, 0, :5]}")
    print(
        f"  Max diff (ref vs fixture): {(output_ref - fixture_output).abs().max().item()}"
    )

    # Custom output
    print(f"\n--- Running Custom Qwen3Attention (SDPA with enable_gqa=True) ---")
    output_custom, _ = attn_custom(
        hidden_states,
        position_embeddings=(fixture_cos, fixture_sin),
        attention_mask=None,
        past_key_values=None,
    )
    print(f"  Custom output[0,0,:5]: {output_custom[0, 0, :5]}")
    print(f"  Fixture output[0,0,:5]: {fixture_output[0, 0, :5]}")
    print(
        f"  Max diff (custom vs fixture): {(output_custom - fixture_output).abs().max().item()}"
    )

    # Compare ref vs custom
    print(f"\n--- Comparison ---")
    print(
        f"  Max diff (ref vs custom): {(output_ref - output_custom).abs().max().item()}"
    )


if __name__ == "__main__":
    main()
