#!/usr/bin/env python3
"""
Generate EXL3 greedy (temp=0) decoding fixtures for end-to-end inference verification.

All core logic (decoder, forward, RoPE, RMS norm) lives in ``q_exl3_common.py``.
This file only handles model paths, tokenizer, generation orchestration, and data I/O.

Usage:
    cd tattletale
    .venv/bin/python testgen/gen_exl3_07_greedy_fixture.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

# ── Add testgen dir to path for importing q_exl3_common ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from q_exl3_common import (
    # utilities
    get_exl3_tensors,
    get_in_features_out_features,
    derive_K,
    load_config,
    # orig (CUDA ground truth)
    reconstruct_orig_exl3,
    linear_forward_orig_exl3,
    # orig / reimpl (EXL3-anchored)
    rms_norm_orig_exl3,
    precompute_freqs_cis_reimpl_exl3,
    apply_rotary_pos_emb_reimpl_exl3,
)

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(_SCRIPT_DIR)  # tests/
MODEL_DIR = os.path.join(BASE_DIR, "hf_models", "Qwen3-0.6B-EXL3-5bpw")
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
OUT_DIR = Path(BASE_DIR) / "fixtures" / "exl3-greedy-decoding"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LAYER_COUNT = 28
DTYPE = torch.float16
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 20

PROMPTS = [
    "Hello how are you?",
    "Do you know the story of this proverb '磨刀不误砍柴功' and why is it so similar to Abraham Lincoln quote?",
]

# ── Determinism ────────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)


# ── EXL3 model forward (without per-layer captures) ───────────────────


class EXL3Model:
    """Compiled EXL3 model for efficient autoregressive generation."""

    def __init__(self, config: dict, tensors: dict, weights: dict):
        self.config = config
        self.tensors = tensors
        self.weights = weights
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.rms_eps = config.get("rms_norm_eps", 1e-6)
        self.rope_theta = config.get("rope_theta", 1000000.0)
        self.max_seq_len = config.get("max_position_embeddings", 40960)
        self.vocab_size = config.get("vocab_size", 151936)

        # Precompute RoPE
        cos, sin = precompute_freqs_cis_reimpl_exl3(
            self.head_dim, self.max_seq_len, theta=self.rope_theta
        )
        self.register_buffer("cos", cos.to(DTYPE).to("cuda:0"))
        self.register_buffer("sin", sin.to(DTYPE).to("cuda:0"))

        # Move norms to CUDA
        self.norms: dict = {}
        for k, v in tensors["_norms"].items():
            self.norms[k] = v.to("cuda:0")

        # Embedding weight
        self.embed_weight = tensors["_embeddings"]["model.embed_tokens.weight"].to("cuda:0")

        # Final norm
        self.final_norm_weight = self.norms.get("model.norm.weight")

        # Check if lm_head is EXL3-quantized or shared
        lm_head_key = "lm_head"
        self.lm_head_exl3 = lm_head_key in weights
        self.lm_weight = weights.get(lm_head_key)
        self.lm_suh = tensors[lm_head_key]["suh"].to("cuda:0") if lm_head_key in tensors else None
        self.lm_svh = tensors[lm_head_key]["svh"].to("cuda:0") if lm_head_key in tensors else None

        # Pre-extract per-layer weights and scales (move to CUDA)
        self.layers: list = []
        for layer_idx in range(LAYER_COUNT):
            prefix = f"model.layers.{layer_idx}"
            layer = {
                "input_ln": self.norms[f"{prefix}.input_layernorm.weight"],
                "post_ln": self.norms[f"{prefix}.post_attention_layernorm.weight"],
                "q_norm": self.norms[f"{prefix}.self_attn.q_norm.weight"],
                "k_norm": self.norms[f"{prefix}.self_attn.k_norm.weight"],
                "q_w": weights[f"{prefix}.self_attn.q_proj"],
                "q_suh": tensors[f"{prefix}.self_attn.q_proj"]["suh"].to("cuda:0"),
                "q_svh": tensors[f"{prefix}.self_attn.q_proj"]["svh"].to("cuda:0"),
                "k_w": weights[f"{prefix}.self_attn.k_proj"],
                "k_suh": tensors[f"{prefix}.self_attn.k_proj"]["suh"].to("cuda:0"),
                "k_svh": tensors[f"{prefix}.self_attn.k_proj"]["svh"].to("cuda:0"),
                "v_w": weights[f"{prefix}.self_attn.v_proj"],
                "v_suh": tensors[f"{prefix}.self_attn.v_proj"]["suh"].to("cuda:0"),
                "v_svh": tensors[f"{prefix}.self_attn.v_proj"]["svh"].to("cuda:0"),
                "o_w": weights[f"{prefix}.self_attn.o_proj"],
                "o_suh": tensors[f"{prefix}.self_attn.o_proj"]["suh"].to("cuda:0"),
                "o_svh": tensors[f"{prefix}.self_attn.o_proj"]["svh"].to("cuda:0"),
                "gate_w": weights[f"{prefix}.mlp.gate_proj"],
                "gate_suh": tensors[f"{prefix}.mlp.gate_proj"]["suh"].to("cuda:0"),
                "gate_svh": tensors[f"{prefix}.mlp.gate_proj"]["svh"].to("cuda:0"),
                "up_w": weights[f"{prefix}.mlp.up_proj"],
                "up_suh": tensors[f"{prefix}.mlp.up_proj"]["suh"].to("cuda:0"),
                "up_svh": tensors[f"{prefix}.mlp.up_proj"]["svh"].to("cuda:0"),
                "down_w": weights[f"{prefix}.mlp.down_proj"],
                "down_suh": tensors[f"{prefix}.mlp.down_proj"]["suh"].to("cuda:0"),
                "down_svh": tensors[f"{prefix}.mlp.down_proj"]["svh"].to("cuda:0"),
            }
            self.layers.append(layer)

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning logits for all positions.

        Args:
            input_ids: [1, seq_len]

        Returns:
            logits: [1, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda:0").unsqueeze(0)

        # Embedding
        h = torch.nn.functional.embedding(input_ids, self.embed_weight)

        for layer_idx in range(LAYER_COUNT):
            ly = self.layers[layer_idx]

            # RMS Norm
            h_norm = rms_norm_orig_exl3(h, ly["input_ln"], self.rms_eps)

            # Q, K, V projections
            q = linear_forward_orig_exl3(h_norm, ly["q_w"], ly["q_suh"], ly["q_svh"])
            k = linear_forward_orig_exl3(h_norm, ly["k_w"], ly["k_suh"], ly["k_svh"])
            v = linear_forward_orig_exl3(h_norm, ly["v_w"], ly["v_suh"], ly["v_svh"])

            # Reshape to multi-head
            q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # QK norm
            q = rms_norm_orig_exl3(q, ly["q_norm"], self.rms_eps)
            k = rms_norm_orig_exl3(k, ly["k_norm"], self.rms_eps)

            # RoPE
            q, k = apply_rotary_pos_emb_reimpl_exl3(q, k, self.cos, self.sin, position_ids)

            # SDPA
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None, dropout_p=0.0, is_causal=True,
                scale=self.head_dim ** -0.5,
                enable_gqa=True,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch, seq_len, self.num_heads * self.head_dim)

            # O projection
            attn_output = linear_forward_orig_exl3(attn_output, ly["o_w"], ly["o_suh"], ly["o_svh"])

            # Residual
            h = h + attn_output

            # MLP
            residual = h
            h_norm = rms_norm_orig_exl3(h, ly["post_ln"], self.rms_eps)

            gate = linear_forward_orig_exl3(h_norm, ly["gate_w"], ly["gate_suh"], ly["gate_svh"])
            up = linear_forward_orig_exl3(h_norm, ly["up_w"], ly["up_suh"], ly["up_svh"])
            gate = torch.nn.functional.silu(gate)
            mlp_output = gate * up
            mlp_output = linear_forward_orig_exl3(mlp_output, ly["down_w"], ly["down_suh"], ly["down_svh"])

            h = residual + mlp_output

        # Final norm
        h = rms_norm_orig_exl3(h, self.final_norm_weight, self.rms_eps)

        # LM head
        if self.lm_head_exl3 and self.lm_weight is not None:
            logits = linear_forward_orig_exl3(h, self.lm_weight, self.lm_suh, self.lm_svh)
        else:
            logits = torch.nn.functional.linear(h, self.embed_weight)

        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 20,
                 eos_token_id: int = 151645) -> tuple:
        """Autoregressive greedy generation (temp=0).

        Args:
            input_ids: [1, seq_len] prompt token IDs.
            max_new_tokens: Maximum number of tokens to generate.
            eos_token_id: End-of-sequence token ID.

        Returns:
            (full_ids, prompt_ids, generated_ids, per_step_logits)
        """
        prompt_ids = input_ids[0].tolist()
        full_ids = input_ids.clone()
        device = input_ids.device

        per_step_logits: list = []

        for step in range(max_new_tokens):
            logits = self.forward(full_ids)
            next_token_logits = logits[0, -1, :]

            chosen_id = next_token_logits.argmax().item()

            top10_vals, top10_idxs = next_token_logits.topk(10)
            per_step_logits.append({
                "step": step,
                "chosen_token": chosen_id,
                "chosen_logit": float(next_token_logits[chosen_id].item()),
                "top10_tokens": top10_idxs.tolist(),
                "top10_logits": top10_vals.tolist(),
            })

            full_ids = torch.cat(
                [full_ids, torch.tensor([[chosen_id]], device=device)], dim=-1
            )

            if chosen_id == eos_token_id:
                break

        generated_ids = full_ids[0].tolist()[len(prompt_ids):]

        return full_ids[0].tolist(), prompt_ids, generated_ids, per_step_logits


# ── Weight reconstruction ──────────────────────────────────────────────


def reconstruct_all_weights(tensors: dict, config: dict) -> dict:
    """Reconstruct all EXL3 linear weights and cache them."""
    cache: dict = {}
    for key, entry in tensors.items():
        if key.startswith("_") or entry.get("trellis") is None:
            continue
        trellis = entry["trellis"].to("cuda:0")
        K = derive_K(trellis)
        mcg = entry.get("mcg") is not None
        mul1 = entry.get("mul1") is not None
        in_f, out_f = get_in_features_out_features(key, trellis, config)
        w = reconstruct_orig_exl3(trellis, K, mcg, mul1, (in_f, out_f))
        cache[key] = w.contiguous()  # [in_features, out_features] — non-transposed, for ext.hgemm (differs from F.linear which needs [out_features, in_features])
        print(f"  Reconstructed {key}: [{in_f}, {out_f}] -> weight [{out_f}, {in_f}]")
    return cache


# ── Tokenizer helpers ──────────────────────────────────────────────────


def load_tokenizer(model_dir: str):
    """Load tokenizer for encoding prompts. Falls back to HF if available."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(model_dir))
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  [info] transformers not available ({e}), using fallback tokenizer")
        return None


def encode_prompt(prompt: str, tokenizer) -> torch.Tensor:
    """Encode a prompt string to token IDs."""
    if tokenizer is not None:
        return tokenizer(prompt, return_tensors="pt").input_ids
    known = {
        "Hello how are you?": [10161, 1355, 527, 499, 30],
    }
    if prompt in known:
        return torch.tensor([known[prompt]])
    print(f"  [WARN] Cannot encode prompt without transformers, using BOS token")
    return torch.tensor([[151643]])


def decode_tokens(token_ids: list, tokenizer) -> str:
    """Decode token IDs to text."""
    if tokenizer is not None:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    return f"<{len(token_ids)} tokens>"


# ── Main ────────────────────────────────────────────────────────────────


def main():
    torch.set_num_threads(4)
    print(f"Model: {MODEL_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    config = load_config()
    print(f"Loading EXL3 tensors...")
    tensors = get_exl3_tensors(MODEL_PATH)

    print(f"\nReconstructing EXL3 weights...")
    weights = reconstruct_all_weights(tensors, config)

    print(f"\nBuilding EXL3 model...")
    model = EXL3Model(config, tensors, weights)

    tokenizer = load_tokenizer(MODEL_DIR)

    for prompt in PROMPTS:
        safe_name = prompt.replace(" ", "_").replace("'", "").replace("?", "")[:40]
        print(f"\n{'=' * 70}")
        print(f"Prompt: {prompt}")
        print(f"{'=' * 70}")

        input_ids = encode_prompt(prompt, tokenizer).to("cuda:0")
        prompt_ids_list = input_ids[0].tolist()

        print(f"  Prompt tokens ({len(prompt_ids_list)}): {prompt_ids_list}")

        full_ids, prompt_ids, generated_ids, step_logits = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=151645,
        )

        full_text = decode_tokens(full_ids, tokenizer)
        generated_text = decode_tokens(generated_ids, tokenizer)

        fixture = {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "full_ids": full_ids,
            "generated_ids": generated_ids,
            "full_text": full_text,
            "generated_text": generated_text,
            "num_prompt_tokens": len(prompt_ids),
            "num_generated_tokens": len(generated_ids),
            "eos_token_id": 151645,
            "bos_token_id": 151643,
            "steps": step_logits,
        }

        out_path = OUT_DIR / f"{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixture, f, indent=2, ensure_ascii=False)

        print(f"  Prompt tokens:  {len(prompt_ids)}")
        print(f"  Generated:      {len(generated_ids)} tokens")
        print(f"  Generated text: {generated_text!r}")
        print(f"  Fixture saved:  {out_path}")

    print(f"\nDone. Fixtures in {OUT_DIR}")


if __name__ == "__main__":
    main()
