#!/usr/bin/env python3
"""
Generate layer intermediates for HF transformers.

Captures the block input and output at each of the 28 layers. Under
the fixture contract v2 the intermediate after_* tensors left the
payload: no suite consumed them and the block input and output carry
the whole external surface the ids suite checks.

Usage:
    cd tattletale
    .venv/bin/python debug/gen_layer_intermediates.py
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import torch as st
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent))

from fixture_stats import (  # noqa: E402
    decision_steps_probed,
    recording_env,
    write_json_zst,
    write_provenance,
    write_text_zst,
)

# ── Config ──────────────────────────────────────────────────────────────
DECISION_PROBE_SCHEMA = "ttt-tf-002-logit-decisions-probe-h2"
MODEL_NAME = "Qwen3-0.6B"
MODEL_PATH = str(Path(__file__).parent.parent / "hf_models" / MODEL_NAME)
OUTPUT_DIR = Path(__file__).parent.parent / "fixtures" / "bf16-03-full-forward-to-logits" / MODEL_NAME
INPUT_TEXT = "Hello, how are you?"
DTYPE = torch.bfloat16
DEVICE = "cpu"

# ── Determinism ─────────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_fixture(output_dir: Path, layer_idx: int, framework: str, metadata: dict, tensors: dict) -> Path:
    """Save intermediates to safetensors + separate metadata.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"layer-{layer_idx:02d}.safetensor"
    filepath = output_dir / filename

    # Sort tensors for deterministic serialization
    sorted_tensors = OrderedDict(
        (name, tensor.detach().cpu().to(DTYPE).contiguous())
        for name, tensor in sorted(tensors.items())
        if tensor is not None
    )

    # Save tensors (no metadata — deterministic)
    serialized = st.save(sorted_tensors, metadata=None)
    with open(filepath, "wb") as f:
        f.write(serialized)

    # Save metadata to separate JSON (deterministic, sorted keys)
    metadata_path = filepath.with_suffix(".metadata.json.zst")
    write_text_zst(metadata_path,
                   json.dumps(metadata, sort_keys=True, indent=2)
                   .encode("utf-8") + b"\n")

    return filepath


def capture_hf_intermediates(model, tokenizer, input_text: str) -> list:
    """
    Capture HF transformer layer intermediates using monkey-patching.
    """
    from transformers import Qwen3ForCausalLM

    captured = [None] * 28

    original_forward = type(model.model.layers[0]).forward

    def instrumented_forward(self, hidden_states, attention_mask=None, position_ids=None,
                             past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        i = list(model.model.layers).index(self)
        intermediates = OrderedDict()

        intermediates["layer_input"] = hidden_states.clone()

        # Sublayer 1: attention
        residual = hidden_states
        h = self.input_layernorm(hidden_states)

        h, _ = self.self_attn(
            hidden_states=h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + h

        # Sublayer 2: mlp
        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)

        h = self.mlp(h)

        hidden_states = residual + h
        intermediates["layer_output"] = hidden_states.clone()

        captured[i] = intermediates
        return hidden_states

    # Patch
    for layer in model.model.layers:
        layer.forward = instrumented_forward.__get__(layer, type(layer))

    # Run forward
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs, use_cache=False)

    # Restore
    for layer in model.model.layers:
        layer.forward = original_forward.__get__(layer, type(layer))

    return captured


def main():
    print(f"Model: {MODEL_PATH}")
    print(f"Input: {INPUT_TEXT}")
    print(f"Device: {DEVICE}")
    print()
    # ── 1. HF Transformers ──────────────────────────────────────────────
    print(f"{'=' * 80}")
    print(f"1. Capturing HF transformer intermediates...")
    print(f"{'=' * 80}")
    from transformers import Qwen3ForCausalLM, AutoTokenizer
    hf_model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH)
    hf_model.eval()
    hf_model = hf_model.to(DEVICE)
    # Preserve inv_freq buffers in float32 — model.to(bfloat16) would corrupt them.
    # bfloat16 loses too much precision for RoPE frequency values (up to 1.2e-3 per element).
    inv_freq = hf_model.model.rotary_emb.inv_freq.float()
    original_inv_freq = hf_model.model.rotary_emb.original_inv_freq.float()

    hf_model = hf_model.to(DTYPE)

    # Restore inv_freq in float32 after dtype conversion
    hf_model.model.rotary_emb.inv_freq = inv_freq
    hf_model.model.rotary_emb.original_inv_freq = original_inv_freq
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    hf_intermediates = capture_hf_intermediates(hf_model, tokenizer, INPUT_TEXT)
    # Save
    hf_dir = OUTPUT_DIR
    hf_logits = hf_model(**tokenizer(INPUT_TEXT, return_tensors="pt"), use_cache=False).logits
    for i, intermediates in enumerate(hf_intermediates):
        if intermediates is None:
            print(f"  Layer {i:02d}: SKIPPED")
            continue
        filepath = save_fixture(
            hf_dir, i, "hf",
            metadata={
                "framework": "hf",
                "model": MODEL_NAME,
                "layer": i,
                "input_text": INPUT_TEXT,
                "input_tokens": tokenizer(INPUT_TEXT)["input_ids"],
                "batch_size": 1,
                "seq_len": len(tokenizer(INPUT_TEXT)["input_ids"]),
                "dtype": "bfloat16",
                "device": "cpu",
            },
            tensors=intermediates,
        )
        print(f"  Layer {i:02d}: {filepath}")

    # Decision projection instead of raw logits: the payload
    # carries one step per position, the argmax id, the top-2 competing
    # pair with f32 logits, and the softmax tail probability beyond the pair.
    # The [1,6,151936] bf16 tensor shrinks to tens of KB, the consumers
    # read argmax and top-2 only.
    print()
    print("Saving final logits decision projection...")
    hf_dir.mkdir(parents=True, exist_ok=True)
    projection = {
        "schema": DECISION_PROBE_SCHEMA,
        "model": MODEL_NAME,
        "input_text": INPUT_TEXT,
        "input_ids": tokenizer(INPUT_TEXT)["input_ids"],
        "vocab_size": int(hf_logits.shape[-1]),
        "steps": decision_steps_probed(hf_logits.detach().cpu().to(torch.float32)),
    }
    write_json_zst(str(hf_dir / "final_logits.decisions.json.zst"), projection)
    provenance = recording_env(
        model=MODEL_NAME,
        generator="testgen/gen_bf16_qwen3_03_full_forward_to_logits.py",
        seed="none (fixed input ids, no sampling)",
        extra={"dtype": "bfloat16"},
    )
    write_provenance(str(hf_dir / "PROVENANCE.md"), list(provenance.items()))
    print(f"  Projection: {hf_dir / 'final_logits.decisions.json.zst'}")
if __name__ == "__main__":
    main()
