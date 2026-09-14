#!/usr/bin/env python3
"""
Generate layer intermediates for HF transformers.

Captures the block input at each of the 28 layers.

  - the fixture contract v2 drops the intermediate after_* tensors, no suite consumed them
  - the block input carries the whole external surface the ids suite checks

invocation:

  - cd <tests dir>, uv run python testgen/gen_bf16_qwen3_03_full_forward_to_logits.py
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import torch as st
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent))

from fixture_stats import (  # noqa, the path insert precedes the import
    argmax_record_from_row,
    grid_of,
    write_argmax_decisions,
    write_text_zst,
)

NUM_POSITIONS = 6
    # Decision records written, one per input position.

# ── Config ──────────────────────────────────────────────────────────────
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

    # Save tensors (no metadata, deterministic)
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

        # sublayer 1 runs the attention
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

        # sublayer 2 runs the mlp
        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)

        h = self.mlp(h)

        hidden_states = residual + h

        captured[i] = intermediates
        return hidden_states

    # patch every layer's forward
    for layer in model.model.layers:
        layer.forward = instrumented_forward.__get__(layer, type(layer))

    # run the forward pass
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs, use_cache=False)

    # restore the original forwards
    for layer in model.model.layers:
        layer.forward = original_forward.__get__(layer, type(layer))

    return captured


def main():
    """Generates the bf16-03 full-forward-to-logits fixtures."""
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
    # Preserve inv_freq buffers in float32, model.to(bfloat16) would corrupt them.
    # bfloat16 loses too much precision for RoPE frequency values (up to 1.2e-3 per element).
    inv_freq = hf_model.model.rotary_emb.inv_freq.float()
    original_inv_freq = hf_model.model.rotary_emb.original_inv_freq.float()

    hf_model = hf_model.to(DTYPE)

    # Restore inv_freq in float32 after dtype conversion
    hf_model.model.rotary_emb.inv_freq = inv_freq
    hf_model.model.rotary_emb.original_inv_freq = original_inv_freq
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    hf_intermediates = capture_hf_intermediates(hf_model, tokenizer, INPUT_TEXT)
    # save the fixtures
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
                # the recorded payload note of the dieted fixture tree,
                # reproduced verbatim, a regen reproduces the recorded
                # values through the instruments
                "note": "layer_output left the payload under the fixture contract v2: "
                        "for layers 0..26 it equals the next layer's recorded "
                        "layer_input byte for byte, and the final block output is "
                        "the layer-27 descriptor sidecar (dmExact).",
            },
            tensors=intermediates,
        )
        print(f"  Layer {i:02d}: {filepath}")

    # the 005 decisions frame of the forward rows, the canonical
    # chains recording code writes it
    #
    # - one record per position over the top-32 logits support
    # - the margin, the tail probability, the drift allowance and the flip cap
    #   consts carry through into the record
    # - the [1,6,151936] bf16 tensor shrinks to tens of KB
    # - the suite reads the frame through assertArgMax
    print()
    print("Saving final logits decisions frame...")
    hf_dir.mkdir(parents=True, exist_ok=True)
    logits_rows = hf_logits.detach().cpu().to(torch.float32)
    if logits_rows.shape[1] < NUM_POSITIONS:
        raise SystemExit(
            f"{MODEL_NAME}: the forward produced {logits_rows.shape[1]} "
            f"positions, the script records {NUM_POSITIONS}")
    records = [argmax_record_from_row(logits_rows[0, pos])
               for pos in range(NUM_POSITIONS)]
    decisions_path = str(hf_dir / "final_logits.decisions.json.zst")
    write_argmax_decisions(decisions_path, "final_logits.decisions", records,
                           grid_of(hf_logits))
    print(f"  Decisions: {decisions_path}")
if __name__ == "__main__":
    main()
