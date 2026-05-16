"""
Generate combined Embedding + LMHead fixtures for Qwen3-0.6B.

This script:
1. Loads embed_tokens.weight directly from the main model file
2. Creates tied LMHead (uses same weight as embedding)
3. Generates test fixtures for both layers

Space-saving: Weights are loaded from tests/hf_models/Qwen3-0.6B/model.safetensors
instead of being saved to a separate file.
"""

import os
import torch
from safetensors import safe_open
from safetensors import torch as st

MODEL_NAME = "Qwen3-0.6B"
FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "fixtures", "layers", f"{MODEL_NAME}-embed-lmhead"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), f"hf_models/{MODEL_NAME}/model.safetensors"
)
FIXED_SEED = 42


def set_seed(seed: int = FIXED_SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_fixture_dir() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_fixture(case_num: int, tensors: dict, metadata: dict) -> str:
    """Save a fixture to safetensors format."""
    filename = f"embed-lmhead-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(FIXTURE_DIR, filename)

    safe_tensors = {}
    for name, tensor in tensors.items():
        if tensor is not None:
            safe_tensors[name] = tensor.detach().cpu().contiguous()

    serialized = st.save(safe_tensors, metadata=metadata)
    with open(filepath, "wb") as f:
        f.write(serialized)

    return filepath


def generate_embed_lmhead_fixtures() -> None:
    """Generate fixtures for Embedding and LMHead layers with tied weights."""
    print(f"Generating {MODEL_NAME} Embedding + LMHead fixtures")
    print("=" * 60)

    ensure_fixture_dir()
    set_seed(FIXED_SEED)

    # Load embed_tokens.weight directly from main model
    print("Loading embed_tokens.weight from main model...")
    with safe_open(MODEL_PATH, framework="pt") as f:
        embed_weight = f.get_tensor("model.embed_tokens.weight")

    vocab_size, hidden_size = embed_weight.shape
    print(f"  vocab_size: {vocab_size}, hidden_size: {hidden_size}")

    # LMHead uses tied weights (same as embedding)
    lm_head_weight = embed_weight

    # Case 00: Single token (batch=1, seq=1)
    print("Generating case 00: single token...")
    input_ids = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)
    embeddings = torch.nn.functional.embedding(input_ids, embed_weight)

    lmhead_input = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)
    logits = torch.nn.functional.linear(lmhead_input, lm_head_weight)

    save_fixture(0, {
        "embed_input_ids": input_ids,
        "embed_output": embeddings,
        "lmhead_input": lmhead_input,
        "lmhead_output": logits,
    }, metadata={
        "model": MODEL_NAME,
        "layer": "model.embed_tokens + lm_head",
        "case": "single_token",
        "tie_word_embeddings": "true",
    })

    # Case 01: Normal batch (batch=2, seq=8)
    print("Generating case 01: normal batch...")
    input_ids = torch.randint(0, vocab_size, (2, 8), dtype=torch.long)
    embeddings = torch.nn.functional.embedding(input_ids, embed_weight)

    lmhead_input = torch.randn(2, 8, hidden_size, dtype=torch.bfloat16)
    logits = torch.nn.functional.linear(lmhead_input, lm_head_weight)

    save_fixture(1, {
        "embed_input_ids": input_ids,
        "embed_output": embeddings,
        "lmhead_input": lmhead_input,
        "lmhead_output": logits,
    }, metadata={
        "model": MODEL_NAME,
        "layer": "model.embed_tokens + lm_head",
        "case": "normal_batch",
        "tie_word_embeddings": "true",
    })

    print("=" * 60)
    print(f"Fixture generation complete!")
    print(f"Fixtures saved to: {FIXTURE_DIR}")
    print(f"Note: Weights loaded from main model (no separate weights file)")


if __name__ == "__main__":
    generate_embed_lmhead_fixtures()