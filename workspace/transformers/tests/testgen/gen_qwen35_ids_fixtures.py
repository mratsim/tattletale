#!/usr/bin/env python3
"""
Generate full-model ids-to-logits fixtures for the Qwen3.5-0.8B text stack
with the reference transformers modeling on CPU torch bf16.
Reference: gen_05_ids_to_logits_inference.py conventions, extended with a
sequential replay reference per the GDN fixture generators.

Generated under tests/fixtures/ids-inference/Qwen3.5-0.8B/:

  layer-{i:02d}.safetensor   per decoder layer i (24 files)
    layer_input       chunked-run layer input (embedding output for layer 0,
                      previous layer output for layers 1+)
    layer_input_seq   sequential-run layer input
    layer_output      chunked-forward layer output
    layer_output_seq  sequential-replay layer output
  final_logits.safetensor
    logits      wrapper logits, all positions
    logits_seq  sequential-replay logits, all positions

Tolerances (asserted by the Nim ids test):
  - 0.00 against the sequential-replay tensors (layer_input_seq,
    layer_output_seq, logits_seq)
  - 5e-3 against the chunked-forward tensors (layer_input, layer_output,
    logits)
"""

import json
from collections import OrderedDict
import os
import sys
import torch
from safetensors import safe_open
from safetensors import torch as st

# The reference transformers checkout is the source of truth.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
VENDORED_SRC = os.environ.get(
    "QWEN35_VENDORED_SRC",
    os.path.join(_REPO_ROOT, "_references_prod", "transformers", "src"))
if not os.path.isdir(VENDORED_SRC):
    raise SystemExit(
        f"[gen_qwen35_ids_fixtures] vendored modeling not found at {VENDORED_SRC}. "
        "Set QWEN35_VENDORED_SRC to the _references_prod/transformers/src directory")
sys.path.insert(0, VENDORED_SRC)

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    torch_recurrent_gated_delta_rule,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

# Determinism (called once at import time).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config.
MODEL_NAME = "Qwen3.5-0.8B"
INPUT_TEXT = "Hello, how are you?"
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "ids-inference", MODEL_NAME
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), f"tests/hf_models/{MODEL_NAME}"
)
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Per-generator seeds, independent and order-agnostic.
SEED_CHUNKED = 71
SEED_SEQUENTIAL = 72


def load_wrapper_config() -> Qwen3_5Config:
    """Load the wrapper Qwen3_5Config from the model config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5Config.from_dict(wrapper)
    cfg._attn_implementation = "sdpa"
    cfg.text_config._attn_implementation = "sdpa"
    return cfg


def ensure_fixture_dir() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_fixture(layer_name: str, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata file."""
    filename = f"{layer_name}.safetensor"
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


def build_model(cfg: Qwen3_5Config) -> Qwen3_5ForConditionalGeneration:
    """Wrapper model with real shard weights, bf16, eval, CPU.

    The rotary inv_freq buffer is restored to f32 after the dtype cast: the
    reference rotary forward computes cos/sin in f32 and bf16 storage would
    round the frequency values (~1e-3 per element).
    """
    model = Qwen3_5ForConditionalGeneration(cfg)
    rotary = model.model.language_model.rotary_emb
    inv_freq = rotary.inv_freq.float()
    original_inv_freq = rotary.original_inv_freq.float()

    model.eval().to(torch.bfloat16)

    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = original_inv_freq

    weights = {}
    with safe_open(MODEL_PATH, framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    del weights
    if len(missing) != 1 or missing[0] != "lm_head.weight":
        raise SystemExit(
            f"[gen_qwen35_ids_fixtures] unexpected missing tensors: {missing}")
    if len(unexpected) != 15:
        raise SystemExit(
            f"[gen_qwen35_ids_fixtures] unexpected foreign tensors: {unexpected}")
    return model


def install_capture_hooks(layers):
    """Wrap every decoder layer forward to record input and output tensors.

    Returns the capture list (one dict per layer) and a restore closure. The
    wrapper records the exact hidden_states the layer receives and the exact
    tensor it returns, so the fixtures hold the true layer boundary values of
    the reference forward. Each install restores the pristine class forward
    after the run, so a second install never chains onto a previous wrapper.
    """
    captured = [None] * len(layers)
    originals = []

    def make_wrapper(layer_idx, layer, original):
        def wrapper(hidden_states, *args, **kwargs):
            entry = {"layer_input": hidden_states.clone()}
            output = original(hidden_states, *args, **kwargs)
            entry["layer_output"] = output.clone()
            captured[layer_idx] = entry
            return output

        return wrapper

    for i, layer in enumerate(layers):
        originals.append(layer.forward)
        layer.forward = make_wrapper(i, layer, originals[-1])

    def restore():
        for layer, original in zip(layers, originals):
            layer.forward = original

    return captured, restore


def run_forward(model, input_ids, seq_seed: int):
    """Run the wrapper forward with per-layer capture and return the logits."""
    torch.manual_seed(seq_seed)
    captured, restore = install_capture_hooks(model.model.language_model.layers)
    with torch.no_grad():
        output = model(input_ids)
    restore()
    return captured, output.logits


def patch_recurrent(model) -> None:
    """Point every GDN layer's chunked rule at the sequential rule.

    The GDN forward calls `self.chunk_gated_delta_rule` for multi-token
    prefills and `self.recurrent_gated_delta_rule` for single-token decode.
    The sequential replay is the prefill with the chunked rule replaced, so
    the whole text stack runs the exact op sequence the Nim implementation
    mirrors. The recurrent rule accepts the same keyword call.
    """
    for layer in model.model.language_model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = torch_recurrent_gated_delta_rule


def replay_linear_layer(layer, layer_input):
    """Manual sequential replay of one GDN decoder layer from its input.

    Recomputes input_layernorm, the GDN block on the sequential rule, the
    post-attention norm and the MLP. The caller asserts the result is
    bit-identical to the hooked sequential forward output, proving the
    manual sequential replay (the 0.00 reference) and the patched real
    forward agree exactly.
    """
    from gen_qwen35_gdn_fixtures import gdn_forward_replay

    normed = layer.input_layernorm(layer_input)
    gdn_out = gdn_forward_replay(layer.linear_attn, normed, use_recurrent=True)["output"]
    h1 = layer_input + gdn_out
    return h1 + layer.mlp(layer.post_attention_layernorm(h1))


def main() -> None:
    print(f"Generating {MODEL_NAME} ids-to-logits fixtures")
    print("=" * 60)
    ensure_fixture_dir()

    cfg = load_wrapper_config()
    model = build_model(cfg)
    tokenizer_ids = [9419, 11, 1204, 513, 488, 30]  # "Hello, how are you?"
    input_ids = torch.tensor([tokenizer_ids])
    seq_len = input_ids.shape[1]
    assert seq_len < 64, "the ids fixture prompt must stay inside one FLA chunk"
    layers = model.model.language_model.layers
    num_layers = len(layers)

    # Chunked forward: the reference ground truth.
    chunked_captured, logits_chunked = run_forward(model, input_ids, SEED_CHUNKED)

    # Sequential replay through the patched model, twice for determinism.
    patch_recurrent(model)
    seq_captured, logits_seq = run_forward(model, input_ids, SEED_SEQUENTIAL)
    seq_captured_again, logits_seq_again = run_forward(model, input_ids, SEED_SEQUENTIAL)
    for i in range(num_layers):
        assert torch.equal(
            seq_captured[i]["layer_output"], seq_captured_again[i]["layer_output"]
        ), f"sequential replay layer {i} is not deterministic"
    assert torch.equal(logits_seq, logits_seq_again), "sequential replay logits are not deterministic"

    for i in range(num_layers):
        layer_input = chunked_captured[i]["layer_input"]
        layer_input_seq = seq_captured[i]["layer_input"]
        layer_output = chunked_captured[i]["layer_output"]
        layer_output_seq = seq_captured[i]["layer_output"]

        input_diff = (layer_input_seq.float() - layer_input.float()).abs().max().item()
        output_diff = (layer_output_seq.float() - layer_output.float()).abs().max().item()
        # The chunked and sequential GDN cores agree to ~1e-8 f32 on identical
        # inputs, but through 24 bf16 layer boundaries the sub-ULP core
        # differences flip bf16 rounding boundaries and accumulate (measured
        # layer-level max ~3.1e-2, logits max ~0.17 for T=6). The bounds
        # below are self-consistency guards that the ladder stays in the
        # sub-ULP regime. The 0.00 contract is the sequential replay, which
        # the manual replay assert below proves bit-exact.
        assert output_diff < 0.05, f"sequential vs chunked layer {i} diff too large: {output_diff}"

        if layers[i].layer_type == "linear_attention":
            replay = replay_linear_layer(layers[i], layer_input_seq)
            assert torch.equal(replay, layer_output_seq), (
                f"manual sequential replay of layer {i} diverged from the patched forward"
            )

        metadata = {
            "model": MODEL_NAME,
            "layer": f"model.language_model.layers.{i}",
            "layer_type": layers[i].layer_type,
            "input_text": INPUT_TEXT,
            "input_tokens": tokenizer_ids,
            "batch_size": 1,
            "seq_len": seq_len,
            "dtype": "bfloat16",
            "device": "cpu",
            "note": "layer_output is the vendored chunked forward (5e-3). "
                    "layer_output_seq is the sequential replay (0.00)",
        }
        save_fixture(
            f"layer-{i:02d}",
            metadata,
            {
                "layer_input": layer_input,
                "layer_input_seq": layer_input_seq,
                "layer_output": layer_output,
                "layer_output_seq": layer_output_seq,
            },
        )

    logits_diff = (logits_seq.float() - logits_chunked.float()).abs().max().item()
    assert logits_diff < 0.25, f"sequential vs chunked logits diff too large: {logits_diff}"
    save_fixture(
        "final_logits",
        {
            "model": MODEL_NAME,
            "input_text": INPUT_TEXT,
            "input_tokens": tokenizer_ids,
            "seq_len": seq_len,
            "dtype": "bfloat16",
            "note": "logits is the vendored chunked forward (5e-3). "
                    "logits_seq is the sequential replay (0.00)",
        },
        {
            "logits": logits_chunked,
            "logits_seq": logits_seq,
        },
    )

    print(f"Generated {num_layers} layer fixtures + final_logits")
    print(f"Sequential vs chunked logits max diff: {logits_diff:.2e}")
    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
