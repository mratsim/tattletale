#!/usr/bin/env python3
"""Generate the 17-key tied-vs-untied synthetic qwen3_5_moe checkpoints under
tests/fixtures/qwen36-tied/, loaded by the tied-head loader falsifier blocks
of the routed-block suite.

What is generated:

  qwen36-tied/tied/{model.safetensors, config.json, tokenizer.json}
  qwen36-tied/untied/{model.safetensors, config.json, tokenizer.json}

Both variants carry the identical 17-key bf16 checkpoint: a one-layer
full-attention text stack under toy geometry, no `lm_head.weight` entry in
either. Only `config.json` differs: `tie_word_embeddings` true (tied) or
false (untied). The tied variant proves the loader derives the head from the
embedding tensor; the untied variant proves the loud refusal when the tie is
declared false with no head entry. A real tokenizer.json is copied next to
each config for the loader.

Element values are deterministic: `1 + 0.5 * row + 0.25 * col`, packed as
big-endian-in-place bf16 round-half-to-even (the upper half word of the
float32) and spilled little-endian, matching the safetensors BF16 layout.

Run:
  cd <worktree root> && python3 workspace/transformers/tests/testgen/gen_qwen36_tied_synthetic.py

Requires the local material at tests/hf_models/Qwen3.6-35B-A3B (gitignored)
for the tokenizer.json copy step.
"""

import os
import shutil
import struct
import sys

HEAD_DIM = 8
NUM_QO = 2
NUM_KV = 1
OUTPUT_Q = NUM_QO * 2 * HEAD_DIM   # q_proj doubles per head: [q | gate]
OUTPUT_KV = NUM_KV * HEAD_DIM

# The 17 keys of the synthetic checkpoint, in emission order. Key names follow
# the model.language_model.* spelling the loader requests.
TENSORS = [
    ("model.language_model.embed_tokens.weight", [4, 4]),
    ("model.language_model.layers.0.input_layernorm.weight", [4]),
    ("model.language_model.layers.0.post_attention_layernorm.weight", [4]),
    ("model.language_model.layers.0.mlp.gate.weight", [2, 4]),
    ("model.language_model.layers.0.mlp.experts.gate_up_proj", [2, 4, 4]),
    ("model.language_model.layers.0.mlp.experts.down_proj", [2, 4, 2]),
    ("model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", [2, 4]),
    ("model.language_model.layers.0.mlp.shared_expert.up_proj.weight", [2, 4]),
    ("model.language_model.layers.0.mlp.shared_expert.down_proj.weight", [4, 2]),
    ("model.language_model.layers.0.mlp.shared_expert_gate.weight", [1, 4]),
    ("model.language_model.layers.0.self_attn.q_proj.weight", [OUTPUT_Q, 4]),
    ("model.language_model.layers.0.self_attn.k_proj.weight", [OUTPUT_KV, 4]),
    ("model.language_model.layers.0.self_attn.v_proj.weight", [OUTPUT_KV, 4]),
    ("model.language_model.layers.0.self_attn.o_proj.weight", [4, OUTPUT_Q]),
    ("model.language_model.layers.0.self_attn.q_norm.weight", [HEAD_DIM]),
    ("model.language_model.layers.0.self_attn.k_norm.weight", [HEAD_DIM]),
    ("model.language_model.norm.weight", [4]),
]

# The wrapper config of the toy checkpoint, byte-for-byte the load contract:
# the architectures wrapper stays the vendor spelling.
CONFIG_TEMPLATE = (
    '{"architectures": ["Qwen3_5MoeForConditionalGeneration"], '
    '"model_type": "qwen3_5_moe", '
    '"text_config": {"vocab_size": 4, "hidden_size": 4, '
    '"num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, '
    '"head_dim": 8, "num_experts": 2, "num_experts_per_tok": 2, '
    '"moe_intermediate_size": 2, "shared_expert_intermediate_size": 2, '
    '"linear_num_key_heads": 1, "linear_key_head_dim": 1, '
    '"linear_num_value_heads": 1, "linear_value_head_dim": 1, '
    '"linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-06, '
    '"hidden_act": "silu", '
    '"max_position_embeddings": 256, "eos_token_id": 1, '
    '"layer_types": ["full_attention"], "tie_word_embeddings": %s}}'
)


def bf16_bytes(value: float) -> bytes:
    lo = struct.pack('<f', float32_of(value))
    (u,) = struct.unpack('<I', lo)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return struct.pack('<H', rounded & 0xFFFF)


def float32_of(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', x))[0]


def tensor_values(shape):
    cols = shape[-1]
    n = 1
    for d in shape:
        n *= d
    return [1.0 + 0.5 * float(i // cols) + 0.25 * float(i % cols) for i in range(n)]


def write_checkpoint(out_dir: str, tie_word_embeddings: bool, tokenizer_src: str) -> None:
    payload = bytearray()
    entries = []
    for name, shape in TENSORS:
        data = b"".join(bf16_bytes(v) for v in tensor_values(shape))
        shape_text = "[" + ", ".join(str(d) for d in shape) + "]"
        entries.append(
            '"%s": {"dtype": "BF16", "shape": %s, "data_offsets": [%d, %d]}'
            % (name, shape_text, len(payload), len(payload) + len(data))
        )
        payload.extend(data)
    header = ("{" + ", ".join(entries) + "}").encode("ascii")
    with open(os.path.join(out_dir, "model.safetensors"), "wb") as f:
        f.write(struct.pack('<Q', len(header)))
        f.write(header)
        f.write(payload)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(CONFIG_TEMPLATE % ("true" if tie_word_embeddings else "false"))
    shutil.copyfile(tokenizer_src, os.path.join(out_dir, "tokenizer.json"))


def main() -> None:
    # Everything derives from this script's own location: testgen/ sits inside
    # the tests/ directory whose fixtures and hf_models it feeds. The campaign
    # root would be the wrong emission target inside a git worktree.
    _here = os.path.dirname(os.path.abspath(__file__))
    tests_root = os.path.dirname(_here)

    tokenizer_src = os.path.join(
        tests_root, "hf_models", "Qwen3.6-35B-A3B", "tokenizer.json")
    if not os.path.isfile(tokenizer_src):
        sys.exit("[gen_qwen36_tied_synthetic] tokenizer.json not found at %s" % tokenizer_src)

    out_root = os.path.join(tests_root, "fixtures", "qwen36-tied")
    for variant, tie in (("tied", True), ("untied", False)):
        out_dir = os.path.join(out_root, variant)
        os.makedirs(out_dir, exist_ok=True)
        write_checkpoint(out_dir, tie, tokenizer_src)
        print("generated %s (tie_word_embeddings: %s)" % (out_dir, tie))


if __name__ == "__main__":
    main()
