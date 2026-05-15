# Fixture Generation for Transformer Layer Testing

## Overview

This document explains the test fixtures generated for the `workspace/transformers` project using real Qwen3-0.6B model weights.

## Approach

**Space-Saving Approach (Updated):**

1. **Load layer weights directly** from `tests/hf_models/Qwen3-0.6B/model.safetensors`
2. **Generate fixtures** using those real weights (no separate weights file)
3. **Only save inputs and outputs** in fixture files

**Benefits:**
- ~90% storage reduction per layer fixture (no ~30 MiB weights file)
- Single source of truth for weights
- Faster fixture generation (no intermediate file I/O)

**Previous approach:** Weights were saved to `Weights-Qwen3-0.6B-layer-8.safetensors` (~30 MiB). This is no longer done.

## Layers Tested

- **norm**: RMSNorm with input_layernorm and post_attention_layernorm
- **mlp**: Gated MLP with SiLU activation
- **attn**: Multi-head attention with GQA and RoPE
- **embedding**: Token embedding lookup (NEW)
- **lmhead**: Language model head with tied embeddings (NEW)
- **transformer_block**: Full transformer block with long residual stream (NEW)

## Test Cases

| Case | Description |
|------|-------------|
| 00 | Normal forward (batch=2, seq=8) |
| 01 | Single token (batch=1, seq=1) |
| 02 | Short sequence (norm: post_attention, mlp: seq=4) |
| 03 | Zeros input |

## Fixture Files

```
fixtures/layers/
├── Qwen3-0.6B-layer-8/           # Existing layer fixtures
│   ├── norm-Qwen3-0.6B-00.safetensor
│   ├── norm-Qwen3-0.6B-01.safetensor
│   ├── norm-Qwen3-0.6B-02.safetensor
│   ├── norm-Qwen3-0.6B-03.safetensor
│   ├── mlp-Qwen3-0.6B-00.safetensor
│   ├── mlp-Qwen3-0.6B-01.safetensor
│   ├── mlp-Qwen3-0.6B-02.safetensor
│   ├── mlp-Qwen3-0.6B-03.safetensor
│   ├── attn-Qwen3-0.6B-00.safetensor
│   └── attn-Qwen3-0.6B-01.safetensor
├── Qwen3-0.6B-embed-lmhead/      # NEW: Embedding + LMHead fixtures
│   ├── embed-lmhead-Qwen3-0.6B-00.safetensor  (single token)
│   └── embed-lmhead-Qwen3-0.6B-01.safetensor  (normal batch)
└── Qwen3-0.6B-block-8/           # NEW: TransformerBlock fixtures
    ├── transformer-block-Qwen3-0.6B-00.safetensor  (single, no residual)
    ├── transformer-block-Qwen3-0.6B-01.safetensor  (seq, no residual)
    ├── transformer-block-Qwen3-0.6B-02.safetensor  (single, with residual)
    └── transformer-block-Qwen3-0.6B-03.safetensor  (seq, with residual)
```

## Fixture Structure

Each safetensor contains:
- `__metadata__`: JSON with model, layer path, case description
- Input tensors
- Output tensors
- Intermediate tensors (for attn: query/key/value states, cos/sin)

## Precision Discipline

### Float32 Buffers That Must Be Preserved

When calling `model.to(torch.bfloat16)`, some internal buffers lose too much
precision. They MUST be saved in float32 before the conversion and restored after:

```python
# Save before model.to(bfloat16)
inv_freq = model.model.rotary_emb.inv_freq.float()
original_inv_freq = model.model.rotary_emb.original_inv_freq.float()

model = model.to(torch.bfloat16)

# Restore after dtype conversion
model.model.rotary_emb.inv_freq = inv_freq
model.model.rotary_emb.original_inv_freq = original_inv_freq
```

Buffer | Why it matters
-------|----------------
`rotary_emb.inv_freq` | RoPE frequency exponents. BF16 loses up to 1.2e-3 per element, causing ~4e-3 cos/sin error that compounds across layers
`rotary_emb.original_inv_freq` | Same as above (used by dynamic RoPE scaling)

## Generating Fixtures

```bash
# Existing layer fixtures (norm, mlp, attn)
cd workspace/transformers
uv run python tests/testgen/gen_layer_fixtures_Qwen3-0.6B.py

# Embedding + LMHead fixtures
uv run python tests/testgen/gen_embed_lmhead_fixtures_Qwen3-0.6B.py

# TransformerBlock fixtures
uv run python tests/testgen/gen_transformer_block_fixtures_Qwen3-0.6B.py
```

## Model Configuration

Qwen3-0.6B:
- hidden_size: 1024
- intermediate_size: 3072
- num_attention_heads: 16
- num_key_value_heads: 8 (GQA ratio: 2)
- head_dim: 128
- rms_norm_eps: 1e-6

## New Fixture Types

### Embedding + LMHead Fixtures

Combined fixtures for testing tied embedding scenarios.

- **Case 00**: Single token (batch=1, seq=1)
- **Case 01**: Normal batch (batch=2, seq=8)

**Metadata:** `tie_word_embeddings: true`

### TransformerBlock Fixtures

Full transformer block forward pass with long residual stream pattern.

- **Case 00**: Single token, no residual (first block, decode)
- **Case 01**: Short sequence, no residual (first block, prefill)
- **Case 02**: Single token, with residual (middle block, decode)
- **Case 03**: Short sequence, with residual (middle block, prefill)

**Inputs:** `input_hidden_states`, `residual` (optional), `position_ids`, `cos`, `sin`

**Outputs:** `output`, `output_residual`

## Long Residual Stream Pattern

The transformer block uses the **long residual stream** pattern (vLLM, SGLang, nano-vllm):

```
residual = x  (or prev_residual)
(h, residual) = attn_norm.forward_with_residual(x, residual)
attn_out = attn(h)
(h2, residual) = mlp_norm.forward_with_residual(h + attn_out, residual)
mlp_out = mlp(h2)
return (h2 + mlp_out, residual)
```

This enables pipeline parallelism and fused norm+residual kernels.