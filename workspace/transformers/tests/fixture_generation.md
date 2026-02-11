# Fixture Generation for Transformer Layer Testing

## Overview

This document explains the test fixtures generated for the `workspace/transformers` project using real Qwen3-0.6B model weights.

## Approach

1. **Extract layer weights** from `tests/hf_models/Qwen3-0.6B/model.safetensors`
2. **Save them** to `tests/fixtures/layers/Qwen3-0.6B-layer-8/Weights-Qwen3-0.6B-layer-8.safetensors`
3. **Generate fixtures** using those real weights

## Weights File

**File**: `Weights-Qwen3-0.6B-layer-8.safetensors`

Contains weights for layer 8:
- `input_layernorm.weight` - [1024]
- `post_attention_layernorm.weight` - [1024]
- `mlp.gate_proj.weight` - [3072, 1024]
- `mlp.up_proj.weight` - [3072, 1024]
- `mlp.down_proj.weight` - [1024, 3072]
- `self_attn.q_proj.weight` - [2048, 1024]
- `self_attn.k_proj.weight` - [1024, 1024]
- `self_attn.v_proj.weight` - [1024, 1024]
- `self_attn.o_proj.weight` - [1024, 2048]
- `self_attn.q_norm.weight` - [128]
- `self_attn.k_norm.weight` - [128]

## Layers Tested

- **norm**: RMSNorm with input_layernorm and post_attention_layernorm
- **mlp**: Gated MLP with SiLU activation
- **attn**: Multi-head attention with GQA and RoPE

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
├── norm-Qwen3-0.6B-00.safetensor
├── norm-Qwen3-0.6B-01.safetensor
├── norm-Qwen3-0.6B-02.safetensor
├── norm-Qwen3-0.6B-03.safetensor
├── mlp-Qwen3-0.6B-00.safetensor
├── mlp-Qwen3-0.6B-01.safetensor
├── mlp-Qwen3-0.6B-02.safetensor
├── mlp-Qwen3-0.6B-03.safetensor
├── attn-Qwen3-0.6B-00.safetensor
└── attn-Qwen3-0.6B-01.safetensor
```

## Fixture Structure

Each safetensor contains:
- `__metadata__`: JSON with model, layer path, case description
- Input tensors
- Output tensors
- Intermediate tensors (for attn: query/key/value states, cos/sin)

## Reproducibility

Fixed seed (42) ensures deterministic inputs:
```python
torch.manual_seed(42)
```

## Generating Fixtures

```bash
cd workspace/transformers
uv run python tests/gen_layer_fixtures_Qwen3-0.6B.py
```

## Model Configuration

Qwen3-0.6B:
- hidden_size: 1024
- intermediate_size: 3072
- num_attention_heads: 16
- num_key_value_heads: 8 (GQA ratio: 2)
- head_dim: 128
- rms_norm_eps: 1e-6
