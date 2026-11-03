# LLM models tested in Tattletale transformers

We reimplement HuggingFace transformers models in Nim, kernel by kernel.

- Validation replays recorded fixture suites against the Python reference:
  per-layer internals, full forward to logits, plus greedy decoding.
- Every model below passes the fixture ladder on Metal (M4 Max)
  and CUDA (2x RTX PRO 6000), in bf16 unless noted.

## Models in order of first test support

| Model | Architectures |
|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | GQA + SWA |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | GDN + gated GQA |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | GDN + gated GQA + MoE |
| [gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it) | GQA + SWA |
| [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | GQA + SWA |
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | GQA + SWA |
| [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | GQA + SWA + PLE |
| [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | GQA + SWA |
| [gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) | GQA + SWA + MoE |
| [Laguna-XS-2.1](https://huggingface.co/Laguna-XS-2.1) | GQA + SWA + MoE |
| [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) | KDA + MLA + MoE |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | MLA + MoE |
| [Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B) | MLA + MoE |
| [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) | KDA + MLA + MoE |
| [North-Mini-Code-1.0](https://huggingface.co/North-Mini-Code-1.0) | GQA + SWA + MoE |

## Attention and layer legend

| Term | Meaning |
|---|---|
| GQA | grouped-query attention: several query heads share one KV head, shrinks the KV cache by the ratio |
| SWA | sliding-window attention: attention restricted to a fixed window of recent tokens, interleaved with full-attention layers |
| Gated GQA | GQA with a per-head output `gate` on the attention projection (Qwen3.5/3.6 full-attention layers) |
| GDN | gated delta net: linear attention with a delta-rule state update and a `forget gate`, constant memory per sequence instead of a KV cache |
| KDA | Kimi delta attention: GDN variant where the decay `gate` is a sigmoid with a learned lower bound (`kda_lower_bound`) |
| MLA | multi-head latent attention: KV compressed into a low-rank latent vector (DeepSeek style), the smallest KV cache per token |
| MoE | mixture of experts: block-sparse FFN where a router picks the top-k experts per token (A3B = 3B active parameters) |
| PLE | per-layer embeddings: an extra token-identity embedding table feeding every layer (Gemma 4 E2B/E4B) |
| RoPE | rotary position embeddings |
| RMSNorm | root-mean-square layer normalization |

All suites replay recorded statistics from the Python reference through our
Nim kernels and assert within honest-rounding bands (bf16 ulp scale).

A green run means the port is mathematically faithful, not merely plausible.
