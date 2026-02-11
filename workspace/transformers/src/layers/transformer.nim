# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Transformer Block with Long Residual Stream Pattern
##
## This module implements a transformer block using the **long residual stream**
## pattern, enabling optimizations for pipeline parallelism and fused kernels.
##
## ## Two Pre-Norm Patterns
##
## Both use pre-norm (normalize before sublayer), but differ in residual handling:
##
## **1. Local residuals** (huggingface, exllamav3):
## ```
## residual = x
## x = attn_norm(x)
## x = attn(x)
## x = residual + x  ← ADD here
## x = mlp_norm(x)
## x = mlp(x)
## x = residual + x  ← ADD here
## ```
##
## **2. Long residual stream** (vLLM, SGLang, nano-vllm):
## ```
## residual = x  ← saved once
## (x, residual) = attn_norm(x, residual)  ← residual passed through
## x = attn(x)
## (x, residual) = mlp_norm(x, residual)  ← x + residual normalized
## x = mlp(x)
## return (x, residual)  ← to next layer
## ```
##
## ## Why Long Residual Stream?
##
## **Pipeline parallelism**: The residual can be split across pipeline stages.
## A layer in stage 1 can return (hidden_states, residual) and the caller
## (stage 2) handles the addition, enabling clean stage boundaries.
##
## **Fused kernels**: The norm + residual addition can be fused into a single
## kernel, reducing memory bandwidth (vLLM's RMSNorm does this).
##
## **Deferred addition**: The addition happens once per layer (inside norm)
## rather than twice (after attn, after MLP). For inference, this saves one
## addition operation.
##
## **Equivalent outputs**: Mathematically, both patterns produce identical
## outputs when residual is the block input:
##   Local: x' = norm(x) + norm(attn(x)) = norm(x) + norm(x + attn(x) - x) = norm(x) + norm(x + attn(x) - x)
##   Long:  x' = norm(x + norm(x + attn(x) - x)) = norm(x + norm(x + attn(x) - x))
##
## ## Architecture
##
## ```
## Input: (x, residual=None or prev_residual)
##   │
##   ▼
## ┌─────────────────────────┐
## │   attn_norm             │  ← forward_with_residual(x, residual)
## │   returns (normed, res) │  ← residual passed through unchanged
## └───────────┬─────────────┘
##             │
##             ▼
## ┌─────────────────────────┐
## │   MH Attention          │
## └───────────┬─────────────┘
##             │
##             ▼
## ┌─────────────────────────┐
## │   mlp_norm              │  ← forward_with_residual(x + attn_out, residual)
## │   adds x + res, normalizes│
## └───────────┬─────────────┘
##             │
##             ▼
## ┌─────────────────────────┐
## │   Gated MLP             │
## └───────────┬─────────────┘
##             │
##             ▼
## Output: (x + mlp_out, residual)  ← to next layer
## ```
##
## ## Usage
##
## For single-layer inference:
##   let (out, _) = block.forward(x, none(TorchTensor), positions, use_cache)
##
## For stacked layers in a model:
##   var residual: TorchTensor
##   for layer in layers:
##     (x, residual) = layer.forward(x, residual, positions, use_cache)
##
## The final model forward typically adds the residual before the final norm:
##   let (normed, _) = final_norm.forward(x + residual, none)

import
  std/options,
  workspace/libtorch as F,
  ./attn,
  ./mlp,
  ./norm

type
  TransformerBlock* = object
    attn_norm*: RmsNorm
    attn*: RopeMHAttention
    mlp_norm*: RmsNorm
    mlp*: GatedMLP

func init*(_: type TransformerBlock, attn_norm: RmsNorm, attn: RopeMHAttention, mlp_norm: RmsNorm, mlp: GatedMLP): TransformerBlock =
  TransformerBlock(
    attn_norm: attn_norm,
    attn: attn,
    mlp_norm: mlp_norm,
    mlp: mlp
  )

func forward*(self: var TransformerBlock, x: TorchTensor, residual: Option[TorchTensor], positions: TorchTensor, use_cache: bool): (TorchTensor, TorchTensor) =
  ## Forward pass for a transformer block with long residual stream.
  ##
  ## This pattern defers residual additions to the norm layers, enabling:
  ## - Pipeline parallelism support (residual can cross stage boundaries)
  ## - Fused norm+residual kernels (vLLM optimization)
  ## - Single addition per layer (instead of two)
  ##
  ## Args:
  ##   x: Input tensor of shape (batch, seq_len, hidden_size)
  ##   residual: Optional residual from previous layer. If None, uses x.
  ##   positions: Position indices for RoPE, shape (batch, seq_len) or similar
  ##   use_cache: Whether to use KV cache for autoregressive decoding
  ##
  ## Returns:
  ##   (output, residual) where:
  ##     - output: Tensor of shape (batch, seq_len, hidden_size)
  ##     - residual: The original input residual, passed through unchanged
  ##
  ## Computation:
  ##   residual = residual.get(x)  # Use x if residual is None
  ##   (h, residual) = self.attn_norm.forward_with_residual(x, residual)
  ##   attn_out = self.attn.forward(h, positions, use_cache)
  ##   (h2, residual) = self.mlp_norm.forward_with_residual(h + attn_out, residual)
  ##   mlp_out = self.mlp.forward(h2)
  ##   (h2 + mlp_out, residual)
  let (h, r) =
    if residual.isSome():
      self.attn_norm.forward_with_residual(x, residual.unsafeGet())
    else:
      (self.attn_norm.forward(x), x)
  let attn_out = self.attn.forward(h, positions, use_cache)
  let (h2, r2) = self.mlp_norm.forward_with_residual(h + attn_out, r)
  let mlp_out = self.mlp.forward(h2)
  (h2 + mlp_out, r2)
