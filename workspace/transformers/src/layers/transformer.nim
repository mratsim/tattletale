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
## **2. Long residual stream** (vLLM, SGLang):
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
##   let (out, _) = block.forward(x, none(Tensor), positions, use_cache)
##
## For stacked layers in a model:
##   var residual: Tensor
##   for layer in layers:
##     (x, residual) = layer.forward(x, residual, positions, use_cache)
##
## The final model forward typically adds the residual before the final norm:
##   let (normed, _) = final_norm.forward(x + residual, none)
##
## INVARIANT:
##
##   At each layer boundary: output + residual == x_local
##
##   where x_local is the output of the equivalent HF local residual pattern:
##     x_local = x + attn(RMSNorm(x)) + mlp(RMSNorm(x + attn(RMSNorm(x))))
##
##   This invariant holds because:
##     - attn_norm.forward_with_residual(x, r) → (RMSNorm(x+r), x+r)
##     - mlp_norm.forward_with_residual(attn_out, r) → (RMSNorm(attn_out+r), attn_out+r)
##     - So output = mlp(RMSNorm(attn_out+r)), residual = attn_out+r
##     - Therefore output + residual = mlp + attn_out + x = x_local
##
##     The invariant is tested in
##     gen_3_block_long_residual.py
##     via test_qwen3_long_residual_3_blocks.nim.


import
  std/options,
  workspace/libtorch as F,
  ./attn,
  ./mlp,
  ./norm

type
  TransformerBlock* = object
    attn_norm*: RmsNorm
    attn*: RopeGQAttention
    mlp_norm*: RmsNorm
    mlp*: GatedMLP

func init*(_: type TransformerBlock, attn_norm: RmsNorm, attn: RopeGQAttention, mlp_norm: RmsNorm, mlp: GatedMLP): TransformerBlock =
  TransformerBlock(
    attn_norm: attn_norm,
    attn: attn,
    mlp_norm: mlp_norm,
    mlp: mlp
  )

proc forward*(self: var TransformerBlock, x: Tensor, residual: Option[Tensor]): (Tensor, Tensor) =
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
  ##
  ## Returns:
  ##   (output, residual) where:
  ##     - output: Tensor of shape (batch, seq_len, hidden_size)
  ##     - residual: The accumulated residual, passed through unchanged
  ##
  ## Note:
  ##   RoPE positions and KV cache are handled internally by the attention layer.
  ##   Call self.attn.rotary.setCache(cos, sin) before forward() to set RoPE.
  ##
  ## Computation:
  ##   residual = residual.get(x)  # Use x if residual is None
  ##   (h, residual) = self.attn_norm.forward_with_residual(x, residual)
  ##   attn_out = self.attn.forward(h)
  ##   (h2, residual) = self.mlp_norm.forward_with_residual(attn_out, residual)
  ##   mlp_out = self.mlp.forward(h2)
  ##   (mlp_out, residual)

  let (h, r) =
    if residual.isSome():
      self.attn_norm.forward_with_residual(x, residual.unsafeGet())
    else:
      (self.attn_norm.forward(x), x)
  let attn_out = self.attn.forward(h)
  let (h2, r2) = self.mlp_norm.forward_with_residual(attn_out, r)
  let mlp_out = self.mlp.forward(h2)
  (mlp_out, r2)
