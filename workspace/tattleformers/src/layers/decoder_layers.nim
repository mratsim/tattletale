# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Decoder layers, built on the long residual stream pattern.
## A decoder block is one DecoderLayer[SequenceMixer, HiddenMixer, Norm]:
## - the input norm and the post-attention norm bracket a sequence mixer
##   and a hidden mixer
## - the Norm parameter selects the block's norm class, RmsNorm or RmsNormOne
##
## The block implements the **long residual stream** pattern:
## pipeline parallelism and fused kernels.
##
## ## Two Pre-Norm Patterns
##
## Both use pre-norm (normalize before sublayer). The residual handling differs:
##
## **1. Local residuals** (huggingface, exllamav3):
## ```
## residual = x
## x = input_layernorm(x)
## x = self_attn(x)
## x = residual + x  ← ADD here
## x = post_attention_layernorm(x)
## x = mlp(x)
## x = residual + x  ← ADD here
## ```
##
## **2. Long residual stream** (vLLM, SGLang):
## ```
## residual = x  ← saved once
## (x, residual) = input_layernorm(x, residual)  ← residual passed through unchanged
## x = self_attn(x)
## (x, residual) = post_attention_layernorm(x, residual)  ← x + residual normalized
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
## **Fused kernels**: the norm + residual addition can fuse into a single
## kernel for less memory bandwidth (the vLLM fused RMSNorm kernel).
##
## **Deferred addition**: one addition per layer, inside the norm
## (not after self_attn and MLP). Inference saves one addition.
##
## **Equivalent outputs**: both patterns produce identical
## outputs for block-input residuals:
##   Local: x' = norm(x) + norm(self_attn(x)) = norm(x) + norm(x + self_attn(x) - x) = norm(x) + norm(x + self_attn(x) - x)
##   Long:  x' = norm(x + norm(x + self_attn(x) - x)) = norm(x + norm(x + self_attn(x) - x))
##
## ## Architecture
##
## ```
## Input: (x, residual=None or prev_residual)
##   │
##   ▼
## ┌─────────────────────────┐
## │   input_layernorm       │  ← forward_with_residual(x, residual)
## │   returns (normed, res) │  ← residual passed through unchanged
## └───────────┬─────────────┘
##             │
##             ▼
## ┌─────────────────────────┐
## │   SequenceMixer         │
## └───────────┬─────────────┘
##             │
##             ▼
## ┌─────────────────────────────┐
## │   post_attention_layernorm  │  ← forward_with_residual(x + self_attn_out, residual)
## │   adds x + res, normalizes  │
## └───────────┬─────────────────┘
##             │
##             ▼
## ┌─────────────────────────┐
## │   HiddenMixer           │
## └───────────┬─────────────┘
##             │
##             ▼
## Output: (x + mlp_out, residual)  ← to next layer
## ```
##
## ## Usage
##
## For a single block:
##   let (out, res) = block(ctx, x, none(Tensor))
##
## Stacked-model usage:
## ```
##   var residual: Option[Tensor]
##   for layer in layers:
##     let out = layer(ctx, x, residual)
##     x = out[0]
##     residual = some(out[1])
## ```
##
## The model final adds the residual before the final norm:
##   let normed = model.norm(x + residual.get(x))
##
## INVARIANT:
##
##   At each layer boundary: output + residual == x_local
##
##   where x_local is the output of the equivalent HF local residual pattern:
##     x_local = x + self_attn(RMSNorm(x)) + mlp(RMSNorm(x + self_attn(RMSNorm(x))))
##
##   Two norm equalities give the invariant:
##     - input_layernorm.forward_with_residual(x, r) → (RMSNorm(x+r), x+r)
##     - post_attention_layernorm.forward_with_residual(self_attn_out, r) → (RMSNorm(self_attn_out+r), self_attn_out+r)
##     - output = mlp(RMSNorm(self_attn_out+r)), residual = self_attn_out+r
##     - output + residual = mlp + self_attn_out + x = x_local
##
##     gen_bf16_qwen3_02_first_8_layers_plus_final.py records it,
##     t_bf16_qwen3_02_first_8_layers_plus_final.nim asserts it.

import
  std/options,
  workspace/tensorcery/src/tensorcery,
  ./norm,
  ../stateful/page_pool,
  ../stateful/inference_context

{.experimental: "callOperator".}


################################################################################
#                            Decoder layer generic                             #
################################################################################

type
  DecoderLayer*[SequenceMixer, HiddenMixer, Norm] = ref object
    ## A decoder-layer transformation over a sequence of token representations.
    ##
    ## Given `X ∈ ℝ^(L×D)`, where `L` is the sequence length and `D`
    ## is the hidden dimension:
    ##
    ##     X (L×D)
    ##       │
    ##       ▼
    ##   SequenceMixer ───► contextualized X' (L×D)
    ##       │
    ##       └────────────► mixer state / context state
    ##       │
    ##       ▼
    ##   HiddenMixer ─────► X'' (L×D)
    ##
    ## `SequenceMixer` contextualizes the sequence: information flows between positions along the sequence axis (`L`).
  ## The mixer may keep inference state for the accumulated context (a KV cache or recurrent state).
    ## - a KV cache
    ## - a recurrent state
    ## Examples: attention, linear attention, SSMs, delta-rule mixers.
    ##
    ## `HiddenMixer` transforms each position's contextualized representation
    ## independently across the `D`-dimensional hidden representation.
    ## Examples: dense/gated MLPs, sparse/MoE MLPs.
    ##
    ## Abstract form (normalization and residual connections ignored).
    ##
    ##     X'' = HiddenMixer(SequenceMixer(X))
    ##
  ## Both mixers preserve the `L×D` representation shape on different dimensions:
  ## `SequenceMixer` mixes across positions (`L`)
  ## and `HiddenMixer` within each position (`D`).
    input_layernorm: Norm
    sequence_mixer: SequenceMixer
    post_attention_layernorm: Norm
    hidden_mixer: HiddenMixer

func init*[SequenceMixer, HiddenMixer, Norm](_: type DecoderLayer[SequenceMixer, HiddenMixer, Norm],
           input_layernorm: Norm,
           sequence_mixer: SequenceMixer,
           post_attention_layernorm: Norm,
           hidden_mixer: HiddenMixer): DecoderLayer[SequenceMixer, HiddenMixer, Norm] =
  ## Take the two norms and the two mixers of one decoder block.
  ##
  ## Args:
  ## - input_layernorm, post_attention_layernorm: the bracketing norms
  ## - sequence_mixer, hidden_mixer: the block's two mixers
  ##
  ## The block owns no layer identity: the KV-cache layer index
  ## and the safetensors key prefix live on the mixers that need them.
  DecoderLayer[SequenceMixer, HiddenMixer, Norm](
    input_layernorm: input_layernorm,
    sequence_mixer: sequence_mixer,
    post_attention_layernorm: post_attention_layernorm,
    hidden_mixer: hidden_mixer
  )

proc forward*[SequenceMixer, HiddenMixer, Norm](
  self: DecoderLayer[SequenceMixer, HiddenMixer, Norm],
  ctx: var InferenceContext[uint32, Page],
  x: Tensor,
  residual: Option[Tensor]
): (Tensor, Tensor) =
  ## Forward pass for one decoder block on the long residual stream.
  ##
  ## Both norms fold the accumulated residual in before normalizing.
## The block never adds the stream to a sublayer output itself.
## The deferred-addition contract lives in this proc.
  ## - a pipeline stage can return (hidden, residual) across a stage boundary
  ## - a fused norm kernel can absorb the add
  ## - each sublayer costs a single addition
  ##
  ## Args:
  ## - ctx: InferenceContext with page refs and RoPE (ctx.pages, ctx.cos, ctx.sin)
  ## - x: Input tensor of shape (1, seq_len, hidden_size)
  ## - residual: The stream carried from the previous block, absent on the first
  ##
  ## Returns:
  ##   (hidden_mixer contribution, accumulated residual). The caller adds
  ##   the pair at the next boundary, or at the model
  ##   final before the final norm.
  ##
  ## The sequence mixer owns positional and cache state. RoPE, KV pages and recurrent state all travel through ctx.
  ##
  ## Computation:
  ## - (h, r) = input_layernorm(x, residual) when a residual was carried,
  ##   otherwise (input_layernorm(x), x)
  ## - mixerOut = sequence_mixer(ctx, h)
  ## - (h2, r2) = post_attention_layernorm(mixerOut, r)
  ##   (hidden_mixer.forward(h2), r2)

  # Explicit forward calls, not the call operator: the lookup does not
  # resolve inside a generic body.
  let (h, r) =
    if residual.isSome():
      forward_with_residual(self.input_layernorm, x, residual.unsafeGet())
    else:
      (self.input_layernorm.forward(x), x)

  let mixerOut = self.sequence_mixer.forward(ctx, h)

  let (h2, r2) = forward_with_residual(self.post_attention_layernorm, mixerOut, r)
  (self.hidden_mixer.forward(h2), r2)

template `()`*[SequenceMixer, HiddenMixer, Norm](layer: DecoderLayer[SequenceMixer, HiddenMixer, Norm],
            ctx: var InferenceContext[uint32, Page],
            x: Tensor,
            residual: Option[Tensor]): untyped =
  layer.forward(ctx, x, residual)
