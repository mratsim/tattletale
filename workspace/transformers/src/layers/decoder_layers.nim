# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Decoder layers, built on the long residual stream pattern.
## A decoder block is one DecoderLayer[SequenceMixer, HiddenMixer, Norm]:
## the input norm and the post-attention norm bracketing a sequence mixer
## and a hidden mixer. The Norm parameter carries the block's norm class,
## RmsNorm or RmsNormOne.
## AnyDecoderLayer is a reference to any instantiation.
## One model can hold several pairings in one sequence.
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
## **Fused kernels**: The norm + residual addition can be fused into a single
## kernel, reducing memory bandwidth (the vLLM fused RMSNorm kernel).
##
## **Deferred addition**: The addition happens once per layer (inside norm)
## rather than twice (after self_attn, after MLP). For inference, this saves one
## addition operation.
##
## **Equivalent outputs**: Mathematically, both patterns produce identical
## outputs when residual is the block input:
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
## For a stacked model:
##   var residual: Option[Tensor]
##   for layer in layers:
##     let out = layer(ctx, x, residual)
##     x = out[0]
##     residual = some(out[1])
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
##   This invariant holds because of two norm equalities:
##     - input_layernorm.forward_with_residual(x, r) → (RMSNorm(x+r), x+r)
##     - post_attention_layernorm.forward_with_residual(self_attn_out, r) → (RMSNorm(self_attn_out+r), self_attn_out+r)
##     - So output = mlp(RMSNorm(self_attn_out+r)), residual = self_attn_out+r
##     - Therefore output + residual = mlp + self_attn_out + x = x_local
##
##     gen_bf16_qwen3_02_first_8_layers_plus_final.py records it,
##     t_bf16_qwen3_02_first_8_layers_plus_final.nim asserts it.


import
  std/options,
  pkg/iface,
  workspace/libtorch as F,
  ./norm,
  ../stateful/inference_context

{.experimental: "callOperator".}


################################################################################
#                            Decoder layer generic                             #
################################################################################

type
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
  ## `SequenceMixer` contextualizes the sequence by allowing
  ## information to flow between positions along the sequence dimension
  ## `L`. It may maintain inference state representing the accumulated context, such as a KV cache
  ## or recurrent state. Examples include attention, linear attention, SSMs,
  ## and delta-rule mixers.
  ##
  ## `HiddenMixer` transforms each position's contextualized representation
  ## independently, mixing information across the `D`-dimensional hidden
  ## representation. Examples include dense/gated MLPs and sparse/MoE MLPs.
  ##
  ## Abstractly, ignoring normalization and residual connections:
  ##
  ##     X'' = HiddenMixer(SequenceMixer(X))
  ##
  ## Both mixers preserve the `L×D` representation shape, while operating
  ## on different dimensions: `SequenceMixer` mixes across positions
  ## (`L`), and `HiddenMixer` mixes features within each position (`D`).
  DecoderLayer*[SequenceMixer, HiddenMixer, Norm] = ref object
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
  ## The block carries no layer identity: the KV-cache layer index
  ## and the safetensors key prefix live on the mixers that need them.
  DecoderLayer[SequenceMixer, HiddenMixer, Norm](
    input_layernorm: input_layernorm,
    sequence_mixer: sequence_mixer,
    post_attention_layernorm: post_attention_layernorm,
    hidden_mixer: hidden_mixer
  )

proc forward*[SequenceMixer, HiddenMixer, Norm](
  self: DecoderLayer[SequenceMixer, HiddenMixer, Norm],
  ctx: var InferenceContext,
  x: Tensor,
  residual: Option[Tensor]
): (Tensor, Tensor) =
  ## Forward pass for one decoder block on the long residual stream.
  ##
  ## Both norms fold the accumulated residual in before normalizing.
  ## The block never adds the stream to a sublayer output itself.
  ## This is the one place that holds the deferred-addition contract:
  ## a pipeline stage can return (hidden, residual) across a stage boundary,
  ## a fused norm kernel can absorb the add, and each sublayer costs
  ## a single addition.
  ##
  ## Args:
  ##   ctx: InferenceContext with page refs and RoPE (ctx.pages, ctx.cos, ctx.sin)
  ##   x: Input tensor of shape (batch, seq_len, hidden_size)
  ##   residual: The stream carried from the previous block, absent on the first
  ##
  ## Returns:
  ##   (hidden_mixer contribution, accumulated residual). The caller adds
  ##   the pair at the next boundary, or at the model
  ##   final before the final norm.
  ##
  ## The sequence mixer owns positional and cache state: RoPE, KV pages
  ## and recurrent state all travel through ctx.
  ##
  ## Computation:
  ##   (h, r) = input_layernorm(x, residual) when a residual was carried,
  ##     otherwise (input_layernorm(x), x)
  ##   mixerOut = sequence_mixer(ctx, h)
  ##   (h2, r2) = post_attention_layernorm(mixerOut, r)
  ##   (hidden_mixer.forward(h2), r2)

  let (h, r) =
    if residual.isSome():
      self.input_layernorm(x, residual.unsafeGet())
    else:
      (self.input_layernorm(x), x)

  let mixerOut = self.sequence_mixer(ctx, h)

  let (h2, r2) = self.post_attention_layernorm(mixerOut, r)
  (self.hidden_mixer.forward(h2), r2)

template `()`*[SequenceMixer, HiddenMixer, Norm](layer: DecoderLayer[SequenceMixer, HiddenMixer, Norm],
            ctx: var InferenceContext,
            x: Tensor,
            residual: Option[Tensor]): untyped =
  layer.forward(ctx, x, residual)

iface *AnyDecoderLayer:
  proc forward(ctx: var InferenceContext, x: Tensor, residual: Option[Tensor]): (Tensor, Tensor)

template `()`*(layer: AnyDecoderLayer,
            ctx: var InferenceContext,
            x: Tensor,
            residual: Option[Tensor]): untyped =
  layer.forward(ctx, x, residual)


# Sandwich decoder layer generic
# ------------------------------

type
  SandwichDecoderLayer*[SequenceMixer, HiddenMixer, Norm] = ref object
    ## Decoder block with the sandwich norm placement (gemma lineage):
    ## four norms bracket the two mixers and both post-norms normalize
    ## their sublayer output alone, the stream adds after the norm.
    ##
    ##     x ──► input_layernorm ──► SequenceMixer ──► post_attention_layernorm ─┐
    ##      ▲                                                                    │
    ##      └──────────────────────────── + ◄────────────────────────────────────┘
    ##      │ = h1
    ##      ├──► pre_feedforward_layernorm ──► HiddenMixer
    ##      │                                    │
    ##      └── + ◄── post_feedforward_layernorm ┘   → block output
    ##
    ## DecoderLayer folds the residual inside both norms, `norm(x + residual)`.
    ## The sandwich pattern normalizes each sublayer output alone, the stream
    ## joins after the norm.
    ##
    ## The two shapes are not interchangeable, the checkpoint config names
    ## the placement of its own blocks.
    input_layernorm: Norm
    sequence_mixer: SequenceMixer
    post_attention_layernorm: Norm
    pre_feedforward_layernorm: Norm
    hidden_mixer: HiddenMixer
    post_feedforward_layernorm: Norm

func init*[SequenceMixer, HiddenMixer, Norm](
    _: type SandwichDecoderLayer[SequenceMixer, HiddenMixer, Norm],
    input_layernorm: Norm,
    sequence_mixer: SequenceMixer,
    post_attention_layernorm: Norm,
    pre_feedforward_layernorm: Norm,
    hidden_mixer: HiddenMixer,
    post_feedforward_layernorm: Norm
): SandwichDecoderLayer[SequenceMixer, HiddenMixer, Norm] =
  ## Take the four norms and the two mixers of one sandwich decoder block.
  ##
  ## Contract:
  ## - the block carries no layer identity
  ## - the KV-cache layer index and the safetensors key prefix live on the mixers that need them
  SandwichDecoderLayer[SequenceMixer, HiddenMixer, Norm](
    input_layernorm: input_layernorm,
    sequence_mixer: sequence_mixer,
    post_attention_layernorm: post_attention_layernorm,
    pre_feedforward_layernorm: pre_feedforward_layernorm,
    hidden_mixer: hidden_mixer,
    post_feedforward_layernorm: post_feedforward_layernorm
  )

proc forward*[SequenceMixer, HiddenMixer, Norm](
  self: SandwichDecoderLayer[SequenceMixer, HiddenMixer, Norm],
  ctx: var InferenceContext,
  x: Tensor,
  residual: Option[Tensor]
): (Tensor, Tensor) =
  ## Forward pass for one sandwich decoder block on the long residual stream.
  ##
  ## Expected input:
  ##
  ## - `ctx`, the InferenceContext carrying page refs and RoPE rows (`ctx.pages`, `ctx.cos`, `ctx.sin`)
  ## - `x`, the previous block's deferred contribution, shape `(batch, seq_len, hidden_size)`
  ## - `residual`, the stream carried from the previous block
  ##   absent on the first block
  ##
  ## Returns:
  ##
  ## - the pair `(post_feedforward_layernorm output, h1)`
  ## - contribution + residual equals the block output
  ## - the caller adds the pair at the next block boundary or at the model
  ##   final before the final norm
  ##
  ## Chain:
  ##
  ## - `hPair = input_layernorm(x, residual)` when a residual was carried
  ## - `hPair = (input_layernorm(x), x)` otherwise
  ## - `mixerOut = sequence_mixer(ctx, hPair[0])`
  ##
  ## - `h1 = h + post_attention_layernorm(mixerOut)`
  ## - `h2 = pre_feedforward_layernorm(h1)`
  ## - `result = (post_feedforward_layernorm(hidden_mixer(h2)), h1)`
  ##
  ## The sequence mixer owns positional and cache state.
  ## RoPE, KV pages and recurrent state all travel through `ctx`.
  let (hNorm, h) =
    if residual.isSome():
      self.input_layernorm(x, residual.unsafeGet())
    else:
      (self.input_layernorm(x), x)

  let mixerOut = self.sequence_mixer(ctx, hNorm)

  let h1 = h + self.post_attention_layernorm(mixerOut)
  let h2 = self.pre_feedforward_layernorm(h1)
  let mlpOut = self.hidden_mixer(h2)
  (self.post_feedforward_layernorm(mlpOut), h1)

template `()`*[SequenceMixer, HiddenMixer, Norm](
    layer: SandwichDecoderLayer[SequenceMixer, HiddenMixer, Norm],
    ctx: var InferenceContext,
    x: Tensor,
    residual: Option[Tensor]): untyped =
  layer.forward(ctx, x, residual)


# ─── Parallel decoder layer generic ────────────────────────────────────────

type
  ParallelDecoderLayer*[SequenceMixer, HiddenMixer, Norm] = ref object
    ## Decoder block with the parallel residual placement (cohere lineage):
    ## ONE input norm feeds both mixers, the block output adds the two sublayer
    ## contributions to the stream.
    ##
    ##     x ──► input_layernorm ──┬──► SequenceMixer ───┐
    ##      ▲                      └──► HiddenMixer ─────┤
    ##      └──────────────────────────── + ◄────────────┘   → block output
    ##
    ## DecoderLayer normalizes the mixer input of a sequential chain, one
    ## norm per sublayer. The parallel shape normalizes once, both mixers
    ## read the same normalized rows, `x + attn(hNorm) + mlp(hNorm)`.
    ##
    ## The two shapes are not interchangeable, the checkpoint config names
    ## the placement of its own blocks.
    input_layernorm: Norm
    sequence_mixer: SequenceMixer
    hidden_mixer: HiddenMixer

func init*[SequenceMixer, HiddenMixer, Norm](
    _: type ParallelDecoderLayer[SequenceMixer, HiddenMixer, Norm],
    input_layernorm: Norm,
    sequence_mixer: SequenceMixer,
    hidden_mixer: HiddenMixer
): ParallelDecoderLayer[SequenceMixer, HiddenMixer, Norm] =
  ## Take the one shared norm and the two mixers of one parallel decoder block.
  ##
  ## Contract:
  ## - the block carries no layer identity
  ## - the KV-cache layer index and the safetensors key prefix live on the mixers that need them
  ParallelDecoderLayer[SequenceMixer, HiddenMixer, Norm](
    input_layernorm: input_layernorm,
    sequence_mixer: sequence_mixer,
    hidden_mixer: hidden_mixer
  )

proc forward*[SequenceMixer, HiddenMixer, Norm](
  self: ParallelDecoderLayer[SequenceMixer, HiddenMixer, Norm],
  ctx: var InferenceContext,
  x: Tensor,
  residual: Option[Tensor]
): (Tensor, Tensor) =
  ## Forward pass for one parallel decoder block on the long residual stream.
  ##
  ## Expected input:
  ##
  ## - `ctx`, the InferenceContext carrying page refs and RoPE rows (`ctx.pages`, `ctx.cos`, `ctx.sin`)
  ## - `x`, the previous block's deferred contribution, shape `(batch, seq_len, hidden_size)`
  ## - `residual`, the stream carried from the previous block,
  ##   absent on the first block
  ##
  ## Returns:
  ##
  ## - the pair `(attn + mlp contributions, carried residual)`
  ## - contribution + residual equals the block output, the reference
  ##   parallel layer output `x + residual + attn(hNorm) + mlp(hNorm)`
  ## - the caller adds the pair at the next block boundary or at the model
  ##   final before the norm
  ##
  ## Chain:
  ##
  ## - `hNorm = input_layernorm(x + residual)` when a residual was carried
  ## - `hNorm = input_layernorm(x)` otherwise, the carried residual is `x`
  ## - `result = (sequence_mixer(ctx, hNorm) + hidden_mixer(hNorm), residual)`
  ##
  ## The sequence mixer owns positional and cache state.
  ## RoPE, KV pages and recurrent state all travel through `ctx`.
  let (hNorm, r) =
    if residual.isSome():
      self.input_layernorm(x, residual.unsafeGet())
    else:
      (self.input_layernorm(x), x)

  let attnOut = self.sequence_mixer(ctx, hNorm)
  let mlpOut = self.hidden_mixer(hNorm)
  (attnOut + mlpOut, r)

template `()`*[SequenceMixer, HiddenMixer, Norm](
    layer: ParallelDecoderLayer[SequenceMixer, HiddenMixer, Norm],
    ctx: var InferenceContext,
    x: Tensor,
    residual: Option[Tensor]): untyped =
  layer.forward(ctx, x, residual)
