# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/importutils,
  std/options,
  workspace/libtorch as F,
  workspace/positron/src/hadamard_transforms,
  ./embedding,
  ../quantizations/datatypes

{.experimental: "callOperator".}

type
  LMHead* = ref object
    ## Language Model Head for projecting hidden states to vocabulary logits.
    ##
    ## Supports tied embeddings (Qwen3-0.6B has tie_word_embeddings=true).
    ##
    ## Input:
    ##   - An externally provided `hidden_states` of shape (batch, seq, hidden_size)
    ##   - A local weight of shape (vocab_size, hidden_size) OR tied embedding
    ##   - Optionally a local bias of shape (vocab_size,)
    ##
    ## Return:
    ##   - Logits of shape (batch, seq, vocab_size) in same dtype as input (BF16)
    case tied: bool
    of false:
      weight: Tensor
    of true:
      tied_embedding: Embedding
    bias: Option[Tensor]
    case quant_format*: QuantFormatKind
    of qBF16:
      discard
    of qExl3:
      suh: Tensor    # [in_features] float16 — Hadamard input scale (EXL3 only)
      svh: Tensor    # [out_features] float16 — Hadamard output scale (EXL3 only)


func init*(_: type LMHead, weight: Tensor, bias = none(Tensor)): LMHead =
  ## Creates an LMHead with explicit weights.
  ##
  ## Args:
  ##   weight: Pre-initialized tensor of shape (vocab_size, hidden_size)
  ##   bias: Optional bias tensor of shape (vocab_size,)
  ##
  ## Returns:
  ##   LMHead with the given weight and optional bias
  LMHead(
    quant_format: qBF16,
    tied: false,
    weight: weight,
    bias: bias,
  )

func initTied*(_: type LMHead, embedding: Embedding, bias = none(Tensor)): LMHead =
  ## Creates an LMHead with tied embedding (shares weights with embedding layer).
  ##
  ## Args:
  ##   embedding: Reference to the embedding layer (weights are shared)
  ##   bias: Optional bias tensor of shape (vocab_size,)
  ##
  ## Returns:
  ##   LMHead with tied embedding weights
  LMHead(
    quant_format: qBF16,
    tied: true,
    tied_embedding: embedding,
    weight: none(Tensor),
    bias: bias,
  )

proc forward*(self: LMHead, hidden_states: Tensor): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   hidden_states: Tensor of shape (batch, seq, hidden_size)
  ##
  ## Returns:
  ##   Logits of shape (batch, seq, vocab_size) in same dtype as input (BF16)
  ##
  ## Computes:
  ##   logits = F.linear(hidden_states, weight, bias)
  ##
  ## Note:
  ##   Returns BF16 if input is BF16 (matches HF transformers).
  ##   Sampling module will upcast to FP32 when needed.

  case self.quant_format
  of qBF16:
    let weight =
      if self.tied:
        privateAccess(Embedding)
        self.tied_embedding.weight
      else:
        self.weight

    result =
      if self.bias.isSome():
        F.linear(hidden_states, weight, self.bias.get())
      else:
        F.linear(hidden_states, weight)
  of qEXL3:
    # TODO: EXL3-quantized lm_head (lm_head has its own trellis)
    raise newException(ValueError, "[ttt] EXL3 lm_head not yet implemented")

template `()`*(layer: LMHead, x: Tensor): untyped =
  forward(layer, x)
