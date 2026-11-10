# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/importutils,
  std/options,
  workspace/tensorcery/src/tensorcery,
  ./embedding

{.experimental: "callOperator".}

type
  LMHead* = ref object
    ## Language Model Head for projecting hidden states to vocabulary logits.
    ##
    ## Supports tied embeddings (Qwen3-0.6B has tie_word_embeddings=true).
    ##
    ## Input:
    ##   - An externally provided `hidden_states` of shape (1, seq, hidden_size)
    ##   - A local weight of shape (vocab_size, hidden_size) OR tied embedding
    ##   - Optionally a local bias of shape (vocab_size,)
    ##
    ## Return:
    ##   - Logits of shape (1, seq, vocab_size) in same dtype as input (BF16)
    case tied: bool
    of false:
      weight: Tensor
    of true:
      tied_embedding: Embedding
    bias: Option[Tensor]
    # quant_format is qBF16 only: the qExl3 branch is unported
    # (the EXL3 path's Hadamard rotations bind libtorch).

proc init*(_: type LMHead, weight: Tensor, bias = none(Tensor)): LMHead =
  ## Creates an LMHead with explicit weights.
  ##
  ## Args:
  ## - weight: pre-initialized tensor of shape (vocab_size, hidden_size)
  ## - bias: optional bias tensor of shape (vocab_size,)
  ##
  ## Returns:
  ## - an LMHead over the given weight with the optional bias
  LMHead(
    tied: false,
    weight: weight,
    bias: bias,
  )

proc initTied*(_: type LMHead, embedding: Embedding, bias = none(Tensor)): LMHead =
  ## Creates an LMHead with a tied embedding (weights shared with the embedding layer).
  ##
  ## Args:
  ## - embedding: reference to the embedding layer (weights are shared)
  ## - bias: optional bias tensor of shape (vocab_size,)
  ##
  ## Returns:
  ## - an LMHead over the tied embedding weights
  LMHead(
    tied: true,
    tied_embedding: embedding,
    bias: bias,
  )

proc forward*(self: LMHead, hidden_states: Tensor): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   hidden_states: Tensor of shape (1, seq, hidden_size)
  ##
  ## Returns:
  ##   Logits of shape (1, seq, vocab_size) in same dtype as input (BF16)
  ##
  ## Computes:
  ##   logits = F.linear(hidden_states, weight, bias)
  ##
  ## Note:
  ##   Returns BF16 if input is BF16 (matches HF transformers).
  ##   Sampling module will upcast to FP32 when needed.

  # F.linear over the matmul op, the Linear layer's projection.
  # Stored (vocab_size, hidden_size) weight feeds the matmul op's
  # BLAS-Bᵀ form directly, the transposition lives in the op (zero copy).
  # Rank-2 matmul over the flattened (tokens, hidden_size) input.
  # Output: leading dimensions restored over the result.

  if self.bias.isSome:
    raise ValueError.newException(
      "LMHead bias is unported: the Qwen3 path runs bias-free " &
      "(reference semantics: F.linear(x, w, b) = x @ weight^T + bias)")

  let weight =
    if self.tied:
      privateAccess(Embedding)
      self.tied_embedding.weight
    else:
      self.weight

  let s = hidden_states.shape()
  var tokens = 1
  for i in 0 ..< s.len - 1:
    tokens *= s[i]
  let vocab = weight.shape()[0]
  let flat = hidden_states.reshape([tokens, s[s.len - 1]])
  let proj = linear(flat, weight)
  var outShape: seq[int]
  for i in 0 ..< s.len - 1:
    outShape.add(s[i])
  outShape.add(vocab)
  proj.reshape(outShape)

template `()`*(layer: LMHead, x: Tensor): untyped =
  forward(layer, x)
