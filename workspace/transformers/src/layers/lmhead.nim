# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  workspace/libtorch as F,
  ./embedding

type
  LMHead* = object
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
    weight*: Option[TorchTensor]
    bias*: Option[TorchTensor]
    tied_embedding*: Option[Embedding]

func init*(_: type LMHead, weight: TorchTensor, bias = none(TorchTensor)): LMHead =
  ## Creates an LMHead with explicit weights.
  ##
  ## Args:
  ##   weight: Pre-initialized tensor of shape (vocab_size, hidden_size)
  ##   bias: Optional bias tensor of shape (vocab_size,)
  ##
  ## Returns:
  ##   LMHead with the given weight and optional bias
  LMHead(
    weight: some(weight),
    bias: bias,
    tied_embedding: none(Embedding)
  )

func initTied*(_: type LMHead, embedding: Embedding, bias = none(TorchTensor)): LMHead =
  ## Creates an LMHead with tied embedding (shares weights with embedding layer).
  ##
  ## Args:
  ##   embedding: Reference to the embedding layer (weights are shared)
  ##   bias: Optional bias tensor of shape (vocab_size,)
  ##
  ## Returns:
  ##   LMHead with tied embedding weights
  LMHead(
    weight: none(TorchTensor),
    bias: bias,
    tied_embedding: some(embedding)
  )

proc forward*(self: LMHead, hidden_states: TorchTensor): TorchTensor =
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
  let weight =
    if self.weight.isSome:
      self.weight.get()
    elif self.tied_embedding.isSome:
      self.tied_embedding.get().weight
    else:
      raise newException(ValueError, "[ttt] Internal Error: LMHead has no weights")
  
  if self.bias.isSome:
    F.linear(hidden_states, weight, self.bias.get())
  else:
    F.linear(hidden_states, weight)