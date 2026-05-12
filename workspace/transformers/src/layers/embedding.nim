# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F

type
  Embedding* = object
    ## Embedding layer for token ID to hidden state conversion.
    ##
    ## Input:
    ##   - An externally provided `input_ids` of shape (batch, seq) or (batch,)
    ##   - A local weight of shape (vocab_size, hidden_size)
    ##
    ## Return:
    ##   - Embeddings of shape (batch, seq, hidden_size) or (batch, hidden_size)
    weight*: Tensor
    vocab_size*: int
    hidden_size*: int

func init*(_: type Embedding, weight: Tensor): Embedding =
  ## Creates an embedding layer from a weight tensor.
  ##
  ## Args:
  ##   weight: Pre-initialized tensor of shape (vocab_size, hidden_size)
  ##
  ## Returns:
  ##   Embedding layer with the given weight
  if weight.numel() == 0:
    raise newException(ValueError, "[ttt] Internal Error: Embedding weight tensor is empty")

  Embedding(
    weight: weight,
    vocab_size: weight.size(0),
    hidden_size: weight.size(1)
  )

proc forward*(self: Embedding, input_ids: Tensor): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   input_ids: Integer tensor of shape (batch, seq) or (batch,)
  ##
  ## Returns:
  ##   Embeddings of shape (batch, seq, hidden_size) or (batch, hidden_size)
  ##
  ## Computes:
  ##   embeddings = weight.index_select(0, input_ids.flatten()).reshape(input_ids.shape + (hidden_size,))
  if self.weight.numel() == 0:
    raise newException(ValueError, "[ttt] Internal Error: Embedding weight tensor is empty")

  # Embedding lookup: select rows from weight based on input_ids
  # For multi-dimensional input_ids, flatten, select, then reshape
  template inputShape: untyped = input_ids.shape
  let flatInput = input_ids.view(-1)
  let selected = self.weight.index_select(0, flatInput)

  # Reshape to (batch, seq, hidden_size) or (batch, hidden_size)
  if inputShape.len == 1:
    # input_ids is (batch,), output should be (batch, hidden_size)
    selected.view([inputShape[0], self.hidden_size])
  else:
    # input_ids is (batch, seq), output should be (batch, seq, hidden_size)
    selected.view([inputShape[0], inputShape[1], self.hidden_size])