# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/tensorcery/src/tensorcery

{.experimental: "callOperator".}

type
  Embedding* = ref object
    ## Embedding layer for token ID to hidden state conversion.
    ##
    ## Input:
    ##   - An externally provided `input_ids` of shape (1, seq) or flat
    ##   - A local weight of shape (vocab_size, hidden_size)
    ##
    ## Return:
    ##   - Embeddings of shape (1, seq, hidden_size) or (numIds, hidden_size)
    weight: Tensor
    vocab_size*: int
    hidden_size*: int

proc init*(_: type Embedding, weight: Tensor): Embedding =
  ## Creates an embedding layer from a weight tensor.
  ##
  ## Args:
  ##   weight: Pre-initialized tensor of shape (vocab_size, hidden_size)
  ##
  ## Returns:
  ##   Embedding layer with the given weight
  block:
    let s = weight.shape()
    doAssert s.len == 2, "[ttt] Internal Error: Embedding weight tensor is rank " & $s.len
    doAssert s[0] > 0 and s[1] > 0, "[ttt] Internal Error: Embedding weight tensor is empty"

  Embedding(
    weight: weight,
    vocab_size: weight.shape()[0],
    hidden_size: weight.shape()[1]
  )

proc forward*(self: Embedding, input_ids: openArray[int], input_shape: openArray[int]): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   input_ids: Integer token ids, passed flat. input_shape gives the id shape: (1, seq) or a one-entry (numIds,) form
  ##
  ## Returns:
  ##   Embeddings of shape (1, seq, hidden_size) or (numIds, hidden_size)
  ##
  ## Computes:
  ##   embeddings = weight[input_ids].reshape(input_ids.shape + (hidden_size,))
  block:
    let s = self.weight.shape()
    doAssert s[0] > 0 and s[1] > 0, "[ttt] Internal Error: Embedding weight tensor is empty"

  # Embedding lookup: select the token vectors from weight based on input_ids
  # input_shape restores over the selected weight entries
  let selected = embed_gather(self.weight, input_ids)

  # Reshape to (1, seq, hidden_size) or (numIds, hidden_size)
  if input_shape.len == 1:
    # a flat id list: output is (numIds, hidden_size)
    selected.reshape([input_shape[0], self.hidden_size])
  else:
    selected.reshape([input_shape[0], input_shape[1], self.hidden_size])

template `()`*(layer: Embedding, input_ids, input_shape: openArray[int]): untyped =
  layer.forward(input_ids, input_shape)
