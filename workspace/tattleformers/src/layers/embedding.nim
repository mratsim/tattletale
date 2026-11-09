# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/tensorcery/src/tensor_datatypes
import ../ops_Ni1o/embed_gather

{.experimental: "callOperator".}

type
  Embedding* = ref object
    ## Embedding layer for token ID to hidden state conversion.
    ##
    ## Input:
    ##   - An externally provided `input_ids` of shape (batch, seq) or (batch,)
    ##   - A local weight of shape (vocab_size, hidden_size)
    ##
    ## Return:
    ##   - Embeddings of shape (batch, seq, hidden_size) or (batch, hidden_size)
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
    let s = weight.node.getImpl().shape()
    var numel = 1
    for i in 0 ..< s.len:
      numel *= s[i]
    doAssert numel > 0, "[ttt] Internal Error: Embedding weight tensor is empty"

  Embedding(
    weight: weight,
    vocab_size: weight.node.getImpl().shape()[0],
    hidden_size: weight.node.getImpl().shape()[1]
  )

proc forward*(self: Embedding, input_ids: openArray[int], input_shape: openArray[int]): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   input_ids: Integer token ids of shape (batch, seq) or (batch,), passed flat. input_shape gives the id shape
  ##
  ## Returns:
  ##   Embeddings of shape (batch, seq, hidden_size) or (batch, hidden_size)
  ##
  ## Computes:
  ##   embeddings = weight[input_ids].reshape(input_ids.shape + (hidden_size,))
  block:
    let s = self.weight.node.getImpl().shape()
    var numel = 1
    for i in 0 ..< s.len:
      numel *= s[i]
    doAssert numel > 0, "[ttt] Internal Error: Embedding weight tensor is empty"

  # Embedding lookup: select the token vectors from weight based on input_ids
  # For multi-dimensional input_ids, flatten, select, then reshape
  let selected = Tensor(
    engine: self.weight.engine,
    node: tn_embed_gather(self.weight.engine, self.weight.node, input_ids))

  # Reshape to (batch, seq, hidden_size) or (batch, hidden_size)
  if input_shape.len == 1:
    # input_ids is (batch,), output should be (batch, hidden_size)
    selected.reshape([input_shape[0], self.hidden_size])
  else:
    selected.reshape([input_shape[0], input_shape[1], self.hidden_size])

template `()`*(layer: Embedding, input_ids, input_shape: openArray[int]): untyped =
  layer.forward(input_ids, input_shape)
