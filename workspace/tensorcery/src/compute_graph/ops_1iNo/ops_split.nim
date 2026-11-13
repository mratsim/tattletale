# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/design_by_contract
import ../../tensors_internals

# ############################################################
#
#                      Split
#
# ############################################################

proc tn_split*(engine: AnyEngine, x: TensorGraph, axis: int, k: int): seq[TensorGraph] {.contracts.} =
  ## Splits `x` along `axis` into `k` equal parts, views into `x`'s buffer.
  let ix = x.getImpl()
  {.precondition: axis >= 0 and axis < ix.shape().p.len.}
  {.precondition: ix.shape().p[axis] mod k == 0.}
  var partShape = @(ix.shape().p)
  partShape[axis] = partShape[axis] div k
  let strides = @(ix.strides().p)
  for i in 0 ..< k:
    result.add(engine.newViewNode(x, partShape, strides,
        i * partShape[axis] * strides[axis], "Split"))
