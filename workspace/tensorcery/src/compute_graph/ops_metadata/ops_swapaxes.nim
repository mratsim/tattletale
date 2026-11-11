# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/iface
import workspace/design_by_contract
import ../../tensors_internals

# ############################################################
#
#                   Swapaxes
#
# ############################################################

proc tn_swapaxes*(engine: AnyEngine, x: TensorNode, axis1, axis2: int): TensorNode {.contracts.} =
  ## A view of `x` with `axis1` and `axis2` exchanged, sharing `x`'s buffer.
  let ix = x.getImpl()
  {.precondition: axis1 >= 0 and axis1 < ix.shape().p.len.}
  {.precondition: axis2 >= 0 and axis2 < ix.shape().p.len.}
  {.precondition: axis1 != axis2.}
  var outShape = @(ix.shape().p)
  swap(outShape[axis1], outShape[axis2])
  var outStrides = @(ix.strides().p)
  swap(outStrides[axis1], outStrides[axis2])
  return engine.newViewNode(x, outShape, outStrides, 0, "Swapaxes")
