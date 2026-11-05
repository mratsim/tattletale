# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shape-op record images: reshape, dimension permutation, range selection.
##
## Shape ops are pure metadata: a new view (shape, strides, offset) over the
## same storage. Their record entries exist for the backward walk only, so the
## image carries the view parameters and no device addresses. Backends decode
## the image nowhere: no kernel runs, submission counts stay untouched.

type ShapeBlob* = object
  ## Recorded view parameters of one shape op.
  ##
  ## - reshape: `a` lists the new dims
  ## - permute: `a` lists the source-axis read order of the output
  ## - slice: `a` lists the first selected index per axis, `b` the selected count per axis
  rank*: int32
  a*, b*: array[4, int32]

proc packShapeBlob*(rank: int, a, b: array[4, int32]): seq[byte] =
  ## Flat byte image of one shape-op record entry, marshaled once at enqueue.
  result = newSeq[byte](sizeof(ShapeBlob))
  var img = ShapeBlob(rank: int32(rank), a: a, b: b)
  copyMem(result[0].addr, img.addr, sizeof(ShapeBlob))

proc decodeShapeBlob*(blob: openArray[byte]): ShapeBlob =
  ## Inverse of packShapeBlob: rebuilds the recorded view parameters.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(ShapeBlob))
