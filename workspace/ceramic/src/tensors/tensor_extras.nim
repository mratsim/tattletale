# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layouts
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors/tensor_datatypes

macro transposedSelect*(layout: Layout, dim0, dim1: static int): untyped =
  let dimCount = layout.rank()
  doAssert dim0 >= 0 and dim0 < dimCount and dim1 >= 0 and dim1 < dimCount,
    "transpose: dimension out of range"
  var call = bindSym"selectDimensions".newCall(layout)
  for i in 0 ..< dimCount:
    call.add newLit(if i == dim0: dim1 elif i == dim1: dim0 else: i)
  result = call

template transpose*(t: AnyTensor, dim0, dim1: static int): auto =
  ## Swaps dimensions `dim0` and `dim1`, zero-copy:
  ## - the shape and stride entries at the two positions exchange
  ## - the memory is untouched and the result reads the same data
  ##
  ## Example: V stores one row of Dv elements per sequence step with the rows
  ## contiguous in memory. transpose(1, 0) reads the same memory with the dimensions exchanged.
  ##   vV = V(batch, kvHead, _, _).transpose(1, 0)  # the shape is (Dv, S)
  make_view(t.data, transposedSelect(t.getLayout(), dim0, dim1))

proc checkPermutation(Is: openArray[int]) {.compileTime.} =
  for m in 0 ..< Is.len:
    var count = 0
    for n in Is:
      if n == m:
        inc count
    doAssert count == 1,
      "permute: dimension " & $m & " appears " & $count & " times, a permutation lists every dimension once"

macro permute*(t: AnyTensor, Is: varargs[int]{lit|`const`}): auto =
  ## Reorders the view's dimensions by index, zero-copy:
  ## dimension `i` of the result reads the input's dimension `Is[i]`.
  var idxs = newSeq[int](Is.len)
  for i, n in Is:
    idxs[i] = n.intVal.int
  idxs.checkPermutation()
  let rankCall = (newDotExpr(t, bindSym"getLayout")).newCall().newDotExpr(bindSym"rank").newCall()
  result = newStmtList()
  result.add bindSym"doAssert".newCall(
    bindSym"==".newCall(newLit(Is.len), rankCall),
    newLit("permute: expects one index per dimension, got " & $Is.len))
  var call = bindSym"selectDimensions".newCall((newDotExpr(t, bindSym"getLayout")).newCall())
  for n in Is:
    call.add n
  result.add bindSym"make_view".newCall(newDotExpr(t, newIdentNode("data")), call)
