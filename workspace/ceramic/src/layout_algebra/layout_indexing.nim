## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout indexing: crd2idx, idx2crd, slice, dice.
{.experimental: "callOperator".}


import std/macros
import std/sequtils
import std/typetraits

import workspace/ceramic/src/int_tuples
import ./layout_indexing_cpu
import ./layout_indexing_gpu
import ./layout_indexing_slicedice
import ./layouts
import ./layouts_unsanctioned_helpers
import ./layout_compiletime
import workspace/ceramic/src/macros/varargs_to_par

export layout_indexing_cpu
export layout_indexing_gpu
export layout_indexing_slicedice

# ═══════════════════════════════════════════════════════════════
#  crd2idx / idx2crd, delegates to layout_indexing_gpu
# ═══════════════════════════════════════════════════════════════

template crd2idx*(layout: Layout; coord: IntOrIntTuple): auto =
  ## Logical-to-memory offset for a coordinate on a Layout, the multiply-add
  ## chain over the layout's strides.
  crd2idx_gpu(layout, coord)

macro idx2crd*(layout: Layout; idx: int or Int): untyped =
  ## Convert linear index to coordinate using a Layout.
  ##
  ## Coordinate = `(idx div stride) mod shape` per dimension, valid for
  ## compact (contiguous) layouts only, where the largest stride sits
  ## at the last dimension.
  ## For non-compact shapes use the shape-based `idx2crd(shape, idx)`.
  ##
  ## Contract:
  ## - excess accumulates at the largest static stride and does not wrap
  ## - rank-1 layouts are the degenerate case of that rule
  ## - pycute decomposes this way, while CuTe C++ mods every leaf and wraps
  ##
  ## A static shape-1 dimension maps to 0 before the division, keeping
  ## stride-0 broadcasts away from it.
  ##
  ## Contract on the guard:
  ## - no runtime shape-1 guard exists
  ## - pycute never sees strides and CuTe C++ guards only the static case
  ## - a dynamic broadcast violates the compact precondition
  let shT = layoutTypeArgs(layout).shapeTy
  let stT = layoutTypeArgs(layout).strideTy
  let sh = newTree(nnkDotExpr, layout, ident"shape")
  let st = newTree(nnkDotExpr, layout, ident"stride")
  if shT.kind != nnkTupleConstr:
    result = quote do:
      when `sh` is Int[1]:
        Int[0]()
      else:
        `idx` div `st`
  else:
    # most-significant leaf = the largest stride, identifiable only
    # when every stride is static, so dynamic strides keep the mod
    let stVals = toSeqStaticInts(stT)
    let maxIdx = stVals.maxIndex
    let allStatic = DynamicSentinel notin stVals
    # for tuple shapes, the quotient runs unmod'd at the largest static stride
    var parts: seq[NimNode] = @[]
    for i in 0 ..< shT.len:
      let s = bindSym"[]".newCall(st, newLit(i))
      let shI = bindSym"[]".newCall(sh, newLit(i))
      let leaf = if allStatic and i == maxIdx:
        quote do: `idx` div `s`
      else:
        quote do: (`idx` div `s`) mod `shI`
      parts.add quote do:
        when `shI` is Int[1]:
          Int[0]()
        else:
          `leaf`
    result = nnkPar.newTree(parts)

# ═══════════════════════════════════════════════════════════════
#  idx2crd, index to coordinate decomposition
# ═══════════════════════════════════════════════════════════════

proc emitCoordTree(profile: NimNode; parts: seq[NimNode]; i: var int): NimNode {.compileTime.} =
  ## Rebuild the shape's nesting over the flat decomposition `parts`.
  if profile.kind in {nnkTupleTy, nnkTupleConstr}:
    result = nnkPar.newTree()
    for j in 0 ..< profile.len:
      result.add emitCoordTree(profile[j], parts, i)
  else:
    result = parts[i]
    inc i

macro idx2crd*(shape: IntOrIntTuple; idx: int or Int): untyped =
  ## Decompose a flat index into a coordinate over SHAPE.
  ##
  ## Returns the coordinate tuple, colexicographic over the leaf sizes,
  ## first dimension fastest.
  ##
  ## Contract:
  ## - excess stays on the last flat leaf, which keeps the full quotient
  ##   and absorbs it without wrapping
  ## - pycute decomposes this way, while CuTe C++ mods every leaf and wraps
  ## - a scalar shape is the degenerate case of that rule, so the coordinate
  ##   is the index itself
  ##
  ## Valid for ANY shape (compact or not), unlike the stride-based
  ## compact-only idx2crd(layout, idx).
  ##
  ## Examples:
  ##
  ##   idx2crd((4, 8), 31)            == (3, 7)
  ##   idx2crd(((4, 8), (2, 2)), 31)  == ((3, 7), (0, 0))
  ##   idx2crd(2, 1)                  == 1
  ##   idx2crd((3, 7, 2), 42)         == (0, 0, 2)
  let shT = shape.getTypeInst()
  if shT.kind in {nnkTupleTy, nnkTupleConstr}:
    let flat = shape.tupleFlatten()
    var parts: seq[NimNode] = @[]
    var q = idx
    for k in 0 ..< flat.len - 1:
      let s = flat[k].leaf
      parts.add newCall(bindSym"mod", q, s)
      q = newCall(bindSym"div", q, s)
    parts.add q
    var i = 0
    result = emitCoordTree(shT, parts, i)
  else:
    result = idx

# ═══════════════════════════════════════════════════════════════
#  layout() call syntax
# ═══════════════════════════════════════════════════════════════

template `()`*(layout: Layout; args: varargs[typed]): auto =
  ## Multi-argument: `L(i, j)` ≡ `L((i, j))`.
  block:
    evalOnceAs(coord, varargs_to_par(args))
    when hasUnderscoreImpl(coord):
      slice(layout, coord)
    else:
      crd2idx(layout, coord)
