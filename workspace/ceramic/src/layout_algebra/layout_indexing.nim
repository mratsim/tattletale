## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout indexing: crd2idx, idx2crd, slice, dice.
{.experimental: "callOperator".}


import std/macros
import std/typetraits

import workspace/ceramic/src/int_tuples
import ./layout_indexing_cpu
import ./layout_indexing_gpu
import ./layouts
import workspace/ceramic/src/macros/varargs_to_par

export layout_indexing_cpu
export layout_indexing_gpu

# ═══════════════════════════════════════════════════════════════
#  crd2idx / idx2crd, delegates to layout_indexing_gpu
# ═══════════════════════════════════════════════════════════════

template crd2idx*(layout: Layout; coord: IntOrIntTuple): auto =
  ## Logical-to-memory offset for a coordinate on a Layout.
  ##
  ## `coord` can be:
  ## - an `int`, decomposed column-major across all dimensions
  ## - a `tuple`, inner product `coord·stride` per dimension
  ## - a static `Int[V]`, same at compile time
  crd2idx(makeIntTuple(coord), layout.shape, layout.stride)

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
    var maxIdx = -1
    var maxV = 0
    var allStatic = true
    for i in 0 ..< stT.len:
      if stT[i].kind == nnkBracketExpr and $stT[i][0] == "Int":
        let v = stT[i][1].intVal
        if v > maxV:
          maxV = v
          maxIdx = i
      else:
        allStatic = false
    # for tuple shapes, the quotient runs unmod'd at the largest static stride
    var parts: seq[NimNode] = @[]
    for i in 0 ..< shT.len:
      let s = newCall(bindSym"[]", st, newLit(i))
      let shI = newCall(bindSym"[]", sh, newLit(i))
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
    let flat = flatLeaves(shape)
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
#  slice and dice, marker-based dimension selection
# ═══════════════════════════════════════════════════════════════

template slice*(target: tuple; selector: typed): auto =
  ## Slice a tuple, keep elements where the selector entry is X.
  ## Elements with a Y, int, or Int selector are dropped.
  filterZipWith(selector, target):
    (when it_a is X: (it_b,)
     elif it_a is Y or it_a is int or it_a is Int: ()
     else: {.error: "slice: selector items must be X, Y, or ints".})

template dice*(target: tuple; selector: typed): auto =
  ## Dice a tuple, keep elements where the selector entry is Y, int, or Int.
  ## Elements with an X selector are dropped.
  filterZipWith(selector, target):
    (when it_a is Y or it_a is int or it_a is Int: (it_b,)
     elif it_a is X: ()
     else: {.error: "dice: selector items must be X, Y, or ints".})

template slice*(target: Layout; selectors: varargs[untyped]): untyped =
  ## Extract a sub-Layout.
  ##
  ## Returns the layout keeping the dimensions marked X or _,
  ## dimensions marked Y, int, or Int are dropped.
  ##
  ## Accepts both varargs and a single tuple argument:
  ## - slice(L, X, Y)     two separate args
  ## - slice(L, (X, Y))   single tuple argument, equivalent
  block:
    evalOnceAs(t, target)
    make_layout(
      slice(t.shape, varargs_to_par(selectors)),
      slice(t.stride, varargs_to_par(selectors)))

template dice*(target: Layout; selectors: varargs[untyped]): untyped =
  ## Extract a sub-Layout.
  ##
  ## Returns the layout keeping the dimensions marked Y, int, or Int,
  ## dimensions marked X are dropped.
  ##
  ## Accepts both varargs and a single tuple argument:
  ## - dice(L, Y, X)     two separate args
  ## - dice(L, (Y, X))   single tuple argument, equivalent
  block:
    evalOnceAs(t, target)
    make_layout(
      dice(t.shape, varargs_to_par(selectors)),
      dice(t.stride, varargs_to_par(selectors)))

# ═══════════════════════════════════════════════════════════════
#  layout() call syntax
# ═══════════════════════════════════════════════════════════════

template hasUnderscoreImpl*(coord: typed): bool =
  when coord is tuple:
    block:
      var found = false
      for c in fields(coord):
        when c.hasUnderscore():
          found = true
      found
  elif coord is int:
    false
  elif coord is Int:
    false
  elif coord is X:
    true
  else:
    {.error: "[ttt] unsupported type: " & typeof(coord).}

macro hasUnderscore*(Cs: varargs[untyped]): bool =
  let r = ident"r"
  result = newStmtList()
  result.add quote do:
    var `r` = false
  for i in 0 ..< Cs.len:
    let Ci = Cs[i]
    result.add quote do:
      `r` = `r` or hasUnderscoreImpl(`Ci`)
  result.add quote do:
    `r`
  result = newBlockStmt(result)

template callImpl(layout: Layout; coord: typed): auto =
  ## Index a layout with a coordinate.
  ## - coord with _ or X → slice, returns a sub-Layout
  ## - all-int coord → crd2idx, returns an offset
  when hasUnderscore(coord):
    slice(layout, coord)
  else:
    crd2idx(layout, coord)

template `()`*(layout: Layout; args: varargs[typed]): auto =
  ## Multi-argument: `L(i, j)` ≡ `L((i, j))`.
  block:
    evalOnceAs(coord, varargs_to_par(args))
    when hasUnderscoreImpl(coord):
      slice(layout, coord)
    else:
      crd2idx(layout, coord)
