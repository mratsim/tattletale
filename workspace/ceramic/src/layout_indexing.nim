## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout indexing: crd2idx, idx2crd, slice, dice.
##
## These are the Layout-consuming wrappers. The raw 3-arg crd2idx
## overloads live in `layout_indexing_gpu.nim`.
{.experimental: "callOperator".}

import std/macros
import std/typetraits

import ./int_tuples
import ./layout_indexing_gpu
import ./layouts
import ./macros/varargs_to_par

export layout_indexing_gpu

# ═══════════════════════════════════════════════════════════════
#  Markers for slice and dice
# ═══════════════════════════════════════════════════════════════

type
  X* = object  ## slice: keep this dimension, dice: drop this dimension
  Y* = object  ## dice: keep this dimension, slice: drop this dimension

const _* = X()  ## value-level marker for free/slice dimensions

# X marker arithmetic: X contributes 0 in inner products
# X*Int[V] returns Int[0] (not plain int) so compile-time constant folding
# preserves the Int type system. X*int is plain int for runtime values.
template `*`*(c: X; s: int): Int[0] = Int[0]()
template `*`*[V: static int](c: X; s: Int[V]): Int[0] = Int[0]()
template `*`*(s: int; c: X): Int[0] = Int[0]()
template `*`*[V: static int](s: Int[V]; c: X): Int[0] = Int[0]()

template makeIntTupleLeaf*(leaf: X): X =
  leaf

template mapLeavesWith*(singleton: X, body: untyped): X =
  singleton

# ═══════════════════════════════════════════════════════════════
#  crd2idx / idx2crd — via layout_indexing_gpu
# ═══════════════════════════════════════════════════════════════
#
#  The raw 3-arg crd2idx overloads live in layout_indexing_gpu.nim.
#  These Layout-consuming wrappers delegate to them.

template crd2idx*(layout: Layout; coord: IntOrIntTuple): auto =
  ## Logical-to-memory offset for a coordinate on a Layout.
  ##
  ## `coord` can be:
  ##   • an `int`   — decomposed column-major across all modes
  ##   • a `tuple`  — inner product `coord·stride` per mode
  ##   • a static `Int[V]` — same, compile-time constant
  ##
  ## External code must use this (or `layout(coord)`) rather than
  ## calling the raw `crd2idx(coord, shape, stride)` directly,
  ## which is module-private to layouts.nim.
  crd2idx(makeIntTuple(coord), layout.shape, layout.stride)

macro idx2crd*(layout: Layout; idx: int or Int): untyped =
  ## Convert linear index to coordinate using a Layout.
  ##
  ##   Cases for `(idx, shape, stride)`:
  ##     shape == 1, stride == 0   →   0  (broadcast — skip division)
  ##     shape == 1, stride != 0   →   0  (size-1 — result always 0)
  ##     shape != 1, stride == 0   ─── invalid layout (unreachable)
  ##     shape != 1, stride != 0   →   (idx div stride) mod shape
  ##
  ## The guard on `shape == 1` matches CuTe's `is_constant<1, Shape>`
  ## check and handles both broadcast modes and trivial dimensions,
  ## avoiding potential division-by-zero on stride-0.
  let lTyp = layout.getTypeInst()
  let shT = lTyp[1]
  let sh = newTree(nnkDotExpr, layout, ident"shape")
  let st = newTree(nnkDotExpr, layout, ident"stride")
  if shT.kind != nnkTupleConstr:
    # Scalar shape: `if shape == 1: 0 else: idx div stride`
    result = quote do:
      (if `sh` == 1: 0 else: `idx` div `st`)
  else:
    # Tuple shape: each mode gets its own guard
    var parts: seq[NimNode] = @[]
    for i in 0 ..< shT.len:
      let s = newCall(bindSym"[]", st, newLit(i))
      let shI = newCall(bindSym"[]", sh, newLit(i))

      parts.add quote do:
        (if (when `shI` is Int: `shI` === Int[1]() else: `shI` == 1): 0 else: (`idx` div `s`) mod `shI`)
    result = nnkPar.newTree(parts)

# ═══════════════════════════════════════════════════════════════
#  Slice and dice — marker-based dimension selection
# ═══════════════════════════════════════════════════════════════

template slice*(target: tuple; selector: typed): auto =
  ## Slice a tuple: keep elements where selector entry is X; drop where it's Y, int, or Int.
  filterZipWith(selector, target):
    (when it_a is X: (it_b,)
     elif it_a is Y or it_a is int or it_a is Int: ()
     else: {.error: "slice: selector items must be X, Y, or ints".})

template dice*(target: tuple; selector: typed): auto =
  ## Dice a tuple: keep elements where selector entry is Y, int, or Int; drop where it's X.
  filterZipWith(selector, target):
    (when it_a is Y or it_a is int or it_a is Int: (it_b,)
     elif it_a is X: ()
     else: {.error: "dice: selector items must be X, Y, or ints".})

template slice*(target: Layout; selectors: varargs[untyped]): untyped =
  ## Extract a sub-Layout: dimensions marked with X / _ are kept; Y, int, Int are dropped.
  ## Accepts both varargs and a single tuple argument:
  ##   slice(L, X, Y)          — two separate args
  ##   slice(L, (X, Y))        — single tuple arg (equivalent)
  block:
    evalOnceAs(t, target)
    make_layout(
      slice(t.shape, varargs_to_par(selectors)),
      slice(t.stride, varargs_to_par(selectors)))

template dice*(target: Layout; selectors: varargs[untyped]): untyped =
  ## Extract a sub-Layout: dimensions marked with Y / int / Int are kept, X are dropped.
  ## Accepts both varargs and a single tuple argument:
  ##   dice(L, Y, X)          — two separate args
  ##   dice(L, (Y, X))        — single tuple arg (equivalent)
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
  ## Layout indexing:
  ##   • coord has _ / X → slice (returns sub-Layout)
  ##   • coord is all ints → crd2idx (returns int)
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