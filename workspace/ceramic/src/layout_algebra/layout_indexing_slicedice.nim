## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option.
## This file may not be copied, modified, or distributed except according to those terms.

## Slice and dice: the X/Y marker sub-language and the dimension
## selection macros built on it.

import std/macros

import workspace/ceramic/src/int_tuples
import ./layouts
import workspace/ceramic/src/macros/varargs_to_par

# ═══════════════════════════════════════════════════════════════
#  slice and dice markers
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

# X coords contribute 0 to any crd2idx inner product, the marker
# arithmetic folds through the scalar leaf dispatch

# ═══════════════════════════════════════════════════════════════
#  slice and dice, marker-based dimension selection
# ═══════════════════════════════════════════════════════════════

macro slice*(target: tuple; selector: typed): untyped =
  ## Slice a tuple, keep elements where the selector entry is X.
  ## Elements with a Y, int, or Int selector are dropped.

  let ty = selector.getTypeInst() # The selector is a type or a tuple of types
  let sel = if ty.kind == nnkBracketExpr: ty[1]
            else: selector
  var builder = TupleBuilderNested.new(1)
  for (selEvent, tgtEvent) in sel.tupleStream().zip(target.tupleStream()):
    if selEvent.kind != tgtEvent.kind:
      error "slice: the selector and the target have different structures", selEvent.leaf
    tgtEvent.onLeaves():
      let raw = selEvent.leafTy
      let selTy = if raw.kind == nnkBracketExpr and raw[0].eqIdent("typeDesc"):
                    raw[1]
                  else: raw
      if selTy.hasType"X":
        builder.append(tgtEvent)
      elif selTy.hasType"Y" or selTy.hasType"int" or (selTy.kind == nnkBracketExpr and selTy[0].hasType"Int"):
        discard
      else:
        error "slice: selector items must be X, Y, or ints", selTy
  result = builder.emit(0).resultTuple

macro dice*(target: tuple; selector: typed): untyped =
  ## Dice a tuple, keep elements where the selector entry is Y, int, or Int.
  ## Elements with an X selector are dropped.

  let ty = selector.getTypeInst() # The selector is a type or a tuple of types
  let sel = if ty.kind == nnkBracketExpr: ty[1]
            else: selector
  var builder = TupleBuilderNested.new(1)
  for (selEvent, tgtEvent) in sel.tupleStream().zip(target.tupleStream()):
    if selEvent.kind != tgtEvent.kind:
      error "slice: the selector and the target have different structures", selEvent.leaf
    tgtEvent.onLeaves():
      let raw = selEvent.leafTy
      let selTy = if raw.kind == nnkBracketExpr and raw[0].eqIdent("typeDesc"):
                    raw[1]
                  else: raw
      if selTy.hasType"Y" or selTy.hasType"int" or (selTy.kind == nnkBracketExpr and selTy[0].hasType"Int"):
        builder.append(tgtEvent)
      elif selTy.hasType"X":
        discard
      else:
        error "dice: selector items must be X, Y, or ints", selTy
  result = builder.emit(0).resultTuple

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
