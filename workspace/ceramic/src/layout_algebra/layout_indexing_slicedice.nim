# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option.
# This file may not be copied, modified, or distributed except according to those terms.

import std/macros

import workspace/ceramic/src/int_tuples
import ./layouts
import ./layout_compiletime
import workspace/ceramic/src/macros/varargs_to_par

# ═══════════════════════════════════════════════════════════════
#  slice and dice markers
# ═══════════════════════════════════════════════════════════════

type
  X* = object  ## marker for free/slice dimensions, kept by slice, dropped by dice
  Y* = object  ## marker for dice dimensions, kept by dice, dropped by slice

const _* = X()  ## value-level marker for free/slice dimensions

# X marker arithmetic: X contributes 0 in inner products
# X*Int[V] returns Int[0] (not plain int) so compile-time constant folding
# preserves the Int type system. X*int is plain int for runtime values.
template `*`*(c: X, s: int): Int[0] = Int[0]()
template `*`*[V: static int](c: X, s: Int[V]): Int[0] = Int[0]()
template `*`*(s: int, c: X): Int[0] = Int[0]()
template `*`*[V: static int](s: Int[V], c: X): Int[0] = Int[0]()

# X coords contribute 0 to any crd2idx inner product, the marker
# arithmetic folds through the scalar leaf dispatch

# ═══════════════════════════════════════════════════════════════
#  slice and dice, marker-based dimension selection
# ═══════════════════════════════════════════════════════════════

macro slice*(target: tuple, selector: typed): untyped =
  ## Slice a tuple, keep elements where the selector entry is X,
  ## elements with an int or Int selector are dropped.
  ##
  ## The selector and target are weakly congruent:
  ## - a selector leaf facing a nested target mode selects the whole mode
  ## - one X keeps an entire subtree, one int drops it

  var builder = TupleBuilderNested.new(1)
  var selStream = selector.tupleStream()
  var tgtStream = target.tupleStream()
  while not selStream.done():
    let selEvent = selStream.next()
    case selEvent.kind
    of kOpen:
      let tgtEvent = tgtStream.next()
      if tgtEvent.kind != kOpen:
        error "slice: a selector mode cannot face a target leaf", selEvent.leafTy
      builder.append(selEvent)
    of kClose:
      discard tgtStream.next()
      builder.append(selEvent)
    of kLeaf:
      let raw = selEvent.leafTy
      # a literal selector leaf carries no type, it is an int by construction
      let selTy = if raw == nil: nil
                  elif raw.kind == nnkBracketExpr and raw[0].eqIdent("typeDesc"):
                    raw[1]
                  else: raw
      let isX = selTy != nil and selTy.hasType"X"
      let tgtEvent = tgtStream.next()
      if tgtEvent.kind == kLeaf:
        if isX:
          builder.append(tgtEvent)
        # int, Int and Y selectors drop the leaf
      else:
        # a selector leaf facing a nested target mode takes the whole mode,
        # the mode's kOpen was consumed as tgtEvent, replayed to open
        # the subtree only when the selector keeps it
        if isX:
          builder.append(tgtEvent)
        var depth = 1
        while true:
          let ev = tgtStream.next()
          if isX:
            builder.append(ev)
          case ev.kind
          of kOpen:
            depth.inc
          of kClose:
            depth.dec
            if depth == 0: break
          of kLeaf:
            discard
  return builder.emit(0)

macro dice*(target: tuple, selector: typed): untyped =
  ## Dice a tuple, keep elements where the selector entry is Y, int, or Int,
  ## elements with an X selector are dropped.
  ##
  ## The selector and target are weakly congruent:
  ## - a selector leaf facing a nested target mode selects the whole mode
  ## - one int keeps an entire subtree, one X drops it

  var builder = TupleBuilderNested.new(1)
  var selStream = selector.tupleStream()
  var tgtStream = target.tupleStream()
  while not selStream.done():
    let selEvent = selStream.next()
    case selEvent.kind
    of kOpen:
      let tgtEvent = tgtStream.next()
      if tgtEvent.kind != kOpen:
        error "dice: a selector mode cannot face a target leaf", selEvent.leafTy
      builder.append(selEvent)
    of kClose:
      discard tgtStream.next()
      builder.append(selEvent)
    of kLeaf:
      let raw = selEvent.leafTy
      # a literal selector leaf carries no type, it is an int by construction
      let selTy = if raw == nil: nil
                  elif raw.kind == nnkBracketExpr and raw[0].eqIdent("typeDesc"):
                    raw[1]
                  else: raw
      let isX = selTy != nil and selTy.hasType"X"
      let tgtEvent = tgtStream.next()
      if tgtEvent.kind == kLeaf:
        if not isX:
          builder.append(tgtEvent)
        # X selectors drop the leaf
      else:
        # a selector leaf facing a nested target mode takes the whole mode,
        # the mode's kOpen was consumed as tgtEvent, replayed to open
        # the subtree only when the selector keeps it
        if not isX:
          builder.append(tgtEvent)
        var depth = 1
        while true:
          let ev = tgtStream.next()
          if not isX:
            builder.append(ev)
          case ev.kind
          of kOpen:
            depth.inc
          of kClose:
            depth.dec
            if depth == 0: break
          of kLeaf:
            discard
  return builder.emit(0)

template slice*(target: Layout, selectors: varargs[untyped]): untyped =
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

template dice*(target: Layout, selectors: varargs[untyped]): untyped =
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

macro hasUnderscore*(Cs: varargs[typed]): bool =
  proc walkImpl(t: NimNode): bool {.compileTime.} =
    ## True if the type subtree contains the X marker
    let ty = t.getTypeInst()
    if ty.kind in [nnkTupleTy, nnkPar, nnkTupleConstr]:
      for i in 0 ..< ty.len:
        if walkImpl(ty[i]):
          return true
      return false
    if ty.kind == nnkBracketExpr:
      # Int[V] instances and friends: the marker never hides in them
      if ty[0].kind == nnkSym and ty[0].strVal == "Int":
        return false
      error "[ttt] unsupported type: " & ty.repr
    if ty.kind == nnkSym:
      if ty.strVal == "X":
        return true
      if ty.strVal == "int" or ty.strVal == "Int":
        return false
      let impl = ty.getImpl()
      if impl.kind == nnkTupleTy:
        for i in 0 ..< impl.len:
          if walkImpl(impl[i]):
            return true
        return false
    error "[ttt] unsupported type: " & ty.repr
  result = newLit(false)
  for i in 0 ..< Cs.len:
    result = newLit(result.boolVal or walkImpl(Cs[i]))

# ═══════════════════════════════════════════════════════════════
#  sliceAxes and diceAxes, reduction-axis selection by axis number
# ═══════════════════════════════════════════════════════════════

proc axesSelector(n: int, axes: seq[int], rest, axis: static string): NimNode =
  ## Per-mode selector: `axis` marker at the axes, `rest` marker elsewhere
  result = nnkPar.newTree()
  for i in 0 ..< n:
    var isAxis = false
    for a in axes:
      if a == i:
        isAxis = true
    if isAxis:
      result.add bindSym(axis)
    else:
      result.add bindSym(rest)

macro sliceAxes*(target: Layout, axes: varargs[int]{lit|`const`}): untyped =
  ## Layout keeping every mode except `axes`.
  ##
  ##   sliceAxes(make_layout((2, 3, 4), (6, 2, 1)), 1)  ==  (2, 4):(6, 1)
  var ax: seq[int]
  for a in axes:
    ax.add a.intVal.int
  let sel = axesSelector(target.rank(), ax, "X", "Y")
  result = bindSym"slice".newCall(target)
  for e in sel:
    result.add e

macro diceAxes*(target: Layout, axes: varargs[int]{lit|`const`}): untyped =
  ## Layout of only the `axes` modes.
  ##
  ##   diceAxes(make_layout((2, 3, 4), (6, 2, 1)), 1)  ==  (3,):(2,)
  var ax: seq[int]
  for a in axes:
    ax.add a.intVal.int
  let sel = axesSelector(target.rank(), ax, "X", "Y")
  result = bindSym"dice".newCall(target)
  for e in sel:
    result.add e

