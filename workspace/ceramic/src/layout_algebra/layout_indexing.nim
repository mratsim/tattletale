## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

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
#  crd2idx, coordinate to logical offset
# ═══════════════════════════════════════════════════════════════

macro crd2idxImpl*(coord, shape, stride: typed): untyped =
  ## One pass over the coord, shape, and stride streams
  ##
  ## Args:
  ## - coord, shape, stride: whole trees
  ## - c, sh, st: a leaf from (coord, shape, stride)
  ##
  ## Returns:
  ## - the offset expression
  ##
  ##   tuple coord, one contribution per leaf triple:
  ##     result += c * st
  ##
  ##   scalar coord, decomposed over the dim's leaves in order:
  ##     remaining = coord
  ##     for each leaf (sh, st) except the last:
  ##       result += (remaining mod sh) * st
  ##       remaining = remaining div sh
  ##     result += remaining * st_last

  var coordStream = TupleStream()
  if coord.isTupleTy():
    coordStream = coord.tupleStream()
  var shapeStream = shape.tupleStream()
  var strideStream = stride.tupleStream()
  var sum: NimNode = nil
  var remaining: NimNode = nil       # the quotient thread, decompose only
  var pending, pendingFinal: NimNode # the pending contribution, decompose only
  var decomposing = false
  var openDepth = -1        # the decompose subtree opens at this depth
  if not coord.isTupleTy():
    # A scalar coord decomposes over the whole layout, the root opens
    # and closes on the shape, the coord stream holds no leaves.
    remaining = coord
    openDepth = shapeStream.next().depth
    discard strideStream.next()
    decomposing = true
  while decomposing or not coordStream.done():
    if decomposing:
      let shapeEvent = shapeStream.next()
      let strideEvent = strideStream.next()
      case shapeEvent.kind
      of kLeaf:
        if strideEvent.kind != shapeEvent.kind:
          error "crd2idx: `shape` and `stride` have different structures", stride
        sum = if sum == nil: pending else: sum + pending
        let shapeLeaf = shapeEvent.leaf
        let strideLeaf = strideEvent.leaf
        let oldRemaining = remaining
        pending = oldRemaining mod shapeLeaf * strideLeaf
        pendingFinal = oldRemaining * strideLeaf
        remaining = oldRemaining div shapeLeaf
      of kClose:
        if shapeEvent.depth == openDepth:
          # the excess stays on the last leaf, it keeps the full quotient
          if pending != nil:
            sum = if sum == nil: pendingFinal
                  else: sum + pendingFinal
            pending = nil
          decomposing = false
      else:
        discard
      continue
    let coordEvent = coordStream.next()
    let shapeEvent = shapeStream.next()
    let strideEvent = strideStream.next()
    if coordEvent.kind == kLeaf and shapeEvent.kind == kOpen:
      # A scalar coord element into a nested dimension, the coord
      # stream holds still until the subtree closes.
      remaining = coordEvent.leaf
      openDepth = shapeEvent.depth
      decomposing = true
      continue
    if shapeEvent.kind != coordEvent.kind or strideEvent.kind != coordEvent.kind:
      error "crd2idx: `coord` and `shape` have different structures", shape
    if coordEvent.kind == kLeaf:
      let prod = coordEvent.leaf * strideEvent.leaf
      sum = if sum == nil: prod else: sum + prod
  if sum == nil:
    error "crd2idx: empty shape", shape
  return sum

macro crd2idx*(layout: Layout, coord: IntOrIntTuple): auto =
  ## Say `layout` is the element order of a tensor in memory.
  ## Say `coord` is a position inside the tensor's dimensions.
  ## `crd2idx` evaluates the layout: it maps the position to the slot
  ## of the element in the backing buffer, a linear offset.
  ##
  ## Returns the inner product of the coordinate with the strides,
  ## `sum(coord[k] * stride[k])` over the layout's dimensions.
  ##
  ## `coord` can be:
  ## - a `tuple`, one element per dimension, the inner product directly
  ## - an `int`, a flat position walked dimension by dimension,
  ##   dimension 0 fastest, `c = c mod sh; c = c div sh` per dimension
  ## - static `Int` values, the offset expression folds at compile time
  ##
  ##   Layout ──▶ destructureLayout ──▶ (shape, stride) ──┐
  ##   coord ─────────────────────────────────────────────┼──▶ crd2idxImpl ──▶ offset
  ##
  ## Example:
  ##   crd2idx(make_layout((4, 8), (1, 4)), (2, 5))
  ##   # → 22                       # 2·1 + 5·4
  ##   crd2idx(make_layout((4, 8), (1, 4)), 9)
  ##   # → 9                        # 9 = (1, 2), 1·1 + 2·4
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  template crd2idxDelegate(c2, sh2, st2) =
    crd2idxImpl(c2, sh2, st2)
  result.add getAst(crd2idxDelegate(coord, sh, st))

template idx2crd*(layout: Layout, idx: int or Int): untyped =
  idx2crd_gpu(layout, idx)

template idx2crd*(shape: IntOrIntTuple, idx: int or Int): untyped =
  idx2crd_gpu(shape, idx)

# ═══════════════════════════════════════════════════════════════
#  inBounds, coordinate validity against a shape
# ═══════════════════════════════════════════════════════════════

func inBoundsJoin(a, b: NimNode): NimNode {.compileTime.} =
  ## Joins one check into the accumulated check, nil is the empty accumulator
  if a == nil: b
  else: nnkInfix.newTree(bindSym"and", a, b)

func inBoundsLeaf(x, s: NimNode): NimNode {.compileTime.} =
  ## `0 <= x and x < s`, both bounds per leaf
  nnkInfix.newTree(bindSym"and", nnkInfix.newTree(bindSym"<=", newLit(0), x), nnkInfix.newTree(bindSym"<", x, s))

macro inBoundsImpl(coord, shape: typed): untyped =
  ## Returns true if a coordinate is within bounds of a shape

  var coordStream = TupleStream()
  if coord.isTupleTy():
    coordStream = coord.tupleStream()
  var shapeStream = shape.tupleStream()
  var ok: NimNode = nil
  var remaining: NimNode = nil       # the quotient thread, decompose only
  var pending, pendingFinal: NimNode # the pending check, decompose only
  var decomposing = false
  var openDepth = -1        # the decompose subtree opens at this depth
  if not coord.isTupleTy():
    # A scalar coord decomposes over the whole shape, the root opens
    # and closes on the shape, the coord stream holds no leaves.
    remaining = coord
    openDepth = shapeStream.next().depth
    decomposing = true
  while decomposing or not coordStream.done():
    if decomposing:
      let shapeEvent = shapeStream.next()
      case shapeEvent.kind
      of kLeaf:
        if pending != nil:
          ok = inBoundsJoin(ok, pending)
        let shapeLeaf = shapeEvent.leaf
        let oldRemaining = remaining
        pending = inBoundsLeaf(oldRemaining mod shapeLeaf, shapeLeaf)
        pendingFinal = inBoundsLeaf(oldRemaining, shapeLeaf)
        remaining = oldRemaining div shapeLeaf
      of kClose:
        if shapeEvent.depth == openDepth:
          # the excess stays on the last leaf, it keeps the full quotient
          ok = inBoundsJoin(ok, pendingFinal)
          pending = nil
          pendingFinal = nil
          decomposing = false
      else:
        discard
      continue
    let coordEvent = coordStream.next()
    let shapeEvent = shapeStream.next()
    if coordEvent.kind == kLeaf and shapeEvent.kind == kOpen:
      # A scalar coord element into a nested dimension, the coord
      # stream holds still until the subtree closes.
      remaining = coordEvent.leaf
      openDepth = shapeEvent.depth
      pending = nil
      pendingFinal = nil
      decomposing = true
      continue
    if shapeEvent.kind != coordEvent.kind:
      error "inBounds: `coord` and `shape` have different structures", shape
    if coordEvent.kind == kLeaf:
      ok = inBoundsJoin(ok, inBoundsLeaf(coordEvent.leaf, shapeEvent.leaf))
  if ok == nil:
    error "inBounds: empty shape", shape
  return ok

template inBounds*(coord: int or Int, shape: IntOrIntTuple): auto =
  ## Returns true if a coordinate is within bounds of a shape
  inBoundsImpl(coord, shape)

template inBounds*(coord: tuple, shape: IntOrIntTuple): auto =
  ## Returns true if a coordinate is within bounds of a shape
  inBoundsImpl(coord, shape)

macro inBounds*(coord: IntOrIntTuple, layout: Layout): auto =
  ## Returns true if a coordinate is within bounds of a shape
  result = newStmtList()
  let (sh, _) = result.destructureLayout(layout)
  template inBoundsDelegate(c2, sh2) =
    inBoundsImpl(c2, sh2)
  result.add getAst(inBoundsDelegate(coord, sh))

# ═══════════════════════════════════════════════════════════════
#  layout() call syntax
# ═══════════════════════════════════════════════════════════════

template `()`*(layout: Layout, args: varargs[typed]): auto =
  ## Multi-argument: `L(i, j)` ≡ `L((i, j))`.
  block:
    evalOnceAs(coord, varargs_to_par(args))
    when hasUnderscoreImpl(coord):
      slice(layout, coord)
    else:
      crd2idx(layout, coord)
