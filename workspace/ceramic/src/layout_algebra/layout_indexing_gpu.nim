## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option.
## This file may not be copied, modified, or distributed except according to those terms.

## GPU-suitable indexing: crd2idx (coord→idx), div/mul/add only,
## and idx2crd (idx→coord), div/mod based.

import std/[macros, typetraits]

import workspace/ceramic/src/int_tuples
import ./layouts
import ./layout_compiletime
import ./layout_indexing_slicedice

# ═══════════════════════════════════════════════════════════════
#  crd2idx over a Layout, one stream pass, no recursion
# ═══════════════════════════════════════════════════════════════

macro crd2idxWalk*(coord, shape, stride: typed): untyped =
  ## One pass over the coord, shape, and stride streams
  ##
  ## c, sh, st a leaf from (coord, shape, stride)
  ##
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
  if coord.getTypeInst().isTupleTy():
    coordStream = coord.tupleStream()
  var shapeStream = shape.tupleStream()
  var strideStream = stride.tupleStream()
  var sum: NimNode = nil
  var remaining: NimNode = nil       # the quotient thread, decompose only
  var pending, pendingFinal: NimNode # the pending contribution, decompose only
  var decomposing = false
  var openDepth = -1        # the decompose subtree opens at this depth
  if not coord.getTypeInst().isTupleTy():
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
  result = sum

macro crd2idx_gpu*(layout: Layout; coord: IntOrIntTuple): auto =
  ## Logical-to-memory offset for a coordinate on a Layout.
  ##
  ## `coord` can be:
  ## - an `int`, decomposed column-major across all dimensions
  ## - a `tuple`, inner product `coord·stride` per dimension
  ## - a static `Int[V]`, same at compile time
  ##
  ##   Layout ──▶ destructureLayout ──▶ (shape, stride) ──┐
  ##   coord (runtime leaves) ──▶ let cc = coord ─────────┼──▶ walk ──▶ offset
  ##   coord (literal/static leaves) ──▶ as-is ───────────┘
  let (sh, st) = destructureLayout(result, layout)
  template crd2idxDelegate(c2, sh2, st2) =
    crd2idxWalk(c2, sh2, st2)
  result = getAst(crd2idxDelegate(coord, sh, st))

macro crd2idx_cpu*(layout: Layout; coord: IntOrIntTuple): auto =
  ## CPU-suffixed crd2idx: multiply-add only, no div/mod,
  ## identical to the GPU path.
  result = getAst(crd2idx_gpu(layout, coord))

# ═══════════════════════════════════════════════════════════════
#  idx2crd, index to coordinate decomposition
# ═══════════════════════════════════════════════════════════════
