## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-suitable indexing: idx2crd (idx→coord), div/mod based.

import std/[macros, sequtils]

import workspace/ceramic/src/int_tuples
import ./layouts
import ./layouts_unsanctioned_helpers
import ./layout_compiletime
import ./layout_indexing_slicedice

# ═══════════════════════════════════════════════════════════════
#  idx2crd, index to coordinate decomposition
# ═══════════════════════════════════════════════════════════════

macro idx2crdGpuImpl(sh, st: typed, idx: int or Int): untyped =
  ## Index-to-coordinate emit for the destructured shape/stride.
  if not sh.getTypeInst().isTupleTy():
    # scalar shape, the broadcast Int[1] case maps to 0
    let shTy = sh.getTypeInst()
    result = if isStaticOne(shTy):
      IntCT(0)
    else:
      idx div st
  else:
    # the leaves, the stream's `leaf` is the bracket expression on the param
    var shDims, shTys, stDims: seq[NimNode] = @[]
    var stVals: seq[int] = @[]
    for shEv in sh.tupleStream():
      if shEv.kind == kLeaf:
        shDims.add shEv.leaf
        shTys.add shEv.leafTy
    for stEv in st.tupleStream():
      if stEv.kind == kLeaf:
        stDims.add stEv.leaf
        stVals.add stEv.leafTy.getStaticInt()
    # most-significant leaf = the largest stride, identifiable only
    # when every stride is static, so dynamic strides keep the mod
    let maxIdx = stVals.maxIndex
    let allStatic = DynamicSentinel notin stVals
    # for tuple shapes, the quotient runs unmod'd at the largest static stride
    var parts: seq[NimNode] = @[]
    for i in 0 ..< shDims.len:
      if isStaticOne(shTys[i]):
        # broadcast dimension, the coordinate is 0 before any division
        parts.add IntCT(0)
      elif allStatic and i == maxIdx:
        parts.add idx div stDims[i]
      else:
        parts.add (idx div stDims[i]) mod shDims[i]
    result = nnkPar.newTree(parts)

macro idx2crd_gpu*(layout: Layout, idx: int or Int): untyped =
  ## Say `layout` is the element order of a tensor in memory.
  ## Say `idx` is an element's slot in the backing buffer.
  ## `idx2crd` reverses `crd2idx`: it maps the slot to the tensor's
  ## coordinate of the element stored there.
  ##
  ## Returns the coordinate `c` with `crd2idx(layout, c) = idx`,
  ## `(idx div stride) mod shape` per dimension.
  ##
  ##        idx ──┬─▶ (idx div st) mod sh ──▶ every dimension's coordinate
  ##              └─▶ idx div st, no mod ──▶ the largest-stride dimension
  ##
  ## A coordinate exists only when the layout is compact:
  ## - the tensor is stored like a plain dense array, consecutive
  ##   elements at consecutive slots, no gaps and no repeats
  ## - column-major, dimension 0 fastest, and row-major, last
  ##   dimension fastest, are both compact
  ## - a strided or broadcast layout is not
  ## - for non-compact layouts use `idx2crd(shape, idx)` on the shapes
  ##
  ## For an `idx` larger than the tensor's size, the coordinate is out of range:
  ## - every dimension's coordinate wraps at its size via `mod`, except
  ##   the dimension with the largest stride: its coordinate is
  ##   `idx div stride` with no `mod`, so it can exceed the size
  ## - so `crd2idx` of the result gives `idx` back, even out of range
  ## - that no-`mod` behavior needs compile-time constant strides:
  ##   with a runtime stride every dimension takes the `mod`, `crd2idx` inverts in range only
  ##
  ## - a static shape-1 dimension maps to 0 before the division, keeping
  ##   a stride-0 broadcast dimension out of the arithmetic
  ##
  ## Example, the size is 32 and `idx = 35`:
  ##   idx2crd(make_layout((4, 8), (1, 4)), 35)
  ##   # → (3, 8)          # dimension 1 keeps 35 div 4 = 8, beyond its size
  ##
  ## Roundtrip after the overrun:
  ##   crd2idx(make_layout((4, 8), (1, 4)), (3, 8))
  ##   # → 35
  ##
  ## Layouts must have a flat, one-level shape.
  ## TODO: nested shape support, say ((2,3), 4).
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  template idx2crdGpuDelegate(s2, t2, i2) =
    idx2crdGpuImpl(s2, t2, i2)
  result.add getAst(idx2crdGpuDelegate(sh, st, idx))


# ═══════════════════════════════════════════════════════════════

macro idx2crd_gpu*(shape: IntOrIntTuple, idx: int or Int): untyped =
  ## Say a tensor packs its elements one after another, with dimension 0
  ## varying fastest, and `idx` points into that packing.
  ## `idx2crd` reverses the packing: it maps the linear position
  ## to the coordinate of the element stored there.
  ##
  ## Returns the coordinate tuple:
  ## - `idx mod sh` per dimension, the integer division `idx div sh`
  ##   carried to the next dimension, dimension 0 fastest
  ## - an `idx` larger than the tensor's size still gives a coordinate:
  ##   the last dimension's coordinate is the plain `idx div sh`, no
  ##   `mod`, so it can exceed the dimension's size
  ## - so `crd2idx` of the result gives `idx` back, even out of range
  ## - a scalar shape is the degenerate case, the coordinate is the index
  ##
  ## Only the sizes matter, so this works for any shape, compact or not.
  ##
  ## Examples:
  ##
  ##   idx2crd((4, 8), 31)            == (3, 7)   # 31 = 3 + 4·7
  ##   idx2crd(((4, 8), (2, 2)), 31)  == ((3, 7), (0, 0))
  ##   idx2crd((3, 7, 2), 42)         == (0, 0, 2)  # excess on the last dimension
  let shTy = shape.getTypeInst()
  if shTy.kind in {nnkTupleTy, nnkTupleConstr}:
    # first pass, the shape's leaves in order
    var sizes: seq[NimNode] = @[]
    var stream = shape.tupleStream()
    while not stream.done():
      let ev = stream.next()
      if ev.kind == kLeaf:
        sizes.add ev.leaf
    # the parts, dimension 0 fastest, the last leaf keeps
    # the plain quotient with no mod
    var parts: seq[NimNode] = @[]
    var q = idx
    for k in 0 ..< sizes.len - 1:
      parts.add q mod sizes[k]
      q = q div sizes[k]
    parts.add q
    # second pass, the parts re-nest over the shape's tree
    var stack: seq[NimNode] = @[]
    var i = 0
    stream = shape.tupleStream()
    while not stream.done():
      let ev = stream.next()
      case ev.kind
      of kOpen:
        stack.add nnkPar.newTree()
      of kLeaf:
        stack[^1].add parts[i]
        inc i
      of kClose:
        let done = stack.pop()
        if stack.len == 0:
          result = done
        else:
          stack[^1].add done
  else:
    result = idx

