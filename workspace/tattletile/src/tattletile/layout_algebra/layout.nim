# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layouts: a shape tree and a congruent stride tree mapping coordinates to offsets.
##
## A Layout evaluates as `L(c) = innerProduct(idx2crd(c, shape), stride)`: every coordinate form reaches the same value.
##
## - the default stride is the compact column-major prefix product of the shape
## - value type, usable in the Nim VM and at runtime
##
## Ported from the frozen python reference implementation (vendored pycute).
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/b4 \
##     --outdir:build/b4 workspace/tattletile/tests/test_layout_algebra_conformance.nim

import ./htuple

type
  Layout* = object
    ## A map from a coordinate domain to a codomain, defined by a shape tree
    ## plus a congruent stride tree.
    ##
    ## A rank-1 layout may carry a scalar shape like `8`, matching the reference implementation.
    shape*: IntTuple
    stride*: IntTuple

# ═══════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════

proc layoutSet*(shape, stride: IntTuple): Layout {.inline.} =
  ## A layout from an already-computed shape and stride, no completion step
  Layout(shape: shape, stride: stride)

proc makeLayout*(shape: IntTuple, stride: IntTuple): Layout =
  ## A layout whose stride is `stride` completed to congruence with `shape` via
  ## the exclusive prefix product, every missing stride continuing from its base.
  ##
  ## Examples: `makeLayout(it(4, 8))` is the compact column-major `(4, 8):(1, 4)`.
  layoutSet(shape, prefixProductT(shape, stride))

proc makeLayout*(shape: IntTuple): Layout =
  ## Compact column-major layout of `shape`, stride base 1
  makeLayout(shape, itv(1))

proc itupleNode*(xs: seq[int]): IntTuple =
  ## A tuple node of integer leaves: `itupleNode(@[4, 2])` is `(4, 2)`.
  ## The empty sequence gives the empty tuple.
  var cs = newSeq[IntTuple](xs.len)
  for i in 0 ..< xs.len:
    cs[i] = itv(xs[i])
  itNode(cs)

proc makeLayout*(shape, stride: seq[int]): Layout =
  ## Layout from flat integer shape and stride lists, one mode per entry
  makeLayout(itupleNode(shape), itupleNode(stride))

proc makeLayout*(shape: seq[int]): Layout =
  ## Compact column-major layout of a flat integer shape list
  makeLayout(itupleNode(shape), itv(1))

# ═══════════════════════════════════════════════════════════════════════
# Analysis: size, rank, cosize, modes
# ═══════════════════════════════════════════════════════════════════════

proc size*(L: Layout): int {.inline.} =
  ## Number of integral coordinates in the layout's domain, the product of the shape leaves.
  getSizeT(L.shape)

proc rank*(L: Layout): int {.inline.} =
  ## Number of top-level modes: 1 for a scalar shape
  if L.shape.kind == tkNode:
    L.shape.children.len
  else:
    1

proc cosize*(L: Layout): IntTuple =
  ## Size of the layout's codomain, one past every offset the layout produces:
  ## the sum of each mode's maximal offset `(s - 1) * d`, plus one.
  ##
  ## - an integer codomain gives an integer leaf
  ## - a codomain over coordinate axes Z^S (ArithTuple strides) gives a tree
  ##   with one bound per axis
  ##
  ## Example: `cosize(makeLayout(it(4, 8), it(eUnit(@[0]), eUnit(@[1]))))` is
  ## `(4, 8)`.
  var acc = ssInt(0)
  for (s, d) in zipLeavesT(L.shape, L.stride):
    if s.leaf.kind != skInt:
      raise layoutError("cosize: non-integer shape leaf")
    acc = ssAdd(acc, ssMul(d.leaf, s.leaf.intVal - 1))
  if acc.kind == skInt:
    return itv(acc.intVal + 1)
  # zero extension folds the +1 in at every axis of the ArithTuple codomain
  proc onesLike(d: StrideScalar): StrideScalar =
    if d.kind == skInt:
      return ssInt(1)
    var cs = newSeq[StrideScalar](d.data.len)
    for i in 0 ..< d.data.len:
      cs[i] = onesLike(d.data[i])
    ssAtup(cs)
  result = asTupleT(ssAdd(acc, onesLike(acc)))

proc modes*(L: Layout): seq[Layout] =
  ## Top-level modes of `L` as sublayouts: mode `i` has shape child `i` plus
  ## stride child `i`. A scalar shape wraps to a single mode.
  let sh = wrapT(L.shape)
  let st = wrapT(L.stride)
  result = newSeq[Layout](sh.children.len)
  for i in 0 ..< sh.children.len:
    result[i] = layoutSet(sh.children[i], st.children[i])

proc getMode*(L: Layout, path: seq[int]): Layout =
  ## Sublayout of `L` at `path`, one mode index per step, negatives counting from the end.
  ## Example: `getMode(L, @[1])` is the second mode of `L`
  if path.len == 0:
    return L
  let sh = getModeT(wrapT(L.shape), @[path[0]])
  let st = getModeT(wrapT(L.stride), @[path[0]])
  let sub = layoutSet(sh, st)
  if path.len == 1:
    return sub
  getMode(sub, path[1 .. ^1])

# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

proc call*(L: Layout, crd: IntTuple): StrideScalar =
  ## Layout evaluated at a coordinate in any form (integral, flat, natural).
  ## Examples: `call(L, it(2, 3)) == call(L, itv(14))` for `(4, 8):(8, 1)`.
  ## Integer coordinates decompose colexicographically against the shape.
  innerProductT(idx2crdT(crd, L.shape), L.stride)

proc call*(L: Layout, i: int): StrideScalar =
  ## Layout evaluated at the flat coordinate `i`
  call(L, itv(i))

proc callInt*(L: Layout, crd: IntTuple): int =
  ## Layout evaluated at a coordinate whose codomain is the integers.
  ##
  ## Raises:
  ## - LayoutError when the stride has a coordinate-axis element
  let v = call(L, crd)
  if v.kind != skInt:
    raise layoutError("callInt: non-integer offset " & ssStr(v))
  v.intVal

proc callInt*(L: Layout, i: int): int =
  ## Layout evaluated at the flat coordinate `i`, integer codomain only.
  ##
  ## Raises:
  ## - LayoutError when the stride has a coordinate-axis element
  callInt(L, itv(i))

# ═══════════════════════════════════════════════════════════════════════
# Equality and printing
# ═══════════════════════════════════════════════════════════════════════

proc `==`*(a, b: Layout): bool =
  ## Two layouts are equal iff their shapes and strides are equal trees
  eqTupleT(a.shape, b.shape) and eqTupleT(a.stride, b.stride)

proc `$`*(L: Layout): string =
  ## Constructor form matching the python reference implementation:
  ## `Layout((4, 8), (1, 4))`, with coordinate-axis strides in basis form like
  ## `Layout((8, 8), (1@0, 1@1))`
  "Layout(" & strT(L.shape) & ", " & strT(L.stride) & ")"

proc strForm*(L: Layout): string =
  ## Compact `shape:stride` form matching the python reference implementation:
  ## `(4, 8):(1, 4)`
  strT(L.shape) & ":" & strT(L.stride)
