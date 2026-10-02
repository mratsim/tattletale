## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU-optimized indexing: wheel-winding iteration (no divmod).
##
## Provides:
##   - CoordWheel[Rank], iterate all logical positions of a layout
##     without divmod, O(1) amortized per step via carry-chain
##   - crd2idx_cpu / idx2crd_cpu, wrappers for useGpuIndexing dispatch
##
## For random-access idx2crd (single flat index → coordinate), there is
## no way around divmod. The wheel-winding only benefits sequential
## iteration over ALL elements.
##
## The CoordWheel can be used directly for custom scans:
##
##   var wheel = initCoordWheel(CoordWheel[2], shape)
##   for _ in 0 ..< totalElements:
##     let off = wheel.coordOffset(strides)
##     ... use off ...
##     wheel.incr(shape)

import workspace/ceramic/src/int_tuples
import ./layouts
import ./layouts_unsanctioned_helpers
import ./layout_compiletime

# ═══════════════════════════════════════════════════════════════
#  CoordWheel, iterate logical positions without divmod
# ═══════════════════════════════════════════════════════════════

type CoordWheel*[Rank: static int] = object
  ## Track a logical coordinate as it advances through a shape.
  ## Initialized to all zeros (first logical position).
  ## `incr` advances by one position via carry-chain (no divmod).
  coord*: array[Rank, int]

func initCoordWheel*[Rank: static int](_: typedesc[CoordWheel[Rank]]): CoordWheel[Rank] =
  ## Wheel at (0, 0, ..., 0), the first logical position.
  CoordWheel[Rank](coord: default(array[Rank, int]))

import workspace/ceramic/src/macros/static_for
func incr*[Rank: static int](wheel: var CoordWheel[Rank]; shape: auto) =
  ## Advance the coordinate by one logical position (carry-chain).
  ##
  ## - in-range carry, the coordinate never reaches the shape
  ## - dim-0 is fastest-changing
  ##
  ## staticFor unrolls the walk, indexing tuples and arrays alike.
  staticFor k, 0, Rank:
    if wheel.coord[k] < int(shape[k]) - 1:
      wheel.coord[k] += 1
      return
    else:
      wheel.coord[k] = 0

func coordOffset*[Rank: static int](wheel: CoordWheel[Rank]; strides: auto): int =
  ## Linear offset = sum(coord[i] * stride[i]), pure multiply-add.
  staticFor i, 0, Rank:
    result += wheel.coord[i] * int(strides[i])

# ═══════════════════════════════════════════════════════════════
#  CPU wrappers, dispatch targets for useGpuIndexing
# ═══════════════════════════════════════════════════════════════

import ./layout_indexing_gpu
import std/macros

func crd2idx_cpu*(layout: Layout; coord: IntOrIntTuple): int {.inline, noInit.} =
  ## CPU-suffixed crd2idx: delegates to the same multiply-add 3-arg.
  ## For tuple coords this is identical to GPU path (no divmod).
  crd2idx(coord, layout.shape, layout.stride)

macro idx2crd_cpu*(layout: Layout; idx: int or Int): untyped =
  ## CPU-suffixed idx2crd: uses same divmod approach.
  ## No wheel-winding alternative for random access.
  ##
  ## TODO: nested shape support (e.g. ((2,3), 4)).
  ##       Currently only handles flat tuple shapes.
  let shT = layoutTypeArgs(layout).shapeTy
  let sh = newTree(nnkDotExpr, layout, ident"shape")
  let st = newTree(nnkDotExpr, layout, ident"stride")
  if shT.kind != nnkTupleConstr:
    result = newCall(bindSym"div", idx, st)
  else:
    var parts: seq[NimNode] = @[]
    for i in 0 ..< shT.len:
      let s = bindSym"[]".newCall(st, newLit(i))
      let shI = bindSym"[]".newCall(sh, newLit(i))
      parts.add newCall(bindSym"mod",
        newCall(bindSym"div", idx, s), shI)
    result = nnkPar.newTree(parts)
