## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU-optimized indexing: wheel-winding iteration (no divmod).
##
## Provides:
##   - CoordWheel[Rank] — iterate all logical positions of a layout
##     without divmod (O(1) amortized per step via carry-chain).
##   - crd2idx_cpu / idx2crd_cpu — wrappers for useGpuIndexing dispatch.
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

import ./int_tuples
import ./layouts

# ═══════════════════════════════════════════════════════════════
#  CoordWheel — iterate logical positions without divmod
# ═══════════════════════════════════════════════════════════════

type CoordWheel*[Rank: static int] = object
  ## Track a logical coordinate as it advances through a shape.
  ## Initialized to all zeros (first logical position).
  ## `incr` advances by one position via carry-chain (no divmod).
  coord*: array[Rank, int]

func initCoordWheel*[Rank: static int](_: typedesc[CoordWheel[Rank]]; shape: auto): CoordWheel[Rank] =
  ## Initialize wheel at (0, 0, ..., 0).
  discard
  CoordWheel[Rank](coord: default(array[Rank, int]))

func initCoordWheel*[Rank: static int](_: typedesc[CoordWheel[Rank]]): CoordWheel[Rank] =
  CoordWheel[Rank](coord: default(array[Rank, int]))

import ./macros/static_for
func incr*[Rank: static int](wheel: var CoordWheel[Rank]; shape: auto) =
  ## Advance coordinate by one logical position (carry-chain).
  ## Innermost dim (dim-0) is fastest-changing, so carry chain starts from dim-0.
  ## Tuples need staticFor (compile-time index), arrays support runtime.
  when shape is tuple:
    staticFor k, 0, Rank:
      if wheel.coord[k] < int(shape[k]) - 1:
        wheel.coord[k] += 1
        return
      else:
        wheel.coord[k] = 0
  else:
    for k in 0 ..< Rank:
      if wheel.coord[k] < int(shape[k]) - 1:
        wheel.coord[k] += 1
        return
      else:
        wheel.coord[k] = 0

import ./macros/static_for
func coordOffset*[Rank: static int](wheel: CoordWheel[Rank]; strides: auto): int =
  ## Compute linear offset = sum(coord[i] * stride[i]).
  ## Pure multiply-add, no divmod.
  ## Tuples need compile-time index (staticFor), arrays support runtime.
  when strides is tuple:
    staticFor i, 0, Rank:
      result += wheel.coord[i] * int(strides[i])
  else:
    for i in 0 ..< Rank:
      result += wheel.coord[i] * int(strides[i])

# ═══════════════════════════════════════════════════════════════
#  CPU wrappers (for useGpuIndexing dispatch)
# ═══════════════════════════════════════════════════════════════
#
#  These delegate to the kernel_indexing_gpu 3-arg functions for
#  tuple-coord crd2idx (which is already multiply-add, no divmod).
#  They exist so callers can do a uniform `crd2idx_cpu` call
#  regardless of whether the underlying impl differs.
#
#  Note: layouts.nim defines the main `crd2idx(layout, coord)`
#  dispatch with `useGpuIndexing` parameter. The `_cpu` suffix
#  here is for code that explicitly wants CPU-optimized semantics.

import ./kernel_indexing_gpu
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
  let lTyp = layout.getTypeInst()
  let shT = lTyp[1]
  let sh = newTree(nnkDotExpr, layout, ident"shape")
  let st = newTree(nnkDotExpr, layout, ident"stride")
  if shT.kind != nnkTupleConstr:
    result = newCall(bindSym"div", idx, st)
  else:
    var parts: seq[NimNode] = @[]
    for i in 0 ..< shT.len:
      let s = newCall(bindSym"[]", st, newLit(i))
      let shI = newCall(bindSym"[]", sh, newLit(i))
      parts.add newCall(bindSym"mod",
        newCall(bindSym"div", idx, s), shI)
    result = nnkPar.newTree(parts)
