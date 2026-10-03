## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, typetraits]
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/int_tuples
import ./layouts
import ./layouts_unsanctioned_helpers
import ./layout_compiletime
import ./layout_indexing_gpu

# ═══════════════════════════════════════════════════════════════
#  CoordWheel, iterate logical positions without div/mod
# ═══════════════════════════════════════════════════════════════

type CoordWheel[Rank: static int] = object
  coord: array[Rank, int]
  offset: int

func initCoordWheel(shape: IntOrIntTuple): auto {.inline.} =
  CoordWheel[shape.rank()]()

func seek(wheel: var CoordWheel, shape, strides: auto, target: int) {.inline.} =
  ## Position the wheel at linear index `target` by peeling:
  ## - each dimension's coordinate counts up by span subtraction,
  ##   the offset by stride adds, no div/mod
  ## - the highest-span dimension counts unbounded, out-of-range
  ##   excess stays visible there
  ## - a span is the product of the faster dimensions, O(span steps)
  ##   per dimension, small for in-range indexes
  var remaining = target
  staticForCountdown k, wheel.Rank - 1, 1:
    var span = 1
    staticFor j, 0, k:
      span *= toIntVal(shape[j])
    while remaining >= span:
      remaining -= span
      wheel.coord[k] += 1
      wheel.offset += toIntVal(strides[k])
  wheel.coord[0] = remaining
  when wheel.Rank == 1:
    wheel.offset += remaining * toIntVal(strides)
  else:
    wheel.offset += remaining * toIntVal(strides[0])

# ═══════════════════════════════════════════════════════════════
#  idx2crd over a Layout, wheel peeling
# ═══════════════════════════════════════════════════════════════

macro idx2crdCpuImpl(sh, st: typed, idx: int or Int): untyped =
  ## Wheel-peeling emit for the destructured shape/stride.
  let wheel = genSym(nskVar, "wheel")
  template wheelPeelEmit(initWheel, seek, wheel, shape, stride, index, tail: untyped): untyped =
    ## Wheel-peeling emit, the wheel and tail share one symbol.
    block:
      var wheel = initWheel(shape)
      seek(wheel, shape, stride, index)
      tail
  let shTy = sh.getTypeInst()
  let tail =
    if shTy.kind == nnkTupleConstr:
      var parts: seq[NimNode] = @[]
      for k in 0 ..< shTy.len:
        parts.add nnkBracketExpr.newTree(
          nnkDotExpr.newTree(wheel, ident"coord"), newLit(k))
      nnkPar.newTree(parts)
    else:
      nnkBracketExpr.newTree(
        nnkDotExpr.newTree(wheel, ident"coord"), newLit(0))
  result = getAst(wheelPeelEmit(
    bindSym"initCoordWheel", bindSym"seek", wheel, sh, st, idx, tail))

macro idx2crd_cpu*(layout: Layout, idx: int or Int): untyped =
  ## Say `layout` is the element order of a tensor in memory.
  ## Say `idx` is an element's slot in the backing buffer.
  ##
  ## `idx2crd_cpu` maps the slot to the coordinate of the element
  ## stored there, like `idx2crd`, one element per dimension,
  ## dimension 0 fastest.
  ##
  ## Returns the coordinate `c` with `crd2idx(layout, c) = idx`:
  ## - the wheel peels the index by span subtraction, no div/mod
  ## - the layout is compact and column-major, dimension 0 fastest:
  ##   consecutive slots hold consecutive elements, no gaps,
  ##   no repeats, stride-0 broadcasts excluded
  ## - the layout has a flat, one-level shape
  ##
  ## TODO: nested shape support, say ((2,3), 4)
  ## TODO: row-major layouts, the spans are stride-ordered products
  ##
  ## An `idx` larger than the tensor's size gives an out-of-range
  ## coordinate that still roundtrips through `crd2idx`.
  ##
  ## Example:
  ##   idx2crd_cpu(make_layout((4, 8), (1, 4)), 22)
  ##   # → (2, 5)
  var bindings = newStmtList()
  let (sh, st) = bindings.destructureLayout(layout)
  template idx2crdCpuDelegate(s2, t2, i2) =
    idx2crdCpuImpl(s2, t2, i2)
  result = getAst(idx2crdCpuDelegate(sh, st, idx))
  echo "DBG emit:\n", result.repr
  if bindings.len != 0:
    result = newTree(nnkBlockStmt, newEmptyNode(),
                     newTree(nnkStmtListExpr, bindings, result))
