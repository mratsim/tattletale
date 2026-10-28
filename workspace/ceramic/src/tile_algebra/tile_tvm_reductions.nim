# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, typetraits]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_compiletime
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/macros/static_for
import ./tiles
import ./tile_matmul

# ############################################################
#
#        Axis reductions through a thread->value map
#
# ############################################################
#
# Tensor cores require specific threads in a warp to own specific part of a tile.
# This significantly complicate reductions done prior to tensor cores Matrix-Multiply-Accumulate (MMA)
# such as within attention kernels.
#
# This file provides a generic implementation of axis reductions
# for n-dimensional tile satisfying arbitrary Thread-Value ownership requirements.
#
# Terms:
#
# lane
#   One thread of the warp.
#   The map's thread coordinates decide which register slots a lane owns.
#   Those slots are the lane's private slice of the tile.
#
# xor shuffle
#   A hardware register exchange between the lanes of a warp: simd_shuffle_xor on Metal, __shfl_xor_sync on CUDA.
#   Lane L receives the value of lane L xor mask, and its own value travels to lane L xor mask.
#   The exchange takes one instruction and no memory traffic.
#
# butterfly
#   A lane reduce built from xor shuffles.
#   Pass p exchanges values with mask 2^p, so the lane distance halves every pass.
#   Four lanes reducing four values v0, v1, v2, v3:
#
#     lanes        0      1      2      3
#     start       v0     v1     v2     v3
#     mask 1     v0+v1  v0+v1  v2+v3  v2+v3
#     mask 2     v0+v1+v2+v3     (every lane)
#
#   After the last pass every lane ends with the total over all lanes.
#
# kept coordinate
#   A coordinate along the axes that survive the reduce.
#   Reducing axis 1 of a (rows, cols) tile keeps the row coordinates.
#
# collapsed stats tile
#   A destination tile with exactly one value per kept coordinate per lane: the reduce axis is gone.
#
# keepdims register tile
#   A destination tile shaped like the register tile: every slot stores the total of its kept coordinate.
#
# Dataflow of one reduce with sum as example:
#
#   register tile src (the lane's slice, placed by the map)
#      │
#      │ fold: each lane combines the slots that share a kept coordinate
#      ▼
#   partials[k] = Σₐ src.data[slot at kept rank k, axis rank a]
#      │
#      │ butterfly: one xor-shuffle pass per tree delta, the lane
#      │ groups merge pairwise until every lane has every total
#      ▼
#   partials[k] = the axis total of kept coordinate k (same on all lanes)
#      │
#      │ write: the destination stores one entry per kept coordinate
#      ▼ dst = the collapsed stats tile or the keepdims register tile
#
# The tree derivation runs per kernel at compile time, in the VM, from the map's layout value:
#
#   map layout ──▶ regCrds(i): tile coordinate of register slot i at lane 0
#       │
#       ├──▶ kept coordinate of slot i ──▶ keptCount, keptRank(i)
#       ├──▶ axis coordinate of slot i ──▶ axisSpan, axisRank(i)
#       └──▶ lane bit b keeps its kept coordinate when its flip adds 0 (laneKept) ──▶ treePasses, treeDelta(x)
#
# reduceSlot(k, a) is the register slot at kept rank k and axis rank a in the map's operand tile.
# The kernel's fold and butterfly are unrolled loops over these compile-time values, and the backend
# compiler folds the calls.

func thrCount[Sh, St](l: Layout[Sh, St]): auto =
  ## Lanes cooperating on the tile, the thread partition's leaf product
  size(l.shape[0][0])

func slotCount[Sh, St](l: Layout[Sh, St]): auto =
  ## Register slots per lane, every mode's leaf product but the thread partition
  size(l.shape) div l.thrCount()

func regCrds[Sh, St](l: Layout[Sh, St], i: int): (int, int) =
  ## Tile coordinate of register slot `i` at lane 0, the flat slot index
  ## scaled by the thread count so the thread modes stay 0
  crd2idx(l, idx2crd(l.shape, i * l.thrCount()))

func regKept[Sh, St](l: Layout[Sh, St], axis: static int, i: int): int =
  ## Kept-axis coordinate of register slot `i`
  when axis == 1:
    regCrds(l, i)[0]
  else:
    regCrds(l, i)[1]

func regAxis[Sh, St](l: Layout[Sh, St], axis: static int, i: int): int =
  ## Reduce-axis coordinate of register slot `i`
  when axis == 1:
    regCrds(l, i)[1]
  else:
    regCrds(l, i)[0]

func laneKept[Sh, St](l: Layout[Sh, St], axis: static int, bit: int): int =
  ## Kept-axis coordinate a lane bit's flip adds, 0 when the bit stays
  ## within one kept-coordinate group
  when axis == 1:
    crd2idx(l, ((1 shl bit, (0, 0))))[0]
  else:
    crd2idx(l, ((1 shl bit, (0, 0))))[1]

func keptCount[Sh, St](l: Layout[Sh, St], axis: static int): int =
  ## Distinct kept coordinates over the lane's registers
  var count = 0
  for i in 0 ..< slotCount(l):
    var first = true
    for j in 0 ..< i:
      if regKept(l, axis, j) == regKept(l, axis, i):
        first = false
    if first:
      count = count + 1
  return count

func axisSpan[Sh, St](l: Layout[Sh, St], axis: static int): int =
  ## Distinct reduce-axis coordinates over the lane's registers, one per kept-coordinate slot group
  slotCount(l) div keptCount(l, axis)

func keptRank[Sh, St](l: Layout[Sh, St], axis: static int, i: int): int =
  ## Rank of register slot `i`'s kept coordinate, ascending over the lane's distinct kept coordinates
  let kc = regKept(l, axis, i)
  var smaller = 0
  for j in 0 ..< slotCount(l):
    if regKept(l, axis, j) < kc:
      smaller = smaller + 1
  return smaller div axisSpan(l, axis)

func axisRank[Sh, St](l: Layout[Sh, St], axis: static int, i: int): int =
  ## Rank of register slot `i`'s axis coordinate, ascending over the lane's distinct axis coordinates
  let ac = regAxis(l, axis, i)
  var smaller = 0
  for j in 0 ..< slotCount(l):
    if regAxis(l, axis, j) < ac:
      smaller = smaller + 1
  return smaller div keptCount(l, axis)

func reduceSlot[Sh, St](l: Layout[Sh, St], axis: static int, k: int, a: int): int =
  ## Register slot holding kept coordinate rank `k` and axis coordinate rank `a`
  for i in 0 ..< slotCount(l):
    if keptRank(l, axis, i) == k and axisRank(l, axis, i) == a:
      return i
  return 0

func treePasses[Sh, St](l: Layout[Sh, St], axis: static int): int =
  ## Butterfly xor passes, the lane bits whose flip stays within one kept-coordinate group
  var passes = 0
  for bit in 0 ..< 32:
    if (1 shl bit) < l.thrCount() and laneKept(l, axis, bit) == 0:
      passes = passes + 1
  return passes

func treeDelta[Sh, St](l: Layout[Sh, St], axis: static int, x: int): int =
  ## Butterfly xor mask of pass `x`, the x-th lane bit whose flip stays within one kept-coordinate group
  var passed = 0
  for bit in 0 ..< 32:
    if (1 shl bit) < l.thrCount() and laneKept(l, axis, bit) == 0:
      if passed == x:
        return 1 shl bit
      passed = passed + 1
  return 0

func checkReduceMap[Sh, St](l: Layout[Sh, St], axis: static int) =
  ## Map contract for the axis reduce: the lane's registers scatter over distinct kept/axis coordinate pairs covering the full slot set
  for i in 0 ..< slotCount(l):
    for j in 0 ..< i:
      doAssert (regKept(l, axis, i), regAxis(l, axis, i)) !=
               (regKept(l, axis, j), regAxis(l, axis, j)),
        "axis reduce: duplicate kept/axis coordinate pairs over the lane's registers"
    doAssert regAxis(l, axis, i) >= 0, "axis reduce: negative axis coordinate"
  doAssert slotCount(l) === keptCount(l, axis) * axisSpan(l, axis),
    "axis reduce: the kept and axis coordinates must cover the slots"

template reducePartials(partials: var array, op, identity: untyped, src: Tile,
    l: static Layout, axis: static int, keptCount, axisSpan: static int) =
  ## Folds the register tile `src` into the per-lane partials `partials`.
  ## Phase 1 combines the lane's slots that share a kept coordinate.
  ## Phase 2 merges lane groups pairwise through xor shuffles.
  ## Every lane must call it unconditionally, the merge is a subgroup collective.
  ##
  ## `op` and `identity` take operator and magic references
  ## (`+`, `max`, `zero`, ...) untyped.
  ##
  ## The static() calls fold the index derivations in the VM.
  ## Runtime loop variables would emit them as standalone functions rescanning the slots per element.
  const passes = treePasses(l, axis)
  staticFor k, 0, keptCount:
    partials[k] = identity(typeof(partials[0]))
    staticFor a, 0, axisSpan:
      partials[k] = op(partials[k], src.data[static(reduceSlot(l, axis, k, a))])
  staticFor x, 0, passes:
    staticFor k, 0, keptCount:
      partials[k] = op(partials[k], simd_shuffle_xor(partials[k], static(treeDelta(l, axis, x))))

proc finalReduce[T, ShD, StD, Sh, St](dst: var Tile[T, ShD, StD], l: static Layout[Sh, St], axis: static int, keptCount: static int, partials: array[keptCount, T]) =
  ## Stores the per-lane partials into `dst`.
  ## A collapsed stats tile stores one slot per kept coordinate.
  ## A keepdims register tile stores every slot's kept coordinate's total
  const dstLen = dst.data.len
  when dstLen == keptCount:
    for k in 0 ..< keptCount:
      {.unroll.}
      dst.data[k] = partials[k]
  else:
    staticFor i, 0, dstLen:
      dst.data[i] = partials[static(keptRank(l, axis, i))]

proc sum*[T, Sh, St, ShD, StD](dst: var Tile[T, ShD, StD], src: Tile[T, Sh, St], axis: static int, map: static ThreadValueMap) {.inline.} =
  ## Sums one axis of the register tile `src` into `dst`, one axis sum per kept coordinate per lane.
  ##
  ## Precondition:
  ## - `src` is the register tile of `map`'s operand
  ## - every lane of the group calls the proc unconditionally
  ##
  ## `dst` stores one entry per kept coordinate: collapsed to the kept count,
  ## or spread over the register slots (keepdims form)
  const l = map.layout
  static: checkReduceMap(l, axis)
  const
    kc = keptCount(l, axis)
    span = axisSpan(l, axis)
  static:
    doAssert kc * span == src.data.len,
      "axis reduce: src must be the register tile of map's operand"
  var partials: array[kc, T]
  partials.reducePartials(`+`, zero, src, l, axis, kc, span)
  dst.finalReduce(l, axis, kc, partials)

proc max*[T, Sh, St, ShD, StD](dst: var Tile[T, ShD, StD], src: Tile[T, Sh, St], axis: static int, map: static ThreadValueMap) {.inline.} =
  ## Takes the maximum of one axis of the register tile `src` into `dst`, one axis maximum per kept
  ## coordinate per lane.
  ##
  ## Precondition:
  ## - `src` is the register tile of `map`'s operand
  ## - every lane of the group calls the proc unconditionally
  ##
  ## `dst` stores one entry per kept coordinate: collapsed to the kept count,
  ## or spread over the register slots (keepdims form)
  const l = map.layout
  static: checkReduceMap(l, axis)
  const
    kc = keptCount(l, axis)
    span = axisSpan(l, axis)
  static:
    doAssert kc * span == src.data.len,
      "axis reduce: src must be the register tile of map's operand"
  var partials: array[kc, T]
  partials.reducePartials(max, negInf, src, l, axis, kc, span)
  dst.finalReduce(l, axis, kc, partials)

proc min*[T, Sh, St, ShD, StD](dst: var Tile[T, ShD, StD], src: Tile[T, Sh, St], axis: static int, map: static ThreadValueMap) {.inline.} =
  ## Takes the minimum of one axis of the register tile `src` into `dst`, one axis minimum per kept
  ## coordinate per lane.
  ##
  ## Precondition:
  ## - `src` is the register tile of `map`'s operand
  ## - every lane of the group calls the proc unconditionally
  ##
  ## `dst` stores one entry per kept coordinate: collapsed to the kept count,
  ## or spread over the register slots (keepdims form)
  const l = map.layout
  static: checkReduceMap(l, axis)
  const
    kc = keptCount(l, axis)
    span = axisSpan(l, axis)
  static:
    doAssert kc * span == src.data.len,
      "axis reduce: src must be the register tile of map's operand"
  var partials: array[kc, T]
  partials.reducePartials(min, posInf, src, l, axis, kc, span)
  dst.finalReduce(l, axis, kc, partials)

func reducedLayout*(l: static Layout, axis: static int): static auto =
  ## Collapsed stats layout of the axis reduce over the thread->value map layout
  ## `l` at `axis`, one value per kept coordinate per lane:
  ## - axis 1: (1, (kept, 1))
  ## - axis 0: (1, (1, kept))
  ##
  ## Example: the attention's running row stats over the QK map:
  ##   const rowStatsL = reducedLayout(qkMap.layout, 1)
  when axis == 1:
    make_layout((1, (Int[keptCount(l, axis)](), 1)))
  else:
    make_layout((1, (1, Int[keptCount(l, axis)]())))
