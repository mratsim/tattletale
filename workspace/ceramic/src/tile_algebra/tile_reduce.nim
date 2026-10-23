# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/math
import workspace/crucible
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/tensors
import ./tiles

# ############################################################
#
#            Two-phase reduce and row strip views
#
# ############################################################

type ReduceOp* = enum
  ## Element-wise reduction operator, selected per call at compile time.
  roSum
  roMax
  roMin

func reduceIdentity*[T](op: static ReduceOp): T =
  ## Returns the neutral element of `op` for `T`,
  ## the starting value of the per-lane combine in `reduce`.
  ##
  ## Contract:
  ## - `roSum`: `zero(T)`
  ## - `roMax` on `SomeFloat` `T`: minus infinity, on ordinals: `low(T)`
  ## - `roMin` on `SomeFloat` `T`: plus infinity, on ordinals: `high(T)`
  when op == roSum:
    result = zero(T)
  elif op == roMax:
    when T is SomeFloat:
      let s = T(-1)
      result = s / zero(T)
    else:
      result = low(T)
  else:
    when T is SomeFloat:
      let s = T(1)
      result = s / zero(T)
    else:
      result = high(T)
  # The float infinities come from a runtime ±1/0 division:
  # Nim cannot carry an infinity literal into device code.
  # `float32(-Inf)` folds to a fp64 literal, the MSL backend rejects it.
  # The folded `-1.0'f32/0.0'f32` emits the invalid MSL token `inff`.

proc reduce*[T, Sh, St](dst: var T, src: Tile[T, Sh, St], op: static ReduceOp, clear: static bool) {.inline.} =
  ## Two-phase warp-scope reduction of a register tile into `dst`.
  ##
  ## Phase 1 combines each lane's owned tile elements into a per-lane partial, starting from `reduceIdentity(op)`.
  ## Phase 2 runs the subgroup butterfly, combines the per-lane partials, and broadcasts the total to every lane.
  ## Butterfly op per `ReduceOp`:
  ## - `roSum`: `reduceAdd`
  ## - `roMax`: `reduceMax`
  ## - `roMin`: `-reduceMax(-x)`
  ##
  ## The reduced size lives in the tile map's replicate dimension, the result collapses into it, and every lane ends holding the total.
  ##
  ## `clear` governs `dst` only:
  ## - `clear = true`: `dst` becomes the result, `dst` is not read
  ## - `clear = false`: `dst` combines with the result
  ##
  ## Contract:
  ## - all lanes must call `reduce` unconditionally, subgroup collectives are non-divergent
  ## - the lanes of one `reduce` call are one simdgroup, the butterfly combines within it only
  ##   lanes of a wider threadgroup hold per-simdgroup partials
  ## - the tile comes from `tile_algebra`'s `load`, a lane's storage holds its owned elements at `data[0 ..< size div lane width]`
  let owned = src.size() div getLaneWidth()
  var acc = reduceIdentity[T](op)
  for i in 0 ..< owned:
    when op == roSum:
      acc += src.data[i]
    elif op == roMax:
      acc = max(acc, src.data[i])
    else:
      acc = min(acc, src.data[i])
  when op == roSum:
    let total = reduceAdd(acc)
  elif op == roMax:
    let total = reduceMax(acc)
  else:
    let total = -reduceMax(-acc)
  when clear:
    dst = total
  else:
    when op == roSum:
      # read plus plain assignment, compound assignment on a `var` scalar
      # parameter lowers to an invalid pointer addition on Metal
      dst = dst + total
    elif op == roMax:
      dst = max(dst, total)
    else:
      dst = min(dst, total)

func rowView*[T, Sh, St](src: TensorView[T, Sh, St], axis: static int, rowFlat: int): auto =
  ## View of one strip of `src` along `axis`, the unit `reduce` consumes.
  ##
  ## `rowFlat` is the flat index over the non-reduced coordinates, the other dimensions in `src`'s own order.
  ## The strip keeps `src`'s strides for the kept dimensions and collapses `axis` into one dimension of the reduced size.
  ##
  ## Decomposition:
  ## - `nonReduced = src.sliceAxes(axis)` keeps the non-reduced dimensions with their real strides
  ## - `reduced = src.diceAxes(axis)` is the strip, one dimension of the reduced size
  ## - `rowCrd = idx2crd(make_layout(nonReduced.shape), rowFlat)` decomposes `rowFlat` on the compact layout,
  ##   `idx2crd` never sees the strided layout
  ## - `offset = crd2idx(nonReduced, rowCrd)` walks the real strides
  ##
  ## The same proc cuts destination strips: a strip of `dst` along `axis` with the same `rowFlat` shares the non-reduced coordinates.
  ## A rank-1 weight over the reduced axis needs no cut, its strip is the full tensor, `rowView(weight, 0, 0)`.
  let
    nonReduced = src.sliceAxes(axis)
    reduced = src.diceAxes(axis)
  when nonReduced.rank() == 0:
    # a rank-1 `src` keeps no non-reduced dimension, the strip is the full tensor
    let offset = 0
  else:
    let
      rowCrd = idx2crd(make_layout(nonReduced.shape), rowFlat)
      offset = crd2idx(nonReduced, rowCrd)
  make_view(src.data +% offset, reduced)
