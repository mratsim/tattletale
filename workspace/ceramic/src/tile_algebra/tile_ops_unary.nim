## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#          Tile unary ops: the per-thread unary maps and init ops
#
# ############################################################
#
# Tile maps iterate the same per-lane slice as the binary maps.
# The col-vec ops are generic over the vpt width.

import ../int_tuples
import ../tensors
import ../atoms
import ./tiles
import ./tile_config
import ./tile_fma_partition
import ./tile_conversions
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  The col-vec unary ops: the attention seeds and the row maps
# ═════════════════════════════════════════════════════════════════════════

proc rsqrt*[rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])];
    src: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = 1/sqrt(src), per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = rsqrt(src.data[i])

proc exp2*[rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])];
    src: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = 2^src, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = exp2(src.data[i])

proc zero*[rowTiles, vpt: static int](
    vec: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## Zeroes the col-vec's slots.
  for i in 0 ..< rowTiles * vpt:
    vec.data[i] = 0.0'f32

proc neg_infty*[rowTiles, vpt: static int](
    vec: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## Seeds each slot with the most-negative finite fp32. Any finite
  ## tile value exceeds it, so the first 3-arg `row_max` replaces it.
  for i in 0 ..< rowTiles * vpt:
    vec.data[i] = -3.402823466e38'f32

proc copy*[rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])];
    src: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = src, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = src.data[i]

# ═════════════════════════════════════════════════════════════════════════
#  The tile unary maps
# ═════════════════════════════════════════════════════════════════════════

proc exp2*[RA, RB: static MmaAtom; R, C: static int; TL: static ThreadLayout](
    dst: var RtLeft[float32, R, C, RA, TL];
    src: RtLeft[float32, R, C, RB, TL]) =
  ## dst = 2^src, per element.
  static:
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.n == RB.mnk.n,
      "exp2: the operand tiles must share the atom's subtile grid"
  const rowTiles = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt:
        dst.frags[n][m].frag[vptI] = exp2(src.frags[n][m].frag[vptI])

proc exp2*[RA, RB: static MmaAtom; R, C: static int; TL: static ThreadLayout](
    dst: var RtRight[float32, R, C, RA, TL];
    src: RtRight[float32, R, C, RB, TL]) =
  ## dst = 2^src, per element.
  static:
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.n == RB.mnk.n,
      "exp2: the operand tiles must share the atom's subtile grid"
  const rowTiles = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for m in countup(thr.tn, colTiles - 1, TL.thrN):
    for n in 0 ..< rowTiles:
      for vptI in 0 ..< vpt:
        dst.frags[m][n].frag[vptI] = exp2(src.frags[m][n].frag[vptI])
