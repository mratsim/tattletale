## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#          Tile binary ops: the per-thread binary and row maps
#
# ############################################################
#
# Map tile params carry separate atoms.
# A shared atom across two params does not unify in Nim.
# Each map guards the shared subtile grid with a static assert instead.

import ../int_tuples
import ../tensors
import ../atoms
import ./tiles
import ./tile_config
import ./tile_fma_partition
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  The col-vec binary ops
# ═════════════════════════════════════════════════════════════════════════

proc add*[rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])];
    src: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]; s: float32) =
  ## dst = src + s, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = src.data[i] + s

proc sub*[rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])];
    lhs: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])];
    rhs: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = lhs − rhs, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = lhs.data[i] - rhs.data[i]

proc mul*[rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])];
    src: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]; s: float32) =
  ## dst = src · s, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = src.data[i] * s

proc mul*[rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])];
    lhs: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])];
    rhs: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = lhs · rhs, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = lhs.data[i] * rhs.data[i]

# ═════════════════════════════════════════════════════════════════════════
#  The tile binary maps
# ═════════════════════════════════════════════════════════════════════════

proc mul*[RA, RB, RC: static MmaAtom; R, C: static int; TL: static ThreadLayout](
    dst: var RtLeft[float32, R, C, RA, TL];
    src: RtLeft[float32, R, C, RB, TL];
    src2: RtLeft[float32, R, C, RC, TL]) =
  ## dst = src · src2, per element.
  static:
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.m == RC.mnk.m and
            RA.mnk.n == RB.mnk.n and RA.mnk.n == RC.mnk.n,
      "mul: the operand tiles must share the atom's subtile grid"
  const rowTiles = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt:
        dst.frags[n][m].frag[vptI] =
          src.frags[n][m].frag[vptI] * src2.frags[n][m].frag[vptI]

proc mul*[RA, RB, RC: static MmaAtom; R, C: static int; TL: static ThreadLayout](
    dst: var RtRight[float32, R, C, RA, TL];
    src: RtRight[float32, R, C, RB, TL];
    src2: RtRight[float32, R, C, RC, TL]) =
  ## dst = src · src2, per element.
  static:
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.m == RC.mnk.m and
            RA.mnk.n == RB.mnk.n and RA.mnk.n == RC.mnk.n,
      "mul: the operand tiles must share the atom's subtile grid"
  const rowTiles = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for m in countup(thr.tn, colTiles - 1, TL.thrN):
    for n in 0 ..< rowTiles:
      for vptI in 0 ..< vpt:
        dst.frags[m][n].frag[vptI] =
          src.frags[m][n].frag[vptI] * src2.frags[m][n].frag[vptI]

proc mul*[RA, RB: static MmaAtom; R, C: static int; TL: static ThreadLayout](
    dst: var RtLeft[float16, R, C, RA, TL];
    src: RtLeft[float16, R, C, RB, TL]; s: float32) =
  ## dst = src · s per element, rounded back to fp16 (RNE).
  static:
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.n == RB.mnk.n,
      "mul: the operand tiles must share the atom's subtile grid"
  const rowTiles = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt:
        dst.frags[n][m].frag[vptI] =
          toFp16(toFp32(src.frags[n][m].frag[vptI]) * s)

# ═════════════════════════════════════════════════════════════════════════
#  The row maps
# ═════════════════════════════════════════════════════════════════════════

proc mul_row*[RA, RB: static MmaAtom; R, C, rowTiles, vpt: static int; TL: static ThreadLayout](
    dst: var RtLeft[float32, R, C, RA, TL];
    src: RtLeft[float32, R, C, RB, TL];
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] · rowVals[r].
  static:
    doAssert rowTiles == R div RA.mnk.m,
      "mul_row: the col-vec subtile count must match the tile's"
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.n == RB.mnk.n,
      "mul_row: the operand tiles must share the atom's subtile grid"
  const rowTiles0 = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt0 = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for n in countup(thr.tm, rowTiles0 - 1, TL.thrM):
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt0:
        dst.frags[n][m].frag[vptI] =
          src.frags[n][m].frag[vptI] *
          rowVals.data[n * vpt + vptI]

proc mul_row*[RA, RB: static MmaAtom; R, C, rowTiles, vpt: static int; TL: static ThreadLayout](
    dst: var RtRight[float32, R, C, RA, TL];
    src: RtRight[float32, R, C, RB, TL];
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] · rowVals[r].
  static:
    doAssert rowTiles == R div RA.mnk.m,
      "mul_row: the col-vec subtile count must match the tile's"
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.n == RB.mnk.n,
      "mul_row: the operand tiles must share the atom's subtile grid"
  const rowTiles0 = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt0 = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for m in countup(thr.tn, colTiles - 1, TL.thrN):
    for n in 0 ..< rowTiles0:
      for vptI in 0 ..< vpt0:
        dst.frags[m][n].frag[vptI] =
          src.frags[m][n].frag[vptI] *
          rowVals.data[n * vpt + vptI]

proc sub_row*[RA, RB: static MmaAtom; R, C, rowTiles, vpt: static int; TL: static ThreadLayout](
    dst: var RtLeft[float32, R, C, RA, TL];
    src: RtLeft[float32, R, C, RB, TL];
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] − rowVals[r].
  static:
    doAssert rowTiles == R div RA.mnk.m,
      "sub_row: the col-vec subtile count must match the tile's"
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.n == RB.mnk.n,
      "sub_row: the operand tiles must share the atom's subtile grid"
  const rowTiles0 = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt0 = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for n in countup(thr.tm, rowTiles0 - 1, TL.thrM):
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt0:
        dst.frags[n][m].frag[vptI] =
          src.frags[n][m].frag[vptI] -
          rowVals.data[n * vpt + vptI]

proc sub_row*[RA, RB: static MmaAtom; R, C, rowTiles, vpt: static int; TL: static ThreadLayout](
    dst: var RtRight[float32, R, C, RA, TL];
    src: RtRight[float32, R, C, RB, TL];
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] − rowVals[r].
  static:
    doAssert rowTiles == R div RA.mnk.m,
      "sub_row: the col-vec subtile count must match the tile's"
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.n == RB.mnk.n,
      "sub_row: the operand tiles must share the atom's subtile grid"
  const rowTiles0 = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt0 = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for m in countup(thr.tn, colTiles - 1, TL.thrN):
    for n in 0 ..< rowTiles0:
      for vptI in 0 ..< vpt0:
        dst.frags[m][n].frag[vptI] =
          src.frags[m][n].frag[vptI] -
          rowVals.data[n * vpt + vptI]

proc div_row*[RA, RB: static MmaAtom; R, C, rowTiles, vpt: static int; TL: static ThreadLayout](
    dst: var RtLeft[float32, R, C, RA, TL];
    src: RtLeft[float32, R, C, RB, TL];
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] / rowVals[r].
  static:
    doAssert rowTiles == R div RA.mnk.m,
      "div_row: the col-vec subtile count must match the tile's"
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.n == RB.mnk.n,
      "div_row: the operand tiles must share the atom's subtile grid"
  const rowTiles0 = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt0 = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for n in countup(thr.tm, rowTiles0 - 1, TL.thrM):
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt0:
        dst.frags[n][m].frag[vptI] =
          src.frags[n][m].frag[vptI] /
          rowVals.data[n * vpt + vptI]

proc div_row*[RA, RB: static MmaAtom; R, C, rowTiles, vpt: static int; TL: static ThreadLayout](
    dst: var RtRight[float32, R, C, RA, TL];
    src: RtRight[float32, R, C, RB, TL];
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] / rowVals[r].
  static:
    doAssert rowTiles == R div RA.mnk.m,
      "div_row: the col-vec subtile count must match the tile's"
    doAssert RA.mnk.m == RB.mnk.m and RA.mnk.n == RB.mnk.n,
      "div_row: the operand tiles must share the atom's subtile grid"
  const rowTiles0 = R div RA.mnk.m
  const colTiles = C div RA.mnk.n
  const vpt0 = toIntVal(RA.valuesPerThread(opA))
  let thr = fmaSlice[RA, TL]()
  for m in countup(thr.tn, colTiles - 1, TL.thrN):
    for n in 0 ..< rowTiles0:
      for vptI in 0 ..< vpt0:
        dst.frags[m][n].frag[vptI] =
          src.frags[m][n].frag[vptI] /
          rowVals.data[n * vpt + vptI]
