## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
import ../int_tuples
import ../tensors
import ./tiles
import ./tile_config
import ./tile_ops_unary
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  Col-vec binary ops
# ═════════════════════════════════════════════════════════════════════════

func add*[T; S; rowTiles, vpt: static int](
    dst: var Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    s: S) =
  ## dst = src + s, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = src.data[i] + s

func sub*[T; rowTiles, vpt: static int](
    dst: var Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    lhs: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    rhs: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = lhs − rhs, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = lhs.data[i] - rhs.data[i]

func mul*[T; S; rowTiles, vpt: static int](
    dst: var Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    s: S) =
  ## dst = src · s, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = src.data[i] * s

func mul*[T; rowTiles, vpt: static int](
    dst: var Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    lhs: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    rhs: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = lhs · rhs, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = lhs.data[i] * rhs.data[i]

# ═════════════════════════════════════════════════════════════════════════
#  Tile binary maps
# ═════════════════════════════════════════════════════════════════════════

func mul*[TIn; TOut; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[TOut, R, C, A],
    src: RtLeft[TIn, R, C, A],
    src2: RtLeft[TIn, R, C, A]) =
  ## dst = src · src2, per element: computed in the destination type.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt:
        dst.frags[n][m].frag[vptI] =
          src.frags[n][m].frag[vptI].to(TOut) * src2.frags[n][m].frag[vptI].to(TOut)

func mul*[TIn; TOut; R, C: static int; A: static MmaAtom](
    dst: var RtRight[TOut, R, C, A],
    src: RtRight[TIn, R, C, A],
    src2: RtRight[TIn, R, C, A]) =
  ## dst = src · src2, per element: computed in the destination type.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for m in 0 ..< colTiles:
    for n in 0 ..< rowTiles:
      for vptI in 0 ..< vpt:
        dst.frags[m][n].frag[vptI] =
          src.frags[m][n].frag[vptI].to(TOut) * src2.frags[m][n].frag[vptI].to(TOut)

func mul*[T; S; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[T, R, C, A],
    src: RtLeft[T, R, C, A],
    s: S) =
  ## dst = src · s per element, computed in T.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt:
        dst.frags[n][m].frag[vptI] = src.frags[n][m].frag[vptI].to(T) * s.to(T)

# ═════════════════════════════════════════════════════════════════════════
#  Row maps
# ═════════════════════════════════════════════════════════════════════════

func mul_row*[TIn; TOut; R, C, rowTiles, vpt: static int; A: static MmaAtom](
    dst: var RtLeft[TOut, R, C, A],
    src: RtLeft[TIn, R, C, A],
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] · rowVals[r], computed in the destination type.
  static:
    doAssert rowTiles == R div A.getM(),
      "mul_row: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  for n in 0 ..< rowTiles0:
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt0:
        dst.frags[n][m].frag[vptI] =
          src.frags[n][m].frag[vptI].to(TOut) * rowVals.data[n * vpt + vptI].to(TOut)

func mul_row*[TIn; TOut; R, C, rowTiles, vpt: static int; A: static MmaAtom](
    dst: var RtRight[TOut, R, C, A],
    src: RtRight[TIn, R, C, A],
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] · rowVals[r], computed in the destination type.
  static:
    doAssert rowTiles == R div A.getM(),
      "mul_row: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  for m in 0 ..< colTiles:
    for n in 0 ..< rowTiles0:
      for vptI in 0 ..< vpt0:
        dst.frags[m][n].frag[vptI] =
          src.frags[m][n].frag[vptI].to(TOut) * rowVals.data[n * vpt + vptI].to(TOut)

func sub_row*[TIn; TOut; R, C, rowTiles, vpt: static int; A: static MmaAtom](
    dst: var RtLeft[TOut, R, C, A],
    src: RtLeft[TIn, R, C, A],
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] − rowVals[r], computed in the destination type.
  static:
    doAssert rowTiles == R div A.getM(),
      "sub_row: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  for n in 0 ..< rowTiles0:
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt0:
        dst.frags[n][m].frag[vptI] =
          src.frags[n][m].frag[vptI].to(TOut) - rowVals.data[n * vpt + vptI].to(TOut)

func sub_row*[TIn; TOut; R, C, rowTiles, vpt: static int; A: static MmaAtom](
    dst: var RtRight[TOut, R, C, A],
    src: RtRight[TIn, R, C, A],
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] − rowVals[r], computed in the destination type.
  static:
    doAssert rowTiles == R div A.getM(),
      "sub_row: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  for m in 0 ..< colTiles:
    for n in 0 ..< rowTiles0:
      for vptI in 0 ..< vpt0:
        dst.frags[m][n].frag[vptI] =
          src.frags[m][n].frag[vptI].to(TOut) - rowVals.data[n * vpt + vptI].to(TOut)

func div_row*[TIn; TOut; R, C, rowTiles, vpt: static int; A: static MmaAtom](
    dst: var RtLeft[TOut, R, C, A],
    src: RtLeft[TIn, R, C, A],
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] / rowVals[r], computed in the destination type.
  static:
    doAssert rowTiles == R div A.getM(),
      "div_row: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  for n in 0 ..< rowTiles0:
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt0:
        dst.frags[n][m].frag[vptI] =
          src.frags[n][m].frag[vptI].to(TOut) / rowVals.data[n * vpt + vptI].to(TOut)

func div_row*[TIn; TOut; R, C, rowTiles, vpt: static int; A: static MmaAtom](
    dst: var RtRight[TOut, R, C, A],
    src: RtRight[TIn, R, C, A],
    rowVals: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[r][c] = src[r][c] / rowVals[r], computed in the destination type.
  static:
    doAssert rowTiles == R div A.getM(),
      "div_row: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  for m in 0 ..< colTiles:
    for n in 0 ..< rowTiles0:
      for vptI in 0 ..< vpt0:
        dst.frags[m][n].frag[vptI] =
          src.frags[m][n].frag[vptI].to(TOut) / rowVals.data[n * vpt + vptI].to(TOut)
