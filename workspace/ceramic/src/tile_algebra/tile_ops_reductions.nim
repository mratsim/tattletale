## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
import ../int_tuples
import ../layout_indexing
import ../tensors
import ./tiles
import ./tile_config
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  The row-reduction shuffle tree
# ═════════════════════════════════════════════════════════════════════════

type ReductionTree* = tuple
  deltas: array[8, int]
  steps: int
  mask: uint32

func getReductionTree*(A: static MmaAtom): ReductionTree =
  ## The row-reduction shuffle tree (deltas, step count, leader mask)
  ## derived from the atom's fragment-column lane coefficients.
  when A.getThreadCount() == 1:
    discard
  else:
    const colCoeffs = block:
      var a: array[5, int]
      for b in 0 .. 4:
        a[b] = toIntVal(crd2idx(A.getLayoutA(), 1 shl b)) div A.getM()
      a
    # Unrolled over the 5 lane bits: each nonzero col coefficient adds
    # its bit's delta (2^b) to the tree and to the leader mask.
    var steps = 0
    when colCoeffs[0] != 0:
      result.deltas[steps] = 1
      result.mask = result.mask or 1'u32
      inc steps
    when colCoeffs[1] != 0:
      result.deltas[steps] = 2
      result.mask = result.mask or 2'u32
      inc steps
    when colCoeffs[2] != 0:
      result.deltas[steps] = 4
      result.mask = result.mask or 4'u32
      inc steps
    when colCoeffs[3] != 0:
      result.deltas[steps] = 8
      result.mask = result.mask or 8'u32
      inc steps
    when colCoeffs[4] != 0:
      result.deltas[steps] = 16
      result.mask = result.mask or 16'u32
      inc steps
    result.steps = steps

# ═════════════════════════════════════════════════════════════════════════
#  The row-reduction family
# ═════════════════════════════════════════════════════════════════════════
proc row_sum*[A: static MmaAtom; R, C, rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: RtLeft[float32, R, C, A]) =
  ## dst[n] = Σ over row n of src, per owned row.
  static:
    doAssert rowTiles == R div A.getM(),
      "row_sum: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  const tree = A.getReductionTree()
  const steps = tree.steps
  const delta0 = tree.deltas[0]
  const delta1 = tree.deltas[1]
  const delta2 = tree.deltas[2]
  static:
    doAssert steps <= 3,
      "row_sum: the unrolled tree supports at most 3 shuffle steps"
  let lane = thread_index_in_threadgroup
  let leader = lane and (not tree.mask)
  for n in 0 ..< rowTiles0:
    var acc = src.frags[n][0].frag[0]
    for vptI in 1 ..< vpt0:
      acc = acc + src.frags[n][0].frag[vptI]
    for m in 1 ..< colTiles:
      for vptI in 0 ..< vpt0:
        acc = acc + src.frags[n][m].frag[vptI]
    when steps >= 1:
      acc = acc + simdShuffleDown(acc, uint32(delta0))
    when steps >= 2:
      acc = acc + simdShuffleDown(acc, uint32(delta1))
    when steps >= 3:
      acc = acc + simdShuffleDown(acc, uint32(delta2))
    acc = simdShuffle(acc, leader)
    for vptI in 0 ..< vpt0:
      dst.data[n * vpt + vptI] = acc

proc row_sum*[A: static MmaAtom; R, C, rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: RtRight[float32, R, C, A]) =
  ## dst[n] = Σ over row n of src, per owned row.
  static:
    doAssert rowTiles == R div A.getM(),
      "row_sum: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  const tree = A.getReductionTree()
  const steps = tree.steps
  const delta0 = tree.deltas[0]
  const delta1 = tree.deltas[1]
  const delta2 = tree.deltas[2]
  static:
    doAssert steps <= 3,
      "row_sum: the unrolled tree supports at most 3 shuffle steps"
  let lane = thread_index_in_threadgroup
  let leader = lane and (not tree.mask)
  for n in 0 ..< rowTiles0:
    var acc = src.frags[0][n].frag[0]
    for vptI in 1 ..< vpt0:
      acc = acc + src.frags[0][n].frag[vptI]
    for m in 1 ..< colTiles:
      for vptI in 0 ..< vpt0:
        acc = acc + src.frags[m][n].frag[vptI]
    when steps >= 1:
      acc = acc + simdShuffleDown(acc, uint32(delta0))
    when steps >= 2:
      acc = acc + simdShuffleDown(acc, uint32(delta1))
    when steps >= 3:
      acc = acc + simdShuffleDown(acc, uint32(delta2))
    acc = simdShuffle(acc, leader)
    for vptI in 0 ..< vpt0:
      dst.data[n * vpt + vptI] = acc

proc row_sum*[A: static MmaAtom; R, C, rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: RtLeft[float32, R, C, A],
    srcAccum: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[n] = srcAccum[n] + Σ over row n of src, the online-softmax running-sum update.
  static:
    doAssert rowTiles == R div A.getM(),
      "row_sum: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  const tree = A.getReductionTree()
  const steps = tree.steps
  const delta0 = tree.deltas[0]
  const delta1 = tree.deltas[1]
  const delta2 = tree.deltas[2]
  static:
    doAssert steps <= 3,
      "row_sum: the unrolled tree supports at most 3 shuffle steps"
  let lane = thread_index_in_threadgroup
  let leader = lane and (not tree.mask)
  for n in 0 ..< rowTiles0:
    var acc = src.frags[n][0].frag[0]
    for vptI in 1 ..< vpt0:
      acc = acc + src.frags[n][0].frag[vptI]
    for m in 1 ..< colTiles:
      for vptI in 0 ..< vpt0:
        acc = acc + src.frags[n][m].frag[vptI]
    when steps >= 1:
      acc = acc + simdShuffleDown(acc, uint32(delta0))
    when steps >= 2:
      acc = acc + simdShuffleDown(acc, uint32(delta1))
    when steps >= 3:
      acc = acc + simdShuffleDown(acc, uint32(delta2))
    acc = simdShuffle(acc, leader)
    for vptI in 0 ..< vpt0:
      dst.data[n * vpt + vptI] = srcAccum.data[n * vpt + vptI] + acc

proc row_sum*[A: static MmaAtom; R, C, rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: RtRight[float32, R, C, A],
    srcAccum: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[n] = srcAccum[n] + Σ over row n of src.
  static:
    doAssert rowTiles == R div A.getM(),
      "row_sum: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  const tree = A.getReductionTree()
  const steps = tree.steps
  const delta0 = tree.deltas[0]
  const delta1 = tree.deltas[1]
  const delta2 = tree.deltas[2]
  static:
    doAssert steps <= 3,
      "row_sum: the unrolled tree supports at most 3 shuffle steps"
  let lane = thread_index_in_threadgroup
  let leader = lane and (not tree.mask)
  for n in 0 ..< rowTiles0:
    var acc = src.frags[0][n].frag[0]
    for vptI in 1 ..< vpt0:
      acc = acc + src.frags[0][n].frag[vptI]
    for m in 1 ..< colTiles:
      for vptI in 0 ..< vpt0:
        acc = acc + src.frags[m][n].frag[vptI]
    when steps >= 1:
      acc = acc + simdShuffleDown(acc, uint32(delta0))
    when steps >= 2:
      acc = acc + simdShuffleDown(acc, uint32(delta1))
    when steps >= 3:
      acc = acc + simdShuffleDown(acc, uint32(delta2))
    acc = simdShuffle(acc, leader)
    for vptI in 0 ..< vpt0:
      dst.data[n * vpt + vptI] = srcAccum.data[n * vpt + vptI] + acc

proc row_max*[A: static MmaAtom; R, C, rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: RtLeft[float32, R, C, A],
    srcAccum: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[n] = max(srcAccum[n], max over row n of src).
  static:
    doAssert rowTiles == R div A.getM(),
      "row_max: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  const tree = A.getReductionTree()
  const steps = tree.steps
  const delta0 = tree.deltas[0]
  const delta1 = tree.deltas[1]
  const delta2 = tree.deltas[2]
  static:
    doAssert steps <= 3,
      "row_max: the unrolled tree supports at most 3 shuffle steps"
  let lane = thread_index_in_threadgroup
  let leader = lane and (not tree.mask)
  for n in 0 ..< rowTiles0:
    var acc = src.frags[n][0].frag[0]
    for vptI in 1 ..< vpt0:
      acc = max(acc, src.frags[n][0].frag[vptI])
    for m in 1 ..< colTiles:
      for vptI in 0 ..< vpt0:
        acc = max(acc, src.frags[n][m].frag[vptI])
    when steps >= 1:
      acc = max(acc, simdShuffleDown(acc, uint32(delta0)))
    when steps >= 2:
      acc = max(acc, simdShuffleDown(acc, uint32(delta1)))
    when steps >= 3:
      acc = max(acc, simdShuffleDown(acc, uint32(delta2)))
    acc = simdShuffle(acc, leader)
    for vptI in 0 ..< vpt0:
      dst.data[n * vpt + vptI] = max(srcAccum.data[n * vpt + vptI], acc)

proc row_max*[A: static MmaAtom; R, C, rowTiles, vpt: static int](
    dst: var Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: RtRight[float32, R, C, A],
    srcAccum: Tensor[float32, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst[n] = max(srcAccum[n], max over row n of src).
  static:
    doAssert rowTiles == R div A.getM(),
      "row_max: the col-vec subtile count must match the tile's"
  const rowTiles0 = R div A.getM()
  const colTiles = C div A.getN()
  const vpt0 = A.getVpt()
  const tree = A.getReductionTree()
  const steps = tree.steps
  const delta0 = tree.deltas[0]
  const delta1 = tree.deltas[1]
  const delta2 = tree.deltas[2]
  static:
    doAssert steps <= 3,
      "row_max: the unrolled tree supports at most 3 shuffle steps"
  let lane = thread_index_in_threadgroup
  let leader = lane and (not tree.mask)
  for n in 0 ..< rowTiles0:
    var acc = src.frags[0][n].frag[0]
    for vptI in 1 ..< vpt0:
      acc = max(acc, src.frags[0][n].frag[vptI])
    for m in 1 ..< colTiles:
      for vptI in 0 ..< vpt0:
        acc = max(acc, src.frags[m][n].frag[vptI])
    when steps >= 1:
      acc = max(acc, simdShuffleDown(acc, uint32(delta0)))
    when steps >= 2:
      acc = max(acc, simdShuffleDown(acc, uint32(delta1)))
    when steps >= 3:
      acc = max(acc, simdShuffleDown(acc, uint32(delta2)))
    acc = simdShuffle(acc, leader)
    for vptI in 0 ..< vpt0:
      dst.data[n * vpt + vptI] = max(srcAccum.data[n * vpt + vptI], acc)

# ═════════════════════════════════════════════════════════════════════════
#  tileKMax: the ragged tile's effective k-end (the branching skip)
# ═════════════════════════════════════════════════════════════════════════

proc tileKMax*(Lengths: ptr UncheckedArray[uint16],
               mTile: uint32): uint32 =
  ## The tile's effective K-loop bound in k-block units:
  ## ceil(max over the tile's rows of Lengths[row] / 16).
  ## The block is the gemm's K staging width (TileK = 16).
  ## The bound shares the K-loop counter's unit.
  ## Inputs:
  ##   Lengths: one uint16 per matrix row, at least (mTile + 1) * 32 entries.
  ##   The tile reads its 32 rows [mTile*32, (mTile+1)*32) unconditionally.
  ##   mTile: the M-tile index (grid.y), the 32-row block of the M dimension this threadgroup covers.
  ##
  ##   Lengths[row] over the tile's rows, 16 columns per k-block
  ##   (▓ valid, · padding, | a block boundary):
  ##
  ##   k-block        0                1                2                3
  ##   row 0  ▓▓▓▓▓▓▓▓▓▓······|················|················|················   L = 10
  ##   row 1  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓|▓···············|················|················   L = 17
  ##   row 2  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓|▓▓▓▓▓▓▓▓▓▓▓▓▓▓··|················|················   L = 30
  ##
  ##   max L = 30 → ⌈30/16⌉ = 2 k-blocks → the K-loop runs blocks 0..1,
  ##   covering 32 ≥ 30 columns for every row.

  let tid = thread_index_in_threadgroup
  var kEff: array[4, uint32]
  for n in 0 ..< 4:
    kEff[n] = uint32(Lengths[mTile * 32'u32 + ((tid + uint32(n * 8)) and 31'u32)])
  var laneMax: uint32
  laneMax = kEff[0]
  for n in 1 ..< 4:
    laneMax = max(laneMax, kEff[n])
  laneMax = max(laneMax, simdShuffleDown(laneMax, 16'u32))
  laneMax = max(laneMax, simdShuffleDown(laneMax, 8'u32))
  laneMax = max(laneMax, simdShuffleDown(laneMax, 4'u32))
  laneMax = max(laneMax, simdShuffleDown(laneMax, 2'u32))
  laneMax = max(laneMax, simdShuffleDown(laneMax, 1'u32))
  laneMax = simdShuffle(laneMax, 0'u32)
  result = (laneMax + 15'u32) div 16'u32
