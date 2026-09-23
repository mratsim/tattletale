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
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  Register/storage conversions
# ═════════════════════════════════════════════════════════════════════════

template to*(x: untyped, TOut: typedesc): untyped =
  when TOut is typeof(x):
    x
  elif TOut is float32:
    toFp32(x)
  elif TOut is float16:
    toFp16(x)
  elif TOut is bfloat16:
    toBf16(x)
  else:
    {.error: "to: unsupported (typeof(x), TOut) type pair".}

# ═════════════════════════════════════════════════════════════════════════
#  Col-vec unary ops: the attention seeds and the row maps
# ═════════════════════════════════════════════════════════════════════════

func rsqrt*[T; rowTiles, vpt: static int](
    dst: var Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = 1/sqrt(src), per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = rsqrt(src.data[i])

func exp2*[T; rowTiles, vpt: static int](
    dst: var Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = 2^src, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = exp2(src.data[i])

func zero*[T; rowTiles, vpt: static int](
    vec: var Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## Zeroes the col-vec's slots.
  for i in 0 ..< rowTiles * vpt:
    vec.data[i] = 0.0'f32

func neg_infty*[T; rowTiles, vpt: static int](
    vec: var Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## Seeds each slot with the most-negative finite fp32. Any finite
  ## tile value exceeds it, so the first 3-arg `row_max` replaces it.
  for i in 0 ..< rowTiles * vpt:
    vec.data[i] = -3.402823466e38'f32

func copy*[T; rowTiles, vpt: static int](
    dst: var Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])],
    src: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]) =
  ## dst = src, per slot.
  for i in 0 ..< rowTiles * vpt:
    dst.data[i] = src.data[i]

func convert*[TIn, TOut; R, C: static int; ADst, ASrc: static MmaAtom](
    dst: var RtLeft[TOut, R, C, ADst],
    src: RtLeft[TIn, R, C, ASrc]) =
  static:
    doAssert typeof(ASrc.getLayoutC()) is typeof(ADst.getLayoutC()),
      "convert: the operand tiles must share the atom's C-fragment layout"
  const rowTiles = R div ADst.getM()
  const colTiles = C div ADst.getN()
  const vpt = ADst.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] = src.frags[n][m].frag[v].to(TOut)

# ═════════════════════════════════════════════════════════════════════════
#  Tile unary maps
# ═════════════════════════════════════════════════════════════════════════

func exp2*[T; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[T, R, C, A],
    src: RtLeft[T, R, C, A]) =
  ## dst = 2^src, per element.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt:
        dst.frags[n][m].frag[vptI] = exp2(src.frags[n][m].frag[vptI])

func exp2*[T; R, C: static int; A: static MmaAtom](
    dst: var RtRight[T, R, C, A],
    src: RtRight[T, R, C, A]) =
  ## dst = 2^src, per element.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for m in 0 ..< colTiles:
    for n in 0 ..< rowTiles:
      for vptI in 0 ..< vpt:
        dst.frags[m][n].frag[vptI] = exp2(src.frags[m][n].frag[vptI])

# ═════════════════════════════════════════════════════════════════════════
#  RNE narrowing (the element-dtype round the maps compose from)
# ═════════════════════════════════════════════════════════════════════════

proc roundToRne*[T](x: float32): T {.device.} =
  ## One round-to-nearest-even of an f32 value into the dtype `T`,
  ## the scalar counterpart of the mma epilogue's single-round contract.
  ##
  ## - `T` is unconstrained
  ## - the body keeps one variant per element dtype, a further dtype
  ##   adds its own variant
  when T is bfloat16:
    x.bfloat16
  else:
    x.to(float16)

# ═════════════════════════════════════════════════════════════════════════
#  Tile element maps
# ═════════════════════════════════════════════════════════════════════════

template map*[TOut, TIn; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[TOut, R, C, A],
    src: RtLeft[TIn, R, C, A],
    f: untyped): untyped =
  ## dst[i] = f(src[i]) per element, over the tile's whole fragment walk.
  ##
  ## Contract:
  ## - the body reads the source element as `x`
  ##
  ## Example, RNE round-and-widen back to f32 over the score tile:
  ##
  ##   scores.map(scores, roundToRne[El](x).float32)  # the softmax input round
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let x {.inject.} = src.frags[n][m].frag[v]
        dst.frags[n][m].frag[v] = f

template map2*[TOut, TIn; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[TOut, R, C, A],
    src1, src2: RtLeft[TIn, R, C, A],
    f: untyped): untyped =
  ## dst[i] = f(src1[i], src2[i]) per element, over the tile's whole fragment walk.
  ##
  ## Contract:
  ## - the body reads the source elements as `x` and `y`
  ## - both operands share one element type and geometry
  ##
  ## Example, KDA chunk-scan pairwise decay per key channel:
  ##
  ##   pdT.map2(cumulogdecayT, cumulogdecayS, exp2((x - y) * Log2e))  # the difference form
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let x {.inject.} = src1.frags[n][m].frag[v]
        let y {.inject.} = src2.frags[n][m].frag[v]
        dst.frags[n][m].frag[v] = f

template roundEl*[El](dst: untyped, src: untyped): untyped =
  ## dst[i] = the src element rounded to the family dtype `El` (RNE),
  ## the per-element rounding chokepoint that collapses the `when El is bfloat16` forks.
  ##
  ## Contract:
  ## - src is one f32 register tile
  ## - dst element = El, the storage round, dst[i] = roundToRne[El](src[i])
  ## - dst element = float32, the round-and-widen, dst[i] = roundToRne[El](src[i]).float32,
  ##   the value re-enters the f32 skeleton rounded to El
  ##
  ## Usage, `dst.roundEl[El](src)` with `El` the family dtype explicit.
  ##
  ## Example, RNE round in place before the softmax:
  ##
  ##   scores.roundEl[El](scores)  # the logits' in-place round
  block:
    when typeof(dst.frags[0][0].frag[0]) is El:
      dst.map(src, roundToRne[El](x))
    elif typeof(dst.frags[0][0].frag[0]) is float32:
      dst.map(src, roundToRne[El](x).float32)
    else:
      {.error: "roundEl: the destination element is neither El nor float32".}

# ═════════════════════════════════════════════════════════════════════════
#  Position-driven tile maps: the attention masks and the column splits
# ═════════════════════════════════════════════════════════════════════════

const Fp32Lowest* = -3.402823466e38'f32
  ## Most-negative finite f32. Any finite tile value exceeds it, so the online
  ## softmax's row max ignores a masked element and exp2(S − m) gives exact zero.

proc bandMask*[R, C: static int; A: static MmaAtom](
    tile: var RtLeft[float32, R, C, A],
    limit, window: int32) =
  ## Banded mask over the S tile's elements.
  ##
  ## Contract:
  ## - element (r, c) is masked to `Fp32Lowest` iff `c > limit + r` or `c < limit + r − window + 1`
  ## - the mask hits exactly the elements the Q·Kᵀ mma produced
  ##
  ## Parameters:
  ## - limit, the block's band offset (`qAbs − kv_idx·8`-class), signed,
  ##   a negative limit masks every column of the row, an unsigned wrap
  ##   would leave them attended
  ## - window, the band width in columns, and the causal mask is the limit
  ##   case where window covers the whole upper half, `window = int32.high`
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let row = laneRowOf(A)
  let col = laneColOf(A)
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let band = limit + int32(row + n * M)
        let c = int32(col + m * N + v)
        if c > band or c < band - window + 1:
          tile.frags[n][m].frag[v] = Fp32Lowest

template splitCols*[TOut; R, H: static int; A: static MmaAtom](
    dstLo, dstHi: var RtLeft[TOut, R, H, A],
    src: untyped): untyped =
  ## Column-half split over a register tile's columns.
  ##
  ## Contract:
  ## - dstLo takes the first column half of src, dstHi the second
  ## - each element converts to the destinations' element type TOut
  ## - src is an (R, 2H) register tile sharing dst's atom, and H must
  ##   cover whole atom columns
  ##
  ## Usage, `lo.splitCols(hi, src)` at the (R, 2H) → two (R, H) split.
  const colTiles = H div A.getN()
  const vpt = A.getVpt()
  const rowTiles = R div A.getM()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dstLo.frags[n][m].frag[v] = src.frags[n][m].frag[v].to(TOut)
        dstHi.frags[n][m].frag[v] = src.frags[n][m + colTiles].frag[v].to(TOut)
