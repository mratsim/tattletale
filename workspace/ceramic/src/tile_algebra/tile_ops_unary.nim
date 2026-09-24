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

proc roundToNearestEven*[T](x: float32): T {.device.} =
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
  ##   scores.map(scores, roundToNearestEven[El](x).float32)  # the softmax input round
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
