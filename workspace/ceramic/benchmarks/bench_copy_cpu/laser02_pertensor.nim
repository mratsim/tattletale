## MIT License
## Copyright (c) 2018 Mamy André-Ratsimbazafy
##
## Adapted from Laser iter02_pertensor.nim for Ceramic tensors.
## Shape/strides passed as array[4, int] instead of Metadata.
##
## Per-tensor strided iteration:
##   Each tensor has its own iter_pos advanced by strides,
##   with backstride adjustment on dimension wrap.

type StrideState*[Rank: static int] = object
  coord: array[Rank, int]
  iterPos: int

proc initStrideState*[Rank: static int](strides, shape: array[Rank, int]): StrideState[Rank] =
  result.iterPos = 0
  for i in 0 ..< Rank:
    result.coord[i] = 0

proc advanceStride*[Rank: static int](state: var StrideState[Rank]; strides, shape: array[Rank, int]) {.inline.} =
  var d = Rank - 1
  while true:
    state.coord[d] += 1
    state.iterPos += strides[d]
    if state.coord[d] < shape[d]:
      break
    # Wrap this dimension
    state.coord[d] = 0
    state.iterPos -= strides[d] * shape[d]  # back over full dimension
    dec d
    if d < 0:
      break

template copy_laser02*[T; Rank: static int](
    dstData: var openArray[T]; dstStrides: array[Rank, int];
    srcData: openArray[T]; srcStrides: array[Rank, int];
    shape: array[Rank, int]) =
  var N = 1
  for i in 0 ..< Rank: N *= shape[i]
  var srcState = initStrideState[Rank](srcStrides, shape)
  var dstState = initStrideState[Rank](dstStrides, shape)
  for _ in 0 ..< N:
    dstData[dstState.iterPos] = srcData[srcState.iterPos]
    advanceStride(srcState, srcStrides, shape)
    advanceStride(dstState, dstStrides, shape)
