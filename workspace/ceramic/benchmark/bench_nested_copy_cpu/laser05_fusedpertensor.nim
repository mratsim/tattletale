## MIT License
## Copyright (c) 2018 Mamy André-Ratsimbazafy
##
## Adapted from Laser iter05_fusedpertensor.nim for Ceramic tensors.
## Shape/strides passed as array[4, int] instead of Metadata.
##
## Fused per-tensor strided iteration:
##   Single shared coord. Each tensor advances its iter_pos
##   via backstride = stride * (shape - 1) on dimension wrap.
##   Coord advance is done ONCE per element (shared across tensors).

type FusedState*[Rank: static int] = object
  coord: array[Rank, int]
  srcPos: int
  dstPos: int

proc initFused*[Rank: static int](srcStrides, dstStrides, shape: array[Rank, int]): FusedState[Rank] =
  result.srcPos = 0
  result.dstPos = 0
  for i in 0 ..< Rank:
    result.coord[i] = 0

proc advanceFused*[Rank: static int](state: var FusedState[Rank]; srcStrides, dstStrides, shape: array[Rank, int]) {.inline.} =
  ## Advance shared coord and per-tensor positions.
  for k in countdown(Rank - 1, 0):
    if state.coord[k] < shape[k] - 1:
      state.coord[k] += 1
      state.srcPos += srcStrides[k]
      state.dstPos += dstStrides[k]
      return
    else:
      state.coord[k] = 0
      state.srcPos -= srcStrides[k] * (shape[k] - 1)  # backstride
      state.dstPos -= dstStrides[k] * (shape[k] - 1)

template copy_laser05*[T; Rank: static int](
    dstData: var openArray[T]; dstStrides: array[Rank, int];
    srcData: openArray[T]; srcStrides: array[Rank, int];
    shape: array[Rank, int]) =
  ## Full copy loop via fused per-tensor strided advance.
  var N = 1
  for i in 0 ..< Rank: N *= shape[i]
  var state = initFused[Rank](srcStrides, dstStrides, shape)
  for _ in 0 ..< N:
    dstData[state.dstPos] = srcData[state.srcPos]
    advanceFused(state, srcStrides, dstStrides, shape)
