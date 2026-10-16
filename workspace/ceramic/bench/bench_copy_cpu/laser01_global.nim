## Adapted from Laser iter01_global.nim for Ceramic tensors.
## Coord wheel-winding + per-element offset (sum(coord[i] * stride[i])).
## No div/mod. Rank is compile-time static.

type CoordState*[Rank: static int] = object
  coord: array[Rank, int]

template prodShape*[Rank: static int](shape: array[Rank, int]): int =
  var sz = shape[0]
  for i in 1 ..< Rank: sz *= shape[i]
  sz

proc offset*[Rank: static int](strides: array[Rank, int]; coord: array[Rank, int]): int {.inline.} =
  result = 0
  for i in 0 ..< Rank:
    result += coord[i] * strides[i]

proc incrCoord*[Rank: static int](state: var CoordState[Rank]; shape: array[Rank, int]) {.inline.} =
  for k in countdown(Rank - 1, 0):
    if state.coord[k] < shape[k] - 1:
      state.coord[k] += 1
      return
    else:
      state.coord[k] = 0

template copy_laser01*[Rank: static int](
    dstData: var openArray[float32]; dstStrides: array[Rank, int];
    srcData: openArray[float32]; srcStrides: array[Rank, int];
    shape: array[Rank, int]) =
  let N = prodShape[Rank](shape)
  var state: CoordState[Rank]
  for _ in 0 ..< N:
    let srcOff = offset[Rank](srcStrides, state.coord)
    let dstOff = offset[Rank](dstStrides, state.coord)
    dstData[dstOff] = srcData[srcOff]
    incrCoord[Rank](state, shape)
