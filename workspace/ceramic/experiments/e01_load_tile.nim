# load_tile: predicated GEMM tile loads via inBounds, the pycute
# OutOfBoundsAccessor pattern in ceramic
#
# References:
# - pycute examples, im2col.py OutOfBoundsAccessor and gemm.ipynb
# - GPU Mode 114 transcript 81:15, the identity coordinate tensor OOB check
# - cutlass cooperative_gemm.hpp and sm80_mma_multistage.hpp
# - ceramic basis-strides-squashed k_layout_copy_gpu.nim, copyFromIfAsync
#
# Layout: A is (M, K) col-major, strides (1, M), the GEMM A convention.
# A tile at block (bm, bk) is a (tileM, tileK) view into the world tensor
# with the world's own strides, offset by crd2idx at the block origin.
# The world coordinate of a tile element is tileCrd + blockOrigin,
# and the load guard is worldCrd.inBounds(worldShape):
#
#   tileCrd --zip+--> worldCrd --inBounds?--> load world(worldCrd)
#                                     \-----> else fill (0)
#
# Two predication granularities:
# - element granularity, a per-element guard, nothing is read OOB
# - chunk granularity, one bool per 16B chunk, a chunk that straddles
#   the world edge zero-fills whole, the copyFromIfAsync ZFILL contract
#
# Tests live in e01_load_tile_tests.nim.
# Throughput numbers live in e01_load_tile_bench.nim.

import std/macros

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

{.experimental: "callOperator".}

proc loadTile*[T, ShT, StT](tile: var TensorView[T, ShT, StT], worldData: ptr T, world: Layout, blockOrigin: tuple) =
  ## Tile <- world, one guarded read per element,
  ## out-of-bounds elements take the fill value instead
  let w = cast[ptr UncheckedArray[T]](worldData)
  let worldShape = world.shape
  for f in 0 ..< tile.size():
    let tileCrd = idx2crd(tile.getLayout(), f)
    let worldCrd = zipDimensionsWith(tileCrd, blockOrigin): it_a + it_b
    let dstOff = toInt(crd2idx(tile.getLayout(), tileCrd))
    if worldCrd.inBounds(worldShape):
      tile.data[dstOff] = w[toInt(crd2idx(world, worldCrd))]
    else:
      tile.data[dstOff] = default(T)

proc chunkPredicates*(world: Layout, tileM, tileK, chunkElems: int, blockOrigin: tuple): seq[bool] =
  ## One bool per chunk of chunkElems elements along the tile's M rows,
  ## true iff every element of the chunk is in bounds of the world
  let worldShape = world.shape
  let numChunks = tileM * tileK div chunkElems
  result = newSeq[bool](numChunks)
  for c in 0 ..< numChunks:
    var all = true
    for e in 0 ..< chunkElems:
      let f = c * chunkElems + e
      let tileCrd = (f mod tileM, f div tileM)
      let worldCrd = zipDimensionsWith(tileCrd, blockOrigin): it_a + it_b
      all = all and worldCrd.inBounds(worldShape)
    result[c] = all

proc loadTileChunked*[T, ShT, StT](tile: var TensorView[T, ShT, StT],
                        worldData: ptr T,
                        world: Layout,
                        blockOrigin: tuple,
                        preds: seq[bool],
                        chunkElems: static int) =
  ## Tile <- world, one guarded read per 16B chunk, ZFILL contract:
  ## pred false zero-fills the whole chunk, pred true copies it whole
  let w = cast[ptr UncheckedArray[T]](worldData)
  let worldShape = world.shape
  for f in 0 ..< tile.size():
    let c = f div chunkElems
    let tileCrd = idx2crd(tile.getLayout(), f)
    let worldCrd = zipDimensionsWith(tileCrd, blockOrigin): it_a + it_b
    let dstOff = toInt(crd2idx(tile.getLayout(), tileCrd))
    if preds[c]:
      tile.data[dstOff] = w[toInt(crd2idx(world, worldCrd))]
    else:
      tile.data[dstOff] = default(T)

proc makeWorld*(M, K: int): auto =
  ## A (M, K) col-major world, element (m, k) = 100·m + k
  var data = newSeq[float32](M * K)
  for m in 0 ..< M:
    for k in 0 ..< K:
      data[m + k * M] = float32(100 * m + k)
  (data, make_layout((M, K), (1, M)))
