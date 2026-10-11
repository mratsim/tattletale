import std/macros
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
{.experimental: "callOperator".}

proc loadTile*[T, ShT, StT](tile: var TensorView[T, ShT, StT], worldData: ptr T, world: Layout, blockOrigin: tuple) =
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
  var data = newSeq[float32](M * K)
  for m in 0 ..< M:
    for k in 0 ..< K:
      data[m + k * M] = float32(100 * m + k)
  (data, make_layout((M, K), (1, M)))
