# load_tile tests: edge cases for the predicated tile loads
# Run: nim c -r workspace/ceramic/experiments/e01_load_tile_tests.nim
#
# Expected tile contents come from the naive guarded read.
# The chunked contract has its own reference: a chunk that straddles
# the world edge zero-fills whole, its in-bounds elements included.

import std/macros

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import ./e01_load_tile

{.experimental: "callOperator".}

proc refTile(M, K, bm, bk, tileM, tileK: int): seq[float32] =
  ## Expected tile contents, the naive guarded read
  result = newSeq[float32](tileM * tileK)
  for j in 0 ..< tileK:
    for i in 0 ..< tileM:
      let (m, k) = (bm * tileM + i, bk * tileK + j)
      result[i + j * tileM] =
        if m >= 0 and m < M and k >= 0 and k < K:
          float32(100 * m + k)
        else:
          0'f32

proc refTileChunked(M, K, bm, bk, tileM, tileK, chunkElems: int): seq[float32] =
  ## Expected chunked-load tile: a chunk that straddles the world
  ## edge zero-fills whole, its in-bounds elements included
  result = refTile(M, K, bm, bk, tileM, tileK)
  for c in 0 ..< result.len div chunkElems:
    var straddles = false
    for e in 0 ..< chunkElems:
      let f = c * chunkElems + e
      let m = bm * tileM + f mod tileM
      let k = bk * tileK + f div tileM
      if not (m >= 0 and m < M and k >= 0 and k < K):
        straddles = true
    if straddles:
      for e in 0 ..< chunkElems:
        result[c * chunkElems + e] = 0'f32

proc checkTile(got, want: seq[float32], name: string) =
  for i in 0 ..< want.len:
    doAssert got[i] == want[i],
      name & ": element " & $i & " got " & $got[i] & " want " & $want[i]

proc runElementLoad(M, K, bm, bk: int, name: string) =
  let (worldData, world) = makeWorld(M, K)
  const tileM = 8
  const tileK = 8
  var buf = newSeq[float32](tileM * tileK)
  var tile = make_view(addr buf[0], make_layout((tileM, tileK), (1, tileM)))
  tile.loadTile(addr worldData[0], world, (bm * tileM, bk * tileK))
  checkTile(buf, refTile(M, K, bm, bk, tileM, tileK), name)

proc runChunkedLoad(M, K, bm, bk: int, name: string) =
  let (worldData, world) = makeWorld(M, K)
  const tileM = 8
  const tileK = 8
  const chunkW = 4   # 4 float32 = 16B, the cp.async chunk
  var buf = newSeq[float32](tileM * tileK)
  var tile = make_view(addr buf[0], make_layout((tileM, tileK), (1, tileM)))
  let preds = chunkPredicates(world, tileM, tileK, chunkW, (bm * tileM, bk * tileK))
  tile.loadTileChunked(addr worldData[0], world, (bm * tileM, bk * tileK), preds, chunkW)
  checkTile(buf, refTileChunked(M, K, bm, bk, tileM, tileK, chunkW), name & " (chunked)")

proc runGranularityContrast(M, K, bm, bk: int) =
  ## Granularity contrast: the straddling chunk zero-fills whole
  ## where the elementwise load keeps the in-bounds elements
  let (worldData, world) = makeWorld(M, K)
  const tileM = 8
  const tileK = 8
  const chunkW = 4
  var buf = newSeq[float32](tileM * tileK)
  var tile = make_view(addr buf[0], make_layout((tileM, tileK), (1, tileM)))
  let origin = (bm * tileM, bk * tileK)
  let preds = chunkPredicates(world, tileM, tileK, chunkW, origin)
  doAssert preds[1] == false          # chunk m 4..7, world 12..15 straddles M = 13
  doAssert preds[0] == true           # chunk m 0..3, world 8..11 fully in bounds
  doAssert preds[2] == true           # chunk k = 1, m 0..3, fully in bounds
  tile.loadTileChunked(addr worldData[0], world, origin, preds, chunkW)
  doAssert buf[4] == 0'f32            # straddling chunk zero-filled whole,
                                      # elementwise load keeps 100*12+0 = 1200 here
  doAssert buf[0] == float32(100 * 8 + 0)  # interior chunk copied

echo "element-granularity loads:"
runElementLoad(32, 32, 1, 1, "interior")                 # nothing OOB
runElementLoad(13, 32, 1, 0, "M edge")                   # rows 13..15 OOB
runElementLoad(32, 17, 0, 2, "K edge")                   # cols 17..23 OOB
runElementLoad(13, 17, 1, 2, "corner")                   # both edges OOB
runElementLoad(5, 8, 0, 0, "world smaller than tile")    # rows 5..7 always OOB.
runElementLoad(8, 8, 0, 0, "exact fit")                  # single block, never OOB.
runElementLoad(8, 8, -1, -1, "negative origin")          # padded view, guards against 0.

block:
  # runtime world shape, the guard stays a runtime bool
  var
    M = 13
    K = 32
  let (worldData, world) = makeWorld(M, K)
  const tileM = 8
  const tileK = 8
  var buf = newSeq[float32](tileM * tileK)
  var tile = make_view(addr buf[0], make_layout((tileM, tileK), (1, tileM)))
  tile.loadTile(addr worldData[0], world, (8, 0))
  checkTile(buf, refTile(M, K, 1, 0, tileM, tileK), "runtime shape")

echo "chunk-granularity loads (copyFromIfAsync / ZFILL contract):"
runChunkedLoad(32, 32, 1, 1, "interior")
runChunkedLoad(13, 32, 1, 0, "M edge")
runChunkedLoad(13, 17, 1, 2, "corner")
runGranularityContrast(13, 32, 1, 0)

echo "GEMM load pattern, all A and B blocks:"

# The full GEMM load legs: A (M, K) and B (N, K) col-major, every block
# loads its (tileM, tileK) tile, and every element is verified against
# the naive guarded read.
block:
  const
    M = 13
    N = 11
    K = 17
    tileM = 8
    tileN = 8
    tileK = 8
  let (aData, aWorld) = makeWorld(M, K)
  var bData = newSeq[float32](N * K)
  for n in 0 ..< N:
    for k in 0 ..< K:
      bData[n + k * N] = float32(1000 * n + k)
  let bWorld = make_layout((N, K), (1, N))

  for bk in 0 ..< (K + tileK - 1) div tileK:
    for bm in 0 ..< (M + tileM - 1) div tileM:
      var buf = newSeq[float32](tileM * tileK)
      var tile = make_view(addr buf[0], make_layout((tileM, tileK), (1, tileM)))
      tile.loadTile(addr aData[0], aWorld, (bm * tileM, bk * tileK))
      checkTile(buf, refTile(M, K, bm, bk, tileM, tileK), "A leg")
    for bn in 0 ..< (N + tileN - 1) div tileN:
      var buf = newSeq[float32](tileN * tileK)
      var tile = make_view(addr buf[0], make_layout((tileN, tileK), (1, tileN)))
      tile.loadTile(addr bData[0], bWorld, (bn * tileN, bk * tileK))
      for j in 0 ..< tileK:
        for i in 0 ..< tileN:
          let (n, k) = (bn * tileN + i, bk * tileK + j)
          let want =
            if n >= 0 and n < N and k >= 0 and k < K:
              float32(1000 * n + k)
            else:
              0'f32
          doAssert buf[i + j * tileN] == want,
            "B leg: (" & $n & "," & $k & ") got " & $buf[i + j * tileN]

echo "load_tile_tests: all green"
