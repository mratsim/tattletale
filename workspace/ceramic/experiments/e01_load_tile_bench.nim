import std/[macros, monotimes, strformat, times]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import ./e01_load_tile
{.experimental: "callOperator".}

const TileM = 128
const TileK = 128
const ChunkW = 4

proc makeWorldLayout(M, K: int, kind: static string): auto =
  when kind == "row-major":
    make_layout((M, K), (K, 1))
  elif kind == "col-major":
    make_layout((M, K), (1, M))
  else:
    make_layout((M, K), (1, K + 7))

proc makeData(M, K, stM, stK: int): seq[float32] =
  result = newSeq[float32](stM * (M - 1) + stK * (K - 1) + 1)
  for m in 0 ..< M:
    for k in 0 ..< K:
      result[m * stM + k * stK] = float32(m + k)

proc bench(name: string, M, K: int, kind: static string, chunked: bool) =
  const edgeM = 3
  const edgeK = 5
  let data = when kind == "row-major": makeData(M, K, K, 1)
             elif kind == "col-major": makeData(M, K, 1, M)
             else: makeData(M, K, 1, K + 7)
  let world = makeWorldLayout(M, K, kind)
  let tile = make_layout((TileM, TileK), (1, TileM))
  var buf = newSeq[float32](TileM * TileK)
  var tview = make_view(addr buf[0], tile)
  let gridM = (M + TileM - 1) div TileM
  let gridK = (K + TileK - 1) div TileK
  var loaded = 0'i64
  var filled = 0'i64
  var best = float64.high
  for pass in 0 ..< 3:
    let t0 = getMonotime()
    loaded = 0
    filled = 0
    for bk in 0 ..< gridK:
      for bm in 0 ..< gridM:
        let origin = (bm * TileM, bk * TileK)
        if chunked:
          let preds = chunkPredicates(world, TileM, TileK, ChunkW, origin)
          tview.loadTileChunked(addr data[0], world, origin, preds, ChunkW)
          for p in preds:
            if p: loaded += ChunkW
            else: filled += ChunkW
        else:
          tview.loadTile(addr data[0], world, origin)
          let inM = min(min(TileM, M - bm * TileM), TileM)
          let inK = min(min(TileK, K - bk * TileK), TileK)
          let inB = max(0, inM) * max(0, inK)
          loaded += inB
          filled += TileM * TileK - inB
    let secs = float64(inNanoseconds(getMonotime() - t0)) / 1e9
    best = min(best, secs)
  let gbs = float64(4 * loaded + 4 * filled) / best / 1e9
  echo &"{name:<38} {kind:<10} chunks={chunked}  loaded={loaded:>9}  filled={filled:>8}  {gbs:8.2f} GB/s"
echo "load_tile bench, tile ", TileM, "x", TileK, " float32, best of 3:"
echo ""
bench("interior", 4096, 4096, "row-major", false)
bench("interior", 4096, 4096, "col-major", false)
bench("strided (ld = K+7), K edge", 4096, 4089, "strided", false)
bench("M edge", 4096 + 3, 4096, "row-major", false)
bench("M edge", 4096 + 3, 4096, "col-major", false)
bench("strided (ld = K+7), M+K edge", 4096 + 3, 4089 + 5, "strided", false)
bench("interior chunked", 4096, 4096, "row-major", true)
bench("interior chunked", 4096, 4096, "col-major", true)
bench("M edge chunked", 4096 + 3, 4096, "row-major", true)
bench("M edge chunked", 4096 + 3, 4096, "col-major", true)
