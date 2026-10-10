# Short bench: e03 one-step tile loads, CPU.
# Small world, few reps, best of 3. Absolute numbers are CPU-gather scale,
# the stride-order ratios are the signal.
import std/[monotimes, strformat, times]
import workspace/crucible
import ./e03_load_tile_oneshot

const mma = TiledMma[float32, float32, float32, 32, 32, 16](atom: APPLE_8x8x8_F32)
const cfgA = mma.getTVCrdLayout(opA)

proc benchLoad(name: static string, M, K, s0, s1: int, reps: int) =
  var buf = newSeq[float32](M*K + 64)
  for i in 0 ..< buf.len: buf[i] = float32(i mod 251)
  let src = make_view(cast[ptr UncheckedArray[float32]](addr buf[0]),
                      makeIntTuple((M, K)), makeIntTuple((s0, s1)))
  let tilesM = (M + 31) div 32
  let tilesK = (K + 15) div 16
  var acc = 0.0
  var best = float.high
  var lastNs = 0i64
  for rep in 0 ..< 3:
    let t0 = getMonotime()
    for r in 0 ..< reps:
      for tm in 0 ..< tilesM:
        for tk in 0 ..< tilesK:
          for tid in 0 ..< 32:
            var rA = rt(float32, mma.getMmaLayout(opA))
            rA.loadTileOneStep(src, cfgA, tid, (tm, tk))
            acc += rA.data[0]
    let ns = inNanoseconds(getMonotime() - t0)
    lastNs = ns
    let dt = ns.float / 1e9
    let bytes = float(reps * tilesM * tilesK * 32 * 16 * 4)
    best = min(best, bytes / dt / 1e9)
  echo &"  {name} {M}x{K} s({s0},{s1}): {best:.2f} GB/s, rep {lastNs} ns, acc {acc:.0f}"

echo "e03 bench (short runs, one-step only):"
benchLoad("one-step", 1024, 1024, 1024, 1, 4)
benchLoad("one-step", 1024, 1024, 1, 1024, 4)
benchLoad("one-step", 1024, 1024, 7, 3, 4)
