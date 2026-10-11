import std/[monotimes, strformat, times]
import workspace/crucible
import ./e04_load_tile_oneshot
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import workspace/ceramic/src/macros/static_for

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
            rA.loadTile(src, cfgA, tid, (tm, tk))
            acc += rA[0]
    let ns = inNanoseconds(getMonotime() - t0)
    lastNs = ns
    let dt = ns.float / 1e9
    let bytes = float(reps * tilesM * tilesK * 32 * 16 * 4)
    best = min(best, bytes / dt / 1e9)
  echo &"  {name} {M}x{K} s({s0},{s1}): {best:.2f} GB/s, rep {lastNs} ns, acc {acc:.0f}"
echo "e04 bench (short runs, load only):"
benchLoad("one-step", 1024, 1024, 1024, 1, 4)
benchLoad("one-step", 1024, 1024, 1, 1024, 4)
benchLoad("one-step", 1024, 1024, 7, 3, 4)
