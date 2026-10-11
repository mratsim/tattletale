import std/[sequtils, sugar, strformat]
import workspace/crucible
import ./e03_load_tile_oneshot
proc hostOneStepChecks: bool =
  const mma = e03_load_tile_oneshot.TiledMma[float32, float32, float32, 32, 32, 16](
      atom: APPLE_8x8x8_F32)
  const cfgA = mma.getTVCrdLayout(opA)
  const cfgC = mma.getTVCrdLayout(opC)
  var ok = true
  block loads:
    for (M, K) in [(32, 32), (70, 50), (32, 41), (33, 16)]:
      for (s0, s1) in [(K, 1), (1, M)]:
        var buf = newSeq[float32](max(M*s0, K*s1) + 64)
        for i in 0 ..< buf.len: buf[i] = float32(i mod 251)
        let src = make_view(cast[ptr UncheckedArray[float32]](addr buf[0]),
                            makeIntTuple((M, K)),
                            makeIntTuple((s0, s1)))
        for tid in [0, 7, 31]:
          for tm in 0 ..< (M + 31) div 32:
            for tk in 0 ..< (K + 15) div 16:
              var rE = rt(float32, mma.getMmaLayout(opA))
              rE.loadTileOneStep(src, cfgA, tid, (tm, tk))
              let rV = view(rE)
              for i in 0 ..< size(rE.getLayout().shape):
                let (v, repC) = idx2crd(rE.getLayout().shape, i)
                let cellFlat = crd2idx(mma.atom.getLayout(opA), (tid, v))
                let cellCrd = idx2crd((8, 8), cellFlat)
                let gm = cellCrd[0] + 8*repC[0] + 32*tm
                let gk = cellCrd[1] + 8*repC[1] + 16*tk
                let want = if gm < M and gk < K: buf[gm*s0 + gk*s1] else: 0'f32
                if rV(i) != want:
                  echo &"  loadOneStep A {M}x{K} s({s0},{s1}) tid{tid} tile({tm},{tk}) i {i}:",
                       &" got {rV(i)} want {want}"
                  ok = false
  block stores:
    for (M, N) in [(32, 32), (70, 50), (32, 45), (40, 45)]:
      for (s0, s1) in [(N, 1), (1, M)]:
        var buf = newSeq[float32](max(M*s0, N*s1) + 64)
        var rC = rt(float32, mma.getMmaLayout(opC))
        let rV = view(rC)
        for i in 0 ..< size(rC.getLayout().shape): rV(i) = float32(1000 + i)
        let dst = make_view(cast[ptr UncheckedArray[float32]](addr buf[0]),
                            makeIntTuple((M, N)),
                            makeIntTuple((s0, s1)))
        for tid in [0, 7, 31]:
          for tm in 0 ..< (M + 31) div 32:
            for tn in 0 ..< (N + 31) div 32:
              for i in 0 ..< buf.len: buf[i] = -1'f32
              var expect = newSeqWith(buf.len, -1'f32)
              for i in 0 ..< size(rC.getLayout().shape):
                let (v, repC) = idx2crd(rC.getLayout().shape, i)
                let cellFlat = crd2idx(mma.atom.getLayout(opC), (tid, v))
                let cellCrd = idx2crd((8, 8), cellFlat)
                let gm = cellCrd[0] + 8*repC[0] + 32*tm
                let gn = cellCrd[1] + 8*repC[1] + 32*tn
                if gm < M and gn < N:
                  expect[gm*s0 + gn*s1] = float32(1000 + i)
              dst.storeTileOneStep(rC, cfgC, tid, (tm, tn))
              for m in 0 ..< M:
                for n in 0 ..< N:
                  if buf[m*s0 + n*s1] != expect[m*s0 + n*s1]:
                    echo &"  storeOneStep C {M}x{N} s({s0},{s1}) tid{tid} tile({tm},{tn}) cell({m},{n}):",
                         &" got {buf[m*s0 + n*s1]} want {expect[m*s0 + n*s1]}"
                    ok = false
  echo "[OK] host one-step: loadTileOneStep/storeTileOneStep"
  ok
doAssert hostOneStepChecks()
echo "e03 one-step checks: all green"
