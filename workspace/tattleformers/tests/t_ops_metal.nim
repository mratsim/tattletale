# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tattleformers ops on metal
##
## Test that the tattleformers ops and layers run end to end on the Metal
## engine in bfloat16:
##
## - ops under test: embed gather, add, matmul, rmsnorm
## - layers under test: Embedding, Linear, RmsNorm
## - one graph through the open eval-walk dispatch, per-node dtype dispatch
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wipM04 --outdir:build/wip \
##     workspace/tattleformers/tests/t_ops_metal.nim

import std/math
import workspace/crucible
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/tattleformers/tests/ops_tests_template
import workspace/tattleformers/src/layers/norm

proc main =
  # RmsNorm(w)(x) through the erased face, judged against an f64 host
  # reference at the kElementwise-class band (2 bf16 ulps).
  # This call sits in the driver, a non-generic module: inside the generic
  # runTest body the call-operator lookup does not resolve.
  block:
    const N = 8
    let engine = gMetalEngine().toAnyEngine()
    var wHost: array[N, bfloat16]
    var xHost: array[2 * N, bfloat16]
    for i in 0 ..< N:
      wHost[i] = (float32(i) * 0.1'f32 + 0.5'f32).to(bfloat16)
    for i in 0 ..< 2 * N:
      xHost[i] = (float32((i * 13) mod 7 - 3) * 0.5'f32).to(bfloat16)
    let w = newTensor(engine, @[N], kBf16)
    w.copyToDeviceFromHost(wHost)
    let x = newTensor(engine, @[2, N], kBf16)
    x.copyToDeviceFromHost(xHost)

    let rms = RmsNorm.init(w)
    let y = rms(x)
    doAssert y.node.getImpl().shape().len == 2
    doAssert y.node.getImpl().shape()[0] == 2
    doAssert y.node.getImpl().shape()[1] == N
    doAssert y.node.getImpl().dtype() == kBf16
    var yRoots = @[y.node]
    y.engine.eval(yRoots.toSpan())
    var got = newSeq[bfloat16](2 * N)
    copyToHostFromDevice(got, y)

    for row in 0 ..< 2:
      var ss = 0.0'f64
      for i in 0 ..< N:
        let v = float64(xHost[row * N + i].to(float32))
        ss += v * v
      let scale = 1.0 / sqrt(ss / float64(N) + 1e-6)
      for i in 0 ..< N:
        let want = (float64(xHost[row * N + i].to(float32)) * scale *
                    float64(wHost[i].to(float32))).to(bfloat16).to(float32)
        let d = abs(ulpKey16(got[row * N + i]) - ulpKey16(want.to(bfloat16)))
        doAssert d <= 2,
          "RmsNorm(w)(x) (" & $row & "," & $i & "): got " &
          $got[row * N + i].to(float32) & ", want " & $want & ", " & $d & " ulps"

  runTest(gMetalEngine(), "Metal")

when isMainModule:
  main()
