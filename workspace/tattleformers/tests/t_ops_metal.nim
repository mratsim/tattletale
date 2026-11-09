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
## - ops under test: embed gather, add, matmul, rmsnorm, rope
## - layers under test: Embedding, Linear, RmsNorm, RotaryPositionEmbedding
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
import workspace/tattleformers/src/layers/ffn
import workspace/tattleformers/src/layers/linear

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

  # GatedDenseFFN(wGate, wUp, wDown)(x) through the erased face.
  # An f64 host reference judges it at the kElementwise-class band
  # (2 bf16 ulps).
  # This call sits in the driver, a non-generic module: inside the generic
  # runTest body the call-operator lookup does not resolve.
  block:
    const
      Tokens = 3
      Hidden = 8
      Inter = 16
    let engine = gMetalEngine().toAnyEngine()
    var xHost = newSeq[bfloat16](Tokens * Hidden)
    var wgHost = newSeq[bfloat16](Inter * Hidden)
    var wuHost = newSeq[bfloat16](Inter * Hidden)
    var wdHost = newSeq[bfloat16](Hidden * Inter)
    for i in 0 ..< Tokens * Hidden:
      xHost[i] = (float32((i * 13) mod 7 - 3) * 0.5'f32).to(bfloat16)
    for i in 0 ..< Inter * Hidden:
      wgHost[i] = (float32((i * 11) mod 17 - 8) * 0.125'f32).to(bfloat16)
      wuHost[i] = (float32((i * 7) mod 13 - 6) * 0.125'f32).to(bfloat16)
    for i in 0 ..< Hidden * Inter:
      wdHost[i] = (float32((i * 17) mod 19 - 9) * 0.125'f32).to(bfloat16)
    let x = newTensor(engine, @[Tokens, Hidden], kBf16)
    x.copyToDeviceFromHost(xHost)
    let wg = newTensor(engine, @[Inter, Hidden], kBf16)
    wg.copyToDeviceFromHost(wgHost)
    let wu = newTensor(engine, @[Inter, Hidden], kBf16)
    wu.copyToDeviceFromHost(wuHost)
    let wd = newTensor(engine, @[Hidden, Inter], kBf16)
    wd.copyToDeviceFromHost(wdHost)

    let ffn = GatedDenseFFN.init(Linear.init(wg), Linear.init(wu), Linear.init(wd))
    let y = ffn(x)
    doAssert y.node.getImpl().shape()[0] == Tokens
    doAssert y.node.getImpl().shape()[1] == Hidden
    doAssert y.node.getImpl().dtype() == kBf16
    var yRoots = @[y.node]
    y.engine.eval(yRoots.toSpan())
    var got = newSeq[bfloat16](Tokens * Hidden)
    copyToHostFromDevice(got, y)

    # f64 host reference of the whole layer.
    # Each stage reference applies the kernel's own rounding points:
    # every projection and the activation round to bfloat16 where the device stores them
    var gate = newSeq[bfloat16](Tokens * Inter)
    var up = newSeq[bfloat16](Tokens * Inter)
    for t in 0 ..< Tokens:
      for n in 0 ..< Inter:
        var accG = 0.0'f32
        var accU = 0.0'f32
        for k in 0 ..< Hidden:
          let xv = xHost[t * Hidden + k].to(float32)
          accG += xv * wgHost[n * Hidden + k].to(float32)
          accU += xv * wuHost[n * Hidden + k].to(float32)
        gate[t * Inter + n] = accG.to(bfloat16)
        up[t * Inter + n] = accU.to(bfloat16)
    var act = newSeq[float32](Tokens * Inter)
    for i in 0 ..< Tokens * Inter:
      let g = float64(gate[i].to(float32))
      let v = float64(up[i].to(float32))
      act[i] = (g / (1.0 + exp(-g)) * v).to(bfloat16).to(float32)
    for t in 0 ..< Tokens:
      for n in 0 ..< Hidden:
        var acc = 0.0'f32
        for k in 0 ..< Inter:
          acc += act[t * Inter + k] * wdHost[n * Inter + k].to(float32)
        let want = acc.to(bfloat16).to(float32)
        let d = abs(ulpKey16(got[t * Hidden + n]) - ulpKey16(want.to(bfloat16)))
        doAssert d <= 2,
          "GatedDenseFFN(w)(x) (" & $t & "," & $n & "): got " &
          $got[t * Hidden + n].to(float32) & ", want " & $want & ", " & $d & " ulps"

  type Mem = typeof(gMetalEngine().allocDevice(bfloat16, 0))
  runTest[Mem](gMetalEngine().toAnyEngine(), "Metal")

when isMainModule:
  main()
