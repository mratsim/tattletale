# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GEMM smokes at the Qwen3-0.6B shapes on Metal.
##
## The two largest GEMMs of the model:
##
## - FFN: (128, 1024) x (1024, 3072)^T, full output judged against
##   a sequential f32 host reference
## - LMHead: (8, 1024) x (1024, 151936)^T, biggest GEMM of the model,
##   judged on a random column slice plus a wall-time report
##
## Judgment: within 4 bfloat16 ulps of the reference per the harness's kReduction-class allowance at depth 1, not byte-exact.

import std/[math, random, strutils, times]
import workspace/crucible
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/tensor_datatypes
import workspace/tattleformers/src/ops_2i1o/matmul
import workspace/tattleformers/src/layers/linear
import workspace/tattleformers/tests/ops_tests_template

export tensor_datatypes

proc runGemmSmokes*[E: HwEngine](engine: E, name: string) =
  var rng = initRand(20261109)

  # ── FFN shape: (128, 1024) x (1024, 3072)^T ───────────────────────────
  block:
    const
      M = 128
      K = 1024
      N = 3072
    var xHost = newSeq[bfloat16](M * K)
    var wHost = newSeq[bfloat16](N * K)
    for i in 0 ..< M * K:
      xHost[i] = rng.rand(-8.0 .. 8.0).to(bfloat16)
    for i in 0 ..< N * K:
      wHost[i] = rng.rand(-8.0 .. 8.0).to(bfloat16)
    let w = newTensor(engine, @[N, K], kBf16)
    w.copyToDeviceFromHost(wHost)
    let layer = Linear.init(w)
    let x = newTensor(engine, @[M, K], kBf16)
    x.copyToDeviceFromHost(xHost)
    let y = layer.forward(x)
    doAssert y.node.getImpl().shape()[0] == M
    doAssert y.node.getImpl().shape()[1] == N
    var yRoots = @[y.node]
    y.engine.eval(yRoots.toSpan())
    var got = newSeq[bfloat16](M * N)
    copyToHostFromDevice(got, y)
    var refO = newSeq[float32](M * N)
    for m in 0 ..< M:
      for n in 0 ..< N:
        var acc = 0.0'f32
        for k in 0 ..< K:
          acc += xHost[m * K + k].to(float32) * wHost[n * K + k].to(float32)
        refO[m * N + n] = acc
    got.checkBf16Ulp(refO, 4, name & ": FFN shape " & $M & "x" & $K & "x" & $N)

  # ── LMHead shape: (8, 1024) x (1024, 151936)^T ────────────────────────
  block:
    const
      M = 8
      K = 1024
      N = 151936
    var xHost = newSeq[bfloat16](M * K)
    var wHost = newSeq[bfloat16](N * K)
    for i in 0 ..< M * K:
      xHost[i] = rng.rand(-8.0 .. 8.0).to(bfloat16)
    for i in 0 ..< N * K:
      wHost[i] = rng.rand(-8.0 .. 8.0).to(bfloat16)
    let w = newTensor(engine, @[N, K], kBf16)
    w.copyToDeviceFromHost(wHost)
    let layer = Linear.init(w)
    let x = newTensor(engine, @[M, K], kBf16)
    x.copyToDeviceFromHost(xHost)
    let y = layer.forward(x)
    doAssert y.node.getImpl().shape()[0] == M
    doAssert y.node.getImpl().shape()[1] == N

    var yRoots = @[y.node]
    let t0 = epochTime()
    y.engine.eval(yRoots.toSpan())
    let evalMs = (epochTime() - t0) * 1000.0

    # random column slice judged against the host reference
    const SliceN = 64
    var cols: array[SliceN, int]
    for i in 0 ..< SliceN:
      cols[i] = rng.rand(0 ..< N)
    var got = newSeq[bfloat16](M * N)
    copyToHostFromDevice(got, y)
    var refO = newSeq[float32](M * SliceN)
    for m in 0 ..< M:
      for j in 0 ..< SliceN:
        let n = cols[j]
        var acc = 0.0'f32
        for k in 0 ..< K:
          acc += xHost[m * K + k].to(float32) * wHost[n * K + k].to(float32)
        refO[m * SliceN + j] = acc
    var gotSlice = newSeq[bfloat16](M * SliceN)
    for m in 0 ..< M:
      for j in 0 ..< SliceN:
        gotSlice[m * SliceN + j] = got[m * N + cols[j]]
    gotSlice.checkBf16Ulp(refO, 4,
      name & ": LMHead shape " & $M & "x" & $K & "x" & $N)

    echo name & ": LMHead GEMM (M=" & $M & ", N=" & $N & ", K=" & $K &
      ") eval wall time: ", evalMs.formatFloat(ffDecimal, 1), " ms"

  echo "tattleformers gemm smokes (", name, "): all checks passed"
