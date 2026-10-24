# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: axpy forward and backward end to end
##
## Test that `engine.setupAutoDiff` turns a `bwd_diff` kernel pair into an AD
## handle whose launches are exact against the Nim closed form:
##
## - forward exact: `dst = alpha * a + b`
## - primitive backward exact and accumulating on caller-owned pre-zeroed buffers
## - convenience backward exact: allocated grads sized off the primals, `d_alpha` a 1-element reduce-kernel buffer
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_axpy.nim

import workspace/crucible

proc axpy(dst, a, b: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (1, 1).} =
  for i in 0 ..< 8:
    dst[i] = alpha * a[i] + b[i]

const axpyPair = bkMetal.bwd_diff(axpy)

type AxpyBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience overload reads that length for its grad allocation.
  p: DevMem
  n: int

proc fillSeq[T](n: int, value: T): seq[T] =
  result = newSeq[T](n)
  for i in 0 ..< n:
    result[i] = value

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  let ad = engine.setupAutoDiff(axpyPair)

  const N = 8
  let aDev = engine.allocDevice(float32, N)
  let bDev = engine.allocDevice(float32, N)
  let outDev = engine.allocDevice(float32, N)
  var aHost: array[N, float32]
  var bHost: array[N, float32]
  for i in 0 ..< N:
    aHost[i] = float32(i + 1)
    bHost[i] = float32(100 + i)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], N)
  engine.copyToDeviceFromHost(bDev, addr bHost[0], N)

  let alpha = 2.5'f32

  block:   # forward: exact against the closed form
    ad.forward(outDev, aDev, bDev, alpha)
    var got: array[N, float32]
    engine.copyToHostFromDevice(addr got[0], outDev, N)
    for i in 0 ..< N:
      let expected = alpha * aHost[i] + bHost[i]
      doAssert got[i] == expected,
        "forward: element " & $i & " got " & $got[i] & " want " & $expected
    echo "[OK] forward exact"

  # backward inputs: the incoming cotangent d_dst
  let dOutDev = engine.allocDevice(float32, N)
  var dOutHost: array[N, float32]
  for i in 0 ..< N:
    dOutHost[i] = float32(2 + i)
  engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

  block:   # primitive backward: caller-owned pre-zeroed buffers, accumulates
    let dADev = engine.allocDevice(float32, N)
    let dBDev = engine.allocDevice(float32, N)
    let dAlphaDev = engine.allocDevice(float32, 1)
    let zeros = fillSeq(N, 0.0'f32)
    engine.copyToDeviceFromHost(dADev, addr zeros[0], N)
    engine.copyToDeviceFromHost(dBDev, addr zeros[0], N)
    engine.copyToDeviceFromHost(dAlphaDev, addr zeros[0], 1)

    ad.backward(dOutDev, dADev, dBDev, dAlphaDev, outDev, aDev, bDev, alpha)
    var dA: array[N, float32]
    var dB: array[N, float32]
    var dAlpha: array[1, float32]
    engine.copyToHostFromDevice(addr dA[0], dADev, N)
    engine.copyToHostFromDevice(addr dB[0], dBDev, N)
    engine.copyToHostFromDevice(addr dAlpha[0], dAlphaDev, 1)
    var dAlphaWant = 0.0'f32
    for i in 0 ..< N:
      doAssert dA[i] == dOutHost[i] * alpha,
        "d_a: element " & $i & " got " & $dA[i] & " want " & $(dOutHost[i] * alpha)
      doAssert dB[i] == dOutHost[i],
        "d_b: element " & $i & " got " & $dB[i] & " want " & $dOutHost[i]
      dAlphaWant += dOutHost[i] * aHost[i]
    doAssert dAlpha[0] == dAlphaWant,
      "d_alpha: got " & $dAlpha[0] & " want " & $dAlphaWant

    # a second launch accumulates into the caller's buffers, scalar cotangent
    # included: the reduce kernel accumulates like the buffer gradients do.
    ad.backward(dOutDev, dADev, dBDev, dAlphaDev, outDev, aDev, bDev, alpha)
    engine.copyToHostFromDevice(addr dA[0], dADev, N)
    engine.copyToHostFromDevice(addr dAlpha[0], dAlphaDev, 1)
    for i in 0 ..< N:
      doAssert dA[i] == 2.0'f32 * dOutHost[i] * alpha,
        "d_a accumulate: element " & $i & " got " & $dA[i] &
        " want " & $(2.0'f32 * dOutHost[i] * alpha)
    doAssert dAlpha[0] == 2.0'f32 * dAlphaWant,
      "d_alpha accumulate: got " & $dAlpha[0] & " want " & $(2.0'f32 * dAlphaWant)
    echo "[OK] primitive backward exact and accumulating"

  block:   # convenience backward: grads allocated from the primals' sizes
    let (d_a, d_b, d_alpha) = ad.backward(dOutDev, outDev,
      AxpyBuf[typeof(aDev)](p: aDev, n: N), AxpyBuf[typeof(bDev)](p: bDev, n: N),
      alpha)
    var gotA: array[N, float32]
    var gotB: array[N, float32]
    var gotAlpha: array[1, float32]
    engine.copyToHostFromDevice(addr gotA[0], d_a, N)
    engine.copyToHostFromDevice(addr gotB[0], d_b, N)
    engine.copyToHostFromDevice(addr gotAlpha[0], d_alpha, 1)
    var dAlphaWant = 0.0'f32
    for i in 0 ..< N:
      doAssert gotA[i] == dOutHost[i] * alpha,
        "convenience d_a: element " & $i & " got " & $gotA[i] &
        " want " & $(dOutHost[i] * alpha)
      doAssert gotB[i] == dOutHost[i],
        "convenience d_b: element " & $i & " got " & $gotB[i] & " want " & $dOutHost[i]
      dAlphaWant += dOutHost[i] * aHost[i]
    doAssert gotAlpha[0] == dAlphaWant,
      "convenience d_alpha: got " & $gotAlpha[0] & " want " & $dAlphaWant
    echo "[OK] convenience backward exact, d_alpha a 1-element device buffer"

when isMainModule:
  runTest()
