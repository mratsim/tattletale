# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: two AD handles chained through a returned gradient
##
## Test that a convenience backward's returned gradient feeds a second
## handle's backward as its incoming cotangent, with no graph machinery:
##
## - `y1 = alpha1 * a + b`, then `y2 = alpha2 * c + d`
## - `h1.backward(d_out1)` returns `d_a1` and `h2.backward` takes it as `d_out2`
## - closed form: `d_c = d_a1 * alpha2` and `d_d = d_a1` and `d_alpha2 = sum(d_a1 * c)`, all exact
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/autodiff/t_metal_ad_chain.nim

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

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  let h1 = engine.setupAutoDiff(axpyPair)
  let h2 = engine.setupAutoDiff(axpyPair)

  const N = 8
  let aDev = engine.allocDevice(float32, N)
  let bDev = engine.allocDevice(float32, N)
  let y1Dev = engine.allocDevice(float32, N)
  let cDev = engine.allocDevice(float32, N)
  let dDev = engine.allocDevice(float32, N)
  let y2Dev = engine.allocDevice(float32, N)
  var aHost: array[N, float32]
  var bHost: array[N, float32]
  var cHost: array[N, float32]
  var dHost: array[N, float32]
  for i in 0 ..< N:
    aHost[i] = float32(i + 1)
    bHost[i] = float32(100 + i)
    cHost[i] = float32(2 * i)
    dHost[i] = float32(7 - i)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], N)
  engine.copyToDeviceFromHost(bDev, addr bHost[0], N)
  engine.copyToDeviceFromHost(cDev, addr cHost[0], N)
  engine.copyToDeviceFromHost(dDev, addr dHost[0], N)

  let alpha1 = 2.5'f32
  let alpha2 = 3.0'f32
  let dOut1Dev = engine.allocDevice(float32, N)
  var dOut1Host: array[N, float32]
  for i in 0 ..< N:
    dOut1Host[i] = float32(2 + i)
  engine.copyToDeviceFromHost(dOut1Dev, addr dOut1Host[0], N)

  # Stage 1: h1's returned gradients.
  # The grads-to-lets hoist is what the chain requires.
  # Launch args consume move-only values through their handles.
  # The wrappers pass inline, the borrow form keeps the caller's buffers.
  let (d_a1, d_b1, d_alpha1) = h1.backward(dOut1Dev, y1Dev,
    AxpyBuf[typeof(aDev)](p: aDev, n: N), AxpyBuf[typeof(bDev)](p: bDev, n: N),
    alpha1)
  h1.forward(y1Dev, aDev, bDev, alpha1)

  var dA1Host: array[N, float32]
  var dB1Host: array[N, float32]
  var dAlpha1Host: array[1, float32]
  engine.copyToHostFromDevice(addr dA1Host[0], d_a1, N)
  engine.copyToHostFromDevice(addr dB1Host[0], d_b1, N)
  engine.copyToHostFromDevice(addr dAlpha1Host[0], d_alpha1, 1)
  var dAlpha1Want = 0.0'f32
  for i in 0 ..< N:
    doAssert dA1Host[i] == dOut1Host[i] * alpha1,
      "stage 1 d_a1: element " & $i & " got " & $dA1Host[i] &
      " want " & $(dOut1Host[i] * alpha1)
    doAssert dB1Host[i] == dOut1Host[i],
      "stage 1 d_b1: element " & $i & " got " & $dB1Host[i] & " want " & $dOut1Host[i]
    dAlpha1Want += dOut1Host[i] * aHost[i]
  doAssert dAlpha1Host[0] == dAlpha1Want,
    "stage 1 d_alpha1: got " & $dAlpha1Host[0] & " want " & $dAlpha1Want
  echo "[OK] stage 1: h1 backward exact"

  # Stage 2: the returned d_a1 feeds h2's backward as its incoming
  # cotangent at slot 0 of the backward signature, no graph machinery.
  let (d_c, d_d, d_alpha2) = h2.backward(d_a1, y2Dev,
    AxpyBuf[typeof(cDev)](p: cDev, n: N), AxpyBuf[typeof(dDev)](p: dDev, n: N),
    alpha2)
  h2.forward(y2Dev, cDev, dDev, alpha2)

  var y1Host: array[N, float32]
  var y2Host: array[N, float32]
  var dCHost: array[N, float32]
  var dDHost: array[N, float32]
  var dAlpha2Host: array[1, float32]
  engine.copyToHostFromDevice(addr y1Host[0], y1Dev, N)
  engine.copyToHostFromDevice(addr y2Host[0], y2Dev, N)
  engine.copyToHostFromDevice(addr dCHost[0], d_c, N)
  engine.copyToHostFromDevice(addr dDHost[0], d_d, N)
  engine.copyToHostFromDevice(addr dAlpha2Host[0], d_alpha2, 1)
  var dAlpha2Want = 0.0'f32
  for i in 0 ..< N:
    doAssert y1Host[i] == alpha1 * aHost[i] + bHost[i],
      "stage 1 forward: element " & $i & " got " & $y1Host[i]
    doAssert y2Host[i] == alpha2 * cHost[i] + dHost[i],
      "stage 2 forward: element " & $i & " got " & $y2Host[i]
    doAssert dCHost[i] == dA1Host[i] * alpha2,
      "stage 2 d_c: element " & $i & " got " & $dCHost[i] &
      " want " & $(dA1Host[i] * alpha2)
    doAssert dDHost[i] == dA1Host[i],
      "stage 2 d_d: element " & $i & " got " & $dDHost[i] & " want " & $dA1Host[i]
    dAlpha2Want += dA1Host[i] * cHost[i]
  doAssert dAlpha2Host[0] == dAlpha2Want,
    "stage 2 d_alpha2: got " & $dAlpha2Host[0] & " want " & $dAlpha2Want
  echo "[OK] stage 2: returned grad feeds the second handle's d_dst, chain exact"

when isMainModule:
  runTest()
