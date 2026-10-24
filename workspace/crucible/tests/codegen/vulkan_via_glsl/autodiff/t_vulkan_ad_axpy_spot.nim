# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Vulkan GLSL (MoltenVK): axpy spot-check
##
## Test that the same AD pair setup runs the Vulkan GLSL route end to end
## on MoltenVK, exact against the Nim closed form:
##
## - forward exact: `out = alpha * a + b`
## - convenience backward exact: `(d_out, out, a, b, alpha)` allocates the grads at the primals' size, folding `d_alpha` via the reduce kernel.
## - `d_alpha` folds from its 1-element device buffer through the reduce kernel.
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/autodiff/t_vulkan_ad_axpy_spot.nim

import workspace/crucible

proc axpy(`out`, a, b: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (1, 1).} =
  for i in 0 ..< 8:
    `out`[i] = alpha * a[i] + b[i]

const axpyPair = bkGlsl.bwd_diff(axpy)

type AxpyBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience overload reads that length for its grad allocation.
  p: DevMem
  n: int

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runVulkan)
  echo "device: ", engine.deviceName()
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
  ad.forward(outDev, aDev, bDev, alpha)
  var got: array[N, float32]
  engine.copyToHostFromDevice(addr got[0], outDev, N)
  for i in 0 ..< N:
    doAssert got[i] == alpha * aHost[i] + bHost[i],
      "forward: element " & $i & " got " & $got[i] &
      " want " & $(alpha * aHost[i] + bHost[i])
  echo "[OK] forward exact"

  let dOutDev = engine.allocDevice(float32, N)
  var dOutHost: array[N, float32]
  for i in 0 ..< N:
    dOutHost[i] = float32(2 + i)
  engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

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
      "d_a: element " & $i & " got " & $gotA[i] & " want " & $(dOutHost[i] * alpha)
    doAssert gotB[i] == dOutHost[i],
      "d_b: element " & $i & " got " & $gotB[i] & " want " & $dOutHost[i]
    dAlphaWant += dOutHost[i] * aHost[i]
  doAssert gotAlpha[0] == dAlphaWant,
    "d_alpha: got " & $gotAlpha[0] & " want " & $dAlphaWant
  echo "[OK] backward exact, d_alpha a 1-element device buffer via the reduce kernel"

when isMainModule:
  runTest()
