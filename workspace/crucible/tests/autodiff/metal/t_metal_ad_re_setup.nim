# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: re-setup invalidates the launch caches
##
## Two same-named kernels with different math, one engine: the second setupAutoDiff must recompile.
## A cached pipeline state for that name would run the old binary:
##
## - first setup forward and backward exact (x2)
## - second setup (same kernel name, +10 math) forward and backward exact
##   against the new source, not the cached pipeline
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_re_setup.nim

import workspace/crucible

type ABuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  p: DevMem
  n: int

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)

  block:   # first kernel named `scale`: multiplies by k
    proc scale(dst, a: ptr UncheckedArray[float32], k: float32) {.global, workgroup: (1, 1).} =
      for i in 0 ..< 4:
        dst[i] = a[i] * k
    const p1 = bkMetal.bwd_diff(scale)
    let ad = engine.setupAutoDiff(p1)
    let aDev = engine.allocDevice(float32, 4)
    let dstDev = engine.allocDevice(float32, 4)
    var aHost: array[4, float32] = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
    engine.copyToDeviceFromHost(aDev, addr aHost[0], 4)
    ad.forward(dstDev, aDev, 2.0'f32)
    var got: array[4, float32]
    engine.copyToHostFromDevice(addr got[0], dstDev, 4)
    for i in 0 ..< 4:
      doAssert got[i] == aHost[i] * 2.0'f32, "p1 forward wrong at " & $i
    let dOutDev = engine.allocDevice(float32, 4)
    var dOutHost: array[4, float32] = [1.0'f32, 1.0'f32, 1.0'f32, 1.0'f32]
    engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], 4)
    let d_a = ad.backward(dOutDev, dstDev, ABuf[typeof(aDev)](p: aDev, n: 4), 2.0'f32)
    var gotA: array[4, float32]
    engine.copyToHostFromDevice(addr gotA[0], d_a.d_a, 4)
    for i in 0 ..< 4:
      doAssert gotA[i] == 2.0'f32, "p1 d_a wrong at " & $i
    echo "[OK] first setup exact (x2 forward, d_a = 2)"

  block:   # second kernel, same name `scale`, different math: adds k
    proc scale(dst, a: ptr UncheckedArray[float32], k: float32) {.global, workgroup: (1, 1).} =
      for i in 0 ..< 4:
        dst[i] = a[i] + k
    const p2 = bkMetal.bwd_diff(scale)
    let ad = engine.setupAutoDiff(p2)
    let aDev = engine.allocDevice(float32, 4)
    let dstDev = engine.allocDevice(float32, 4)
    var aHost: array[4, float32] = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
    engine.copyToDeviceFromHost(aDev, addr aHost[0], 4)
    ad.forward(dstDev, aDev, 10.0'f32)
    var got: array[4, float32]
    engine.copyToHostFromDevice(addr got[0], dstDev, 4)
    for i in 0 ..< 4:
      doAssert got[i] == aHost[i] + 10.0'f32,
        "p2 forward wrong at " & $i & " got " & $got[i] &
        " want " & $(aHost[i] + 10.0'f32) & " (a stale cached pipeline ran)"
    let dOutDev = engine.allocDevice(float32, 4)
    var dOutHost: array[4, float32] = [1.0'f32, 1.0'f32, 1.0'f32, 1.0'f32]
    engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], 4)
    let d_a = ad.backward(dOutDev, dstDev, ABuf[typeof(aDev)](p: aDev, n: 4), 10.0'f32)
    var gotA: array[4, float32]
    engine.copyToHostFromDevice(addr gotA[0], d_a.d_a, 4)
    for i in 0 ..< 4:
      doAssert gotA[i] == 1.0'f32, "p2 d_a wrong at " & $i
    echo "[OK] second setup recompiled, exact (+10 forward, d_a = 1)"

when isMainModule:
  runTest()
