# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: elif conditions in the backward
##
## A branch chain whose elif condition reads a local re-derives that local
## in the backward kernel, so the backward compiles and the elif-gated
## cotangent matches the closed form:
##
## - the backward compiles on Metal (a raw `else if (flag)` has no declaration there)
## - the elif-gated cotangent is exact against the closed form
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_elif.nim

import workspace/crucible

proc pick(dst, alt, a: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  for i in 0 ..< 8:
    let flag = a[i] > 0.5'f32
    if a[i] > 1.0'f32:
      dst[i] = a[i]
    elif flag:
      alt[i] = a[i] * 2.0'f32

const pickPair = bkMetal.bwd_diff(pick)

type ABuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience overload reads that length for its grad allocation.
  p: DevMem
  n: int

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  let ad = engine.setupAutoDiff(pickPair)

  const N = 8
  let aDev = engine.allocDevice(float32, N)
  let dstDev = engine.allocDevice(float32, N)
  let altDev = engine.allocDevice(float32, N)
  var aHost: array[N, float32]
  for i in 0 ..< N:
    aHost[i] = float32(i) * 0.25'f32
  engine.copyToDeviceFromHost(aDev, addr aHost[0], N)

  ad.forward(dstDev, altDev, aDev)
  var gotDst: array[N, float32]
  var gotAlt: array[N, float32]
  engine.copyToHostFromDevice(addr gotDst[0], dstDev, N)
  engine.copyToHostFromDevice(addr gotAlt[0], altDev, N)
  for i in 0 ..< N:
    let wantDst = if aHost[i] > 1.0'f32: aHost[i] else: 0.0'f32
    # the elif runs only when the if is false: 0.5 < a[i] <= 1.0
    let wantAlt = if aHost[i] > 0.5'f32 and aHost[i] <= 1.0'f32:
      aHost[i] * 2.0'f32
    else:
      0.0'f32
    doAssert gotDst[i] == wantDst, "dst: element " & $i & " got " & $gotDst[i]
    doAssert gotAlt[i] == wantAlt, "alt: element " & $i & " got " & $gotAlt[i]
  echo "[OK] forward exact through the branch chain"

  let dDstDev = engine.allocDevice(float32, N)
  let dAltDev = engine.allocDevice(float32, N)
  var dDstHost: array[N, float32]
  var dAltHost: array[N, float32]
  for i in 0 ..< N:
    dDstHost[i] = 1.0'f32
    dAltHost[i] = 1.0'f32
  engine.copyToDeviceFromHost(dDstDev, addr dDstHost[0], N)
  engine.copyToDeviceFromHost(dAltDev, addr dAltHost[0], N)

  let d_a = ad.backward(dDstDev, dAltDev, dstDev, altDev,
    ABuf[typeof(aDev)](p: aDev, n: N))
  var gotA: array[N, float32]
  engine.copyToHostFromDevice(addr gotA[0], d_a.d_a, N)
  for i in 0 ..< N:
    # dst[i] = a[i] gated by a[i] > 1. alt[i] = a[i] * 2 is gated by
    # the elif's local flag (a[i] > 0.5), re-derived in the backward.
    # The elif branch runs only when the if branch did not.
    var want = 0.0'f32
    if aHost[i] > 1.0'f32:
      want += dDstHost[i]
    elif aHost[i] > 0.5'f32:
      want += dAltHost[i] * 2.0'f32
    doAssert gotA[i] == want,
      "d_a: element " & $i & " got " & $gotA[i] & " want " & $want
  echo "[OK] backward exact, the elif's local condition re-derived"

when isMainModule:
  runTest()
