# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: the forward output at a non-zero parameter slot
##
## Test that the pair's launches bind by printed parameter order when the forward output is not parameter 0:
##
## - forward `bump(a, dst)` writing dst: the launch binds the caller's output at dst's slot, exact against the closed form
## - backward convenience and primitive bind by name with the output mid-signature, gradients exact
## - a scalar trailing the mid-signature output keeps its constant-buffer slot, d_alpha exact via the reduce kernel
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_outslot.nim

import workspace/crucible

proc bump(a, dst: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  ## Forward output declared second: dst is the pair's primary output, a the only primal.
  for i in 0 ..< 8:
    dst[i] = 2.0'f32 * a[i]

proc scale(a, dst: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (1, 1).} =
  ## Forward output mid-signature with a scalar after it: the launch must keep
  ## the scalar's constant-buffer slot while the output slots into position 1.
  for i in 0 ..< 8:
    dst[i] = alpha * a[i]

const bumpKernels = cgMetal.bwd_diff(bump)
const scaleKernels = cgMetal.bwd_diff(scale)

type HostBuf[DevMem] = object
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

  const N = 8
  var aHost: array[N, float32]
  var dOutHost: array[N, float32]
  for i in 0 ..< N:
    aHost[i] = 0.5'f32 + 0.125'f32 * float32(i mod 7)
    dOutHost[i] = 1.0'f32 + 0.25'f32 * float32(i mod 5)

  block:   # bump: output at parameter 1, no scalar
    let ad = engine.setupAutoDiff(bumpKernels)
    let aDev = engine.allocDevice(float32, N)
    let outDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], N)

    ad.forward(outDev, aDev)
    var got: array[N, float32]
    engine.copyToHostFromDevice(addr got[0], outDev, N)
    for i in 0 ..< N:
      let expected = 2.0'f32 * aHost[i]
      doAssert got[i] == expected,
        "bump forward: element " & $i & " got " & $got[i] & " want " & $expected
    echo "[OK] bump forward exact, output bound at parameter 1"

    let dOutDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

    let (d_a,) = ad.backward(dOutDev, HostBuf[typeof(aDev)](p: aDev, n: N), outDev)
    var gotA: array[N, float32]
    engine.copyToHostFromDevice(addr gotA[0], d_a, N)
    for i in 0 ..< N:
      let expected = 2.0'f32 * dOutHost[i]
      doAssert gotA[i] == expected,
        "bump d_a: element " & $i & " got " & $gotA[i] & " want " & $expected
    echo "[OK] bump convenience backward exact, d_a allocated from the primal"

    let dADev = engine.allocDevice(float32, N)
    let dDstDev = engine.allocDevice(float32, N)
    let zeros = fillSeq(N, 0.0'f32)
    engine.copyToDeviceFromHost(dADev, addr zeros[0], N)
    engine.copyToDeviceFromHost(dDstDev, addr dOutHost[0], N)
    ad.backward(dADev, dDstDev, aDev, outDev)
    var gotAcc: array[N, float32]
    engine.copyToHostFromDevice(addr gotAcc[0], dADev, N)
    for i in 0 ..< N:
      let expected = 2.0'f32 * dOutHost[i]
      doAssert gotAcc[i] == expected,
        "bump primitive d_a: element " & $i & " got " & $gotAcc[i] & " want " & $expected
    echo "[OK] bump primitive backward exact, gradients in mirror order"

  block:   # scale: output mid-signature, a scalar after it
    let ad = engine.setupAutoDiff(scaleKernels)
    let alpha = 2.0'f32
    let aDev = engine.allocDevice(float32, N)
    let outDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], N)

    ad.forward(outDev, aDev, alpha)
    var got: array[N, float32]
    engine.copyToHostFromDevice(addr got[0], outDev, N)
    for i in 0 ..< N:
      let expected = alpha * aHost[i]
      doAssert got[i] == expected,
        "scale forward: element " & $i & " got " & $got[i] & " want " & $expected
    echo "[OK] scale forward exact, output at parameter 1 with a scalar at parameter 2"

    let dOutDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

    let (d_a, d_alpha) = ad.backward(dOutDev,
      HostBuf[typeof(aDev)](p: aDev, n: N), outDev, alpha)
    var gotA: array[N, float32]
    var gotAlpha: array[1, float32]
    engine.copyToHostFromDevice(addr gotA[0], d_a, N)
    engine.copyToHostFromDevice(addr gotAlpha[0], d_alpha, 1)
    var dAlphaWant = 0.0'f32
    for i in 0 ..< N:
      let expected = alpha * dOutHost[i]
      doAssert gotA[i] == expected,
        "scale d_a: element " & $i & " got " & $gotA[i] & " want " & $expected
      dAlphaWant += dOutHost[i] * aHost[i]
    doAssert gotAlpha[0] == dAlphaWant,
      "scale d_alpha: got " & $gotAlpha[0] & " want " & $dAlphaWant
    echo "[OK] scale convenience backward exact, d_alpha via the reduce kernel"

when isMainModule:
  runTest()
