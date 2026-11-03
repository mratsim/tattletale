# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: atomic and barrier builtins in the derivation
##
## Test that atomic builtins differentiate as statements and barriers keep their backward position, bitwise-exact against the closed form.
##
## - atomicFetchAdd into a shared slot: the value's gradient is the target
##   slot's incoming gradient, accumulated atomically into the shadow adjoint
## - threadgroup_barrier: the backward keeps the barrier before the gradient write
## - the fixtures are integers, every atomic ordering is exact in float32
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_atomic_barrier.nim

import std/strutils
import workspace/crucible

proc atomicAccK(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (16, 1).} =
  ## Sixteen threads atomic-add their element into one shared slot.
  let j = thread_position_in_threadgroup.x
  discard atomicFetchAdd(addr dst[0], a[j])

proc barrierMapK(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (16, 1).} =
  ## A barrier between the coordinate read and the map write, the backward keeps it.
  let j = thread_position_in_threadgroup.x
  threadgroup_barrier()
  dst[j] = a[j] * 4.0'f32

const atomicAccPair = cgMetal.bwd_diff(atomicAccK)
const barrierMapPair = cgMetal.bwd_diff(barrierMapK)

type SyncBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience overload reads that length for its grad allocation.
  p: DevMem
  n: int

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  const T = 16

  block:   # atomicFetchAdd: forward accumulates, the value's gradient is the slot's
    let ad = engine.setupAutoDiff(atomicAccPair)
    let aDev = engine.allocDevice(float32, T)
    let dstDev = engine.allocDevice(float32, 1)
    var aHost: array[T, float32]
    for i in 0 ..< T:
      aHost[i] = float32(i + 1)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], T)

    ad.forward(dstDev, aDev)
    var dstGot: array[1, float32]
    engine.copyToHostFromDevice(addr dstGot[0], dstDev, 1)
    var sumWant = 0.0'f32
    for i in 0 ..< T:
      sumWant += aHost[i]
    doAssert dstGot[0] == sumWant,
      "atomic forward: got " & $dstGot[0] & " want " & $sumWant

    let dDstDev = engine.allocDevice(float32, 1)
    let dDstHost: array[1, float32] = [3.5'f32]
    engine.copyToDeviceFromHost(dDstDev, addr dDstHost[0], 1)
    let grads = ad.backward(dDstDev, dstDev, SyncBuf[typeof(aDev)](p: aDev, n: T))
    var daGot: array[T, float32]
    engine.copyToHostFromDevice(addr daGot[0], grads.d_a, T)
    for j in 0 ..< T:
      doAssert daGot[j] == dDstHost[0],
        "atomic d_a: j " & $j & " got " & $daGot[j] & " want " & $dDstHost[0]
    echo "[OK] atomicFetchAdd backward: the value's gradient is the slot's incoming gradient"

  block:   # threadgroup_barrier: same sync structure in the backward
    doAssert "threadgroup_barrier" in barrierMapPair.fwdSource,
      "the forward source must keep the barrier"
    doAssert "threadgroup_barrier" in barrierMapPair.bwdSource,
      "the backward source must mirror the barrier"
    doAssert barrierMapPair.bwdSource.find("threadgroup_barrier") <
      barrierMapPair.bwdSource.find("d_a["),
      "the backward keeps the barrier before the gradient write"
    let ad = engine.setupAutoDiff(barrierMapPair)
    let aDev = engine.allocDevice(float32, T)
    let dstDev = engine.allocDevice(float32, T)
    var aHost: array[T, float32]
    var dDstHost: array[T, float32]
    for i in 0 ..< T:
      aHost[i] = 0.5'f32 + 0.25'f32 * float32(i)
      dDstHost[i] = 1.0'f32 + 0.5'f32 * float32(i mod 3)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], T)

    ad.forward(dstDev, aDev)
    var dstGot: array[T, float32]
    engine.copyToHostFromDevice(addr dstGot[0], dstDev, T)
    for j in 0 ..< T:
      let want = aHost[j] * 4.0'f32
      doAssert dstGot[j] == want,
        "barrier forward: j " & $j & " got " & $dstGot[j] & " want " & $want

    let dDstDev = engine.allocDevice(float32, T)
    engine.copyToDeviceFromHost(dDstDev, addr dDstHost[0], T)
    let grads = ad.backward(dDstDev, dstDev, SyncBuf[typeof(aDev)](p: aDev, n: T))
    var daGot: array[T, float32]
    engine.copyToHostFromDevice(addr daGot[0], grads.d_a, T)
    for j in 0 ..< T:
      let want = dDstHost[j] * 4.0'f32
      doAssert daGot[j] == want,
        "barrier d_a: j " & $j & " got " & $daGot[j] & " want " & $want
    echo "[OK] threadgroup_barrier mirrors into the backward, map values exact"

when isMainModule:
  runTest()
