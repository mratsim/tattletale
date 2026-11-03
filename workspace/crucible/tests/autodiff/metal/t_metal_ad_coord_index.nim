# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: coordinate builtins as zero-gradient index terms
##
## Test that coordinate-builtin reads differentiate as primal index terms with bitwise-exact values.
##
## - the vector read substitutes untouched into the backward's gather indices.
## - the flat int read substitutes the same way into the scatter indices.
## - the reads carry zero gradient and emit no gradient statements.
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_coord_index.nim

import workspace/crucible

proc coordIndex(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  ## Vector-coordinate variant: the read substitutes into the backward's indices.
  let gx = thread_position_in_threadgroup.x
  for i in 0 ..< 8:
    dst[i] = a[gx + i] * 2.0'f32

proc coordFlat(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  ## Flat-int variant: `thread_index_in_threadgroup` is an int builtin read.
  let t = thread_index_in_threadgroup
  for i in 0 ..< 8:
    dst[i] = a[t + i] * 3.0'f32

type CoordBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience overload reads that length for its grad allocation.
  p: DevMem
  n: int

const coordIndexPair = cgMetal.bwd_diff(coordIndex)
const coordFlatPair = cgMetal.bwd_diff(coordFlat)

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  const N = 8

  block:   # vector coordinate: the backward's indices carry the read
    let ad = engine.setupAutoDiff(coordIndexPair)
    let aDev = engine.allocDevice(float32, N)
    let dstDev = engine.allocDevice(float32, N)
    var aHost: array[N, float32]
    var dDstHost: array[N, float32]
    for i in 0 ..< N:
      aHost[i] = 0.5'f32 + 0.25'f32 * float32(i)
      dDstHost[i] = 1.5'f32 - 0.125'f32 * float32(i)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], N)
    ad.forward(dstDev, aDev)
    var dstGot: array[N, float32]
    engine.copyToHostFromDevice(addr dstGot[0], dstDev, N)
    for i in 0 ..< N:
      let want = aHost[i] * 2.0'f32
      doAssert dstGot[i] == want,
        "coord forward: i " & $i & " got " & $dstGot[i] & " want " & $want

    let dDstDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(dDstDev, addr dDstHost[0], N)
    let grads = ad.backward(dDstDev, dstDev, CoordBuf[typeof(aDev)](p: aDev, n: N))
    var daGot: array[N, float32]
    engine.copyToHostFromDevice(addr daGot[0], grads.d_a, N)
    for i in 0 ..< N:
      let want = dDstHost[i] * 2.0'f32
      doAssert daGot[i] == want,
        "coord d_a: i " & $i & " got " & $daGot[i] & " want " & $want
    echo "[OK] thread_position_in_threadgroup.x flows through the derivation"

  block:   # flat coordinate: same contract for the int builtin
    let ad = engine.setupAutoDiff(coordFlatPair)
    let aDev = engine.allocDevice(float32, N)
    let dstDev = engine.allocDevice(float32, N)
    var aHost: array[N, float32]
    var dDstHost: array[N, float32]
    for i in 0 ..< N:
      aHost[i] = 1.0'f32 + 0.5'f32 * float32(i)
      dDstHost[i] = 0.5'f32 + 0.25'f32 * float32(i)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], N)
    ad.forward(dstDev, aDev)
    var dstGot: array[N, float32]
    engine.copyToHostFromDevice(addr dstGot[0], dstDev, N)
    for i in 0 ..< N:
      let want = aHost[i] * 3.0'f32
      doAssert dstGot[i] == want,
        "flat forward: i " & $i & " got " & $dstGot[i] & " want " & $want

    let dDstDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(dDstDev, addr dDstHost[0], N)
    let grads = ad.backward(dDstDev, dstDev, CoordBuf[typeof(aDev)](p: aDev, n: N))
    var daGot: array[N, float32]
    engine.copyToHostFromDevice(addr daGot[0], grads.d_a, N)
    for i in 0 ..< N:
      let want = dDstHost[i] * 3.0'f32
      doAssert daGot[i] == want,
        "flat d_a: i " & $i & " got " & $daGot[i] & " want " & $want
    echo "[OK] thread_index_in_threadgroup flows through the derivation"

when isMainModule:
  runTest()
