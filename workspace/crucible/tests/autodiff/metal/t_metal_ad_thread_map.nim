# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: thread-parallel elementwise map backward
##
## Test that a coordinate-driven map kernel derives a race-free thread-parallel backward,
## bitwise-exact against the analytic closed form and matching the CPU central differences:
##
## - the pair launches at the forward's declared workgroup (16, 1), one thread per element pair
## - the backward writes only the thread's own gradient slots, no atomics
## - coordinate builtins are zero-gradient index terms, the reads pass through primal
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_thread_map.nim

import workspace/crucible

proc mapPair(dst, acc, a, w, b: ptr UncheckedArray[float32]) {.global, workgroup: (16, 1).} =
  ## Thread j owns the even slot of dst and the odd slot of acc:
  ## `dst[2j] = a[2j] * w[j]`, `acc[2j+1] = a[2j+1] * b[j]`.
  let j = thread_position_in_threadgroup.x
  dst[2 * j] = a[2 * j] * w[j]
  acc[2 * j + 1] = a[2 * j + 1] * b[j]

proc mapFlat(dst, a, b: ptr UncheckedArray[float32]) {.global, workgroup: (16, 1).} =
  ## Flat-index variant: `thread_index_in_threadgroup` is an int builtin read.
  let t = thread_index_in_threadgroup
  dst[t] = a[t] * b[t]

const mapPairKernels = cgMetal.bwd_diff(mapPair)
const mapFlatKernels = cgMetal.bwd_diff(mapFlat)

type MapBuf[DevMem] = object
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
  const T = 16      # threads, one threadgroup
  const S = 32      # stride-2 buffers, one slot pair per thread

  let aDev = engine.allocDevice(float32, S)
  let wDev = engine.allocDevice(float32, T)
  let bDev = engine.allocDevice(float32, T)
  let dstDev = engine.allocDevice(float32, S)
  let accDev = engine.allocDevice(float32, S)
  var aHost: array[S, float32]
  var wHost: array[T, float32]
  var bHost: array[T, float32]
  for i in 0 ..< S:
    aHost[i] = 0.5'f32 + 0.125'f32 * float32(i mod 8)
  for i in 0 ..< T:
    wHost[i] = 0.75'f32 + 0.25'f32 * float32(i mod 4)
    bHost[i] = 1.5'f32 - 0.125'f32 * float32(i mod 5)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], S)
  engine.copyToDeviceFromHost(wDev, addr wHost[0], T)
  engine.copyToDeviceFromHost(bDev, addr bHost[0], T)

  block:   # forward: exact against the closed form
    let ad = engine.setupAutoDiff(mapPairKernels)
    ad.forward(dstDev, accDev, aDev, wDev, bDev)
    var dstGot: array[S, float32]
    var accGot: array[S, float32]
    engine.copyToHostFromDevice(addr dstGot[0], dstDev, S)
    engine.copyToHostFromDevice(addr accGot[0], accDev, S)
    for j in 0 ..< T:
      let dstWant = aHost[2 * j] * wHost[j]
      let accWant = aHost[2 * j + 1] * bHost[j]
      doAssert dstGot[2 * j] == dstWant,
        "map forward dst: j " & $j & " got " & $dstGot[2 * j] & " want " & $dstWant
      doAssert accGot[2 * j + 1] == accWant,
        "map forward acc: j " & $j & " got " & $accGot[2 * j + 1] & " want " & $accWant
    echo "[OK] map forward exact, thread-parallel launch (16, 1)"

  # incoming gradients of the two forward outputs
  let dDstDev = engine.allocDevice(float32, S)
  let dAccDev = engine.allocDevice(float32, S)
  var dDstHost: array[S, float32]
  var dAccHost: array[S, float32]
  for i in 0 ..< S:
    dDstHost[i] = 0.25'f32 + 0.5'f32 * float32(i mod 3)
    dAccHost[i] = 1.0'f32 - 0.25'f32 * float32(i mod 4)
  engine.copyToDeviceFromHost(dDstDev, addr dDstHost[0], S)
  engine.copyToDeviceFromHost(dAccDev, addr dAccHost[0], S)

  # CPU central finite differences of the loss over the closed-form forward
  proc fdA(k: int): float64 =
    let h = 1.0e-3'f32
    var ap = aHost
    var am = aHost
    ap[k] = ap[k] + h
    am[k] = am[k] - h
    var lp = 0.0
    var lm = 0.0
    for j in 0 ..< T:
      lp += float64(dDstHost[2 * j]) * float64(ap[2 * j] * wHost[j])
      lp += float64(dAccHost[2 * j + 1]) * float64(ap[2 * j + 1] * bHost[j])
      lm += float64(dDstHost[2 * j]) * float64(am[2 * j] * wHost[j])
      lm += float64(dAccHost[2 * j + 1]) * float64(am[2 * j + 1] * bHost[j])
    result = (lp - lm) / (2.0 * float64(h))

  block:   # convenience backward: bitwise analytic + finite differences
    let ad = engine.setupAutoDiff(mapPairKernels)
    let grads = ad.backward(dDstDev, dAccDev, dstDev, accDev,
      MapBuf[typeof(aDev)](p: aDev, n: S),
      MapBuf[typeof(wDev)](p: wDev, n: T),
      MapBuf[typeof(bDev)](p: bDev, n: T))
    var daGot: array[S, float32]
    var dwGot: array[T, float32]
    var dbGot: array[T, float32]
    engine.copyToHostFromDevice(addr daGot[0], grads.d_a, S)
    engine.copyToHostFromDevice(addr dwGot[0], grads.d_w, T)
    engine.copyToHostFromDevice(addr dbGot[0], grads.d_b, T)
    for j in 0 ..< T:
      # analytic closed form, same f32 expression tree as the backward emits
      let daEvenWant = dDstHost[2 * j] * wHost[j]
      let daOddWant = dAccHost[2 * j + 1] * bHost[j]
      let dwWant = dDstHost[2 * j] * aHost[2 * j]
      let dbWant = dAccHost[2 * j + 1] * aHost[2 * j + 1]
      doAssert daGot[2 * j] == daEvenWant,
        "d_a even: j " & $j & " got " & $daGot[2 * j] & " want " & $daEvenWant
      doAssert daGot[2 * j + 1] == daOddWant,
        "d_a odd: j " & $j & " got " & $daGot[2 * j + 1] & " want " & $daOddWant
      doAssert dwGot[j] == dwWant,
        "d_w: j " & $j & " got " & $dwGot[j] & " want " & $dwWant
      doAssert dbGot[j] == dbWant,
        "d_b: j " & $j & " got " & $dbGot[j] & " want " & $dbWant
      # CPU central differences, one coordinate at a time
      let daEvenFd = fdA(2 * j)
      let daOddFd = fdA(2 * j + 1)
      doAssert abs(float64(daGot[2 * j]) - daEvenFd) <= 1.0e-5 + 1.0e-3 * abs(daEvenFd),
        "d_a fd even: j " & $j & " got " & $daGot[2 * j] & " want " & $daEvenFd
      doAssert abs(float64(daGot[2 * j + 1]) - daOddFd) <= 1.0e-5 + 1.0e-3 * abs(daOddFd),
        "d_a fd odd: j " & $j & " got " & $daGot[2 * j + 1] & " want " & $daOddFd
    echo "[OK] map backward bitwise vs analytic, matches finite differences"

  block:   # primitive backward: caller-owned pre-zeroed buffers, accumulates
    let ad = engine.setupAutoDiff(mapPairKernels)
    let daDev = engine.allocDevice(float32, S)
    let dwDev = engine.allocDevice(float32, T)
    let dbDev = engine.allocDevice(float32, T)
    let zerosS = fillSeq(S, 0.0'f32)
    let zerosT = fillSeq(T, 0.0'f32)
    engine.copyToDeviceFromHost(daDev, addr zerosS[0], S)
    engine.copyToDeviceFromHost(dwDev, addr zerosT[0], T)
    engine.copyToDeviceFromHost(dbDev, addr zerosT[0], T)
    ad.backward(dDstDev, dAccDev, daDev, dwDev, dbDev, dstDev, accDev, aDev, wDev, bDev)
    ad.backward(dDstDev, dAccDev, daDev, dwDev, dbDev, dstDev, accDev, aDev, wDev, bDev)
    var daGot: array[S, float32]
    engine.copyToHostFromDevice(addr daGot[0], daDev, S)
    for j in 0 ..< T:
      doAssert daGot[2 * j] == 2.0'f32 * (dDstHost[2 * j] * wHost[j]),
        "d_a accumulate even: j " & $j & " got " & $daGot[2 * j]
      doAssert daGot[2 * j + 1] == 2.0'f32 * (dAccHost[2 * j + 1] * bHost[j]),
        "d_a accumulate odd: j " & $j & " got " & $daGot[2 * j + 1]
    echo "[OK] map primitive backward accumulates into caller buffers"

  block:   # flat-index variant: thread_index_in_threadgroup differentiates too
    let ad = engine.setupAutoDiff(mapFlatKernels)
    let xDev = engine.allocDevice(float32, T)
    let yDev = engine.allocDevice(float32, T)
    let outDev = engine.allocDevice(float32, T)
    engine.copyToDeviceFromHost(xDev, addr aHost[0], T)
    engine.copyToDeviceFromHost(yDev, addr wHost[0], T)
    ad.forward(outDev, xDev, yDev)
    var outGot: array[T, float32]
    engine.copyToHostFromDevice(addr outGot[0], outDev, T)
    for t in 0 ..< T:
      let want = aHost[t] * wHost[t]
      doAssert outGot[t] == want,
        "flat forward: t " & $t & " got " & $outGot[t] & " want " & $want
    let dOutDev = engine.allocDevice(float32, T)
    engine.copyToDeviceFromHost(dOutDev, addr dDstHost[0], T)
    let grads = ad.backward(dOutDev, outDev,
      MapBuf[typeof(xDev)](p: xDev, n: T),
      MapBuf[typeof(yDev)](p: yDev, n: T))
    var daGot: array[T, float32]
    engine.copyToHostFromDevice(addr daGot[0], grads.d_a, T)
    for t in 0 ..< T:
      let want = dDstHost[t] * wHost[t]
      doAssert daGot[t] == want,
        "flat d_a: t " & $t & " got " & $daGot[t] & " want " & $want
    echo "[OK] flat-index map backward exact"

when isMainModule:
  runTest()
