# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: thread-parallel axpy backward at workgroup (1024, 1)
##
## Test that the axpy kernel's thread-parallel form derives a race-free backward at workgroup (1024, 1):
##
## - each thread owns its strided slice `i = t + k * 1024`, the alpha gradient included
## - the convenience backward is exact against the closed form and the CPU central differences,
##   bitwise across launches, the primitive backward accumulates into caller-owned buffers
## - the runtime trip count and both coordinate-index variants stay exact under
##   multi-thread workgroups
## - a write index that does not separate the threads refuses at pair expansion.
##   the workgroup geometry appears in the message (locked in t_ad_expansion_refusals)
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_axpy_thread_parallel.nim

import workspace/crucible

proc axpyTP(dst, a, b: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (1024, 1).} =
  ## Thread t owns the strided slice `i = t + k * 1024`, 4 slots per thread.
  let t = thread_position_in_threadgroup.x
  for k in 0 ..< 4:
    let i = t + k * 1024
    dst[i] = alpha * a[i] + b[i]

proc axpyTPN(dst, a, b: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global, workgroup: (4, 1).} =
  ## Runtime trip count: thread t owns the strided slice `i = t + k * 4`.
  let t = thread_index_in_threadgroup
  for k in 0 ..< n:
    let i = t + k * 4
    dst[i] = alpha * a[i] + b[i]

proc coordIndexTP(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (8, 1).} =
  ## Thread-parallel coordinate-index kernel: thread gx owns the strided
  ## slice `j = gx + i * 8`, the backward's gather indices carry the read.
  let gx = thread_position_in_threadgroup.x
  for i in 0 ..< 8:
    let j = gx + i * 8
    dst[j] = a[j] * 2.0'f32

const axpyTPKernels = cgMetal.bwd_diff(axpyTP)
const axpyTPNKernels = cgMetal.bwd_diff(axpyTPN)
const coordIndexTPKernels = cgMetal.bwd_diff(coordIndexTP)

proc coordFlatTP(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (8, 1).} =
  ## Flat thread identity: thread t owns the strided slice `j = t + i * 8`.
  let t = thread_index_in_threadgroup
  for i in 0 ..< 8:
    let j = t + i * 8
    dst[j] = a[j] * 3.0'f32

const coordFlatTPKernels = cgMetal.bwd_diff(coordFlatTP)

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
  let ad = engine.setupAutoDiff(axpyTPKernels)

  const T = 1024    # threads, one threadgroup
  const K = 4       # strided slots per thread
  const N = T * K

  let aDev = engine.allocDevice(float32, N)
  let bDev = engine.allocDevice(float32, N)
  let outDev = engine.allocDevice(float32, N)
  var aHost: array[N, float32]
  var bHost: array[N, float32]
  for i in 0 ..< N:
    aHost[i] = 0.5'f32 + 0.125'f32 * float32(i mod 7)
    bHost[i] = 1.5'f32 - 0.0625'f32 * float32(i mod 11)
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
    echo "[OK] forward exact, thread-parallel launch (1024, 1)"

  # incoming gradient d_dst
  let dOutDev = engine.allocDevice(float32, N)
  var dOutHost: array[N, float32]
  for i in 0 ..< N:
    dOutHost[i] = 0.25'f32 + 0.5'f32 * float32(i mod 5)
  engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

  # CPU central finite differences of the loss over the closed-form forward
  proc fdA(k: int): float64 =
    let h = 1.0e-3'f32
    var ap = aHost
    var am = aHost
    ap[k] = ap[k] + h
    am[k] = am[k] - h
    var lp = 0.0
    var lm = 0.0
    for i in 0 ..< N:
      lp += float64(dOutHost[i]) * float64(alpha * ap[i] + bHost[i])
      lm += float64(dOutHost[i]) * float64(alpha * am[i] + bHost[i])
    result = (lp - lm) / (2.0 * float64(h))

  block:   # convenience backward: bitwise analytic + finite differences
    let (d_a, d_b, d_alpha) = ad.backward(dOutDev, outDev,
      AxpyBuf[typeof(aDev)](p: aDev, n: N), AxpyBuf[typeof(bDev)](p: bDev, n: N),
      alpha)
    # determinism across launches: a second call into fresh grads matches bitwise
    let (d_a2, d_b2, d_alpha2) = ad.backward(dOutDev, outDev,
      AxpyBuf[typeof(aDev)](p: aDev, n: N), AxpyBuf[typeof(bDev)](p: bDev, n: N),
      alpha)
    var gotA: array[N, float32]
    var gotB: array[N, float32]
    var gotAlpha: array[1, float32]
    var gotA2: array[N, float32]
    var gotB2: array[N, float32]
    var gotAlpha2: array[1, float32]
    engine.copyToHostFromDevice(addr gotA[0], d_a, N)
    engine.copyToHostFromDevice(addr gotB[0], d_b, N)
    engine.copyToHostFromDevice(addr gotAlpha[0], d_alpha, 1)
    engine.copyToHostFromDevice(addr gotA2[0], d_a2, N)
    engine.copyToHostFromDevice(addr gotB2[0], d_b2, N)
    engine.copyToHostFromDevice(addr gotAlpha2[0], d_alpha2, 1)
    for i in 0 ..< N:
      doAssert gotA2[i] == gotA[i],
        "d_a across launches: element " & $i & " first " & $gotA[i] & " second " & $gotA2[i]
      doAssert gotB2[i] == gotB[i],
        "d_b across launches: element " & $i & " first " & $gotB[i] & " second " & $gotB2[i]
    doAssert gotAlpha2[0] == gotAlpha[0],
      "d_alpha across launches: first " & $gotAlpha[0] & " second " & $gotAlpha2[0]
    var dAlphaWant = 0.0'f32
    for i in 0 ..< N:
      let daWant = dOutHost[i] * alpha
      doAssert gotA[i] == daWant,
        "d_a: element " & $i & " got " & $gotA[i] & " want " & $daWant
      doAssert gotB[i] == dOutHost[i],
        "d_b: element " & $i & " got " & $gotB[i] & " want " & $dOutHost[i]
      dAlphaWant += dOutHost[i] * aHost[i]
      let daFd = fdA(i)
      doAssert abs(float64(gotA[i]) - daFd) <= 1.0e-5 + 1.0e-3 * abs(daFd),
        "d_a fd: element " & $i & " got " & $gotA[i] & " want " & $daFd
    doAssert gotAlpha[0] == dAlphaWant,
      "d_alpha: got " & $gotAlpha[0] & " want " & $dAlphaWant
    echo "[OK] convenience backward bitwise vs analytic, matches finite differences"

  block:   # primitive backward: caller-owned pre-zeroed buffers, accumulates
    let dADev = engine.allocDevice(float32, N)
    let dBDev = engine.allocDevice(float32, N)
    let dAlphaDev = engine.allocDevice(float32, 1)
    let zeros = fillSeq(N, 0.0'f32)
    engine.copyToDeviceFromHost(dADev, addr zeros[0], N)
    engine.copyToDeviceFromHost(dBDev, addr zeros[0], N)
    engine.copyToDeviceFromHost(dAlphaDev, addr zeros[0], 1)

    ad.backward(dOutDev, dADev, dBDev, dAlphaDev, outDev, aDev, bDev, alpha)
    ad.backward(dOutDev, dADev, dBDev, dAlphaDev, outDev, aDev, bDev, alpha)
    var dA: array[N, float32]
    var dAlpha: array[1, float32]
    engine.copyToHostFromDevice(addr dA[0], dADev, N)
    engine.copyToHostFromDevice(addr dAlpha[0], dAlphaDev, 1)
    var dAlphaWant = 0.0'f32
    for i in 0 ..< N:
      doAssert dA[i] == 2.0'f32 * (dOutHost[i] * alpha),
        "d_a accumulate: element " & $i & " got " & $dA[i]
      dAlphaWant += dOutHost[i] * aHost[i]
    doAssert dAlpha[0] == 2.0'f32 * dAlphaWant,
      "d_alpha accumulate: got " & $dAlpha[0] & " want " & $(2.0'f32 * dAlphaWant)
    echo "[OK] primitive backward exact and accumulating"


  block:   # runtime trip count: scratch length and reduce trip count derive per launch
    let ad = engine.setupAutoDiff(axpyTPNKernels)
    const T = 4      # threads, one threadgroup
    const K = 3      # loop iterations per thread, a kernel parameter at launch
    const N = T * K
    let aDev = engine.allocDevice(float32, N)
    let bDev = engine.allocDevice(float32, N)
    let outDev = engine.allocDevice(float32, N)
    var aHost: array[N, float32]
    var bHost: array[N, float32]
    for i in 0 ..< N:
      aHost[i] = 0.5'f32 + 0.125'f32 * float32(i mod 7)
      bHost[i] = 1.5'f32 - 0.0625'f32 * float32(i mod 11)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], N)
    engine.copyToDeviceFromHost(bDev, addr bHost[0], N)
    let alpha = 1.5'f32

    ad.forward(outDev, aDev, bDev, alpha, int32(K))
    var got: array[N, float32]
    engine.copyToHostFromDevice(addr got[0], outDev, N)
    for i in 0 ..< N:
      doAssert got[i] == alpha * aHost[i] + bHost[i],
        "runtime forward: element " & $i & " got " & $got[i]

    let dOutDev = engine.allocDevice(float32, N)
    var dOutHost: array[N, float32]
    for i in 0 ..< N:
      dOutHost[i] = 0.25'f32 + 0.5'f32 * float32(i mod 5)
    engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

    let (d_a, d_b, d_alpha) = ad.backward(dOutDev, outDev,
      AxpyBuf[typeof(aDev)](p: aDev, n: N), AxpyBuf[typeof(bDev)](p: bDev, n: N),
      alpha, int32(K))
    var gotA: array[N, float32]
    var gotB: array[N, float32]
    var gotAlpha: array[1, float32]
    engine.copyToHostFromDevice(addr gotA[0], d_a, N)
    engine.copyToHostFromDevice(addr gotB[0], d_b, N)
    engine.copyToHostFromDevice(addr gotAlpha[0], d_alpha, 1)
    var dAlphaWant = 0.0'f32
    for i in 0 ..< N:
      doAssert gotA[i] == dOutHost[i] * alpha,
        "runtime d_a: element " & $i & " got " & $gotA[i]
      doAssert gotB[i] == dOutHost[i],
        "runtime d_b: element " & $i & " got " & $gotB[i]
      dAlphaWant += dOutHost[i] * aHost[i]
    doAssert gotAlpha[0] == dAlphaWant,
      "runtime d_alpha: got " & $gotAlpha[0] & " want " & $dAlphaWant
    echo "[OK] runtime trip count: scratch and reduce derive per launch, bitwise"

    block:   # a second n in the same run: a fresh buffer pair, scratch and reduce re-derive
      const K2 = 2
      const N2 = T * K2
      let a2Dev = engine.allocDevice(float32, N2)
      let b2Dev = engine.allocDevice(float32, N2)
      let out2Dev = engine.allocDevice(float32, N2)
      var a2Host: array[N2, float32]
      var b2Host: array[N2, float32]
      for i in 0 ..< N2:
        a2Host[i] = 0.5'f32 + 0.125'f32 * float32(i mod 7)
        b2Host[i] = 1.5'f32 - 0.0625'f32 * float32(i mod 11)
      engine.copyToDeviceFromHost(a2Dev, addr a2Host[0], N2)
      engine.copyToDeviceFromHost(b2Dev, addr b2Host[0], N2)
      let alpha2 = 2.0'f32

      ad.forward(out2Dev, a2Dev, b2Dev, alpha2, int32(K2))
      var got2: array[N2, float32]
      engine.copyToHostFromDevice(addr got2[0], out2Dev, N2)
      for i in 0 ..< N2:
        doAssert got2[i] == alpha2 * a2Host[i] + b2Host[i],
          "runtime n=2 forward: element " & $i & " got " & $got2[i]

      let dOut2Dev = engine.allocDevice(float32, N2)
      var dOut2Host: array[N2, float32]
      for i in 0 ..< N2:
        dOut2Host[i] = 0.25'f32 + 0.5'f32 * float32(i mod 5)
      engine.copyToDeviceFromHost(dOut2Dev, addr dOut2Host[0], N2)

      let (d_a2, d_b2, d_alpha2) = ad.backward(dOut2Dev, out2Dev,
        AxpyBuf[typeof(a2Dev)](p: a2Dev, n: N2),
        AxpyBuf[typeof(b2Dev)](p: b2Dev, n: N2), alpha2, int32(K2))
      var gotA2: array[N2, float32]
      var gotB2: array[N2, float32]
      var gotAlpha2: array[1, float32]
      engine.copyToHostFromDevice(addr gotA2[0], d_a2, N2)
      engine.copyToHostFromDevice(addr gotB2[0], d_b2, N2)
      engine.copyToHostFromDevice(addr gotAlpha2[0], d_alpha2, 1)
      var dAlpha2Want = 0.0'f32
      for i in 0 ..< N2:
        doAssert gotA2[i] == dOut2Host[i] * alpha2,
          "runtime n=2 d_a: element " & $i & " got " & $gotA2[i]
        doAssert gotB2[i] == dOut2Host[i],
          "runtime n=2 d_b: element " & $i & " got " & $gotB2[i]
        dAlpha2Want += dOut2Host[i] * a2Host[i]
      doAssert gotAlpha2[0] == dAlpha2Want,
        "runtime n=2 d_alpha: got " & $gotAlpha2[0] & " want " & $dAlpha2Want
      echo "[OK] runtime trip count at a second n: scratch and reduce re-derive, bitwise"

  block:   # coordinate-index kernel thread-parallel: gather indices carry the read
    let ad = engine.setupAutoDiff(coordIndexTPKernels)
    const T = 8      # threads, one threadgroup
    const K = 8      # strided slots per thread
    const N = T * K
    let aDev = engine.allocDevice(float32, N)
    let dstDev = engine.allocDevice(float32, N)
    var aHost: array[N, float32]
    var dOutHost: array[N, float32]
    for i in 0 ..< N:
      aHost[i] = 0.5'f32 + 0.25'f32 * float32(i mod 3)
      dOutHost[i] = 1.5'f32 - 0.125'f32 * float32(i mod 11)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], N)

    ad.forward(dstDev, aDev)
    var dstGot: array[N, float32]
    engine.copyToHostFromDevice(addr dstGot[0], dstDev, N)
    for i in 0 ..< N:
      doAssert dstGot[i] == 2.0'f32 * aHost[i],
        "coord forward: element " & $i & " got " & $dstGot[i]

    let dOutDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

    proc fdA(k: int): float64 =
      let h = 1.0e-3'f32
      var ap = aHost
      var am = aHost
      ap[k] = ap[k] + h
      am[k] = am[k] - h
      var lp = 0.0
      var lm = 0.0
      for j in 0 ..< N:
        lp += float64(dOutHost[j]) * float64(2.0'f32 * ap[j])
        lm += float64(dOutHost[j]) * float64(2.0'f32 * am[j])
      result = (lp - lm) / (2.0 * float64(h))

    let (d_a,) = ad.backward(dOutDev, dstDev, AxpyBuf[typeof(aDev)](p: aDev, n: N))
    var gotA: array[N, float32]
    engine.copyToHostFromDevice(addr gotA[0], d_a, N)
    for i in 0 ..< N:
      doAssert gotA[i] == 2.0'f32 * dOutHost[i],
        "coord d_a: element " & $i & " got " & $gotA[i]
      let daFd = fdA(i)
      doAssert abs(float64(gotA[i]) - daFd) <= 1.0e-5 + 1.0e-3 * abs(daFd),
        "coord d_a fd: element " & $i & " got " & $gotA[i] & " want " & $daFd
    echo "[OK] coordinate-index thread-parallel: gather indices carry the read, bitwise"

  block:   # coordinate-index flat thread identity under a multi-thread workgroup
    let ad = engine.setupAutoDiff(coordFlatTPKernels)
    const T = 8      # threads, one threadgroup
    const K = 8      # strided slots per thread
    const N = T * K
    let aDev = engine.allocDevice(float32, N)
    let dstDev = engine.allocDevice(float32, N)
    var aHost: array[N, float32]
    var dOutHost: array[N, float32]
    for i in 0 ..< N:
      aHost[i] = 0.5'f32 + 0.25'f32 * float32(i mod 3)
      dOutHost[i] = 1.5'f32 - 0.125'f32 * float32(i mod 11)
    engine.copyToDeviceFromHost(aDev, addr aHost[0], N)

    ad.forward(dstDev, aDev)
    var dstGot: array[N, float32]
    engine.copyToHostFromDevice(addr dstGot[0], dstDev, N)
    for i in 0 ..< N:
      doAssert dstGot[i] == 3.0'f32 * aHost[i],
        "coordFlat forward: element " & $i & " got " & $dstGot[i]

    let dOutDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

    proc fdA(k: int): float64 =
      let h = 1.0e-3'f32
      var ap = aHost
      var am = aHost
      ap[k] = ap[k] + h
      am[k] = am[k] - h
      var lp = 0.0
      var lm = 0.0
      for j in 0 ..< N:
        lp += float64(dOutHost[j]) * float64(3.0'f32 * ap[j])
        lm += float64(dOutHost[j]) * float64(3.0'f32 * am[j])
      result = (lp - lm) / (2.0 * float64(h))

    let (d_a,) = ad.backward(dOutDev, dstDev, AxpyBuf[typeof(aDev)](p: aDev, n: N))
    var gotA: array[N, float32]
    engine.copyToHostFromDevice(addr gotA[0], d_a, N)
    for i in 0 ..< N:
      doAssert gotA[i] == 3.0'f32 * dOutHost[i],
        "coordFlat d_a: element " & $i & " got " & $gotA[i]
      let daFd = fdA(i)
      doAssert abs(float64(gotA[i]) - daFd) <= 1.0e-5 + 1.0e-3 * abs(daFd),
        "coordFlat d_a fd: element " & $i & " got " & $gotA[i] & " want " & $daFd
    echo "[OK] coordinate-index flat thread identity: forward exact, backward bitwise"

when isMainModule:
  runTest()
