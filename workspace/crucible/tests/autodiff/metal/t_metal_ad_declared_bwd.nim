# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: the declared-backward escape hatch end to end
##
## Test that a hand-written backward declared through `bwd_diff(fwd, bwd)`
## launches exactly like the transposed pair and that the save wiring keeps
## the forward outputs alive between `.forward` and `.backward`:
##
## - declared axpy backward ≡ transposed axpy backward, exact, primitive and convenience, same backward signature
## - the declared backward accumulates the scalar cotangent in-kernel, its primitive doubles on a second call,
##   the transposed rewrites from a fresh scratch, the documented accumulation-contract difference
## - FA-shaped stub (q, k, v, dst, lse): the forward writes dst and lse,
##   the convenience backward reads those same device buffers and produces the closed-form d_q/d_k/d_v,
##   the save wiring round-trips one buffer set
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_declared_bwd.nim

import workspace/crucible

# ── Fixtures ──────────────────────────────────────────────────────────────

proc axpy(dst, a, b: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (1, 1).} =
  for i in 0 ..< 8:
    dst[i] = alpha * a[i] + b[i]

proc axpy_bwd(d_dst, d_a, d_b, d_alpha: ptr UncheckedArray[float32],
              dst, a, b: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (1, 1).} =
  # Hand-written mirror of the transposed axpy backward: same signature,
  # the scalar cotangent accumulated in-kernel, no scratch and no reduce kernel.
  for i in 0 ..< 8:
    d_a[i] += d_dst[i] * alpha
    d_b[i] += d_dst[i]
    d_alpha[0] += d_dst[i] * a[i]

proc sdpa_fwd(q, k, v, dst, lse: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  # FA-shaped stub: five forward parameters, dst and lse are the saved
  # forward outputs. Trivial math, no SDPA.
  for i in 0 ..< 8:
    dst[i] = q[i] + k[i] + v[i]
  lse[0] = dst[0] + 1.0'f32

proc sdpa_fwd_bwd(d_q, d_k, d_v, d_dst, d_lse: ptr UncheckedArray[float32],
                  q, k, v, dst, lse: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  # Declared backward in mirror order, gradients first:
  # (d_q, d_k, d_v, d_dst, d_lse) in forward-parameter order,
  # block 2 the forward signature verbatim.
  # Reads dst and lse, the saved outputs, so the save wiring is observable.
  for i in 0 ..< 8:
    d_q[i] += d_dst[i] * dst[i] * 0.5'f32
    d_k[i] += d_dst[i] * lse[0]
    d_v[i] += d_dst[i] + d_lse[0]
  d_lse[0] += d_dst[0]

const axpyPair = bkMetal.bwd_diff(axpy)
const axpyDeclaredPair = bkMetal.bwd_diff(axpy, axpy_bwd)
const sdpaPair = bkMetal.bwd_diff(sdpa_fwd, sdpa_fwd_bwd, save =  [dst, lse])

# ── Host helpers ──────────────────────────────────────────────────────────

type HostBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience overload reads that length for its grad allocation.
  p: DevMem
  n: int

proc fillSeq[T](n: int, value: T): seq[T] =
  result = newSeq[T](n)
  for i in 0 ..< n:
    result[i] = value

proc copyBack[E](engine: E, dev: auto, len: int): seq[float32] =
  ## Device buffer contents as a host sequence.
  result = newSeq[float32](len)
  engine.copyToHostFromDevice(addr result[0], dev, len)

proc runTest() =   # private: engines are destroyed at return
  # One engine per kernel-name namespace: ingestADKernel caches by kernel
  # name per engine, and both pairs ingest "axpy_bwd" with different sources.
  var engineT = newEngine(runMetal)   # transposed axpy pair
  var engineD = newEngine(runMetal)   # declared axpy pair and the sdpa stub

  const N = 8
  let aDev = engineD.allocDevice(float32, N)
  let bDev = engineD.allocDevice(float32, N)
  let outDev = engineD.allocDevice(float32, N)
  var aHost: array[N, float32]
  var bHost: array[N, float32]
  for i in 0 ..< N:
    aHost[i] = float32(i + 1)
    bHost[i] = float32(100 + i)
  engineD.copyToDeviceFromHost(aDev, addr aHost[0], N)
  engineD.copyToDeviceFromHost(bDev, addr bHost[0], N)
  let alpha = 2.5'f32

  let dOutDev = engineD.allocDevice(float32, N)
  var dOutHost: array[N, float32]
  for i in 0 ..< N:
    dOutHost[i] = float32(2 + i)
  engineD.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

  var dAlphaWant = 0.0'f32
  for i in 0 ..< N:
    dAlphaWant += dOutHost[i] * aHost[i]

  block:   # declared axpy: primitive backward, one call, exact
    let ad = engineD.setupAutoDiff(axpyDeclaredPair)
    ad.forward(outDev, aDev, bDev, alpha)
    var got: array[N, float32]
    engineD.copyToHostFromDevice(addr got[0], outDev, N)
    for i in 0 ..< N:
      let expected = alpha * aHost[i] + bHost[i]
      doAssert got[i] == expected,
        "declared forward: element " & $i & " got " & $got[i] &
        " want " & $expected

    let dADev = engineD.allocDevice(float32, N)
    let dBDev = engineD.allocDevice(float32, N)
    let dAlphaDev = engineD.allocDevice(float32, 1)
    let zeros = fillSeq(N, 0.0'f32)
    engineD.copyToDeviceFromHost(dADev, addr zeros[0], N)
    engineD.copyToDeviceFromHost(dBDev, addr zeros[0], N)
    engineD.copyToDeviceFromHost(dAlphaDev, addr zeros[0], 1)

    ad.backward(dOutDev, dADev, dBDev, dAlphaDev, outDev, aDev, bDev, alpha)
    let dA = engineD.copyBack(dADev, N)
    let dB = engineD.copyBack(dBDev, N)
    let dAlpha = engineD.copyBack(dAlphaDev, 1)
    for i in 0 ..< N:
      doAssert dA[i] == dOutHost[i] * alpha,
        "declared d_a: element " & $i & " got " & $dA[i] &
        " want " & $(dOutHost[i] * alpha)
      doAssert dB[i] == dOutHost[i],
        "declared d_b: element " & $i & " got " & $dB[i] & " want " & $dOutHost[i]
    doAssert dAlpha[0] == dAlphaWant,
      "declared d_alpha: got " & $dAlpha[0] & " want " & $dAlphaWant

    # Second call: the declared kernel accumulates in-kernel with no scratch rewrite,
    # the caller's buffers double.
    ad.backward(dOutDev, dADev, dBDev, dAlphaDev, outDev, aDev, bDev, alpha)
    let dA2 = engineD.copyBack(dADev, N)
    let dAlpha2 = engineD.copyBack(dAlphaDev, 1)
    for i in 0 ..< N:
      doAssert dA2[i] == 2.0'f32 * dOutHost[i] * alpha,
        "declared d_a accumulate: element " & $i & " got " & $dA2[i] &
        " want " & $(2.0'f32 * dOutHost[i] * alpha)
    doAssert dAlpha2[0] == 2.0'f32 * dAlphaWant,
      "declared d_alpha accumulate: got " & $dAlpha2[0] &
      " want " & $(2.0'f32 * dAlphaWant)
    stdout.writeLine("[OK] declared axpy primitive backward exact, accumulates in-kernel")

  block:   # transposed axpy: primitive backward, one call, exact
    let ad = engineT.setupAutoDiff(axpyPair)
    ad.forward(outDev, aDev, bDev, alpha)
    let dADev = engineT.allocDevice(float32, N)
    let dBDev = engineT.allocDevice(float32, N)
    let dAlphaDev = engineT.allocDevice(float32, 1)
    let zeros = fillSeq(N, 0.0'f32)
    engineT.copyToDeviceFromHost(dADev, addr zeros[0], N)
    engineT.copyToDeviceFromHost(dBDev, addr zeros[0], N)
    engineT.copyToDeviceFromHost(dAlphaDev, addr zeros[0], 1)

    ad.backward(dOutDev, dADev, dBDev, dAlphaDev, outDev, aDev, bDev, alpha)
    let dA = engineT.copyBack(dADev, N)
    let dB = engineT.copyBack(dBDev, N)
    let dAlpha = engineT.copyBack(dAlphaDev, 1)
    for i in 0 ..< N:
      doAssert dA[i] == dOutHost[i] * alpha,
        "transposed d_a: element " & $i & " got " & $dA[i] &
        " want " & $(dOutHost[i] * alpha)
      doAssert dB[i] == dOutHost[i],
        "transposed d_b: element " & $i & " got " & $dB[i] & " want " & $dOutHost[i]
    doAssert dAlpha[0] == dAlphaWant,
      "transposed d_alpha: got " & $dAlpha[0] & " want " & $dAlphaWant
    stdout.writeLine("[OK] transposed axpy primitive backward exact")

  block:   # declared ≡ transposed: convenience backward, element-wise exact
    let adD = engineD.setupAutoDiff(axpyDeclaredPair)
    let adT = engineT.setupAutoDiff(axpyPair)
    adD.forward(outDev, aDev, bDev, alpha)
    adT.forward(outDev, aDev, bDev, alpha)

    let (d_aD, d_bD, d_alphaD) = adD.backward(dOutDev, outDev,
      HostBuf[typeof(aDev)](p: aDev, n: N), HostBuf[typeof(bDev)](p: bDev, n: N),
      alpha)
    let (d_aT, d_bT, d_alphaT) = adT.backward(dOutDev, outDev,
      HostBuf[typeof(aDev)](p: aDev, n: N), HostBuf[typeof(bDev)](p: bDev, n: N),
      alpha)

    let gotAD = engineD.copyBack(d_aD, N)
    let gotBD = engineD.copyBack(d_bD, N)
    let gotAlphaD = engineD.copyBack(d_alphaD, 1)
    let gotAT = engineT.copyBack(d_aT, N)
    let gotBT = engineT.copyBack(d_bT, N)
    let gotAlphaT = engineT.copyBack(d_alphaT, 1)
    for i in 0 ..< N:
      doAssert gotAD[i] == gotAT[i],
        "convenience d_a: element " & $i & " declared " & $gotAD[i] &
        " transposed " & $gotAT[i]
      doAssert gotAD[i] == dOutHost[i] * alpha,
        "convenience declared d_a: element " & $i & " got " & $gotAD[i] &
        " want " & $(dOutHost[i] * alpha)
      doAssert gotBD[i] == gotBT[i],
        "convenience d_b: element " & $i & " declared " & $gotBD[i] &
        " transposed " & $gotBT[i]
      doAssert gotBD[i] == dOutHost[i],
        "convenience declared d_b: element " & $i & " got " & $gotBD[i] &
        " want " & $dOutHost[i]
    doAssert gotAlphaD[0] == gotAlphaT[0],
      "convenience d_alpha: declared " & $gotAlphaD[0] &
      " transposed " & $gotAlphaT[0]
    doAssert gotAlphaD[0] == dAlphaWant,
      "convenience declared d_alpha: got " & $gotAlphaD[0] & " want " & $dAlphaWant
    stdout.writeLine("[OK] declared convenience backward ≡ transposed, element-wise exact")

  block:   # FA-shaped stub: the saved outputs round-trip one buffer set
    let ad = engineD.setupAutoDiff(sdpaPair)

    let qDev = engineD.allocDevice(float32, N)
    let kDev = engineD.allocDevice(float32, N)
    let vDev = engineD.allocDevice(float32, N)
    let lseDev = engineD.allocDevice(float32, 1)
    var qHost: array[N, float32]
    var kHost: array[N, float32]
    var vHost: array[N, float32]
    for i in 0 ..< N:
      qHost[i] = float32(i + 1)
      kHost[i] = float32(10 + i)
      vHost[i] = float32(20 + i)
    engineD.copyToDeviceFromHost(qDev, addr qHost[0], N)
    engineD.copyToDeviceFromHost(kDev, addr kHost[0], N)
    engineD.copyToDeviceFromHost(vDev, addr vHost[0], N)

    # Poison the save targets: the forward must overwrite the sentinels,
    # the backward must read them back from these exact buffers.
    let sentinels = fillSeq(N, -7.5'f32)
    engineD.copyToDeviceFromHost(outDev, addr sentinels[0], N)
    engineD.copyToDeviceFromHost(lseDev, addr sentinels[0], 1)

    ad.forward(outDev, qDev, kDev, vDev, lseDev)
    let outGot = engineD.copyBack(outDev, N)
    let lseGot = engineD.copyBack(lseDev, 1)
    for i in 0 ..< N:
      let expected = qHost[i] + kHost[i] + vHost[i]
      doAssert outGot[i] == expected,
        "stub forward dst: element " & $i & " got " & $outGot[i] &
        " want " & $expected
    doAssert lseGot[0] == outGot[0] + 1.0'f32,
      "stub forward lse: got " & $lseGot[0] & " want " & $(outGot[0] + 1.0'f32)

    # Incoming cotangents: d_dst distinctive, d_lse seeded nonzero, keeping d_v cotangent terms value-visible
    var dLseHost: array[1, float32] = [0.5'f32]
    let dLseDev = engineD.allocDevice(float32, 1)
    engineD.copyToDeviceFromHost(dLseDev, addr dLseHost[0], 1)

    let grads = ad.backward(dOutDev, dLseDev,
      HostBuf[typeof(qDev)](p: qDev, n: N),
      HostBuf[typeof(kDev)](p: kDev, n: N),
      HostBuf[typeof(vDev)](p: vDev, n: N),
      outDev, lseDev)

    let dQ = engineD.copyBack(grads.d_q, N)
    let dK = engineD.copyBack(grads.d_k, N)
    let dV = engineD.copyBack(grads.d_v, N)
    let dLse = engineD.copyBack(dLseDev, 1)
    for i in 0 ..< N:
      # d_q reads dst[i], the saved forward output of these exact buffers
      doAssert dQ[i] == dOutHost[i] * outGot[i] * 0.5'f32,
        "stub d_q: element " & $i & " got " & $dQ[i] &
        " want " & $(dOutHost[i] * outGot[i] * 0.5'f32)
      # d_k reads lse[0], the saved forward output of these exact buffers
      doAssert dK[i] == dOutHost[i] * lseGot[0],
        "stub d_k: element " & $i & " got " & $dK[i] &
        " want " & $(dOutHost[i] * lseGot[0])
      doAssert dV[i] == dOutHost[i] + dLseHost[0],
        "stub d_v: element " & $i & " got " & $dV[i] &
        " want " & $(dOutHost[i] + dLseHost[0])
    doAssert dLse[0] == dOutHost[0] + dLseHost[0],
      "stub d_lse: got " & $dLse[0] & " want " & $(dOutHost[0] + dLseHost[0])
    stdout.writeLine("[OK] FA-shaped stub: saved outputs round-trip through the handle")

when isMainModule:
  runTest()
