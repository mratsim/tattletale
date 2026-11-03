# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: rmsnorm forward and backward end to end, thread-parallel
##
## Test that the declared thread-parallel rmsnorm pair (workgroup (2, 1)) matches the analytic and finite-difference references.
##
## - forward: `y[r][i] = x[r][i] * rrms[r] * w[i]`, `rrms[r] = rsqrt(mean(x[r]²) + eps)` exports into its residual slot
## - the backward is declared: the derived pair refuses, `d_w` collects both r values' contributions in one shared slot
## - the hand-written backward accumulates `d_w` through `atomicFetchAdd`, two float adds commute, the result is order-independent
## - the differentiated loss includes the statistic's direct term `Σ d_rrms*rrms` and the incoming `d_rrms` gradient path is live
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_rmsnorm.nim

import std/math
import workspace/crucible

proc rmsnormPlain(y, x, w, rrms: ptr UncheckedArray[float32]) {.global, workgroup: (2, 1).} =
  ## One thread per r: thread r normalizes r's eight columns.
  let r = thread_position_in_threadgroup.x
  let b = r * 8
  let ssq = x[b + 0] * x[b + 0] + x[b + 1] * x[b + 1] +
            x[b + 2] * x[b + 2] + x[b + 3] * x[b + 3] +
            x[b + 4] * x[b + 4] + x[b + 5] * x[b + 5] +
            x[b + 6] * x[b + 6] + x[b + 7] * x[b + 7]
  let inv = rsqrt(ssq / 8.0'f32 + 1.0e-6'f32)
  rrms[r] = inv
  y[b + 0] = x[b + 0] * inv * w[0]
  y[b + 1] = x[b + 1] * inv * w[1]
  y[b + 2] = x[b + 2] * inv * w[2]
  y[b + 3] = x[b + 3] * inv * w[3]
  y[b + 4] = x[b + 4] * inv * w[4]
  y[b + 5] = x[b + 5] * inv * w[5]
  y[b + 6] = x[b + 6] * inv * w[6]
  y[b + 7] = x[b + 7] * inv * w[7]

proc rmsnormPlain_bwd(d_y, d_x, d_w, d_rrms, y, x, w, rrms: ptr UncheckedArray[float32]) {.global, workgroup: (2, 1).} =
  ## Declared backward, the forward's thread mapping: one thread per r.
  ## `d_w[i]` collects both r values' contributions through atomicFetchAdd,
  ## two float adds commute so the slot value is order-independent.
  let r = thread_position_in_threadgroup.x
  let b = r * 8
  let inv = rrms[r]
  var dot = 0.0'f32
  for k in 0 ..< 8:
    dot += d_y[b + k] * w[k] * x[b + k]
  let statPath = -inv * inv * inv / 8.0'f32
  for i in 0 ..< 8:
    d_x[b + i] += d_y[b + i] * w[i] * inv + (dot + d_rrms[r]) * statPath * x[b + i]
    discard atomicFetchAdd(addr d_w[i], d_y[b + i] * x[b + i] * inv)

const rmsnormPair = cgMetal.bwd_diff(rmsnormPlain, rmsnormPlain_bwd, save =  [rrms])

type RmsBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience overload reads that length for its grad allocation.
  p: DevMem
  n: int

const
  RmsRows = 2
  RmsCols = 8
  N = RmsRows * RmsCols
  Eps = 1.0e-6

proc fillSeq[T](n: int, value: T): seq[T] =
  result = newSeq[T](n)
  for i in 0 ..< n:
    result[i] = value

proc xHostAt(i: int): float32 =
  ## Deterministic positive inputs with varied magnitudes.
  0.3'f32 + 0.17'f32 * float32((7 * i + 3) mod 11)

proc wHostAt(i: int): float32 =
  0.5'f32 + 0.1'f32 * float32(i mod 5)

proc dYHostAt(i: int): float32 =
  0.25'f32 * float32((i mod 5) - 2)

proc dRrmsHostAt(r: int): float32 =
  if r == 0:
    0.5'f32
  else:
    -0.25'f32

# ── CPU references, float64 ───────────────────────────────────────────────

proc rrms64(x: openArray[float32], r: int): float64 =
  ## Statistic of index `r`: float64 evaluation of the kernel's formula.
  var ssq = 0.0
  for i in 0 ..< RmsCols:
    let v = float64(x[r * RmsCols + i])
    ssq += v * v
  result = 1.0 / sqrt(ssq / float64(RmsCols) + Eps)

proc loss64(x, w, dY, dRrms: openArray[float32]): float64 =
  ## Differentiated loss: the y term plus the statistic's direct term `Σ d_rrms*rrms`.
  for r in 0 ..< RmsRows:
    let inv = rrms64(x, r)
    for i in 0 ..< RmsCols:
      let y = float64(x[r * RmsCols + i]) * inv * float64(w[i])
      result += float64(dY[r * RmsCols + i]) * y
    result += float64(dRrms[r]) * inv

proc analyticDx(x, w, dY, dRrms: openArray[float32], r, i: int): float64 =
  ## dL/dx[r][i]: elementwise term, statistic-path term, incoming d_rrms term.
  let inv = rrms64(x, r)
  var dot = 0.0
  for k in 0 ..< RmsCols:
    dot += float64(dY[r * RmsCols + k]) * float64(w[k]) * float64(x[r * RmsCols + k])
  let statPath = -inv * inv * inv * float64(x[r * RmsCols + i]) / float64(RmsCols)
  result = float64(dY[r * RmsCols + i]) * float64(w[i]) * inv +
           (dot + float64(dRrms[r])) * statPath

proc analyticDw(x, w, dY, dRrms: openArray[float32], i: int): float64 =
  ## dL/dw[i]: the per-column sum over both r values.
  for r in 0 ..< RmsRows:
    result += float64(dY[r * RmsCols + i]) * float64(x[r * RmsCols + i]) * rrms64(x, r)

proc relClose(got, want, rtol, atol: float64): bool =
  abs(got - want) <= atol + rtol * abs(want)

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  let ad = engine.setupAutoDiff(rmsnormPair)

  var xHost: array[N, float32]
  var wHost: array[RmsCols, float32]
  var dYHost: array[N, float32]
  var dRrmsHost: array[RmsRows, float32]
  for i in 0 ..< N:
    xHost[i] = xHostAt(i)
    dYHost[i] = dYHostAt(i)
  for i in 0 ..< RmsCols:
    wHost[i] = wHostAt(i)
  for r in 0 ..< RmsRows:
    dRrmsHost[r] = dRrmsHostAt(r)

  let xDev = engine.allocDevice(float32, N)
  let wDev = engine.allocDevice(float32, RmsCols)
  let yDev = engine.allocDevice(float32, N)
  let rrmsDev = engine.allocDevice(float32, RmsRows)
  engine.copyToDeviceFromHost(xDev, addr xHost[0], N)
  engine.copyToDeviceFromHost(wDev, addr wHost[0], RmsCols)

  block:
    # forward: y and the exported statistic against the closed form
    ad.forward(yDev, xDev, wDev, rrmsDev)
    var yGot: array[N, float32]
    var rrmsGot: array[RmsRows, float32]
    engine.copyToHostFromDevice(addr yGot[0], yDev, N)
    engine.copyToHostFromDevice(addr rrmsGot[0], rrmsDev, RmsRows)
    for r in 0 ..< RmsRows:
      let invWant = rrms64(xHost, r)
      doAssert relClose(float64(rrmsGot[r]), invWant, 1.0e-6, 1.0e-7),
        "rrms: row " & $r & " got " & $rrmsGot[r] & " want " & $invWant
      for i in 0 ..< RmsCols:
        let yWant = float64(xHost[r * RmsCols + i]) * invWant * float64(wHost[i])
        doAssert relClose(float64(yGot[r * RmsCols + i]), yWant, 1.0e-6, 1.0e-7),
          "y: r " & $r & " i " & $i & " got " & $yGot[r * RmsCols + i] &
          " want " & $yWant
    echo "[OK] forward exact at workgroup (2, 1), rrms exported"

  let dYDev = engine.allocDevice(float32, N)
  let dRrmsDev = engine.allocDevice(float32, RmsRows)
  engine.copyToDeviceFromHost(dYDev, addr dYHost[0], N)
  engine.copyToDeviceFromHost(dRrmsDev, addr dRrmsHost[0], RmsRows)

  # CPU central finite differences of the loss, one coordinate at a time
  proc fdDx(r, i: int): float64 =
    let h = 1.0e-3'f32
    var xp = xHost
    var xm = xHost
    xp[r * RmsCols + i] = xp[r * RmsCols + i] + h
    xm[r * RmsCols + i] = xm[r * RmsCols + i] - h
    result = (loss64(xp, wHost, dYHost, dRrmsHost) -
              loss64(xm, wHost, dYHost, dRrmsHost)) / (2.0 * float64(h))
  proc fdDw(i: int): float64 =
    let h = 1.0e-3'f32
    var wp = wHost
    var wm = wHost
    wp[i] = wp[i] + h
    wm[i] = wm[i] - h
    result = (loss64(xHost, wp, dYHost, dRrmsHost) -
              loss64(xHost, wm, dYHost, dRrmsHost)) / (2.0 * float64(h))

  block:
    # convenience backward: grads sized off the primals, exact twice
    let (d_x, d_w) = ad.backward(dYDev, dRrmsDev,
      RmsBuf[typeof(yDev)](p: yDev, n: N),
      RmsBuf[typeof(xDev)](p: xDev, n: N),
      RmsBuf[typeof(wDev)](p: wDev, n: RmsCols),
      rrmsDev)
    var dxGot: array[N, float32]
    var dwGot: array[RmsCols, float32]
    engine.copyToHostFromDevice(addr dxGot[0], d_x, N)
    engine.copyToHostFromDevice(addr dwGot[0], d_w, RmsCols)
    for r in 0 ..< RmsRows:
      for i in 0 ..< RmsCols:
        let wantAnalytic = analyticDx(xHost, wHost, dYHost, dRrmsHost, r, i)
        let wantFd = fdDx(r, i)
        doAssert relClose(float64(dxGot[r * RmsCols + i]), wantAnalytic, 1.0e-5, 1.0e-6),
          "d_x analytic: r " & $r & " i " & $i & " got " & $dxGot[r * RmsCols + i] &
          " want " & $wantAnalytic
        doAssert relClose(float64(dxGot[r * RmsCols + i]), wantFd, 1.0e-3, 1.0e-5),
          "d_x finite diff: r " & $r & " i " & $i & " got " & $dxGot[r * RmsCols + i] &
          " want " & $wantFd
    for i in 0 ..< RmsCols:
      let wantAnalytic = analyticDw(xHost, wHost, dYHost, dRrmsHost, i)
      let wantFd = fdDw(i)
      doAssert relClose(float64(dwGot[i]), wantAnalytic, 1.0e-5, 1.0e-6),
        "d_w analytic: i " & $i & " got " & $dwGot[i] & " want " & $wantAnalytic
      doAssert relClose(float64(dwGot[i]), wantFd, 1.0e-3, 1.0e-5),
        "d_w finite diff: i " & $i & " got " & $dwGot[i] & " want " & $wantFd
    echo "[OK] convenience backward: d_x/d_w match analytic and finite differences"

    # a second call rewrites from fresh zeroed grads, same values
    let (d_x2, d_w2) = ad.backward(dYDev, dRrmsDev,
      RmsBuf[typeof(yDev)](p: yDev, n: N),
      RmsBuf[typeof(xDev)](p: xDev, n: N),
      RmsBuf[typeof(wDev)](p: wDev, n: RmsCols),
      rrmsDev)
    var dx2Arr: array[N, float32]
    var dw2Arr: array[RmsCols, float32]
    engine.copyToHostFromDevice(addr dx2Arr[0], d_x2, N)
    engine.copyToHostFromDevice(addr dw2Arr[0], d_w2, RmsCols)
    for i in 0 ..< N:
      doAssert dx2Arr[i] == dxGot[i], "d_x rewrite: element " & $i
    for i in 0 ..< RmsCols:
      doAssert dw2Arr[i] == dwGot[i], "d_w rewrite: element " & $i
    echo "[OK] convenience backward rewrites from zeroed grads, atomic d_w order-independent"

  block:
    # primitive backward: pre-zeroed caller buffers, accumulates
    let dxDev = engine.allocDevice(float32, N)
    let dwDev = engine.allocDevice(float32, RmsCols)
    let zerosN = fillSeq(N, 0.0'f32)
    let zerosC = fillSeq(RmsCols, 0.0'f32)
    engine.copyToDeviceFromHost(dxDev, addr zerosN[0], N)
    engine.copyToDeviceFromHost(dwDev, addr zerosC[0], RmsCols)
    ad.backward(dYDev, dxDev, dwDev, dRrmsDev, yDev, xDev, wDev, rrmsDev)
    ad.backward(dYDev, dxDev, dwDev, dRrmsDev, yDev, xDev, wDev, rrmsDev)
    var dxGot: array[N, float32]
    var dwGot: array[RmsCols, float32]
    engine.copyToHostFromDevice(addr dxGot[0], dxDev, N)
    engine.copyToHostFromDevice(addr dwGot[0], dwDev, RmsCols)
    for r in 0 ..< RmsRows:
      for i in 0 ..< RmsCols:
        let want = 2.0 * analyticDx(xHost, wHost, dYHost, dRrmsHost, r, i)
        doAssert relClose(float64(dxGot[r * RmsCols + i]), want, 1.0e-5, 1.0e-6),
          "d_x accumulate: r " & $r & " i " & $i & " got " & $dxGot[r * RmsCols + i] &
          " want " & $want
    for i in 0 ..< RmsCols:
      let want = 2.0 * analyticDw(xHost, wHost, dYHost, dRrmsHost, i)
      doAssert relClose(float64(dwGot[i]), want, 1.0e-5, 1.0e-6),
        "d_w accumulate: i " & $i & " got " & $dwGot[i] & " want " & $want
    echo "[OK] primitive backward accumulates into caller-owned buffers"

when isMainModule:
  runTest()
