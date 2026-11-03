# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: simdgroup reduction backward
##
## Test that reduceAdd and reduceMax derive with bitwise-exact values against the analytic closed form and the CPU central differences.
##
## - reduceAdd: every lane's gradient is the incoming gradient, summed again over the lane set
## - reduceMax: the argmax lane's gradient only, the mask recomputes the primal max
## - the fixtures are integers, every summation order is exact in float32
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_reduce.nim

import workspace/crucible

proc reduceSumK(y, a: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1).} =
  ## One simdgroup of 32 lanes, lane l reduces `a` and writes slot l.
  let lane = thread_position_in_threadgroup.x
  y[lane] = reduceAdd(a[lane])

proc reduceMaxK(y, a: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1).} =
  ## Same mapping, the max variant masks the gradient to the argmax lane.
  let lane = thread_position_in_threadgroup.x
  y[lane] = reduceMax(a[lane])

const reduceSumPair = cgMetal.bwd_diff(reduceSumK)
const reduceMaxPair = cgMetal.bwd_diff(reduceMaxK)

type ReduceBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience overload reads that length for its grad allocation.
  p: DevMem
  n: int

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  const L = 32     # lanes, one full simdgroup

  let aDev = engine.allocDevice(float32, L)
  let yDev = engine.allocDevice(float32, L)
  var aHost: array[L, float32]
  for i in 0 ..< L:
    aHost[i] = float32(i + 1)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], L)

  var sumWant = 0.0'f32
  for i in 0 ..< L:
    sumWant += aHost[i]

  block:   # reduceAdd: forward broadcast, every lane's gradient is the incoming sum
    let ad = engine.setupAutoDiff(reduceSumPair)
    ad.forward(yDev, aDev)
    var yGot: array[L, float32]
    engine.copyToHostFromDevice(addr yGot[0], yDev, L)
    for l in 0 ..< L:
      doAssert yGot[l] == sumWant,
        "reduceAdd forward: lane " & $l & " got " & $yGot[l] & " want " & $sumWant

    let dYDev = engine.allocDevice(float32, L)
    var dYHost: array[L, float32]
    for i in 0 ..< L:
      dYHost[i] = float32((i mod 7) + 1)
    engine.copyToDeviceFromHost(dYDev, addr dYHost[0], L)

    var dSumWant = 0.0'f32
    for i in 0 ..< L:
      dSumWant += dYHost[i]

    let grads = ad.backward(dYDev, yDev, ReduceBuf[typeof(aDev)](p: aDev, n: L))
    var daGot: array[L, float32]
    engine.copyToHostFromDevice(addr daGot[0], grads.d_a, L)
    for l in 0 ..< L:
      doAssert daGot[l] == dSumWant,
        "reduceAdd d_a: lane " & $l & " got " & $daGot[l] & " want " & $dSumWant
    echo "[OK] reduceAdd backward: every lane's gradient is the incoming sum"

  block:   # reduceMax: the argmax lane's gradient only
    let ad = engine.setupAutoDiff(reduceMaxPair)
    ad.forward(yDev, aDev)
    var yGot: array[L, float32]
    engine.copyToHostFromDevice(addr yGot[0], yDev, L)
    for l in 0 ..< L:
      doAssert yGot[l] == aHost[L - 1],
        "reduceMax forward: lane " & $l & " got " & $yGot[l] &
        " want " & $aHost[L - 1]

    let dYDev = engine.allocDevice(float32, L)
    var dYHost: array[L, float32]
    for i in 0 ..< L:
      dYHost[i] = float32((i mod 5) + 2)
    engine.copyToDeviceFromHost(dYDev, addr dYHost[0], L)

    var dSumWant = 0.0'f32
    for i in 0 ..< L:
      dSumWant += dYHost[i]

    let grads = ad.backward(dYDev, yDev, ReduceBuf[typeof(aDev)](p: aDev, n: L))
    var daGot: array[L, float32]
    engine.copyToHostFromDevice(addr daGot[0], grads.d_a, L)
    for l in 0 ..< L:
      let want = if l == L - 1: dSumWant else: 0.0'f32
      doAssert daGot[l] == want,
        "reduceMax d_a: lane " & $l & " got " & $daGot[l] & " want " & $want
    echo "[OK] reduceMax backward: the argmax lane carries the incoming sum"

  block:   # CPU central differences over the closed-form forward, reduceAdd
    let ad = engine.setupAutoDiff(reduceSumPair)
    let dYDev = engine.allocDevice(float32, L)
    var dYHost: array[L, float32]
    for i in 0 ..< L:
      dYHost[i] = 0.5'f32 + 0.25'f32 * float32(i mod 4)
    engine.copyToDeviceFromHost(dYDev, addr dYHost[0], L)

    proc lossAt(av: openArray[float32]): float64 =
      # Sum evaluation in float64: the f32 sum of 528 +/- h rounds
      # asymmetrically and a linear central difference inherits
      # that rounding as a bias.
      var s = 0.0'f64
      for i in 0 ..< L:
        s += float64(av[i])
      for l in 0 ..< L:
        result += float64(dYHost[l]) * s

    let grads = ad.backward(dYDev, yDev, ReduceBuf[typeof(aDev)](p: aDev, n: L))
    var daGot: array[L, float32]
    engine.copyToHostFromDevice(addr daGot[0], grads.d_a, L)
    let h = 1.0e-3'f32
    for k in 0 ..< L:
      var ap = aHost
      var am = aHost
      ap[k] = ap[k] + h
      am[k] = am[k] - h
      let want = (lossAt(ap) - lossAt(am)) / (2.0 * float64(h))
      doAssert abs(float64(daGot[k]) - want) <= 1.0e-5 + 1.0e-3 * abs(want),
        "reduceAdd fd: lane " & $k & " got " & $daGot[k] & " want " & $want
    echo "[OK] reduceAdd backward matches the CPU central differences"

when isMainModule:
  runTest()
