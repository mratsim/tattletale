# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Math-builtin cotangents through a launched Metal kernel
##
## Test that exp, ln, sqrt and rsqrt transpose to their mechanical cotangent rules, value-exact against the closed form and finite differences
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_ad_math_builtins.nim

import std/math
import workspace/crucible

proc `=~`[T: SomeFloat, U: SomeFloat](x: T, y: U): bool =
  ## Relative 1e-5 / absolute 1e-6 floating-point comparison
  abs(float64(x) - float64(y)) <= max(1.0e-5 * max(abs(float64(x)), abs(float64(y))), 1.0e-6)

proc fExp(x: float64): float64 = exp(x)
proc dfExp(x: float64): float64 = exp(x)
proc fLn(x: float64): float64 = ln(x)
proc dfLn(x: float64): float64 = 1.0 / x
proc fSqrt(x: float64): float64 = sqrt(x)
proc dfSqrt(x: float64): float64 = 0.5 / sqrt(x)
proc fRsqrt(x: float64): float64 = 1.0 / sqrt(x)
proc dfRsqrt(x: float64): float64 = -0.5 / (x * sqrt(x))

proc fd(f: proc(x: float64): float64 {.nimcall.}, x: float64): float64 =
  ## Central finite difference, h small enough for 1e-5 relative checks in float64
  let h = 1.0e-3
  (f(x + h) - f(x - h)) / (2.0 * h)

proc expKernel(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  for i in 0 ..< 8:
    dst[i] = exp(a[i])

proc lnKernel(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  for i in 0 ..< 8:
    dst[i] = ln(a[i])

proc sqrtKernel(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  for i in 0 ..< 8:
    dst[i] = sqrt(a[i])

proc rsqrtKernel(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  for i in 0 ..< 8:
    dst[i] = rsqrt(a[i])

const expPair = bkMetal.bwd_diff(expKernel)
const lnPair = bkMetal.bwd_diff(lnKernel)
const sqrtPair = bkMetal.bwd_diff(sqrtKernel)
const rsqrtPair = bkMetal.bwd_diff(rsqrtKernel)

type MathBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience backward reads that length for its grad allocation.
  p: DevMem
  n: int

template checkBuiltin(name: string, ad: typed, f, df: untyped) =
  ## Forward value, then convenience backward: d_a against the closed form and the finite difference
  block:
    ad.forward(dstDev, aDev)
    var gotDst: array[N, float32]
    engine.copyToHostFromDevice(addr gotDst[0], dstDev, N)
    for i in 0 ..< N:
      doAssert gotDst[i] =~ f(float64(aHost[i])),
        name & " forward: element " & $i & " got " & $gotDst[i]

    let grads = ad.backward(dDstDev, dstDev, MathBuf[typeof(aDev)](p: aDev, n: N))
    var gotDa: array[N, float32]
    engine.copyToHostFromDevice(addr gotDa[0], grads.d_a, N)
    for i in 0 ..< N:
      let wantClosed = df(float64(aHost[i])) * float64(dDstHost[i])
      let wantFd = fd(f, float64(aHost[i])) * float64(dDstHost[i])
      doAssert gotDa[i] =~ wantClosed,
        name & " d_a closed form: element " & $i & " got " & $gotDa[i] & " want " & $wantClosed
      doAssert gotDa[i] =~ wantFd,
        name & " d_a finite diff: element " & $i & " got " & $gotDa[i] & " want " & $wantFd
    stdout.writeLine("[OK] " & name & ": forward, d_a closed form and finite difference")

proc runTest() =
  var engine = newEngine(runMetal)
  const N = 8

  let aDev = engine.allocDevice(float32, N)
  let dstDev = engine.allocDevice(float32, N)
  let dDstDev = engine.allocDevice(float32, N)
  var aHost: array[N, float32]
  var dDstHost: array[N, float32]
  for i in 0 ..< N:
    aHost[i] = 0.5'f32 + 0.25'f32 * float32(i)
    dDstHost[i] = if (i mod 2) == 0: 0.5'f32 else: 1.5'f32
  engine.copyToDeviceFromHost(aDev, addr aHost[0], N)
  engine.copyToDeviceFromHost(dDstDev, addr dDstHost[0], N)

  block:  # exp: d_a = d_dst * exp(a)
    let ad = engine.setupAutoDiff(expPair)
    checkBuiltin("exp", ad, fExp, dfExp)

  block:  # ln: d_a = d_dst / a, the reciprocal of the primal
    let ad = engine.setupAutoDiff(lnPair)
    checkBuiltin("ln", ad, fLn, dfLn)

  block:  # sqrt: d_a = d_dst / (2*sqrt(a))
    let ad = engine.setupAutoDiff(sqrtPair)
    checkBuiltin("sqrt", ad, fSqrt, dfSqrt)

  block:  # rsqrt: d_a = -d_dst / (2*a*sqrt(a))
    let ad = engine.setupAutoDiff(rsqrtPair)
    checkBuiltin("rsqrt", ad, fRsqrt, dfRsqrt)

when isMainModule:
  runTest()
