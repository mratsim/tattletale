# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAutoDiff on Metal: the loop index converts through a differentiated value
##
## `dst[i] = a[i] * float32(i)`: the loop index is primal and the conversion
## carries a zero cotangent, nothing is lost:
##
## - the kernel transposes (it refused before the primal-conv rule)
## - d_a[i] = float32(i) * d_dst[i], value-checked on Metal
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_conv_index.nim

import workspace/crucible

proc weight(dst, a: ptr UncheckedArray[float32], n: int32) {.global, workgroup: (1, 1).} =
  for i in 0 ..< n:
    dst[i] = a[i] * float32(i)

const weightPair = bkMetal.bwd_diff(weight)

type ABuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  p: DevMem
  n: int

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  let ad = engine.setupAutoDiff(weightPair)

  const N = 8
  let aDev = engine.allocDevice(float32, N)
  let dstDev = engine.allocDevice(float32, N)
  var aHost: array[N, float32]
  for i in 0 ..< N:
    aHost[i] = float32(i + 2)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], N)

  ad.forward(dstDev, aDev, int32(N))
  var got: array[N, float32]
  engine.copyToHostFromDevice(addr got[0], dstDev, N)
  for i in 0 ..< N:
    doAssert got[i] == aHost[i] * float32(i),
      "forward: element " & $i & " got " & $got[i]
  echo "[OK] forward exact"

  let dOutDev = engine.allocDevice(float32, N)
  var dOutHost: array[N, float32]
  for i in 0 ..< N:
    dOutHost[i] = float32(3 + i)
  engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], N)

  let d_a = ad.backward(dOutDev, dstDev, ABuf[typeof(aDev)](p: aDev, n: N), int32(N))
  var gotA: array[N, float32]
  engine.copyToHostFromDevice(addr gotA[0], d_a.d_a, N)
  for i in 0 ..< N:
    doAssert gotA[i] == dOutHost[i] * float32(i),
      "d_a: element " & $i & " got " & $gotA[i] &
      " want " & $(dOutHost[i] * float32(i))
  echo "[OK] backward exact, the loop index conversion is primal"

when isMainModule:
  runTest()
