# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## fp32 GEMM bench on Vulkan (MoltenVK): flops actually done / wall-clock.
##
## The 8x4 variant falsifies a register-pressure hypothesis about spilled fragments.
## A spill would shift the 8x4/4x8 ratio away from 1.0 as the A fragment doubles
## and the B fragment halves per k-tile.
##
## Run:
##   cd tattletale (worktree root)
##   nim c -r --hints:off --warnings:off -d:release \
##     --outdir:build/tests/bench_gemm_vulkan --nimcache:nimcache/tests/bench_gemm_vulkan \
##     workspace/tensorcery/tests/manual_bench_gemm_vulkan.nim

import std/[times, strformat]
import workspace/crucible
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/layout_algebra
import workspace/tensorcery/src/kernels/k_gemm_vulkan
import workspace/ceramic/tests/ceramic_testutils

proc gemmTileVulkan8x4(d: distinct TensorView, a: distinct TensorView, b: distinct TensorView, alpha, beta: float32) =
  ## 8x4-cell schedule, each thread owns 8 m-lines x 4 n-columns instead
  ## of 4x8. The fragment budget is unchanged, the A fragment doubles
  ## and the B fragment halves per k-tile. The kernel boundary ships 4x8,
  ## this variant is bench-only.
  let
    thrID = getLocalThreadID()
    rowBlock = thrID div 8
    colBlock = thrID mod 8
    m0 = 32 * get_group_id(0)
    n0 = 32 * get_group_id(1)
    kTiles = ceil_div(a.layout.shape[1], 2)
  block:
    var rD: tile(d.T, make_layout((32, (1, 1))))
    staticFor i, 0, 32:
      rD[i] = zero(d.T)
    var rA: tile(a.T, make_layout((16, (1, 1))))
    var rB: tile(b.T, make_layout((8, (1, 1))))
    for kt in 0 ..< kTiles:
      let k0 = 2 * kt
      let aInterior = (m0 + 32 <= a.layout.shape[0]) and (k0 + 2 <= a.layout.shape[1])
      let bInterior = (n0 + 32 <= b.layout.shape[0]) and (k0 + 2 <= b.layout.shape[1])
      staticFor kk, 0, 2:
        staticFor m, 0, 8:
          rA[m + 8 * kk] = if aInterior:
              a[(m0 + 8 * rowBlock + m, k0 + kk)]
            else:
              a.paddedLoad((m0 + 8 * rowBlock + m, k0 + kk), zero(a.T))
      staticFor kk, 0, 2:
        staticFor n, 0, 4:
          rB[n + 4 * kk] = if bInterior:
              b[(n0 + 4 * colBlock + n, k0 + kk)]
            else:
              b.paddedLoad((n0 + 4 * colBlock + n, k0 + kk), zero(b.T))
      staticFor n, 0, 4:
        staticFor m, 0, 8:
          var acc = rD[m + 8 * n]
          staticFor kk, 0, 2:
            acc = acc + rA[m + 8 * kk] * rB[n + 4 * kk]
          rD[m + 8 * n] = acc
    staticFor n, 0, 4:
      staticFor m, 0, 8:
        let crd = (m0 + 8 * rowBlock + m, n0 + 4 * colBlock + n)
        if crd.inBounds(d.layout.shape):
          let dPrev = if beta != 0.0'f32: d.paddedLoad(crd, zero(d.T)) else: zero(d.T)
          d[crd] = alpha * rD[m + 8 * n] + beta * dPrev

proc benchOnce(engine: VulkanEngine, m, n, k, iters: int): float64 =
  ## Returns achieved TFLOP/s over `iters` back-to-back launches
  let
    aHost = makeHostTensor(float32, (m, k), (k, 1))
    bHost = makeHostTensor(float32, (n, k), (k, 1))
    dHost = makeHostTensor(float32, (m, n), (n, 1))
  for i in 0 ..< aHost.len:
    aHost.data[i] = float32((i * 37 mod 19) - 9) * 0.125'f32
  for i in 0 ..< bHost.len:
    bHost.data[i] = float32((i * 53 mod 23) - 11) * 0.125'f32
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  let flops = 2.0 * float64(m) * float64(n) * float64(k) * float64(iters)
  let t0 = epochTime()
  for i in 0 ..< iters:
    engine.launchDeviceFn gemmTileVulkan(dDev, aDev, bDev, 1.0'f32, 0.0'f32) do:
      grid = (ceil_div(m, 32), ceil_div(n, 32))
      blk = (32, 1)
      lang = glsl
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, 1)
  let wall = epochTime() - t0
  result = flops / wall / 1.0e12

proc benchOnce8x4(engine: VulkanEngine, m, n, k, iters: int): float64 =
  ## Same harness as benchOnce over the 8x4 schedule
  let
    aHost = makeHostTensor(float32, (m, k), (k, 1))
    bHost = makeHostTensor(float32, (n, k), (k, 1))
    dHost = makeHostTensor(float32, (m, n), (n, 1))
  for i in 0 ..< aHost.len:
    aHost.data[i] = float32((i * 37 mod 19) - 9) * 0.125'f32
  for i in 0 ..< bHost.len:
    bHost.data[i] = float32((i * 53 mod 23) - 11) * 0.125'f32
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  let flops = 2.0 * float64(m) * float64(n) * float64(k) * float64(iters)
  let t0 = epochTime()
  for i in 0 ..< iters:
    engine.launchDeviceFn gemmTileVulkan8x4(dDev, aDev, bDev, 1.0'f32, 0.0'f32) do:
      grid = (ceil_div(m, 32), ceil_div(n, 32))
      blk = (32, 1)
      lang = glsl
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, 1)
  let wall = epochTime() - t0
  result = flops / wall / 1.0e12

proc check8x4Agrees(engine: VulkanEngine) =
  ## Ratio conclusions need the 8x4 variant to compute the same result
  ## as the shipped 4x8 kernel. Both launch once on a small ragged shape,
  ## their outputs must agree to fp32 reassociation slack.
  const m = 97
  const n = 53
  const k = 45
  var
    aHost = makeHostTensor(float32, (m, k), (k, 1))
    bHost = makeHostTensor(float32, (n, k), (k, 1))
    d4 = makeHostTensor(float32, (m, n), (n, 1))
    d8 = makeHostTensor(float32, (m, n), (n, 1))
  for i in 0 ..< aHost.len:
    aHost.data[i] = float32((i * 37 mod 19) - 9) * 0.125'f32
  for i in 0 ..< bHost.len:
    bHost.data[i] = float32((i * 53 mod 23) - 11) * 0.125'f32
  for i in 0 ..< d4.len:
    d4.data[i] = 0.0'f32
    d8.data[i] = 0.0'f32
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    d4Dev = engine.bridge(d4)
    d8Dev = engine.bridge(d8)
  engine.launchDeviceFn gemmTileVulkan(d4Dev, aDev, bDev, 1.0'f32, 0.0'f32) do:
    grid = (ceil_div(m, 32), ceil_div(n, 32))
    blk = (32, 1)
    lang = glsl
  engine.launchDeviceFn gemmTileVulkan8x4(d8Dev, aDev, bDev, 1.0'f32, 0.0'f32) do:
    grid = (ceil_div(m, 32), ceil_div(n, 32))
    blk = (32, 1)
    lang = glsl
  engine.copyToHostFromDevice(addr d4.data[0], d4Dev.data, d4.len)
  engine.copyToHostFromDevice(addr d8.data[0], d8Dev.data, d8.len)
  assertAllClose(d8.view(), d4.view(), rtol = 1e-6, abstol = 1e-7,
    msg = "8x4 variant must agree with the shipped 4x8 kernel")

proc runTest() =
  var engine = newEngine(runVulkan)
  echo engine.deviceName()
  const M = 4096
  const N = 4096
  const K = 4096
  # warmup
  discard benchOnce(engine, M, N, K, 2)
  # quiet window: 3 rounds of 20 back-to-back launches, best round
  var best = 0.0
  for r in 0 ..< 3:
    let tf = benchOnce(engine, M, N, K, 20)
    echo "round ", r, ": ", fmt"{tf:.3f}", " TFLOP/s"
    if tf > best:
      best = tf
  echo "BEST 4x8: ", fmt"{best:.3f}", " TFLOP/s (fp32, 4096^3, MoltenVK)"
  # falsifier: the 8x4 variant under the same harness, agreement first
  check8x4Agrees(engine)
  echo "8x4 agrees with 4x8 on a 97x53x45 ragged shape"
  discard benchOnce8x4(engine, M, N, K, 2)
  var best8x4 = 0.0
  for r in 0 ..< 3:
    let tf = benchOnce8x4(engine, M, N, K, 20)
    echo "round 8x4 ", r, ": ", fmt"{tf:.3f}", " TFLOP/s"
    if tf > best8x4:
      best8x4 = tf
  echo "BEST 8x4: ", fmt"{best8x4:.3f}", " TFLOP/s (fp32, 4096^3, MoltenVK)"
  echo "ratio 8x4/4x8: ", fmt"{best8x4/best:.3f}"

when isMainModule:
  runTest()
