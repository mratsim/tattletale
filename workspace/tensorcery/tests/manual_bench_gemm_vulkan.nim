# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## fp32 GEMM bench on Vulkan (MoltenVK): flops actually done / wall-clock.
##
## Run:
##   cd tattletale (worktree root)
##   nim c -r --hints:off --warnings:off -d:release \
##     --outdir:build/tests/bench_gemm_vulkan --nimcache:nimcache/tests/bench_gemm_vulkan \
##     workspace/tensorcery/tests/manual_bench_gemm_vulkan.nim

import std/[times, math, strformat]
import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/layout_algebra
import workspace/tensorcery/src/kernels/k_gemm_vulkan
import workspace/ceramic/tests/ceramic_testutils

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
    if tf > best: best = tf
  echo "BEST: ", fmt"{best:.3f}", " TFLOP/s (fp32, 4096^3, MoltenVK)"

when isMainModule:
  runTest()
