# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile-algebra GEMM through the Vulkan GLSL route: the k_gemm_vulkan kernel.
##
## Test that gemmVulkan computes d = alpha·A·Bᵀ + beta·d correctly
## on MoltenVK through glslang, across the alpha/beta contract:
##
## - interior 64x64x8, grid (2, 2), alpha = 1.5, beta = 0
## - ragged 97x53x45, grid (4, 2), alpha = 1.5, beta = 0
## - accumulate 64x64x8, beta = 0.5 (d's prior content survives, scaled)
##
## Run:
##   cd tattletale (worktree root)
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/t_kernel_gemm_vulkan --nimcache:nimcache/tests/t_kernel_gemm_vulkan \
##     workspace/tensorcery/tests/t_kernel_gemm_vulkan.nim

import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/layout_algebra
import workspace/tensorcery/src/kernels/k_gemm_vulkan
import workspace/ceramic/tests/ceramic_testutils

proc gemmRef(expected, aHost, bHost, cHost: HostTensor, m, n, k: int, alpha, beta: float32) =
  ## expected = alpha·A·Bᵀ + beta·c accumulated in float64 over the float32 inputs.
  ## A (m, k) and B (n, k) have stride (k, 1), c (m, n) has stride (n, 1).
  for mm in 0 ..< m:
    for nn in 0 ..< n:
      var acc = 0.0'f64
      for kk in 0 ..< k:
        acc += float64(aHost[mm, kk]) * float64(bHost[nn, kk])
      expected[mm, nn] = float32(float64(alpha) * acc + float64(beta) * float64(cHost[mm, nn]))

proc fillPattern(ht: HostTensor) =
  ## Deterministic small-magnitude fill, |v| <= 1.125
  for i in 0 ..< ht.len:
    ht.data[i] = float32((i * 37 mod 19) - 9) * 0.125'f32

proc runGemmCase(engine: VulkanEngine, m, n, k: int, alpha, beta: float32, msg: string) =
  var
    aHost = makeHostTensor(float32, (m, k), (k, 1))
    bHost = makeHostTensor(float32, (n, k), (k, 1))
    cHost = makeHostTensor(float32, (m, n), (n, 1))
    dHost = makeHostTensor(float32, (m, n), (n, 1))
    expected = makeHostTensor(float32, (m, n), (n, 1))
  fillPattern(aHost)
  fillPattern(bHost)
  fillPattern(cHost)
  for i in 0 ..< dHost.len:
    dHost.data[i] = cHost.data[i]
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn gemmTileVulkan(dDev, aDev, bDev, alpha, beta) do:
    grid = (ceil_div(m, 32), ceil_div(n, 32))
    blk = (32, 1)
    lang = glsl
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  gemmRef(expected, aHost, bHost, cHost, m, n, k, alpha, beta)
  assertAllClose(dHost.view(), expected.view(), rtol = 1e-4, abstol = 1e-3,
    msg = msg)

proc runTest() =
  var engine = newEngine(runVulkan)
  echo engine.deviceName()

  runGemmCase(engine, 64, 64, 8, 1.5'f32, 0.0'f32, "interior 64x64x8, grid (2, 2)")
  runGemmCase(engine, 97, 53, 45, 1.5'f32, 0.0'f32, "ragged 97x53x45, grid (4, 2)")
  runGemmCase(engine, 64, 64, 8, 1.5'f32, 0.5'f32, "accumulate 64x64x8, beta 0.5")

  echo "t_kernel_gemm_vulkan: ALL GREEN"

when isMainModule:
  runTest()
