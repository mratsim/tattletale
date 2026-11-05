# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Lead POC: a ceramic tile-algebra GEMM tile through the Vulkan GLSL route
##
## Test that a register-fragment tile kernel computes D = alpha·A·Bᵀ
## correctly on MoltenVK through glslang:
##
## - thread->value map loads, tile-grid loop, `{.unroll.}` inner FMA body
## - runtime tensor shapes (no static shape baked into the kernel)
## - both fixtures launch multi-workgroup grids in x and y, so the tile
##   addressing runs through gl_WorkGroupID and gl_NumWorkGroups
##
## Run:
##   cd tattletale (worktree root)
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/t_poc_tile_vulkan --nimcache:nimcache/tests/t_poc_tile_vulkan \
##     workspace/tensorcery/tests/t_poc_tile_vulkan.nim

import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import workspace/ceramic/tests/ceramic_testutils

# ═══════════════════════════════════════════════════════════════════════
#  Thread->value maps of the 4x8-cell schedule
# ═══════════════════════════════════════════════════════════════════════
#
# Fragment layouts of a 32-thread schedule over one (M32, K2) A tile,
# one (N32, K2) B tile and one (M32, N32) D tile, int strides only:
# - A: thread t addresses A coordinates 4·(t div 4) ..+3 for both k, the t mod 4 half reloads them (stride 0)
# - B: thread t addresses B coordinates 8·(t mod 4) ..+7, both k
# - D: thread t owns one disjoint 4x8 output cell, V col-major
#   (m_local, n_local), offset m_local + 32·n_local

const
  fragA = make_layout(((4, 8), (4, 2)), ((0, 4), (1, 32)))
  fragB = make_layout(((4, 8), (8, 2)), ((8, 0), (1, 32)))
  fragD = make_layout(((4, 8), (4, 8)), ((256, 4), (1, 32)))

proc tileGemm(d: distinct TensorView, a: distinct TensorView, b: distinct TensorView, alpha: float32) =
  ## D = alpha·A·Bᵀ over 32x32 output tiles: one threadgroup per tile,
  ## 32 threads each owning one 4x8 output cell, k-tiles 2 deep,
  ## per-thread register fragments loaded through the thread->value maps.
  const
    # (T, V) -> flat tile offset: fragment layouts folded through the flat tile layout, no cell repetitions
    mapA = compose(make_layout((32, 2), (1, 32)), fragA)
    mapB = compose(make_layout((32, 2), (1, 32)), fragB)
    mapD = compose(make_layout((32, 32), (1, 32)), fragD)
    tvmA = toThreadValueMap(mapA, (32, 2))
    tvmB = toThreadValueMap(mapB, (32, 2))
    tvmD = toThreadValueMap(mapD, (32, 32))
    regA = make_layout((8, (1, 1)))
    regB = make_layout((16, (1, 1)))
    regD = make_layout((32, (1, 1)))

  let thrID = getLocalThreadID()
  if thrID < getLaneWidth():
    let
      dGeom = d.tileWith((32, 32))
      aGeom = a.tileWith((32, 2))
      dGrid = dGeom.getTileGrid()
      (_, kTiles) = aGeom.getTileGrid()
      gridStride = getGridSize()
    for tileID in countup(dGrid.getLocalTileID(), dGrid.size() - 1, gridStride):
      let (m0, n0) = idx2crd(dGrid, tileID)
      var rD: tile(d.T, regD)
      rD.fillWith(zero(d.T))
      for k in 0 ..< kTiles:
        var rA: tile(a.T, regA)
        var rB: tile(b.T, regB)
        rA.load(a, tvmA, thrID, (m0, k))
        rB.load(b, tvmB, thrID, (n0, k))
        # register order is the fragment layouts' V axis, first mode fastest:
        # a[m + 4·k], b[n + 8·k], d[m + 4·n], 64 FMA per k-tile
        {.unroll.}
        for n in 0 ..< 8:
          {.unroll.}
          for m in 0 ..< 4:
            var acc = rD[m + 4 * n]
            {.unroll.}
            for kk in 0 ..< 2:
              acc = acc + rA[m + 4 * kk] * rB[n + 8 * kk]
            rD[m + 4 * n] = acc
      if alpha != 1.0'f32:
        for i in 0 ..< 32:
          rD[i] = alpha * rD[i]
      d.store(rD, tvmD, thrID, (m0, n0))

# ═══════════════════════════════════════════════════════════════════════
#  CPU f64 reference and fixtures
# ═══════════════════════════════════════════════════════════════════════

proc gemmRef(expected, aHost, bHost: HostTensor, m, n, k: int, alpha: float32) =
  ## expected = alpha·A·Bᵀ accumulated in float64 over the float32 inputs,
  ## A (m, k) with stride (k, 1), B (n, k) with stride (k, 1)
  for mm in 0 ..< m:
    for nn in 0 ..< n:
      var acc = 0.0'f64
      for kk in 0 ..< k:
        acc += float64(aHost[mm, kk]) * float64(bHost[nn, kk])
      expected[mm, nn] = float32(float64(alpha) * acc)

proc fillPattern(ht: HostTensor) =
  ## Deterministic small-magnitude fill, |v| <= 1.125
  for i in 0 ..< ht.len:
    ht.data[i] = float32((i * 37 mod 19) - 9) * 0.125'f32

proc runTileGemmCase(engine: VulkanEngine, m, n, k: int, msg: string) =
  ## One launch with runtime tensor shapes, the result compared against
  ## the float64 reference
  var
    aHost = makeHostTensor(float32, (m, k), (k, 1))
    bHost = makeHostTensor(float32, (n, k), (k, 1))
    dHost = makeHostTensor(float32, (m, n), (n, 1))
    expected = makeHostTensor(float32, (m, n), (n, 1))
  fillPattern(aHost)
  fillPattern(bHost)
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn tileGemm(dDev, aDev, bDev, 1.5'f32) do:
    grid = (ceil_div(m, 32), ceil_div(n, 32))
    blk = (32, 1)
    lang = glsl
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  gemmRef(expected, aHost, bHost, m, n, k, 1.5'f32)
  assertAllClose(dHost.view(), expected.view(), rtol = 1e-4, abstol = 1e-3,
    msg = msg)

proc runTest() =
  var engine = newEngine(runVulkan)
  echo engine.deviceName()

  # multi-workgroup grid in x and y, one k-tile deep
  runTileGemmCase(engine, 64, 64, 8, "interior 64x64x8, grid (2, 2)")
  # ragged edges on m, n and k, multiple k-tiles
  runTileGemmCase(engine, 97, 53, 45, "ragged 97x53x45, grid (4, 2)")

  echo "t_poc_tile_vulkan: ALL GREEN"

when isMainModule:
  runTest()
