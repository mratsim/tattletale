# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile-algebra GEMM tile through the Vulkan GLSL route
##
## Test that a register-fragment tile kernel computes D = alpha·A·Bᵀ
## correctly on MoltenVK through glslang:
##
## - direct fragment addressing over runtime tensor shapes, `{.unroll.}`
##   inner FMA body
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
#  Fragment maps of the 4x8-cell schedule
# ═══════════════════════════════════════════════════════════════════════
#
# Fragment maps of a 32-thread schedule over one (M32, K2) A tile,
# one (N32, K2) B tile and one (M32, N32) D tile, int strides only:
# - A: thread t addresses A coordinates 4·(t div 4) ..+3 for both k, the t mod 4 half reloads them (stride 0)
# - B: thread t addresses B coordinates 8·(t mod 4) ..+7, both k
# - D: thread t owns one disjoint 4x8 output cell, V col-major
#   (m_local, n_local), offset m_local + 32·n_local
#
# Fragment addressing is direct runtime arithmetic, the static-map substitution loses composed
# coordinate strides in the CIRv2 frontend when a static param contains them

proc tileGemm(d: distinct TensorView, a: distinct TensorView, b: distinct TensorView, alpha: float32) =
  ## D = alpha·A·Bᵀ over 32x32 output tiles: one threadgroup per tile,
  ## 32 threads each owning one disjoint 4x8 output cell, k-tiles 2 deep.
  ##
  ## Fragment addressing of the 4x8-cell schedule, thread t:
  ##
  ## - D cell: tile m-lines 4*(t div 4) ..+3, tile n-columns 8*(t mod 4) ..+7
  ## - A fragment: the thread's 4 cell m-lines x 2 k, rA[m + 4*kk]
  ## - B fragment: the thread's 8 cell n-columns x 2 k, rB[n + 8*kk]
  ## - cell value (m, n) = sum over kk of A(m, kk) * B(n, kk), rD[m + 4*n]
  ##
  ## Runtime params only: shapes, strides and the tile grid are runtime values.
  ## Ragged edges read padWith and drop out-of-bounds stores.
  let
    thrID = getLocalThreadID()
    rowBlock = thrID div 4
    colBlock = thrID mod 4
    m0 = 32 * get_group_id(0)
    n0 = 32 * get_group_id(1)
    kTiles = ceil_div(a.layout.shape[1], 2)
  block:
    var rD: tile(d.T, make_layout((32, (1, 1))))
    rD.fillWith(zero(d.T))
    for kt in 0 ..< kTiles:
      let k0 = 2 * kt
      var rA: tile(a.T, make_layout((8, (1, 1))))
      var rB: tile(b.T, make_layout((16, (1, 1))))
      {.unroll.}
      for kk in 0 ..< 2:
        {.unroll.}
        for m in 0 ..< 4:
          rA[m + 4 * kk] = a.paddedLoad((m0 + 4 * rowBlock + m, k0 + kk), zero(a.T))
      {.unroll.}
      for kk in 0 ..< 2:
        {.unroll.}
        for n in 0 ..< 8:
          rB[n + 8 * kk] = b.paddedLoad((n0 + 8 * colBlock + n, k0 + kk), zero(b.T))
      {.unroll.}
      for n in 0 ..< 8:
        {.unroll.}
        for m in 0 ..< 4:
          var acc = rD[m + 4 * n]
          {.unroll.}
          for kk in 0 ..< 2:
            acc = acc + rA[m + 4 * kk] * rB[n + 8 * kk]
          rD[m + 4 * n] = acc
    for n in 0 ..< 8:
      for m in 0 ..< 4:
        let crd = (m0 + 4 * rowBlock + m, n0 + 8 * colBlock + n)
        if crd.inBounds(d.layout.shape):
          d[crd] = alpha * rD[m + 4 * n]

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
  engine.copyToHostFromDevice(dHost.data.toOpenArray(0, dHost.len - 1), dDev.data)
  gemmRef(expected, aHost, bHost, m, n, k, 1.5'f32)
  assertAllClose(dHost.view(), expected.view(), rtol = 1e-4, abstol = 1e-3,
    msg = msg)

proc runTest() =
  var engine = newEngine(runVulkan)
  echo engine.deviceName()

  # multi-workgroup grid in x and y, one k-tile deep
  engine.runTileGemmCase(64, 64, 8, "interior 64x64x8, grid (2, 2)")
  # ragged edges on m, n and k, multiple k-tiles
  engine.runTileGemmCase(97, 53, 45, "ragged 97x53x45, grid (4, 2)")

  echo "t_poc_tile_vulkan: ALL GREEN"

when isMainModule:
  runTest()
