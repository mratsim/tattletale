# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan GEMM kernel, 4x8-cell schedule: the device body.
##
## Schedule (proven on Metal in the C10D15 records, 5.5-5.8 TF fp32 class):
## 32 threads per threadgroup, each owning one disjoint 4x8 output cell,
## k-tiles 2 deep, register fragments, no shared memory, no subgroup use.

import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu

proc gemmTileVulkan*(d: distinct TensorView, a: distinct TensorView, b: distinct TensorView, alpha, beta: float32) =
  ## d = alpha·A·Bᵀ + beta·d, one threadgroup per 32x32 output tile,
  ## 32 threads each owning one disjoint 4x8 cell, k-tiles 2 deep.
  ##
  ## Fragment addressing of the 4x8-cell schedule, thread t:
  ##
  ## - D cell: tile m-lines 4*(t div 4) ..+3, tile n-columns 8*(t mod 4) ..+7
  ## - A fragment: the thread's 4 cell m-lines x 2 k (rA[m + 4*kk])
  ## - B fragment: the thread's 8 cell n-columns x 2 k (rB[n + 8*kk])
  ## - cell value (m, n) = sum over kk of A(m, kk) * B(n, kk) in rD[m + 4*n]
  ##
  ## Runtime params only. Ragged edges read padWith, out-of-bounds stores drop.
  ## beta = 0 never reads d.
  ##
  ## Fragment addressing is direct runtime arithmetic. The CIRv2 frontend
  ## loses composed coordinate strides when substituting a static param
  ## whose fields contain them.
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
          let dPrev = if beta != 0.0'f32: d.paddedLoad(crd, zero(d.T)) else: zero(d.T)
          d[crd] = alpha * rD[m + 4 * n] + beta * dPrev
