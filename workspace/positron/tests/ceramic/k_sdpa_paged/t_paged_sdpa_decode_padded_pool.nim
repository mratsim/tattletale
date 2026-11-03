# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Padded-pool replay: pool entries may sit packed (entry stride == Hkv * D)
## or padded (entry stride > Hkv * D), both must read the same keys and values.
##
## - one sequence, two pages, one kv tile per page, page ids 1 and 2
## - both pools hold identical logical KV entries, padded gaps hold garbage
## - pass: the padded run matches the packed run
##
## Run from the repository root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/t_paged_sdpa_decode_padded_pool --nimcache:nimcache/t_paged_sdpa_decode_padded_pool \
##     workspace/positron/tests/ceramic/k_sdpa_paged/t_paged_sdpa_decode_padded_pool.nim

import std/random
import workspace/crucible
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import workspace/positron/tests/ceramic/k_sdpa_paged/kernel_sdpa_paged_template

const
  D = 16             # head dim, f32 path like the base suite's tiny_f32
  PageSize = 32      # one kv tile per page, matching the fixture suite's schedule
  Seqlen = 40        # two pages: a full tile then an 8-entry tail
  RowPadK = 3        # padded K pool: entry stride D + RowPadK
  RowPadV = 5        # padded V pool: entry stride D + RowPadV

proc makeFlatI32Tensor(n: int): auto =
  ## Rank-1 int32 host tensor for cache seqlens. Built by hand:
  ## makeHostTensor rejects rank-1 shapes (see sdpa_paged_fixtures.nim).
  let layout = make_layout((n,))
  HostTensor[int32, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[int32]](allocShared(n * sizeof(int32))),
    offset: 0,
    layout: layout)

proc fillPagedPool[T, Sh, St](ht: HostTensor[T, Sh, St], rng: var Rand) =
  ## Fill a (numPages, numLayers, pageSize, Hkv, D) pool: garbage in every buffer slot,
  ## then deterministic values in [1, 2] on the logical entries.
  ## Padding slots left by an entry stride above Hkv * D keep their garbage.
  for i in 0 ..< toInt(ht.layout.cosize()):
    ht.data[i] = -7.0'f32
  for p in 0 ..< ht.layout.shape[0]:
    for l in 0 ..< ht.layout.shape[1]:
      for r in 0 ..< ht.layout.shape[2]:
        for h in 0 ..< ht.layout.shape[3]:
          for j in 0 ..< D:
            ht[p, l, r, h, j] = rng.rand(1.0 .. 2.0)

proc copyLogicalEntries[T, ShD, StD, ShS, StS](dst: HostTensor[T, ShD, StD], src: HostTensor[T, ShS, StS]) =
  ## Copy logical pool entries src -> dst, same shape, dst may be padded.
  for p in 0 ..< dst.layout.shape[0]:
    for l in 0 ..< dst.layout.shape[1]:
      for r in 0 ..< dst.layout.shape[2]:
        for h in 0 ..< dst.layout.shape[3]:
          for j in 0 ..< D:
            dst[p, l, r, h, j] = src[p, l, r, h, j]

proc replayPaddedPool() =
  var rng = initRand(42)
  var engine = newEngine(runMetal)

  # packed pool: entry stride == Hkv * D
  # padded pools: entry stride == Hkv * D + pad
  let
    packedK = makeHostTensor(float32, (3, 1, PageSize, 1, D), LayoutRight)
    packedV = makeHostTensor(float32, (3, 1, PageSize, 1, D), LayoutRight)
    paddedK = makeHostTensor(float32, (3, 1, PageSize, 1, D),
      (PageSize * (D + RowPadK), PageSize * (D + RowPadK), D + RowPadK, D, 1))
    paddedV = makeHostTensor(float32, (3, 1, PageSize, 1, D),
      (PageSize * (D + RowPadV), PageSize * (D + RowPadV), D + RowPadV, D, 1))
  doAssert toInt(paddedK.layout.stride[2]) > 1 * D, "padded K pool must have padded rows"
  doAssert toInt(paddedV.layout.stride[2]) > 1 * D, "padded V pool must have padded rows"

  # identical logical KV entries in both pools
  fillPagedPool(packedK, rng)
  fillPagedPool(packedV, rng)
  fillPagedPool(paddedK, rng)
  fillPagedPool(paddedV, rng)
  copyLogicalEntries(paddedK, packedK)
  copyLogicalEntries(paddedV, packedV)

  let
    qh = makeHostTensor(float32, (1, 1, 1, D), LayoutRight)
    oh = makeHostTensor(float32, (1, 1, 1, D), LayoutRight)
  for h in 0 ..< D:
    qh[0, 0, 0, h] = rng.rand(1.0 .. 2.0)

  let
    bt = makeHostTensor(int32, (1, 2), LayoutRight)
  var cs = makeFlatI32Tensor(1)
  bt[0, 0] = 1
  bt[0, 1] = 2   # page ids 1 and 2, pool pages 0 and 2 stay unread
  cs[0] = Seqlen

  let
    scaleVal = 0.125'f32
    layerVal = int32(0)

  proc runOnPool(kh, vh: auto): auto =
    let
      qDev = engine.bridge(qh)
      kDev = engine.bridge(kh)
      vDev = engine.bridge(vh)
      btDev = engine.bridge(bt)
      csDev = engine.bridge(cs)
      oDev = engine.bridge(oh)
    engine.launchDeviceFn sdpaPagedCombo(
        oDev, qDev, kDev, vDev, btDev, csDev, scaleVal, D, D,
        PageSize, false, low(int32), layerVal) do:
      grid = (1, 1, 1)
      blk = (32, 1)
    var outHost = makeHostTensor(float32, (1, 1, 1, D), LayoutRight)
    engine.copyToHostFromDevice(addr outHost.data[0], oDev.data, outHost.len())
    outHost

  let gotPacked = runOnPool(packedK, packedV)
  let gotPadded = runOnPool(paddedK, paddedV)

  assertAllClose(gotPadded.view(), gotPacked.view(), rtol = 1e-5, abstol = 1e-4,
                 msg = "padded-pool decode must match the packed-pool decode")
  echo "padded pool replay ok: padded rows read the same keys and values"

replayPaddedPool()
