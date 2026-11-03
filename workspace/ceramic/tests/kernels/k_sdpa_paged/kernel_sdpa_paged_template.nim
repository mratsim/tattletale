# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Paged SDPA suite machinery shared by the drivers, per backend.
## The runtime parts are ordinary procs, one copy total.
## Only the runner monomorphizes per dtype and head dim, all in one compile.

import std/[os, math]
import workspace/math
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_sdpa_paged/sdpa_paged_fixtures
import workspace/ceramic/src/kernels/k_tile_sdpa_paged

export tile_algebra, layout_algebra, tensors, int_tuples

const SdpaPagedTile = 32
  ## 32x32x32 mma schedule tile, one kv tile per page at the fixture page size

proc sdpaPagedCheckMeta(meta: SdpaPagedFixtureMetadata, dtype: string, headDim: int) =
  doAssert meta.dtype == dtype, meta.name & ": dtype " & meta.dtype
  doAssert meta.Dk == headDim and meta.Dv == headDim, meta.name & ": head dim mismatch"
  doAssert meta.Hq mod meta.Hkv == 0,
    "fixture " & meta.name & ": Hq " & $meta.Hq & " is not a multiple of Hkv " & $meta.Hkv
  doAssert meta.pageSize == SdpaPagedTile,
    "fixture " & meta.name & ": page size " & $meta.pageSize & " != the tile's kv tile size " & $SdpaPagedTile
  for seqlen in meta.cacheSeqlens:
    doAssert seqlen <= meta.maxPages * meta.pageSize,
      "fixture " & meta.name & ": seqlen " & $seqlen & " exceeds the table capacity"

proc sdpaPagedCompare(got, expected: auto, msg: string, lowPrec: bool) =
  let (rtol, abstol) = if lowPrec: (1e-3'f64, 1e-2'f64) else: (1e-5'f64, 1e-4'f64)
  assertAllClose(got.view(), expected.view(), rtol = rtol, abstol = abstol, msg = msg)

proc sdpaPagedLoadAndBridge[E](engine: E, Elem: typedesc, meta: SdpaPagedFixtureMetadata): auto =
  let dir = SdpaPagedFixturesDir / meta.name
  let
    qh = loadFixtureTensor(dir, "q.bin", (meta.numSeqs, meta.Hq, 1, meta.Dk), Elem)
    kh = loadFixtureTensor(dir, "k.bin",
                           (meta.numPages, meta.numLayers, meta.pageSize, meta.Hkv, meta.Dk), Elem)
    vh = loadFixtureTensor(dir, "v.bin",
                           (meta.numPages, meta.numLayers, meta.pageSize, meta.Hkv, meta.Dv), Elem)
    bh = loadFixtureTensor(dir, "block_table.bin", (meta.numSeqs, meta.maxPages), int32)
    ch = loadFlatFixtureTensor(dir, "cache_seqlens.bin", meta.numSeqs)
    oh = makeHostTensor(Elem, (meta.numSeqs, meta.Hq, 1, meta.Dv), LayoutRight)
    expected = loadFixtureTensor(dir, "o.bin", meta.expectedShape, float32)
  doAssert xorHash(qh) == meta.xorhashChecksum("q.bin"), meta.name & ": q.bin xorhash mismatch"
  doAssert xorHash(kh) == meta.xorhashChecksum("k.bin"), meta.name & ": k.bin xorhash mismatch"
  doAssert xorHash(vh) == meta.xorhashChecksum("v.bin"), meta.name & ": v.bin xorhash mismatch"
  doAssert xorHash(bh) == meta.xorhashChecksum("block_table.bin"), meta.name & ": block_table.bin xorhash mismatch"
  doAssert xorHash(ch) == meta.xorhashChecksum("cache_seqlens.bin"), meta.name & ": cache_seqlens.bin xorhash mismatch"
  doAssert xorHash(expected) == meta.checksum, meta.name & ": o.bin checksum mismatch"
  (engine.bridge(qh), engine.bridge(kh), engine.bridge(vh),
   engine.bridge(bh), engine.bridge(ch), engine.bridge(oh), oh, expected)

proc sdpaPagedCombo*(o, q, kvPoolK, kvPoolV: distinct TensorView,
                     blockTable, cacheSeqlens: distinct TensorView,
                     scale: float32, Dk, Dv, PageSize: static int, layer: int32) =
  paged_sdpa_decode(
    tiledMma(float32, q.T, kvPoolK.T, SdpaPagedTile, SdpaPagedTile, SdpaPagedTile),
    tiledMma(float32, o.T, kvPoolV.T, SdpaPagedTile, SdpaPagedTile, SdpaPagedTile),
    o, q, kvPoolK, kvPoolV, blockTable, cacheSeqlens,
    scale, Dk, Dv, PageSize, layer)

proc sdpaPagedReplayCase(engine: MetalEngine, Elem: typedesc, HeadDim: static int, name: string) =
  let meta = parseSdpaPagedMetadata(SdpaPagedFixturesDir / name)
  sdpaPagedCheckMeta(meta, $Elem, HeadDim)
  let (qDev, kDev, vDev, btDev, csDev, oDev, oh, expected) =
    sdpaPagedLoadAndBridge(engine, Elem, meta)
  let
    scaleVal = meta.scale
    layerVal = int32(meta.layer)

  engine.launchDeviceFn sdpaPagedCombo(
      oDev, qDev, kDev, vDev, btDev, csDev, scaleVal, HeadDim, HeadDim,
      SdpaPagedTile, layerVal) do:
    grid = (1, meta.Hq, meta.numSeqs)
    blk = (32, 1)

  engine.copyToHostFromDevice(addr oh.data[0], oDev.data, oh.len())
  let got = oh.to(float32)
  sdpaPagedCompare(got, expected, "fixture " & meta.name & " (" & $Elem & ")",
                   not (Elem is float32))
  echo "replay ok: ", meta.name, " (", $Elem, ")"

proc replayPagedFixture*(engine: MetalEngine, Elem: typedesc, HeadDim: static int, names: openArray[string]) =
  for name in names:
    sdpaPagedReplayCase(engine, Elem, HeadDim, name)
