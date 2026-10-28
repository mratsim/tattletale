# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SDPA suite machinery shared by the drivers, per backend.
## The runtime parts are plain procs, one copy total.
## Only the runner monomorphizes per dtype, head dim and masking combo, all in one compile.

import std/[os, math]
import workspace/math
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_sdpa/sdpa_fixtures
import workspace/ceramic/src/kernels/k_tile_sdpa

export tile_algebra
export k_tile_sdpa

const SdpaTile = 32
  ## 32x32x32 mma schedule tile, threadgroups step query rows by it

proc sdpaCheckMeta(meta: SdpaFixtureMetadata, dtype: string, headDim: int, isCausal, windowed: bool) =
  doAssert meta.dtype == dtype, meta.name & ": dtype " & meta.dtype
  doAssert meta.Dk == headDim and meta.Dv == headDim, meta.name & ": head dim mismatch"
  doAssert meta.causal == isCausal, meta.name & ": causal flag mismatch"
  doAssert (meta.windowSize > 0) == windowed, meta.name & ": window flag mismatch"
  doAssert meta.Hq mod meta.Hkv == 0,
    "fixture " & meta.name & ": Hq " & $meta.Hq & " is not a multiple of Hkv " & $meta.Hkv

proc sdpaCompare(got, expected: auto, msg: string, lowPrec: bool) =
  let (rtol, abstol) = if lowPrec: (1e-3'f64, 1e-2'f64) else: (1e-5'f64, 1e-4'f64)
  assertAllClose(got.view(), expected.view(), rtol = rtol, abstol = abstol, msg = msg)

proc sdpaLoadAndBridge[E](engine: E, Elem: typedesc, meta: SdpaFixtureMetadata): auto =
  let
    qh = loadFixtureTensor(SdpaFixturesDir / meta.name, "q.bin",
                           (meta.B, meta.Hq, meta.L, meta.Dk), Elem)
    kh = loadFixtureTensor(SdpaFixturesDir / meta.name, "k.bin",
                           (meta.B, meta.Hkv, meta.S, meta.Dk), Elem)
    vh = loadFixtureTensor(SdpaFixturesDir / meta.name, "v.bin",
                           (meta.B, meta.Hkv, meta.S, meta.Dv), Elem)
    oh = makeHostTensor(Elem, (meta.B, meta.Hq, meta.L, meta.Dv), LayoutRight)
    expected = loadFixtureTensor(SdpaFixturesDir / meta.name, "o.bin",
                                 meta.expectedShape, float32)
  doAssert xorHash(qh) == meta.xorhashChecksum("q.bin"), meta.name & ": q.bin xorhash mismatch"
  doAssert xorHash(kh) == meta.xorhashChecksum("k.bin"), meta.name & ": k.bin xorhash mismatch"
  doAssert xorHash(vh) == meta.xorhashChecksum("v.bin"), meta.name & ": v.bin xorhash mismatch"
  doAssert xorHash(expected) == meta.checksum, meta.name & ": o.bin checksum mismatch"
  (engine.bridge(qh), engine.bridge(kh), engine.bridge(vh), engine.bridge(oh), oh, expected)

proc sdpaCombo*(o, q, k, v: distinct TensorView, scale: float32, Dk, Dv: static int, isCausal, windowed: static bool, windowSize: int = low(int32)) =
  # Requires export: the launch machinery re-emits the callee name into the importing module.
  scaled_dot_product_attention(
    tiledMma(float32, q.T, k.T, SdpaTile, SdpaTile, SdpaTile),
    tiledMma(float32, o.T, v.T, SdpaTile, SdpaTile, SdpaTile),
    o, q, k, v, scale, Dk, Dv, isCausal, windowed, windowSize)

proc sdpaReplayCase(engine: MetalEngine, Elem: typedesc, HeadDim: static int, isCausal, windowed: static bool, name: string) =
  let meta = parseSdpaMetadata(SdpaFixturesDir / name)
  sdpaCheckMeta(meta, $Elem, HeadDim, isCausal, windowed)
  let (qDev, kDev, vDev, oDev, oh, expected) = sdpaLoadAndBridge(engine, Elem, meta)
  let
    scaleVal = meta.scale
    qTiles = ceil_div(meta.L, SdpaTile)

  engine.launchDeviceFn sdpaCombo(
      oDev, qDev, kDev, vDev, scaleVal, HeadDim, HeadDim, isCausal, windowed,
      when windowed: meta.windowSize else: low(int32)) do:
    grid = (qTiles, meta.Hq, meta.B)
    blk = (32, 1)

  engine.copyToHostFromDevice(addr oh.data[0], oDev.data, oh.len())
  let got = oh.to(float32)
  sdpaCompare(got, expected, "fixture " & meta.name & " (" & $Elem & ")",
              not (Elem is float32))
  echo "replay ok: ", meta.name, " (", $Elem, ")"

proc replayFixture*(engine: MetalEngine, Elem: typedesc, HeadDim: static int, isCausal, windowed: static bool, names: openArray[string]) =
  for name in names:
    sdpaReplayCase(engine, Elem, HeadDim, isCausal, windowed, name)
