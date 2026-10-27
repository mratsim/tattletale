# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SDPA fixture suite: one (head-dim, dtype, masking) combination per compile.
##
## Runtime dispatch over the three axes would instantiate every combination, each paying a full
## bridge codegen at Nim-compile time, so compile-time defines lock one combination per binary:
##
##   -d:sdpaDk=16|32|64|128   -d:sdpaDtype=f32|f16|bf16   -d:sdpaMask=plain|causal|windowed|causal_window
##
## Run from the repository root, one compile per combination, one subprocess per matching fixture:
##   nim c -r --maxLoopIterationsVM:100000000 -d:sdpaDk=16 -d:sdpaDtype=f32 -d:sdpaMask=plain \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/k_sdpa/t_scaled_dot_product_attention_metal.nim

import std/[os, osproc, math]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_sdpa/sdpa_fixtures
import workspace/ceramic/src/kernels/k_tile_sdpa

const SdpaTile = 32
  ## 32x32x32 mma schedule tile, threadgroups step query rows by it

const SdpaFixtureNames = [
  "tiny_f32", "gqa2_f32", "batched_gqa_f32", "causal_f32",
  "windowed_f32", "causal_window_f32",
  "gqa4_f32", "gqa8_f32", "gqa8_causal_f32",
  "d128_causal_f32", "scale_f32", "cross_attn_f32",
  "tiny_f16", "causal_gqa2_f16", "decode_gqa8_f16",
  "gqa4_f16", "gqa8_causal_f16",
  "windowed_gqa8_f16", "windowed_decode_f16", "windowed_f16",
  "tiny_bf16", "gqa2_bf16"]

const sdpaDk {.intdefine.} = 16
  ## the static head dim this compile instantiates

const sdpaDtype {.strdefine.} = "f32"
  ## the element type this compile instantiates: "f32", "f16" or "bf16"

const sdpaMask {.strdefine.} = "plain"
  ## the masking variant this compile instantiates:
  ## "plain", "causal", "windowed" or "causal_window"

when sdpaDtype == "f32":
  type Elem = float32
elif sdpaDtype == "f16":
  type Elem = float16
else:
  type Elem = bfloat16

proc replay[E: HwEngine](engine: E, meta: SdpaFixtureMetadata, dir: string) =
  ## Bridges one fixture, launches the kernel, compares against o.bin
  # host-side validation of the head-count contract: the device procs
  # cannot raise on a violated one, the composed layouts would misindex
  doAssert meta.Hq mod meta.Hkv == 0,
    "fixture " & meta.name & ": Hq " & $meta.Hq & " is not a multiple of Hkv " & $meta.Hkv
  let
    qh = loadFixtureTensor(dir, "q.bin", (meta.B, meta.Hq, meta.L, meta.Dk), Elem)
    kh = loadFixtureTensor(dir, "k.bin", (meta.B, meta.Hkv, meta.S, meta.Dk), Elem)
    vh = loadFixtureTensor(dir, "v.bin", (meta.B, meta.Hkv, meta.S, meta.Dv), Elem)
    oh = makeHostTensor(Elem, (meta.B, meta.Hq, meta.L, meta.Dv), LayoutRight)
    expected = loadFixtureTensor(dir, "o.bin", meta.expectedShape, float32)
  # loaded bytes match the generator's recorded per-file checksums
  doAssert xorHash(qh) == meta.xorhashChecksum("q.bin"), meta.name & ": q.bin xorhash mismatch"
  doAssert xorHash(kh) == meta.xorhashChecksum("k.bin"), meta.name & ": k.bin xorhash mismatch"
  doAssert xorHash(vh) == meta.xorhashChecksum("v.bin"), meta.name & ": v.bin xorhash mismatch"
  doAssert xorHash(expected) == meta.checksum, meta.name & ": o.bin checksum mismatch"

  let
    qDev = engine.bridge(qh)
    kDev = engine.bridge(kh)
    vDev = engine.bridge(vh)
    oDev = engine.bridge(oh)
    scaleVal = meta.scale
    qTiles = (meta.L + SdpaTile - 1) div SdpaTile

  when sdpaMask == "causal_window":
    engine.launchDeviceFn scaled_dot_product_attention(
        tiledMma(float32, Elem, Elem, SdpaTile, SdpaTile, SdpaTile),
        tiledMma(float32, Elem, Elem, SdpaTile, SdpaTile, SdpaTile),
        oDev, qDev, kDev, vDev, scaleVal, sdpaDk, sdpaDk, true, true,
        meta.windowSize) do:
      grid = (qTiles, meta.Hq, meta.B)
      blk = (32, 1)
  elif sdpaMask == "windowed":
    engine.launchDeviceFn scaled_dot_product_attention(
        tiledMma(float32, Elem, Elem, SdpaTile, SdpaTile, SdpaTile),
        tiledMma(float32, Elem, Elem, SdpaTile, SdpaTile, SdpaTile),
        oDev, qDev, kDev, vDev, scaleVal, sdpaDk, sdpaDk, false, true,
        meta.windowSize) do:
      grid = (qTiles, meta.Hq, meta.B)
      blk = (32, 1)
  elif sdpaMask == "causal":
    engine.launchDeviceFn scaled_dot_product_attention(
        tiledMma(float32, Elem, Elem, SdpaTile, SdpaTile, SdpaTile),
        tiledMma(float32, Elem, Elem, SdpaTile, SdpaTile, SdpaTile),
        oDev, qDev, kDev, vDev, scaleVal, sdpaDk, sdpaDk, true) do:
      grid = (qTiles, meta.Hq, meta.B)
      blk = (32, 1)
  else:
    engine.launchDeviceFn scaled_dot_product_attention(
        tiledMma(float32, Elem, Elem, SdpaTile, SdpaTile, SdpaTile),
        tiledMma(float32, Elem, Elem, SdpaTile, SdpaTile, SdpaTile),
        oDev, qDev, kDev, vDev, scaleVal, sdpaDk, sdpaDk) do:
      grid = (qTiles, meta.Hq, meta.B)
      blk = (32, 1)

  engine.copyToHostFromDevice(addr oh.data[0], oDev.data, oh.len())
  # comparison is f64 against the f32 truth, kernel output widened
  var got = makeHostTensor(float32, meta.expectedShape, LayoutRight)
  for i in 0 ..< got.len():
    got.data[i] = to(oh.data[i], float32)
  when Elem is float32:
    let (rtol, abstol) = (1e-5, 1e-4)
  else:
    let (rtol, abstol) = (1e-3, 1e-2)
  assertAllClose(got.view(), expected.view(), rtol = rtol, abstol = abstol,
    msg = "fixture " & meta.name & " (" & $Elem & ")")
  echo "replay ok: ", meta.name, " (", $Elem, ")"

proc comboMatches(meta: SdpaFixtureMetadata): bool =
  ## Whether the fixture's meta matches the combination locked at compile time
  let dtypeOk = when sdpaDtype == "f16": meta.dtype == "float16"
                elif sdpaDtype == "bf16": meta.dtype == "bfloat16"
                else: meta.dtype == "float32"
  let maskOk = when sdpaMask == "causal_window":
                 meta.causal and meta.windowSize > 0
               elif sdpaMask == "windowed":
                 (not meta.causal) and meta.windowSize > 0
               elif sdpaMask == "causal":
                 meta.causal and meta.windowSize == 0
               else:
                 (not meta.causal) and meta.windowSize == 0
  result = meta.Dk == sdpaDk and meta.Dv == sdpaDk and dtypeOk and maskOk

proc runFixture(name: string) =
  ## One engine per process, function-local, released on return
  let meta = parseSdpaMetadata(SdpaFixturesDir / name)
  doAssert comboMatches(meta),
    "fixture " & name & " does not match the compiled combination"
  var engine = newEngine(runMetal)
  replay(engine, meta, SdpaFixturesDir / name)

proc main() =
  echo "combination: Dk=", sdpaDk, " dtype=", sdpaDtype, " mask=", sdpaMask
  if paramCount() > 0:
    runFixture(paramStr(1))
  else:
    var ran = 0
    for name in SdpaFixtureNames:
      if comboMatches(parseSdpaMetadata(SdpaFixturesDir / name)):
        let (outp, code) = execCmdEx(getAppFilename() & " " & name)
        stdout.write(outp)
        doAssert code == 0, "fixture " & name & " failed in its subprocess"
        ran += 1
    doAssert ran > 0, "no fixture matches the compiled combination"
    echo "all SDPA fixture replays pass for the combination (", ran, " fixtures)"

main()
