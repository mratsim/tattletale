# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Scaled dot product attention on Metal, replayed against the python-generated
## fixture corpus with one fixture per subprocess. The suite compiles one dtype
## leg per binary, default f32, with the `-d:SdpaLegF16` and `-d:SdpaLegBf16`
## flags picking the other legs. A fixture-name argument replays that fixture.
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/k_sdpa/t_scaled_dot_product_attention_metal.nim
##
## Fixtures regenerate with `fixtures/gen_sdpa_fixtures.py` under `uv run --group test-vectors`.

import std/[os, osproc, math]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/fp_conv/fp_conv
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_sdpa/sdpa_fixtures
import workspace/ceramic/src/kernels/k_tile_sdpa

const SdpaTile = 32
  ## 32x32x32 mma schedule tile, threadgroups step query rows by it

when defined(SdpaLegF16):
  const SdpaFixtureNames = [
    "tiny_f16", "causal_gqa2_f16", "decode_gqa8_f16",
    "gqa4_f16", "gqa8_causal_f16"]
elif defined(SdpaLegBf16):
  const SdpaFixtureNames = ["tiny_bf16", "gqa2_bf16"]
else:
  const SdpaFixtureNames = [
    "tiny_f32", "gqa2_f32", "batched_gqa_f32", "causal_f32",
    "mask_f32", "causal_mask_gqa4_f32", "mask_broadcast_f32",
    "gqa4_f32", "gqa8_f32", "gqa8_causal_f32", "gqa8_mask_f32",
    "d128_causal_f32", "scale_f32", "cross_attn_f32", "mask_gqa_heads_f32"]

proc replay[E: HwEngine, T](engine: E, meta: SdpaFixtureMeta, dir: string, _: typedesc[T], Dk, Dv: static int) =
  ## Bridges one fixture, launches the kernel, compares against o.bin
  # host-side validation of the head-count contract: the device procs
  # cannot raise on a violated one, the composed layouts would misindex
  doAssert meta.Hq mod meta.Hkv == 0,
    "fixture " & meta.name & ": Hq " & $meta.Hq & " is not a multiple of Hkv " & $meta.Hkv
  let
    qh = loadFixtureTensor[T](dir, "q.bin", (meta.B, meta.Hq, meta.L, meta.Dk))
    kh = loadFixtureTensor[T](dir, "k.bin", (meta.B, meta.Hkv, meta.S, meta.Dk))
    vh = loadFixtureTensor[T](dir, "v.bin", (meta.B, meta.Hkv, meta.S, meta.Dv))
    oh = makeHostTensor(T, (meta.B, meta.Hq, meta.L, meta.Dv),
      (meta.Hq * meta.L * meta.Dv, meta.L * meta.Dv, meta.Dv, 1))
    expected = loadFixtureTensor[float32](dir, "o.bin", meta.expectedShape)

  let
    qDev = engine.bridge(qh)
    kDev = engine.bridge(kh)
    vDev = engine.bridge(vh)
    oDev = engine.bridge(oh)
    scaleVal = meta.scale
    qTiles = (meta.L + SdpaTile - 1) div SdpaTile

  if meta.mask == "additive":
    let
      nBytes = int(getFileSize(dir / "mask.bin"))
      Hm = nBytes div (4 * meta.B * meta.S * meta.S)
      mh = loadFixtureTensor[float32](dir, "mask.bin", (meta.B, Hm, meta.S, meta.S))
    doAssert Hm == 1 or Hm == meta.Hkv,
      "fixture " & meta.name & ": mask head count " & $Hm & " is neither 1 nor Hkv " & $meta.Hkv
      maskDev = engine.bridge(mh)
    if meta.causal:
      engine.launchDeviceFn scaled_dot_product_attention(
          tiledMma(float32, T, T, SdpaTile, SdpaTile, SdpaTile),
          tiledMma(float32, T, T, SdpaTile, SdpaTile, SdpaTile),
          oDev, qDev, kDev, vDev, maskDev, scaleVal, Dk, Dv, true) do:
        grid = (qTiles, meta.Hq, meta.B)
        blk = (32, 1)
    else:
      engine.launchDeviceFn scaled_dot_product_attention(
          tiledMma(float32, T, T, SdpaTile, SdpaTile, SdpaTile),
          tiledMma(float32, T, T, SdpaTile, SdpaTile, SdpaTile),
          oDev, qDev, kDev, vDev, maskDev, scaleVal, Dk, Dv, false) do:
        grid = (qTiles, meta.Hq, meta.B)
        blk = (32, 1)
  elif meta.causal:
    engine.launchDeviceFn scaled_dot_product_attention(
        tiledMma(float32, T, T, SdpaTile, SdpaTile, SdpaTile),
        tiledMma(float32, T, T, SdpaTile, SdpaTile, SdpaTile),
        oDev, qDev, kDev, vDev, scaleVal, Dk, Dv, true) do:
      grid = (qTiles, meta.Hq, meta.B)
      blk = (32, 1)
  else:
    engine.launchDeviceFn scaled_dot_product_attention(
        tiledMma(float32, T, T, SdpaTile, SdpaTile, SdpaTile),
        tiledMma(float32, T, T, SdpaTile, SdpaTile, SdpaTile),
        oDev, qDev, kDev, vDev, scaleVal, Dk, Dv) do:
      grid = (qTiles, meta.Hq, meta.B)
      blk = (32, 1)

  engine.copyToHostFromDevice(addr oh.data[0], oDev.data, oh.len())
  # the comparison is f64 against the f32 truth, the kernel output widened
  var got = makeHostTensor(float32, meta.expectedShape,
    (meta.Hq * meta.L * meta.Dv, meta.L * meta.Dv, meta.Dv, 1))
  for i in 0 ..< got.len():
    got.data[i] = to(oh.data[i], float32)
  when T is float32:
    let (rtol, abstol) = (1e-5, 1e-4)
  else:
    let (rtol, abstol) = (1e-3, 1e-2)
  assertAllClose(got.view(), expected.view(), rtol = rtol, abstol = abstol,
    msg = "fixture " & meta.name & " (" & $T & ")")
  echo "replay ok: ", meta.name, " (", $T, ")"

proc runFixture(name: string) =
  ## One engine per process, function-local, released on return
  let dir = SdpaFixturesDir / name
  let meta = readSdpaMeta(dir)
  var engine = newEngine(runMetal)
  template dispatchD(T: typedesc) =
    case meta.Dk
    of 16: replay(engine, meta, dir, T, 16, 16)
    of 32: replay(engine, meta, dir, T, 32, 32)
    of 64: replay(engine, meta, dir, T, 64, 64)
    of 128: replay(engine, meta, dir, T, 128, 128)
    else: doAssert false, "no static head-dim case for " & $meta.Dk
  
when defined(SdpaLegF16):
    dispatchD(float16)
  elif defined(SdpaLegBf16):
    dispatchD(bfloat16)
  else:
    dispatchD(float32)

proc main =
  if paramCount() == 0:
    for name in SdpaFixtureNames:
      let (outp, code) = execCmdEx(getAppFilename() & " " & name)
      stdout.write(outp)
      doAssert code == 0, "fixture " & name & " failed in its subprocess"
    echo "all SDPA fixture replays pass (", SdpaFixtureNames.len, " fixtures)"
  else:
    runFixture(paramStr(1))

main()
