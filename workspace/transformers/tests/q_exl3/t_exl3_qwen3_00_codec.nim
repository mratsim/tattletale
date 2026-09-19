# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 trellis decode of the recorded weight slices, every reconstructed
## weight hash must equal the recorded production kernel hash.
## Run through the test_tf_exl3_qwen3_00_codec task in config.nims.

import
  std/algorithm,
  std/os,
  std/json,
  std/strutils,
  workspace/libtorch,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/transformers/src/quantizations/exl3,
  workspace/transformers/tests/harness/harness

from workspace/transformers/tests/harness import zstdReadFixture

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-00-codec" / "Qwen3-0.6B-EXL3-5bpw"

proc main(): bool =
  ## Replays the trellis decode of every recorded weight slice.
  ##
  ## The decode is a pure function of the recorded bytes, so the reconstructed
  ## weight hash must equal the recorded kernel hash.
  var fixtureList: seq[string] = @[]
  for dir in walkPattern(FixtureDir / "model_layers_*"):
    fixtureList.add(dir)
  fixtureList.sort()
  if fixtureList.len == 0:
    raise newException(IOError,  # noqa  # the codec round-trip fails on missing fixtures, the instruments bind recorded tensors
      "no EXL3 codec fixtures found at " & FixtureDir)

  for fixturePath in fixtureList:
    let metaJson = parseJson(zstdReadFixture(fixturePath / "metadata.json.zst"))
    let layerKey = metaJson["layer_key"].getStr()
    let expectedHash = metaJson["weight_hash"].getStr()

    var st = Safetensor.open(fixturePath / "fixture.safetensors")
    let trellis = st.getTensorOwned("trellis")

    # Decode the packed trellis to [in_features, out_features], transpose
    # to [out_features, in_features], hash the weight words.
    let weight = exl3_reconstruct(trellis,
      metaJson["K"].getInt(), metaJson["cb"].getInt(),
      metaJson["in_features"].getInt(), metaJson["out_features"].getInt())
    let computedHash = weight.t().contiguous().hash_tensor().toHex.toLowerAscii

    if computedHash != expectedHash:
      raise newException(HarnessCheckError,  # noqa  # the codec asserts byte equality of a hash, no band instrument exists for it
        "layer " & layerKey & " decode hash " & computedHash &
        " != recorded " & expectedHash)
    echo "layer " & layerKey & " matches the recorded kernel hash"
  result = true

when isMainModule:
  runCppTest("exl3 qwen3 codec", main)
