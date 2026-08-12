# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 decoder — verify reconstructed weights against production CUDA kernel fixtures

import
  std/memfiles,
  std/os,
  std/strutils,
  std/strformat,
  std/sequtils,
  std/algorithm,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/transformers/src/quantizations/exl3

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-codec"

proc runTests*() =
  # Enumerate all layer fixture dirs
  var fixtureList: seq[string] = @[]
  for f in walkPattern(FixtureDir / "model_layers_*"):
    fixtureList.add(f)
  fixtureList.sort()
  let total = fixtureList.len
  if total == 0:
    raise newException(AssertionDefect, "[ttt] No EXL3 codec fixtures found at " & FixtureDir)

  runCppTest &"EXL3 decoder: {total} layers (K=5, cb=0) match production CUDA kernel":
    proc(): bool =
      var passed = 0
      var failed = 0

      for fixturePath in fixtureList:
        let metaJson = parseJson(readFile(fixturePath / "metadata.json"))
        let key = metaJson["layer_key"].getStr
        let expectedHash = metaJson["weight_hash"].getStr

        var memFile = memFiles.open(fixturePath / "fixture.safetensors", mode = fmRead)
        defer: close(memFile)
        var st = safetensors.load(memFile)

        let trellis = st.getTensorOwned("trellis")
        let K = metaJson["K"].getInt
        let cb = metaJson["cb"].getInt
        let inF = metaJson["in_features"].getInt
        let outF = metaJson["out_features"].getInt

        # Decode packed trellis to [in_features, out_features] then transpose to [out, in]
        let weight = exl3_reconstruct(trellis, K, cb, inF, outF).t().contiguous()

        let computedHash = weight.hash_tensor().toHex.toLowerAscii

        if computedHash == expectedHash:
          inc passed
        else:
          inc failed
          echo &"  ❌ {key}: hash mismatch (got {computedHash}, expected {expectedHash})"

      if failed > 0:
        echo &"\n  Summary: {passed}/{total} passed, {failed} failed"
        raise newException(AssertionDefect, &"[ttt] {failed} layers failed hash check")

      echo &"  Summary: {passed}/{total} layers verified ✓"
      true

when isMainModule:
  runTests()
