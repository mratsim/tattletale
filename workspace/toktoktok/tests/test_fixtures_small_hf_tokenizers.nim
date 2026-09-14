# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/unittest
import std/os
import pkg/jsony

import workspace/toktoktok/src/deserializers
import workspace/zstd/zstd_highlevel
import workspace/toktoktok/src/pipeline
import pytoktoktok

let
  SourcePathAbs = absolutePath(currentSourcePath())
  FixturesDir = SourcePathAbs.parentDir() /
    "fixtures" / "small"
  TokenizersDir = SourcePathAbs.parentDir() /
    "tokenizers"

const PullCap = 4096

proc pullAll(pipeline: TokPipeline, text: string): seq[int] =
  ## Run:
  ##   nim test_toktoktok  # from the worktree root
  ## Whole-input encode as a bounded-consumption loop, the one-shot
  ##
  ## convenience lives in the tests, never in the binding, so every row
  ## exercises bounded machine consumption and the drain discipline.
  pipeline.resetText(text)
  var taken = 0
  while true:
    var batch = 0
    for id in pipeline.items:
      result.add id
      inc taken
      inc batch
      if batch == PullCap:
        break
    if batch < PullCap:
      break

type
  CodecFixture = object
    name: string
    text: string
    tokenIds: seq[int]
    tokenizer: string

proc runHfTokenizersTests() =
  suite "HF Tokenizers Fixtures Tests":
    const HfFixtures = [
      ("gpt2", "gpt2-tokenizer.json"),
      ("llama3", "llama3-tokenizer.json"),
      ("minimax-m2.1", "minimax-m2.1-tokenizer.json"),
      ("glm-4.7", "glm-4.7-tokenizer.json"),
      ("exaone", "exaone-tokenizer.json"),
      ("step-3.5-flash", "step-3.5-flash-tokenizer.json"),
    ]

    for pair in HfFixtures:
      let fixtureName = pair[0]
      let hfFile = pair[1]
      let fixturePath = FixturesDir / "hf_" & fixtureName
      let hfPath = TokenizersDir / hfFile
      let testName = "HF tokenizers library fixture (" & fixtureName & ")"

      # the fixture container is the recorded .json.zst frame
      doAssert fileExists(fixturePath & ".json.zst"),
        "Fixture not found: " & fixturePath
      doAssert fileExists(hfPath), "HF tokenizer not found: " & hfPath

      # the frames are recorded from the HF tokenizers library, whose
      # added-token ids the specials-active pipeline emits the same way
      let loaded = load_tokenizer_hf(hfPath)
      let content = readFile(fixturePath & ".json.zst").zstdDecompress(string)
      let fixtures = content.fromJson(seq[CodecFixture])

      for fixture in fixtures:
        test "HF tokenizer fixture - " & fixture.name & " (" & fixtureName & ")":
          let result = pullAll(loaded.pipe, fixture.text)
          check result == fixture.tokenIds

when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runHfTokenizersTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
