# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
## Special pre-tokenization recorded vectors over the failing-vector
## frames of the recorded upstream report, asserted through the pull
## chain against the recorded ids.
## See the upstream report at https://github.com/mratsim/tattletale/issues/22

import std/[monotimes, times]
import std/unittest
import std/os
import pkg/jsony

import workspace/toktoktok/src/deserializers
import workspace/zstd/zstd_highlevel
import ../pytoktoktok
import ../pull_chunks

let
  SourcePathAbs = absolutePath(currentSourcePath())
  FixturesDir = SourcePathAbs.parentDir().parentDir() / "fixtures"
  TokenizersDir = SourcePathAbs.parentDir().parentDir() /
    "tokenizers"

type
  CodecFixture = object
    name: string
    text: string
    tokenIds: seq[int]
    tokenizer: string

proc loadPipeline(hfFile, tiktokenFile, pattern: string): PipelineRef =
  ## One pipeline per frame config, HF checkpoints through the HF loader,
  ## the kimik2.5 rank file through the tiktoken loader
  ## (the frame's tokenizer field names the checkpoint either way).
  if hfFile.len > 0:
    load_tokenizer_hf(TokenizersDir / hfFile)
  else:
    load_tokenizer_tiktoken(TokenizersDir / tiktokenFile, pattern)

proc runRecordedVectorTests() =
  ## Special-pretokenization failing vectors as recorded frames
  ## (recorded upstream in mratsim/tattletale#22, see the module header).
  ##
  ## Pull-chain rows resolve the flat-join divergence class,
  ## covering step-3.5-flash and kimik2.5,
  ## asserted against the recorded reference ids.
  ##
  ## Exaone frame rows stay skipped, their recorded ids come from the HF
  ## tokenizers library, whose AddedVocabulary splits added tokens out with longest-match.
  ##
  ## Converted-tiktoken special semantics apply instead (same-start tie by table order), a documented engine-flavor divergence.
  suite "Special pre-tokenization recorded vectors":
    const VectorFrames = [
      ("kimik2.5", "special_pretok_kimik2.5.json.zst", "", "kimik2.5.tiktoken", "kimik2.5"),
      ("step-3.5-flash", "special_pretok_step-3.5-flash.json.zst",
        "step-3.5-flash-tokenizer.json", "", ""),
      ("exaone", "special_pretok_exaone.json.zst", "exaone-tokenizer.json", "", ""),
    ]

    for frame in VectorFrames:
      let (configName, frameFile, hfFile, tiktokenFile, pattern) = frame
      let framePath = FixturesDir / frameFile

      doAssert fileExists(framePath), "Fixture frame not found: " & framePath

      let loaded = loadPipeline(hfFile, tiktokenFile, pattern)
      let content = readFile(framePath).zstdDecompress(string)
      let fixtures = content.fromJson(seq[CodecFixture])

      for fixture in fixtures:
        test "Special pre-tokenization vector - " & fixture.name &
            " (" & configName & ")":
          if configName == "exaone":
            echo "[SKIPPED] ", fixture.name, " (", configName,
              ") recorded ", $fixture.tokenIds.len,
              " ids (HF added-token longest-match class), pull chain not asserted"
            skip()
          else:
            let result = pullAll(loaded.pipe, fixture.text)
            check result == fixture.tokenIds

when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runRecordedVectorTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
