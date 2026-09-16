import std/[monotimes, times]
import std/unittest
import std/os
import pkg/jsony

import workspace/toktoktok/src/deserializers
import workspace/zstd/zstd_highlevel
import pytoktoktok
import pull_chunks

let
  SourcePathAbs = absolutePath(currentSourcePath())
  FixturesDir = SourcePathAbs.parentDir() /
    "fixtures" / "small"
  TokenizersDir = SourcePathAbs.parentDir() /
    "tokenizers"

type
  CodecFixture = object
    name: string
    text: string
    tokenIds: seq[int]
    tokenizer: string

proc runTiktokenFixturesTests() =
  suite "Tiktoken Fixtures Tests":
    const TiktokenFixtures = [
      ("r50k_base", "r50k_base.tiktoken", "r50k"),
      ("p50k_base", "p50k_base.tiktoken", "p50k"),
      ("cl100k_base", "cl100k_base.tiktoken", "cl100k"),
      ("o200k_base", "o200k_base.tiktoken", "o200k"),
      ("kimik2.5", "kimik2.5.tiktoken", "kimik2.5"),
    ]

    for config in TiktokenFixtures:
      let (fixtureName, tiktokenFile, pattern) = config
      let fixturePath = FixturesDir / "tiktoken_" & fixtureName
      let tiktokenPath = TokenizersDir / tiktokenFile

      # the fixture container is the recorded .json.zst frame
      doAssert fileExists(fixturePath & ".json.zst"),
        "Fixture not found: " & fixturePath
      doAssert fileExists(tiktokenPath), "Tiktoken not found: " & tiktokenPath

      let loaded = load_tokenizer_tiktoken_ordinary(tiktokenPath, pattern)
      let content = readFile(fixturePath & ".json.zst").zstdDecompress(string)
      let fixtures = content.fromJson(seq[CodecFixture])

      for fixture in fixtures:
        test "Tiktoken fixture - " & fixture.name & " (" & fixtureName & ")":
          let result = pullAll(loaded.pipe, fixture.text)
          check result == fixture.tokenIds

when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runTiktokenFixturesTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
