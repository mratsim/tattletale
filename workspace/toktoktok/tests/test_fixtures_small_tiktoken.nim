import std/unittest
import std/os
import pkg/jsony

import workspace/toktoktok

const FIXTURES_DIR = currentSourcePath().parentDir() / "fixtures" / "small"
const TOKENIZERS_DIR = currentSourcePath().parentDir() / "tokenizers"

type
  CodecFixture* = object
    name*: string
    text*: string
    tokenIds*: seq[int]
    tokenizer*: string

proc runTiktokenFixturesTests*() =
  suite "Tiktoken Fixtures Tests":
    const TiktokenFixtures = [
      ("r50k_base", "r50k_base.tiktoken", R50kRegexp),
      ("p50k_base", "p50k_base.tiktoken", P50kRegexp),
      ("cl100k_base", "cl100k_base.tiktoken", Cl100kRegexp),
      ("o200k_base", "o200k_base.tiktoken", O200kRegexp),
      ("kimik2.5", "kimik2.5.tiktoken", KimiK25Regexp),
    ]

    for config in TiktokenFixtures:
      let (fixtureName, tiktokenFile, regexp) = config
      let fixturePath = FIXTURES_DIR / "tiktoken_" & fixtureName & ".json"
      let tiktokenPath = TOKENIZERS_DIR / tiktokenFile
      let testName = "Tiktoken fixture (" & fixtureName & ")"

      test testName:
        doAssert fileExists(fixturePath), "Fixture not found: " & fixturePath
        doAssert fileExists(tiktokenPath), "Tiktoken not found: " & tiktokenPath

        let tokenizer = loadTiktokenizer(tiktokenPath, regexp)
        let content = readFile(fixturePath)
        let fixtures = content.fromJson(seq[CodecFixture])

        for fixture in fixtures:
          let result = tokenizer.encodeOrdinary(fixture.text)
          check result == fixture.tokenIds

when isMainModule:
  runTiktokenFixturesTests()
