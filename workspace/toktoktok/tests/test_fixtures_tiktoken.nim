import std/unittest
import std/os
import pkg/jsony

import workspace/toktoktok

const FIXTURES_DIR = currentSourcePath().parentDir() / "fixtures" / "codec"
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
      ("r50k_base", "r50k_base.tiktoken"),
      ("p50k_base", "p50k_base.tiktoken"),
      ("cl100k_base", "cl100k_base.tiktoken"),
      ("o200k_base", "o200k_base.tiktoken"),
    ]

    for pair in TiktokenFixtures:
      let fixtureName = pair[0]
      let tiktokenFile = pair[1]
      let fixturePath = FIXTURES_DIR / fixtureName & ".json"
      let tiktokenPath = TOKENIZERS_DIR / tiktokenFile
      let testName = "Tiktoken fixture (" & fixtureName & ")"

      test testName:
        doAssert fileExists(fixturePath), "Fixture not found: " & fixturePath
        doAssert fileExists(tiktokenPath), "Tiktoken not found: " & tiktokenPath

        let tokenizer = loadTiktokenizer(tiktokenPath)
        let content = readFile(fixturePath)
        let fixtures = content.fromJson(seq[CodecFixture])

        for fixture in fixtures:
          let result = tokenizer.encodeOrdinary(fixture.text)
          check result == fixture.tokenIds

when isMainModule:
  runTiktokenFixturesTests()
