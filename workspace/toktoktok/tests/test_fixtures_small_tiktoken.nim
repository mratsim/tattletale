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

        test "Chinese historical paragraph issue merging '。\\n' (" & fixtureName & ")":
          let original = "紅。白\n髮漁樵江渚上，慣看秋月春風。一壺濁酒喜相逢：古今多少事，都付笑談中。\n\n　　話說天下大勢，分久必合，合久必分：周末七國分爭，并入於秦。及秦滅之後，楚\n、漢分爭，又并入於漢。漢朝自高祖斬白蛇而起義，一統天下。後來光武中興，傳至獻\n帝，遂分為三國。推其致亂之由，殆始於桓、靈二帝。桓帝禁錮善類，崇信宦官。及桓\n帝崩，靈帝即位，大將軍竇武、太傅陳蕃，共相輔佐。時"
          let encoded = tokenizer.encodeOrdinary(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded == original

when isMainModule:
  runTiktokenFixturesTests()
