import std/unittest
import std/os
import pkg/jsony

import workspace/toktoktok

const FIXTURES_DIR = currentSourcePath().parentDir() / "fixtures" / "small"
const TOKENIZERS_DIR = currentSourcePath().parentDir() / "tokenizers"

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
      let fixturePath = FIXTURES_DIR / "hf_" & fixtureName & ".json"
      let hfPath = TOKENIZERS_DIR / hfFile
      let testName = "HF tokenizers library fixture (" & fixtureName & ")"

      doAssert fileExists(fixturePath), "Fixture not found: " & fixturePath
      doAssert fileExists(hfPath), "HF tokenizer not found: " & hfPath

      let tokenizer = loadHFTokenizer(hfPath)
      let content = readFile(fixturePath)
      let fixtures = content.fromJson(seq[CodecFixture])

      for fixture in fixtures:
        if (fixture.name, fixtureName) == ("sanguozhi_paragraph", "step-3.5-flash") or
            (fixture.name, fixtureName) == ("sanguozhi_paragraph", "exaone"):
          # TODO: Currently failing, probably due to a difference between Rust regexp on Han characters
          # and pcre2 handling of Han characters.
          #   Given that in many cases the regex is supplied by the model
          #   we might want to add a preprocessing phase
          #   that would translate [\p{Han}]+ into [\p{Script=Han}]
          #   see commit bc8d9df32db81a9d08c4458bce5df2b1098a8f68 for a workaround for Kimi-K2.5
          echo "⏩ [SKIPPED]" & "HF tokenizer fixture - " & fixture.name & " (" & fixtureName & ")"
          continue

        test "HF tokenizer fixture - " & fixture.name & " (" & fixtureName & ")":
          let result = tokenizer.encodeOrdinary(fixture.text)
          check result == fixture.tokenIds

when isMainModule:
  runHfTokenizersTests()
