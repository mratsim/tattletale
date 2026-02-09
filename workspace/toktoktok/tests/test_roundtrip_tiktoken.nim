import std/unittest
import std/os

import workspace/toktoktok

const TOKENIZERS_DIR = currentSourcePath().parentDir() / "tokenizers"

proc runTiktokenizerTests*() =
  suite "Tiktokenizer Tests":

    test "load tiktokenizer file not found":
      expect TokenizerError:
        discard loadTiktokenizer("nonexistent.tiktoken", Gpt2Regexp)

    const TiktokenPairs = [
      ("r50k_base", "r50k_base.tiktoken", R50kRegexp),
      ("p50k_base", "p50k_base.tiktoken", P50kRegexp),
      ("cl100k_base", "cl100k_base.tiktoken", Cl100kRegexp),
      ("o200k_base", "o200k_base.tiktoken", O200kRegexp),
      ("kimik2.5", "kimik2.5.tiktoken", KimiK25Regexp),
    ]

    for tokenizerPair in TiktokenPairs:
      let (name, filename, regexp) = tokenizerPair
      let path = TOKENIZERS_DIR / filename

      test "load and decode (" & name & ")":
        doAssert fileExists(path), name & " tiktokenizer not found: " & path
        let tokenizer = loadTiktokenizer(path, regexp)
        let encoded = tokenizer.encode("Hello, world!")
        check encoded.len > 0

        let decoded = decodeToString(tokenizer, encoded)
        check decoded.len >= 5 and decoded[0..4] == "Hello"

      test "byte encoding roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path, regexp)
        let text = "Hello, world!"
        let encoded = tokenizer.encode(text)
        let decodedStr = decodeToString(tokenizer, encoded)
        check decodedStr == text

      test "CJK roundtrip - Chinese (" & name & ")":
        let tokenizer = loadTiktokenizer(path, regexp)
        let original = "你好世界"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "CJK roundtrip - Japanese (" & name & ")":
        let tokenizer = loadTiktokenizer(path, regexp)
        let original = "こんにちは"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "CJK roundtrip - Korean (" & name & ")":
        let tokenizer = loadTiktokenizer(path, regexp)
        let original = "안녕하세요 세계"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Russian roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path, regexp)
        let original = "Привет мир"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Hebrew roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path, regexp)
        let original = "שלום עולם"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Khmer roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path, regexp)
        let original = "សួស្តីពិភពលោក"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Emoji roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path, regexp)
        let original = "Hello 🌍 World! 🎉"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Mixed CJK and English roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path, regexp)
        let original = "Hello 世界 こんにちは 안녕"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

when isMainModule:
  runTiktokenizerTests()
