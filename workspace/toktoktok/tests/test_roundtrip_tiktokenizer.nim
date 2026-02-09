import std/unittest
import std/os

import workspace/toktoktok

const TOKENIZERS_DIR = currentSourcePath().parentDir() / "tokenizers"

proc runTiktokenizerTests*() =
  suite "Tiktokenizer Tests":

    test "load tiktokenizer file not found":
      expect TokenizerError:
        discard loadTiktokenizer("nonexistent.tiktoken")

    const TiktokenPairs = [
      ("r50k_base", "r50k_base.tiktoken"),
      ("p50k_base", "p50k_base.tiktoken"),
      ("cl100k_base", "cl100k_base.tiktoken"),
      ("o200k_base", "o200k_base.tiktoken")
    ]

    for tokenizerPair in TiktokenPairs:
      let (name, filename) = tokenizerPair
      let path = TOKENIZERS_DIR / filename

      test "load and decode (" & name & ")":
        doAssert fileExists(path), name & " tiktokenizer not found: " & path
        let tokenizer = loadTiktokenizer(path)
        let encoded = tokenizer.encode("Hello, world!")
        check encoded.len > 0

        let decoded = decodeToString(tokenizer, encoded)
        check decoded.len >= 5 and decoded[0..4] == "Hello"

      test "byte encoding roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path)
        let text = "Hello, world!"
        let encoded = tokenizer.encode(text)
        let decodedStr = decodeToString(tokenizer, encoded)
        check decodedStr == text

      test "CJK roundtrip - Chinese (" & name & ")":
        let tokenizer = loadTiktokenizer(path)
        let original = "你好世界"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "CJK roundtrip - Japanese (" & name & ")":
        let tokenizer = loadTiktokenizer(path)
        let original = "こんにちは"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "CJK roundtrip - Korean (" & name & ")":
        let tokenizer = loadTiktokenizer(path)
        let original = "안녕하세요 세계"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Russian roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path)
        let original = "Привет мир"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Hebrew roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path)
        let original = "שלום עולם"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Khmer roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path)
        let original = "សួស្តីពិភពលោក"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Emoji roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path)
        let original = "Hello 🌍 World! 🎉"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Mixed CJK and English roundtrip (" & name & ")":
        let tokenizer = loadTiktokenizer(path)
        let original = "Hello 世界 こんにちは 안녕"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

when isMainModule:
  runTiktokenizerTests()
