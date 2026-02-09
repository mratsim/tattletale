import std/unittest
import std/os

import workspace/toktoktok

const TOKENIZERS_DIR = currentSourcePath().parentDir() / "tokenizers"

proc runHfTokenizerTests*() =
  suite "HF Tokenizer Tests":



    test "load tokenizer file not found":
      expect TokenizerError:
        discard loadHFTokenizer("nonexistent.json")

    const TokenizerPairs = [
      ("gpt2", "gpt2-tokenizer.json"),
      ("llama3", "llama3-tokenizer.json"),
      ("minimax-m2.1", "minimax-m2.1-tokenizer.json"),
      ("glm-4.7", "glm-4.7-tokenizer.json"),
      ("exaone", "exaone-tokenizer.json"),
      ("step-3.5-flash", "step-3.5-flash-tokenizer.json"),
    ]

    for tokenizerPair in TokenizerPairs:
      let (name, filename) = tokenizerPair
      let path = TOKENIZERS_DIR / filename

      test "load and decode (" & name & ")":
        doAssert fileExists(path), name & " tokenizer not found: " & path
        let tokenizer = loadHFTokenizer(path)
        let encoded = tokenizer.encode("Hello, world!")
        check encoded.len > 0

        let decoded = decodeToString(tokenizer, encoded)
        check decoded.len >= 5 and decoded[0..4] == "Hello"

      test "byte encoding roundtrip (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let text = "Hello, world!"
        let encoded = tokenizer.encode(text)
        let decodedStr = decodeToString(tokenizer, encoded)
        check decodedStr == text

      test "CJK roundtrip - Chinese (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "你好世界"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "CJK roundtrip - Japanese (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "こんにちは"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "CJK roundtrip - Korean (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "안녕하세요 세계"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Russian roundtrip (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "Привет мир"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Hebrew roundtrip (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "שלום עולם"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Khmer roundtrip (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "សួស្តីពិភពលោក"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Emoji roundtrip (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "Hello 🌍 World! 🎉"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Mixed CJK and English roundtrip (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "Hello 世界 こんにちは 안녕"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Chinese historical paragraph regression (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "紅。白\n髮漁樵江渚上，慣看秋月春風。一壺濁酒喜相逢：古今多少事，都付笑談中。\n\n　　話說天下大勢，分久必合，合久必分：周末七國分爭，并入於秦。及秦滅之後，楚\n、漢分爭，又并入於漢。漢朝自高祖斬白蛇而起義，一統天下。後來光武中興，傳至獻\n帝遂分為三國。推其致亂之由，殆始於桓、靈二帝。桓帝禁錮善類，崇信宦官。及桓\n帝崩，靈帝即位，大將軍竇武"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

      test "Jules Verne passage with runic characters (" & name & ")":
        let tokenizer = loadHFTokenizer(path)
        let original = "En voici le fac-similé exact.  Je tiens à faire connaître ces signes bizarres, car ils amenèrent le professeur Lidenbrock et son neveu à entreprendre la plus étrange expédition du dix-neuvième siècle:\n\n    ᛯ  . ᛦ ᚳ ᛚ ᛚ ᚼ    ᛅ ᚼ ᛦ ᛅ ᚢ ᛅ ᛚ    ᚼ ᛅ ᛅ ᚴ ᛁ ᚦ ᛅ\n    ᚼ ᛎ ᛏ ᚼ ᚼ ᛘ ᚠ    ᚢ ᚳ ᛏ ᛅ ᛁ ᛅ ᚠ    ᚳ ᛁ ᛅ ᚦ ᛦ ᚴ ᛅ\n    ᚴ ᛏ  , ᚼ ᛐ ᛘ ᚳ    ᛐ ᛏ ᛦ ᛐ ᛏ ᛅ_ᚼ_  _ᚼ_ᛐ ᚭ ᚦ ᛦ ᛦ ᚳ\n    ᛅ ᛘ ᛏ ᚳ ᛐ ᛅ_ᛁ_   ᚳ ᚢ ᛐ ᛅ ᚴ ᛏ       ᛦ ᛦ ᛁ ᛚ_ᚼ_ᛐ\n   _ᛐ_ᛏ ᚢ ᛐ ᛐ ᛦ        . ᚳ ᚼ ᚴ ᛦ ᚴ       ᛁ ᛅ ᛐ ᛐ ᚲ ᚼ\n    ᚴ ᚴ ᚦ ᛦ ᛘ ᛁ       ᛅ ᛅ ᚢ ᛏ ᚢ ᛚ       ᚠ ᛦ ᛐ ᚳ ᛏ ᚢ\n    ᚦ ᛏ  , ᛁ ᛐ ᚴ       ᚭ ᚼ ᛅ ᛁ ᚲ ᚭ      _ᚴ_ᛅ ᚦ ᛁ ᛁ_ᛦ_"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

when isMainModule:
  runHfTokenizerTests()
