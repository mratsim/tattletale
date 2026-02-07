import std/unittest
import std/os

import workspace/toktoktok

const TOKENIZERS_DIR = currentSourcePath().parentDir() / "tokenizers"

proc runToktoktokTests*() =
  suite "BPETokenizer Tests":

    test "decode single byte token":
      let tokenizer = BPETokenizer.init()
      let result = decodeToken(tokenizer, 65)
      check result == "A"

    test "decode unknown token ID raises error":
      let tokenizer = BPETokenizer.init()
      expect TokenizerError:
        discard decodeToken(tokenizer, 256)

    test "decode negative token ID raises error":
      let tokenizer = BPETokenizer.init()
      expect TokenizerError:
        discard decodeToken(tokenizer, -1)

    test "decode to string":
      let tokenizer = BPETokenizer.init()
      let result = decodeToString(tokenizer, @[65, 66, 67])
      check result == "ABC"

    test "token count":
      let tokenizer = BPETokenizer.init()
      check tokenizer.tokenCount == 0

    test "is special token":
      let tokenizer = BPETokenizer.init()
      check not tokenizer.isSpecialToken(0)

    test "load tokenizer file not found":
      expect TokenizerError:
        discard loadTokenizerJson("nonexistent.json")

    test "load and decode GPT-2 tokenizer":
      let gpt2_path = TOKENIZERS_DIR / "gpt2-tokenizer.json"
      if not file_exists(gpt2_path):
        skip()
      else:
        let tokenizer = loadTokenizerJson(gpt2_path)
        check tokenizer.tokenCount > 0

        let encoded = tokenizer.encode("Hello, world!")
        check encoded.len > 0

        let decoded = decodeToString(tokenizer, encoded)
        check decoded.len >= 5 and decoded[0..4] == "Hello"

    test "GPT-2 encode 'Hey there dear friend!'":
      let gpt2_path = TOKENIZERS_DIR / "gpt2-tokenizer.json"
      if not file_exists(gpt2_path):
        skip()
      else:
        let tokenizer = loadTokenizerJson(gpt2_path)
        let encoded = tokenizer.encode("Hey there dear friend!")
        check encoded.len > 0

        let decoded = decodeToString(tokenizer, encoded)
        check decoded.len > 0

    test "load and decode Llama 3 tokenizer":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let tokenizer = loadTokenizerJson(llama3_path)
        check tokenizer.tokenCount > 0

        let encoded = tokenizer.encode("Hello, world!")
        check encoded.len > 0

        let decoded = decodeToString(tokenizer, encoded)
        check decoded.len >= 5 and decoded[0..4] == "Hello"

    test "Llama 3 decode roundtrip":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let tokenizer = loadTokenizerJson(llama3_path)
        let original = "Hello, world!"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded.len > 0

    test "encode special tokens (GPT-2)":
      let gpt2_path = TOKENIZERS_DIR / "gpt2-tokenizer.json"
      if not file_exists(gpt2_path):
        skip()
      else:
        discard loadTokenizerJson(gpt2_path)
        check true

when isMainModule:
  runToktoktokTests()
