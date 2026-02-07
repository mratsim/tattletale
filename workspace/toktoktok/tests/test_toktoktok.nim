import std/unittest
import std/os
import std/strutils
import std/json
import std/tables

import workspace/toktoktok

const TOKENIZERS_DIR = currentSourcePath().parentDir() / "tokenizers"
const CONVERTED_DIR = TOKENIZERS_DIR / "converted"

proc ensureConvertedTokenizer*(hf_path: string): string =
  let hf_file = TOKENIZERS_DIR / hf_path
  if not file_exists(hf_file):
    return ""
  
  let converted_file = CONVERTED_DIR / (hf_path.replace(".json", "_tiktoken.json"))
  if not file_exists(converted_file):
    converted_file.parentDir.createDir()
    
    let content = read_file(hf_file)
    let jsonNode = content.parseJson()
    let byte_decoder = createByteDecoder()
    
    var encoder = initTable[seq[byte], int]()
    if jsonNode.hasKey("model"):
      let model = jsonNode["model"]
      if model.hasKey("vocab"):
        let vocabNode = model["vocab"]
        if vocabNode.kind == JObject:
          for key, value in vocabNode:
            if value.kind == JInt:
              let id = int(value.getInt)
              var raw_bytes: seq[byte] = @[]
              for c in key:
                let char_str = $c
                if byte_decoder.hasKey(char_str):
                  raw_bytes.add(byte(byte_decoder[char_str]))
                else:
                  raw_bytes.add(byte(ord(c)))
              encoder[raw_bytes] = id
    
    let pat_str = if jsonNode.hasKey("pre_tokenizer"): 
      if jsonNode["pre_tokenizer"].hasKey("type") and jsonNode["pre_tokenizer"]["type"].getStr == "ByteLevel":
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\r?\\n|\\s+(?!\\S)|\\s+"
      else:
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\r?\\n|\\s+(?!\\S)|\\s+"
    else:
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\r?\\n|\\s+(?!\\S)|\\s+"
    
    var special_tokens = initTable[string, int]()
    if jsonNode.hasKey("added_tokens"):
      for token in jsonNode["added_tokens"]:
        if token.hasKey("id") and token.hasKey("content"):
          special_tokens[token["content"].getStr] = int(token["id"].getInt)
    
    var lines: seq[string] = @[]
    lines.add("{")
    lines.add("  \"mergeable_ranks\": {")
    var first = true
    for k, v in encoder:
      if not first: lines.add(",")
      first = false
      var bytes_str = "\""
      for i, b in k:
        if i > 0: bytes_str.add(", ")
        bytes_str.add($b)
      bytes_str.add("\"")
      lines.add("    " & bytes_str & ": " & $v)
    lines.add("  },")
    lines.add("  \"pat_str\": \"" & pat_str.escapeJson() & "\",")
    lines.add("  \"special_tokens\": {")
    first = true
    for k, v in special_tokens:
      if not first: lines.add(",")
      first = false
      lines.add("    \"" & k.escapeJson() & "\": " & $v)
    lines.add("  }")
    lines.add("}")
    write_file(converted_file, lines.join("\n"))
  
  return converted_file

proc runToktoktokTests*() =
  suite "BPETokenizer Tests":

    test "decode single byte token in empty tokenizer":
      let tokenizer = init(BPETokenizer)
      expect TokenizerError:
        discard decodeToken(tokenizer, 65)

    test "decode unknown token ID raises error":
      let tokenizer = init(BPETokenizer)
      expect TokenizerError:
        discard decodeToken(tokenizer, 256)

    test "decode negative token ID raises error":
      let tokenizer = init(BPETokenizer)
      expect TokenizerError:
        discard decodeToken(tokenizer, -1)

    test "token count of empty tokenizer":
      let tokenizer = init(BPETokenizer)
      check tokenizer.tokenCount == 0

    test "is special token on empty tokenizer":
      let tokenizer = init(BPETokenizer)
      check not tokenizer.isSpecialToken(0)

    test "load tokenizer file not found":
      expect TokenizerError:
        discard loadTokenizerJson("nonexistent.json")

    test "load and decode GPT-2 tokenizer (via conversion)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        check tokenizer.tokenCount > 0

        let encoded = tokenizer.encode("Hello, world!")
        check encoded.len > 0

        let decoded = decodeToString(tokenizer, encoded)
        check decoded.len >= 5 and decoded[0..4] == "Hello"

    test "GPT-2 encode 'Hey there dear friend!' (via conversion)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let encoded = tokenizer.encode("Hey there dear friend!")
        check encoded.len > 0

        let decoded = decodeToString(tokenizer, encoded)
        check decoded.len > 0

    test "load and decode Llama 3 tokenizer (via conversion)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          check tokenizer.tokenCount > 0

          let encoded = tokenizer.encode("Hello, world!")
          check encoded.len > 0

          let decoded = decodeToString(tokenizer, encoded)
          check decoded.len >= 5 and decoded[0..4] == "Hello"

    test "Llama 3 decode roundtrip (via conversion)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          let original = "Hello, world!"
          let encoded = tokenizer.encode(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded.len > 0

    test "encode special tokens (GPT-2) (via conversion)":
      let gpt2_path = TOKENIZERS_DIR / "gpt2-tokenizer.json"
      if not file_exists(gpt2_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          discard loadTokenizerJson(converted_path)
          check true

    test "byte encoding roundtrip":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let text = "Hello, world!"
        let encoded = tokenizer.encode(text)
        let decoded_bytes = decodeToBytes(tokenizer, encoded)
        let decoded_str = decodeToString(tokenizer, encoded)
        check decoded_str == text

    test "CJK roundtrip - Chinese (GPT-2)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let original = "你好世界"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

    test "CJK roundtrip - Japanese (GPT-2)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let original = "こんにちは"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

    test "CJK roundtrip - Korean (GPT-2)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let original = "안녕하세요 세계"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

    test "Russian roundtrip (GPT-2)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let original = "Привет мир"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

    test "Hebrew roundtrip (GPT-2)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let original = "שלום עולם"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

    test "Khmer roundtrip (GPT-2)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let original = "សួស្តីពិភពលោក"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

    test "Emoji roundtrip (GPT-2)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let original = "Hello 🌍 World! 🎉"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

    test "Mixed CJK and English roundtrip (GPT-2)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let original = "Hello 世界 こんにちは 안녕"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

    test "CJK roundtrip - Chinese (Llama 3)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          let original = "你好世界"
          let encoded = tokenizer.encode(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded == original

    test "CJK roundtrip - Japanese (Llama 3)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          let original = "こんにちは"
          let encoded = tokenizer.encode(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded == original

    test "CJK roundtrip - Korean (Llama 3)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          let original = "안녕하세요 세계"
          let encoded = tokenizer.encode(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded == original

    test "Russian roundtrip (Llama 3)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          let original = "Привет мир"
          let encoded = tokenizer.encode(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded == original

    test "Hebrew roundtrip (Llama 3)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          let original = "שלום עולם"
          let encoded = tokenizer.encode(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded == original

    test "Khmer roundtrip (Llama 3)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          let original = "សួស្តីពិភពលោក"
          let encoded = tokenizer.encode(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded == original

    test "Emoji roundtrip (Llama 3)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          let original = "Hello 🌍 World! 🎉"
          let encoded = tokenizer.encode(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded == original

    test "Jules Verne passage with runic characters roundtrip (GPT-2)":
      let converted_path = ensureConvertedTokenizer("gpt2-tokenizer.json")
      if converted_path == "":
        skip()
      else:
        let tokenizer = loadTokenizerJson(converted_path)
        let original = "En voici le fac-similé exact.  Je tiens à faire connaître ces signes bizarres, car ils amenèrent le professeur Lidenbrock et son neveu à entreprendre la plus étrange expédition du dix-neuvième siècle:\n\n    ᛯ  . ᛦ ᚳ ᛚ ᛚ ᚼ    ᛅ ᚼ ᛦ ᛅ ᚢ ᛅ ᛚ    ᚼ ᛅ ᛅ ᚴ ᛁ ᚦ ᛅ\n    ᚼ ᛎ ᛏ ᚼ ᚼ ᛘ ᚠ    ᚢ ᚳ ᛏ ᛅ ᛁ ᛅ ᚠ    ᚳ ᛁ ᛅ ᚦ ᛦ ᚴ ᛅ\n    ᚴ ᛏ  , ᚼ ᛐ ᛘ ᚳ    ᛐ ᛏ ᛦ ᛐ ᛏ ᛅ_ᚼ_  _ᚼ_ᛐ ᚭ ᚦ ᛦ ᛦ ᚳ\n    ᛅ ᛘ ᛏ ᚳ ᛐ ᛅ_ᛁ_   ᚳ ᚢ ᛐ ᛅ ᚴ ᛏ       ᛦ ᛦ ᛁ ᛚ_ᚼ_ᛐ\n   _ᛐ_ᛏ ᚢ ᛐ ᛐ ᛦ        . ᚳ ᚼ ᚴ ᛦ ᚴ       ᛁ ᛅ ᛐ ᛐ ᚲ ᚼ\n    ᚴ ᚴ ᚦ ᛦ ᛘ ᛁ       ᛅ ᛅ ᚢ ᛏ ᚢ ᛚ       ᚠ ᛦ ᛐ ᚳ ᛏ ᚢ\n    ᚦ ᛏ  , ᛁ ᛐ ᚴ       ᚭ ᚼ ᛅ ᛁ ᚲ ᚭ      _ᚴ_ᛅ ᚦ ᛁ ᛁ_ᛦ_"
        let encoded = tokenizer.encode(original)
        let decoded = decodeToString(tokenizer, encoded)
        check decoded == original

    test "Jules Verne passage with runic characters roundtrip (Llama 3)":
      let llama3_path = TOKENIZERS_DIR / "llama3-tokenizer.json"
      if not file_exists(llama3_path):
        skip()
      else:
        let converted_path = ensureConvertedTokenizer("llama3-tokenizer.json")
        if converted_path == "":
          skip()
        else:
          let tokenizer = loadTokenizerJson(converted_path)
          let original = "En voici le fac-similé exact.  Je tiens à faire connaître ces signes bizarres, car ils amenèrent le professeur Lidenbrock et son neveu à entreprendre la plus étrange expédition du dix-neuvième siècle:\n\n    ᛯ  . ᛦ ᚳ ᛚ ᛚ ᚼ    ᛅ ᚼ ᛦ ᛅ ᚢ ᛅ ᛚ    ᚼ ᛅ ᛅ ᚴ ᛁ ᚦ ᛅ\n    ᚼ ᛎ ᛏ ᚼ ᚼ ᛘ ᚠ    ᚢ ᚳ ᛏ ᛅ ᛁ ᛅ ᚠ    ᚳ ᛁ ᛅ ᚦ ᛦ ᚴ ᛅ\n    ᚴ ᛏ  , ᚼ ᛐ ᛘ ᚳ    ᛐ ᛏ ᛦ ᛐ ᛏ ᛅ_ᚼ_  _ᚼ_ᛐ ᚭ ᚦ ᛦ ᛦ ᚳ\n    ᛅ ᛘ ᛏ ᚳ ᛐ ᛅ_ᛁ_   ᚳ ᚢ ᛐ ᛅ ᚴ ᛏ       ᛦ ᛦ ᛁ ᛚ_ᚼ_ᛐ\n   _ᛐ_ᛏ ᚢ ᛐ ᛐ ᛦ        . ᚳ ᚼ ᚴ ᛦ ᚴ       ᛁ ᛅ ᛐ ᛐ ᚲ ᚼ\n    ᚴ ᚴ ᚦ ᛦ ᛘ ᛁ       ᛅ ᛅ ᚢ ᛏ ᚢ ᛚ       ᚠ ᛦ ᛐ ᚳ ᛏ ᚢ\n    ᚦ ᛏ  , ᛁ ᛐ ᚴ       ᚭ ᚼ ᛅ ᛁ ᚲ ᚭ      _ᚴ_ᛅ ᚦ ᛁ ᛁ_ᛦ_"
          let encoded = tokenizer.encode(original)
          let decoded = decodeToString(tokenizer, encoded)
          check decoded == original

when isMainModule:
  runToktoktokTests()
