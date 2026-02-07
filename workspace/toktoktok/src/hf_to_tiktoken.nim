# HF to Tiktoken Converter
# Converts HuggingFace tokenizer JSON to tiktoken format (raw bytes + pattern + special tokens)
# This module is used internally for loading HF tokenizer files directly.
# For Python integration, use hf_converter.py instead.

import std/tables
import std/json
import std/os
import std/strutils

type
  TiktokenFormat* = object
    mergeable_ranks*: Table[seq[byte], int]
    pat_str*: string
    special_tokens*: Table[string, int]

  TokenizerParseError* = object of ValueError

proc bytesToUnicode*(): Table[int, string] =
  var bs: seq[int] = @[]
  for i in 33..126:
    bs.add(i)
  for i in 161..172:
    bs.add(i)
  for i in 174..255:
    bs.add(i)

  var cs: seq[string] = @[]
  for b in bs:
    cs.add($chr(b))

  var n = 0
  for b in 0..255:
    var found = false
    for existing in bs:
      if existing == b:
        found = true
        break
    if not found:
      bs.add(b)
      if n < 256:
        cs.add($chr(256 + n))
      else:
        cs.add($chr(b))
      n += 1

  result = init_table[int, string]()
  for i in 0..<bs.len:
    let byte_val = bs[i]
    let char_str = cs[i]
    result[byte_val] = char_str

proc unicodeToBytes*(): Table[string, int] =
  var byte_table = bytesToUnicode()
  result = init_table[string, int]()
  for k, v in byte_table:
    let char_str = v
    if char_str.len == 1:
      let char_code = ord(char_str[0])
      if char_code < 256:
        result[char_str] = char_code
      else:
        result[char_str] = k
    else:
      result[char_str] = k

proc hfTokenToRawBytes*(token: string, byteDecoder: Table[string, int]): seq[byte] =
  var raw_bytes: seq[byte] = @[]
  for c in token:
    let char_str = $c
    if byteDecoder.hasKey(char_str):
      raw_bytes.add(byte(byte_decoder[char_str]))
    else:
      raw_bytes.add(byte(ord(c)))
  raw_bytes

proc extractPatternFromJson*(jsonNode: JsonNode): string =
  if jsonNode.hasKey("pre_tokenizer"):
    let pre_tok = jsonNode["pre_tokenizer"]
    if pre_tok.hasKey("type"):
      if pre_tok["type"].getStr == "ByteLevel":
        return "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\r?\\n|\\s+(?!\\S)|\\s+"
      elif pre_tok["type"].getStr == "Sequence":
        if pre_tok.hasKey("pretokenizers"):
          for item in pre_tok["pretokenizers"]:
            if item.hasKey("type") and item["type"].getStr == "Split":
              if item.hasKey("pattern"):
                let pattern_info = item["pattern"]
                if pattern_info.hasKey("Regex"):
                  return pattern_info["Regex"].getStr

  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\r?\\n|\\s+(?!\\S)|\\s+"

proc extractSpecialTokensFromJson*(jsonNode: JsonNode): Table[string, int] =
  result = init_table[string, int]()

  if jsonNode.hasKey("added_tokens"):
    let added_tokens = jsonNode["added_tokens"]
    for token in added_tokens:
      if token.hasKey("id") and token.hasKey("content"):
        let id = int(token["id"].getInt)
        let content_str = token["content"].getStr
        result[content_str] = id

proc convertHfToTiktokenJson*(hfTokenizerPath: string): JsonNode =
  if not file_exists(hf_tokenizer_path):
    raise new_exception(TokenizerParseError, "Tokenizer file not found: " & hf_tokenizer_path)

  let content = read_file(hf_tokenizer_path)
  if content.len == 0:
    raise new_exception(TokenizerParseError, "Tokenizer file is empty: " & hf_tokenizer_path)

  let jsonNode = content.parseJson()

  if not jsonNode.hasKey("model"):
    raise new_exception(TokenizerParseError, "Missing 'model' section in tokenizer")

  let model = jsonNode["model"]
  let byte_decoder = unicodeToBytes()

  var mergeable_ranks = init_table[seq[byte], int]()

  if model.hasKey("vocab"):
    let vocabNode = model["vocab"]
    if vocabNode.kind == JObject:
      for key, value in vocabNode:
        if value.kind == JInt:
          let id = int(value.getInt)
          let raw_bytes = hfTokenToRawBytes(key, byteDecoder)
          mergeable_ranks[raw_bytes] = id
    elif vocabNode.kind == JArray:
      for item in vocabNode:
        if item.kind == JArray:
          let id = int(item[1].getInt)
          let token_str = item[0].getStr
          let raw_bytes = hfTokenToRawBytes(token_str, byteDecoder)
          mergeable_ranks[raw_bytes] = id

  let pat_str = extractPatternFromJson(jsonNode)

  let special_tokens = extractSpecialTokensFromJson(jsonNode)

  var result_json = newJObject()
  var ranks_json = newJObject()
  for bytes, rank in mergeable_ranks:
    var bytes_list = newJArray()
    for b in bytes:
      bytes_list.add(%b)
    ranks_json[$bytes_list] = %rank
  result_json["mergeable_ranks"] = ranks_json
  result_json["pat_str"] = %pat_str
  var special_json = newJObject()
  for token, id in special_tokens:
    special_json[token] = %id
  result_json["special_tokens"] = special_json

  result = result_json

proc saveTiktokenFormat*(tiktokenFormat: TiktokenFormat, outputPath: string) =
  var lines: seq[string] = @[]
  lines.add("{")

  lines.add("  \"mergeable_ranks\": {")
  var first = true
  for bytes, rank in tiktoken_format.mergeable_ranks:
    if not first:
      lines.add(",")
    else:
      first = false
    var bytes_str = ""
    for i, b in bytes:
      if i > 0: bytes_str.add(", ")
      bytes_str.add($b)
    lines.add("    [" & bytes_str & "]: " & $rank)
  lines.add("  },")

  lines.add("  \"pat_str\": \"" & tiktoken_format.pat_str.escapeJson() & "\",")
  lines.add("  \"special_tokens\": {")
  first = true
  for token, id in tiktoken_format.special_tokens:
    if not first:
      lines.add(",")
    else:
      first = false
    lines.add("    \"" & token.escapeJson() & "\": " & $id)
  lines.add("  }")

  lines.add("}")

  write_file(output_path, lines.join("\n"))

proc `$`*(bytes: seq[byte]): string =
  result = "["
  for i, b in bytes:
    if i > 0: result.add(", ")
    result.add($b)
  result.add("]")

proc `$`*(tiktoken_format: TiktokenFormat): string =
  result = "TiktokenFormat(\n"
  result.add("  mergeable_ranks: " & $tiktoken_format.mergeable_ranks.len & " entries,\n")
  result.add("  pat_str: \"" & tiktoken_format.pat_str & "\",\n")
  result.add("  special_tokens: " & $tiktoken_format.special_tokens.len & " entries\n")
  result.add(")")
