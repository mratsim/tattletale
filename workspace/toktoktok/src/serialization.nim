# HF to Tiktoken Converter
# Converts HuggingFace tokenizer JSON to tiktoken format (raw bytes + pattern + special tokens)
# This module is used internally for loading HF tokenizer files directly.
# For Python integration, use hf_converter.py instead.

import std/tables
import std/os
import std/strutils
import std/options
import std/json
import pkg/jsony

type
  HFSpecialToken* = object
    id*: int
    content*: string

  HFTokenizerModel* = object
    vocab*: OrderedTable[string, int]
    `type`*: string

  HFTokenizer* = object
    model*: HFTokenizerModel
    pre_tokenizer*: Option[OrderedTable[string, string]]
    added_tokens*: Option[seq[HFSpecialToken]]

  TiktokenFormat* = object
    mergeable_ranks*: OrderedTable[string, int]
    pat_str*: string
    special_tokens*: OrderedTable[string, int]

  TokenizerParseError* = object of ValueError

proc parseHook*(src: string, pos: var int, value: var seq[byte]) =
  var strVal: string
  parseHook(src, pos, strVal)
  value = @[]
  if strVal.startsWith("[") and strVal.endsWith("]"):
    let inner = strVal[1..^2]
    for part in inner.split(", "):
      if part.len > 0:
        value.add(byte(parseInt(part)))

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

proc extractPatternFromJson*(hfTokenizer: HFTokenizer): string =
  if hfTokenizer.pre_tokenizer.isSome:
    let pre_tok = hfTokenizer.pre_tokenizer.get
    if pre_tok.hasKey("type"):
      let tokType = pre_tok["type"]
      if tokType == "ByteLevel":
        return "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\r?\\n|\\s+(?!\\S)|\\s+"

  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\r?\\n|\\s+(?!\\S)|\\s+"

proc extractSpecialTokensFromJson*(hfTokenizer: HFTokenizer): OrderedTable[string, int] =
  result = init_ordered_table[string, int]()

  if hfTokenizer.added_tokens.isSome:
    for token in hfTokenizer.added_tokens.get:
      result[token.content] = token.id

proc convertHfToTiktoken*(hfTokenizerPath: string): TiktokenFormat =
  if not file_exists(hf_tokenizer_path):
    raise new_exception(TokenizerParseError, "Tokenizer file not found: " & hf_tokenizer_path)

  let content = read_file(hf_tokenizer_path)
  if content.len == 0:
    raise new_exception(TokenizerParseError, "Tokenizer file is empty: " & hf_tokenizer_path)

  let hfTokenizer = content.fromJson(HFTokenizer)

  if hfTokenizer.model.vocab.len == 0:
    raise new_exception(TokenizerParseError, "Missing 'model.vocab' section in tokenizer")

  let byte_decoder = unicodeToBytes()

  var mergeable_ranks = init_ordered_table[string, int]()

  for token, rank in hfTokenizer.model.vocab:
    mergeable_ranks[token] = rank

  let pat_str = extractPatternFromJson(hfTokenizer)

  let special_tokens = extractSpecialTokensFromJson(hfTokenizer)

  TiktokenFormat(
    mergeable_ranks: mergeable_ranks,
    pat_str: pat_str,
    special_tokens: special_tokens
  )

proc convertHfToTiktokenJson*(hfTokenizerPath: string): string =
  let format = convertHfToTiktoken(hfTokenizerPath)
  format.toJson()

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
