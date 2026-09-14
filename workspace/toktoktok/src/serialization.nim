# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/base64
import std/strutils
## Checkpoint serialization for the PCRE2-era codec path, HF json records, tiktoken rank tables, and the byte decoder.

import std/options
import std/tables
import std/unicode
import pkg/jsony
import ./tokenizers_regexps

type
  ## HF json Split-step pattern record, the json spells the key "Regex".
  HFtokRegexp* = object
    Regex*: string

  ## Deserialized checkpoint (mergeable ranks plus pattern plus special tokens).
  TiktokenFormat* = object
    mergeableRanks*: OrderedTable[seq[byte], int]
    pattern*: TokRegexp
    specialTokens*: OrderedTable[string, int]

  ## Raised for malformed checkpoint content on the parse path.
  TokenizerParseError* = object of ValueError

  ## Deserialized HF tokenizer.json root record (jsony target).
  HFTokenizer* = object
    version*: string
    truncation*: string
    padding*: string
    addedTokens*: seq[HFSpecialToken]
    preTokenizer*: HFPreTokenizer
    postProcessor*: HFPostProcessor
    decoder*: HFDecoder
    model*: HFTokenizerModel

  ## Deserialized HF model section (jsony target).
  HFTokenizerModel* = object
    vocab*: OrderedTable[string, int]
    dropout*: string
    unkToken*: string
    continuingSubwordPrefix*: string
    endOfWordSuffix*: string
    fuseUnk*: bool
    `type`*: string
    pattern*: TokRegexp

  ## Deserialized HF pre_tokenizer section (jsony target).
  HFPreTokenizer* = object
    addPrefixSpace*: bool
    trimOffsets*: bool
    `type`*: string
    pretokenizers*: seq[HFPretokenizerStep]
    useRegex*: Option[bool]

  ## One HF pre_tokenizer chain step (jsony target).
  HFPretokenizerStep* = object
    `type`*: string
    pattern*: HFtokRegexp
    behavior*: string
    invert*: bool
    addPrefixSpace*: bool
    trimOffsets*: bool
    useRegex*: Option[bool]

  ## Deserialized HF post_processor section (jsony target).
  HFPostProcessor* = object
    addPrefixSpace*: bool
    trimOffsets*: bool
    `type`*: string

  ## Deserialized HF decoder section (jsony target).
  HFDecoder* = object
    addPrefixSpace*: bool
    trimOffsets*: bool
    `type`*: string

  ## One HF added-token special entry (jsony target).
  HFSpecialToken* = object
    content*: string
    id*: int
    lstrip*: bool
    normalized*: bool
    rstrip*: bool
    singleWord*: bool
    special*: bool

template toBytes*(str: string): seq[byte] =
  ## String bytes as a seq[byte] copy.
  @(toOpenArrayByte(str, 0, str.len - 1))

proc initByteDecoder*(): Table[uint32, int] =
  ## GPT-2 bytes-to-unicode inverse map, codepoint to original byte
  ## value (printable bytes map to themselves, the rest shift to 256+n).
  result = initTable[uint32, int]()
  var bs = newSeq[int]()
  var cs = newSeq[int]()

  for b in ord('!')..ord('~'):
    bs.add(b)
    cs.add(b)
  for b in 0x00A1..0x00AC:
    bs.add(b)
    cs.add(b)
  for b in 0x00AE..0x00FF:
    bs.add(b)
    cs.add(b)

  var n = 0
  for b in 0..<256:
    var found = false
    for x in bs:
      if x == b:
        found = true
        break
    if not found:
      bs.add(b)
      cs.add(256 + n)
      n += 1

  for i in 0..<cs.len:
    result[uint32(cs[i])] = bs[i]

proc renameHook*(v: var HFTokenizer, key: var string) =
  ## jsony rename hook, snake_case json keys map to the camelCase fields.
  if key == "added_tokens":
    key = "addedTokens"
  elif key == "pre_tokenizer":
    key = "preTokenizer"
  elif key == "post_processor":
    key = "postProcessor"

proc renameHook*(v: var HFTokenizerModel, key: var string) =
  ## jsony rename hook, snake_case json keys map to the camelCase fields.
  if key == "unk_token":
    key = "unkToken"
  elif key == "continuing_subword_prefix":
    key = "continuingSubwordPrefix"
  elif key == "end_of_word_suffix":
    key = "endOfWordSuffix"
  elif key == "fuse_unk":
    key = "fuseUnk"

proc renameHook*(v: var HFPreTokenizer, key: var string) =
  ## jsony rename hook, snake_case json keys map to the camelCase fields.
  if key == "add_prefix_space":
    key = "addPrefixSpace"
  elif key == "trim_offsets":
    key = "trimOffsets"

proc renameHook*(v: var HFPostProcessor, key: var string) =
  ## jsony rename hook, snake_case json keys map to the camelCase fields.
  if key == "add_prefix_space":
    key = "addPrefixSpace"
  elif key == "trim_offsets":
    key = "trimOffsets"

proc renameHook*(v: var HFDecoder, key: var string) =
  ## jsony rename hook, snake_case json keys map to the camelCase fields.
  if key == "add_prefix_space":
    key = "addPrefixSpace"
  elif key == "trim_offsets":
    key = "trimOffsets"

proc renameHook*(v: var HFSpecialToken, key: var string) =
  ## jsony rename hook, snake_case json keys map to the camelCase fields.
  if key == "single_word":
    key = "singleWord"
  elif key == "special_tokens":
    key = "specialTokens"

proc renameHook*(v: var HFtokRegexp, key: var string) =
  ## jsony rename hook, snake_case json keys map to the camelCase fields.
  if key == "Regex":
    key = "Regex"

proc deserializeHfTokenizer*(jsonContent: string): HFTokenizer =
  ## Parses one HF tokenizer.json (jsony into the HF record types).
  jsonContent.fromJson(HFTokenizer)

proc deserializeTiktokenizer*(content: string, regexp = R50kRegexp): TiktokenFormat =
  ## Expected input:
  ## - one "base64 rank" pair per line, '#' comment lines and empties skipped.
  ## Output:
  ## - the parsed TiktokenFormat with the given pattern, specialTokens stays empty.
  let lines = content.splitLines()
  var mergeableRanks = initOrderedTable[seq[byte], int]()

  for line in lines:
    if line.len == 0:
      continue
    if line.startsWith("#"):
      continue

    let parts = line.split(" ")
    if parts.len < 2:
      raise newException(TokenizerParseError, "Invalid tiktoken line: " & line)

    let encodedToken = parts[0]
    let rankStr = parts[1]
    let rank = parseInt(rankStr)
    let decodedTokenStr = decode(encodedToken)
    let decodedTokenBytes = toBytes(decodedTokenStr)
    mergeableRanks[decodedTokenBytes] = rank

  TiktokenFormat(
    mergeableRanks: mergeableRanks,
    pattern: regexp,
    specialTokens: initOrderedTable[string, int]()
  )

proc convertHfToTiktoken*(hf: HFTokenizer): TiktokenFormat =
  ## Converts one deserialized HF checkpoint into tiktoken shape,
  ## conversion contract:
  ## - vocab keys walk back to raw bytes through the byte decoder, keys
  ##   with non-decodable codepoints are dropped.
  ## - the pattern field carries the model-level Regex, else the Split
  ##   steps of the pre_tokenizer chain joined with '|', else the GPT-2
  ##   default pattern for ByteLevel pre-tokenizers.
  ## - a checkpoint with none of those is a load error.

  var pattern: TokRegexp

  if hf.model.pattern.regexp.len > 0:
    pattern = hf.model.pattern
  else:
    var splitPatterns: seq[string] = @[]
    if hf.preTokenizer.pretokenizers.len > 0:
      for step in hf.preTokenizer.pretokenizers:
        if step.type == "Split" and step.pattern.Regex.len > 0:
          splitPatterns.add(step.pattern.Regex)

    if splitPatterns.len > 0:
      pattern = TokRegexp(regexp: splitPatterns.join("|"))
    elif hf.preTokenizer.type == "ByteLevel":
      let useByteLevelDefault = hf.preTokenizer.useRegex.get(true)
      if useByteLevelDefault:
        pattern = Gpt2Regexp
      else:
        raise newException(ValueError, "Error: the HuggingFace tokenizer JSON file is missing regexp information.")
    elif hf.preTokenizer.pretokenizers.len > 0:
      var useByteLevelDefault = false
      for step in hf.preTokenizer.pretokenizers:
        if step.type == "ByteLevel":
          useByteLevelDefault = step.useRegex.get(true)
          break

      if useByteLevelDefault:
        pattern = Gpt2Regexp
      else:
        raise newException(ValueError, "Error: the HuggingFace tokenizer JSON file is missing regexp information.")
    else:
      raise newException(ValueError, "Error: the HuggingFace tokenizer JSON file is missing regexp information.")

  var mergeableRanks = initOrderedTable[seq[byte], int]()

  if hf.model.vocab.len > 0:
    let byteDecoder = initByteDecoder()
    var convertedCount = 0
    var failedCount = 0
    for key, rank in hf.model.vocab:
      var bytesSeq: seq[byte] = @[]
      let keyRunes = toRunes(key)
      for c in keyRunes:
        let runeVal = uint32(c)
        let byteVal = byteDecoder.getOrDefault(runeVal, -1)
        if byteVal >= 0:
          bytesSeq.add(byte(byteVal))
      if bytesSeq.len == keyRunes.len:
        mergeableRanks[bytesSeq] = rank
        inc convertedCount
      else:
        inc failedCount

  let byteRankStart = 1000000  # High rank for byte tokens
  for i in 0..<256:
    let byteSeq = @[byte(i)]
    if not mergeableRanks.hasKey(byteSeq):
      mergeableRanks[byteSeq] = byteRankStart + i

  var specialTokens = initOrderedTable[string, int]()
  for token in hf.addedTokens:
    specialTokens[token.content] = token.id

  TiktokenFormat(
    mergeableRanks: mergeableRanks,
    pattern: pattern,
    specialTokens: specialTokens
  )
