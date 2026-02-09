# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/base64
import std/strutils
import std/options
import std/tables
import std/unicode
import pkg/jsony
import ./tokenizers_regexps

type
  HFtokRegexp* = object
    Regex*: string

  TiktokenFormat* = object
    mergeableRanks*: OrderedTable[seq[byte], int]
    pattern*: TokRegexp
    specialTokens*: OrderedTable[string, int]

  TokenizerParseError* = object of ValueError

  HFTokenizer* = object
    version*: string
    truncation*: string
    padding*: string
    addedTokens*: seq[HFSpecialToken]
    preTokenizer*: HFPreTokenizer
    postProcessor*: HFPostProcessor
    decoder*: HFDecoder
    model*: HFTokenizerModel

  HFTokenizerModel* = object
    vocab*: OrderedTable[string, int]
    dropout*: string
    unkToken*: string
    continuingSubwordPrefix*: string
    endOfWordSuffix*: string
    fuseUnk*: bool
    `type`*: string
    pattern*: TokRegexp

  HFPreTokenizer* = object
    addPrefixSpace*: bool
    trimOffsets*: bool
    `type`*: string
    pretokenizers*: seq[HFPretokenizerStep]
    useRegex*: Option[bool]

  HFPretokenizerStep* = object
    `type`*: string
    pattern*: HFtokRegexp
    behavior*: string
    invert*: bool
    addPrefixSpace*: bool
    trimOffsets*: bool
    useRegex*: Option[bool]

  HFPostProcessor* = object
    addPrefixSpace*: bool
    trimOffsets*: bool
    `type`*: string

  HFDecoder* = object
    addPrefixSpace*: bool
    trimOffsets*: bool
    `type`*: string

  HFSpecialToken* = object
    content*: string
    id*: int
    lstrip*: bool
    normalized*: bool
    rstrip*: bool
    singleWord*: bool
    special*: bool

template toBytes*(str: string): seq[byte] =
  @(toOpenArrayByte(str, 0, str.len - 1))

proc initByteDecoder*(): Table[uint32, int] =
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
  if key == "added_tokens":
    key = "addedTokens"
  elif key == "pre_tokenizer":
    key = "preTokenizer"
  elif key == "post_processor":
    key = "postProcessor"

proc renameHook*(v: var HFTokenizerModel, key: var string) =
  if key == "unk_token":
    key = "unkToken"
  elif key == "continuing_subword_prefix":
    key = "continuingSubwordPrefix"
  elif key == "end_of_word_suffix":
    key = "endOfWordSuffix"
  elif key == "fuse_unk":
    key = "fuseUnk"

proc renameHook*(v: var HFPreTokenizer, key: var string) =
  if key == "add_prefix_space":
    key = "addPrefixSpace"
  elif key == "trim_offsets":
    key = "trimOffsets"

proc renameHook*(v: var HFPostProcessor, key: var string) =
  if key == "add_prefix_space":
    key = "addPrefixSpace"
  elif key == "trim_offsets":
    key = "trimOffsets"

proc renameHook*(v: var HFDecoder, key: var string) =
  if key == "add_prefix_space":
    key = "addPrefixSpace"
  elif key == "trim_offsets":
    key = "trimOffsets"

proc renameHook*(v: var HFSpecialToken, key: var string) =
  if key == "single_word":
    key = "singleWord"
  elif key == "special_tokens":
    key = "specialTokens"

proc renameHook*(v: var HFtokRegexp, key: var string) =
  if key == "Regex":
    key = "Regex"

proc deserializeHfTokenizer*(jsonContent: string): HFTokenizer =
  jsonContent.fromJson(HFTokenizer)

proc deserializeTiktokenizer*(content: string, regexp = R50kRegexp): TiktokenFormat =
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

  var pattern: TokRegexp

  if hf.model.pattern.regexp.len > 0:
    pattern = hf.model.pattern
  else:
    var foundPattern = false
    when defined(debug_pretokenizers):
      echo "DEBUG: preTokenizer.type = ", hf.preTokenizer.type
      echo "DEBUG: pretokenizers.len = ", hf.preTokenizer.pretokenizers.len
    if hf.preTokenizer.pretokenizers.len > 0:
      for step in hf.preTokenizer.pretokenizers:
        when defined(debug_pretokenizers):
          echo "DEBUG: step.type = ", step.type, ", pattern.Regex.len = ", step.pattern.Regex.len
        if step.type == "Split" and step.pattern.Regex.len > 0:
          pattern = TokRegexp(regexp: step.pattern.Regex)
          foundPattern = true
          break

    if not foundPattern:
      var useByteLevelDefault = false
      if hf.preTokenizer.type == "ByteLevel":
        useByteLevelDefault = hf.preTokenizer.useRegex.get(true)
      elif hf.preTokenizer.pretokenizers.len > 0:
        for step in hf.preTokenizer.pretokenizers:
          if step.type == "ByteLevel":
            useByteLevelDefault = step.useRegex.get(true)
            break

      if useByteLevelDefault:
        pattern = Gpt2Regexp
        foundPattern = true
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
        else:
          when defined(debug_bytes):
            echo "DEBUG: failed to convert rune ", int(c), " '", c, "' for key: ", key
      if bytesSeq.len == keyRunes.len:
        mergeableRanks[bytesSeq] = rank
        inc convertedCount
      else:
        inc failedCount
    when defined(debug_bytes):
      echo "DEBUG: converted ", convertedCount, " vocab entries, failed ", failedCount

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
