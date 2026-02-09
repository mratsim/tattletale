# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/tables
import std/strutils
import std/base64
import pkg/jsony
import ./tokenizers_regexps

type
  HFTokenizer* = object
    version*: string
    truncation*: string
    padding*: string
    addedTokens*: seq[HFSpecialToken]
    normalizer*: string
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

  HFPretokenizerStep* = object
    `type`*: string
    pattern*: TokRegexp
    behavior*: string
    invert*: bool
    addPrefixSpace*: bool
    trimOffsets*: bool
    useRegex*: bool

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

  TiktokenFormat* = object
    mergeableRanks*: OrderedTable[seq[byte], int]
    pattern*: TokRegexp
    specialTokens*: OrderedTable[string, int]


  TokenizerParseError* = object of ValueError

template toBytes*(str: string): seq[byte] =
  @(toOpenArrayByte(str, 0, str.len - 1))

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

proc renameHook*(v: var TokRegexp, key: var string) =
  if key == "Regex":
    key = "regex"

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

  if hf.model.pattern.regexp.len == 0:
    raise newException(ValueError, "Error: the HuggingFace tokenizer JSON file is missing regexp information.")

  var mergeableRanks = initOrderedTable[seq[byte], int]()

  if hf.model.vocab.len > 0:
    for key, rank in hf.model.vocab:
      let bytesSeq = toBytes(key)
      if bytesSeq.len > 0:
        mergeableRanks[bytesSeq] = rank

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
    pattern: hf.model.pattern,
    specialTokens: specialTokens
  )
