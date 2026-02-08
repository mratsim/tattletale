# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/tables
import std/options
import std/strutils
import pkg/jsony

type
  HFTokenizer* = object
    version*: Option[string]
    truncation*: Option[string]
    padding*: Option[string]
    addedTokens*: Option[seq[HFSpecialToken]]
    normalizer*: Option[string]
    preTokenizer*: Option[HFPreTokenizer]
    postProcessor*: Option[HFPostProcessor]
    decoder*: Option[HFDecoder]
    model*: HFTokenizerModel

  HFTokenizerModel* = object
    vocab*: OrderedTable[string, int]
    dropout*: Option[string]
    unkToken*: Option[string]
    continuingSubwordPrefix*: Option[string]
    endOfWordSuffix*: Option[string]
    fuseUnk*: Option[bool]
    `type`*: string

  HFPreTokenizer* = object
    addPrefixSpace*: Option[bool]
    trimOffsets*: Option[bool]
    `type`*: string

  HFPostProcessor* = object
    addPrefixSpace*: Option[bool]
    trimOffsets*: Option[bool]
    `type`*: string

  HFDecoder* = object
    addPrefixSpace*: Option[bool]
    trimOffsets*: Option[bool]
    `type`*: string

  HFSpecialToken* = object
    content*: string
    id*: int
    lstrip*: Option[bool]
    normalized*: Option[bool]
    rstrip*: Option[bool]
    singleWord*: Option[bool]
    special*: Option[bool]

  TiktokenFormat* = object
    mergeableRanks*: OrderedTable[seq[byte], int]
    patStr*: string
    specialTokens*: OrderedTable[string, int]

  TiktokenFile* = object
    patStr*: string
    mergeableRanks*: OrderedTable[string, int]
    specialTokens*: OrderedTable[string, int]

  TokenizerParseError* = object of ValueError

const DefaultPat* = "'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s0-9a-zA-Z]+|\\r?\\n|\\s+(?!\\S)|\\s+"

proc toBytes*(str: string): seq[byte] =
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

proc newHook*(v: var HFTokenizerModel) =
  v.`type` = ""

proc newHook*(v: var HFSpecialToken) =
  v.special = some(false)
  v.singleWord = some(false)
  v.lstrip = some(false)
  v.rstrip = some(false)
  v.normalized = some(true)

proc newHook*(v: var HFPreTokenizer) =
  discard

proc newHook*(v: var HFPostProcessor) =
  discard

proc newHook*(v: var HFDecoder) =
  discard

proc deserializeHfTokenizer*(jsonContent: string): HFTokenizer =
  jsonContent.fromJson(HFTokenizer)

proc deserializeTiktoken*(jsonContent: string): TiktokenFile =
  jsonContent.fromJson(TiktokenFile)

proc convertHfToTiktoken*(hf: HFTokenizer): TiktokenFormat =
  var mergeableRanks = initOrderedTable[seq[byte], int]()

  if hf.model.vocab.len > 0:
    for key, rank in hf.model.vocab:
      let bytesSeq = toBytes(key)
      if bytesSeq.len > 0:
        mergeableRanks[bytesSeq] = rank

  # Ensure individual bytes are always in the encoder (for UTF-8 fallback)
  # These get very high ranks (low priority) so they're only used when no merge is available
  let byteRankStart = 1000000  # High rank for byte tokens
  for i in 0..<256:
    let byteSeq = @[byte(i)]
    if not mergeableRanks.hasKey(byteSeq):
      mergeableRanks[byteSeq] = byteRankStart + i

  var specialTokens = initOrderedTable[string, int]()
  if hf.addedTokens.isSome:
    for token in hf.addedTokens.get:
      specialTokens[token.content] = token.id

  TiktokenFormat(
    mergeableRanks: mergeableRanks,
    patStr: DefaultPat,
    specialTokens: specialTokens
  )
