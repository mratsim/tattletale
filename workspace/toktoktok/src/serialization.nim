# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/tables
import std/options
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
    mergeableRanks*: OrderedTable[string, int]
    patStr*: string
    specialTokens*: OrderedTable[string, int]

  TokenizerParseError* = object of ValueError

const DefaultPat = "'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s0-9a-zA-Z]+|\\r?\\n|\\s+(?!\\S)|\\s+"

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

proc convertHfToTiktoken*(hf: HFTokenizer): TiktokenFormat =
  var mergeableRanks = initOrderedTable[string, int]()

  if hf.model.vocab.len > 0:
    for key, rank in hf.model.vocab:
      var bytesSeq: seq[int] = @[]
      var valid = true
      for c in key:
        let codePoint = ord(c)
        if codePoint < 128 or (codePoint >= 161 and codePoint <= 172) or (codePoint >= 174 and codePoint <= 255):
          bytesSeq.add(codePoint)
        elif codePoint >= 256:
          bytesSeq.add(codePoint - 128)
        else:
          valid = false
          break
      if valid and bytesSeq.len > 0:
        var bytesStr = "["
        for i, b in bytesSeq:
          if i > 0: bytesStr.add(", ")
          bytesStr.add($b)
        bytesStr.add("]")
        mergeableRanks[bytesStr] = rank

  var specialTokens = initOrderedTable[string, int]()
  if hf.addedTokens.isSome:
    for token in hf.addedTokens.get:
      specialTokens[token.content] = token.id

  TiktokenFormat(
    mergeableRanks: mergeableRanks,
    patStr: DefaultPat,
    specialTokens: specialTokens
  )
