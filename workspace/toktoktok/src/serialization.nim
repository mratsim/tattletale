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
    pattern*: string

  HFPreTokenizer* = object
    addPrefixSpace*: bool
    trimOffsets*: bool
    `type`*: string
    pretokenizers*: seq[HFPretokenizerStep]

  HFPretokenizerStep* = object
    `type`*: string
    pattern*: HFPretokenizerRegex
    behavior*: string
    invert*: bool
    addPrefixSpace*: bool
    trimOffsets*: bool
    useRegex*: bool

  HFPretokenizerRegex* = object
    regex*: string

  HFSplitPreTokenizer* = object
    regex*: string
    behavior*: string
    invert*: bool

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
    patStr*: string
    specialTokens*: OrderedTable[string, int]


  TokenizerParseError* = object of ValueError

const DefaultPat* = r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s0-9a-zA-Z]+|\r?\n|\s+(?!\S)|\s+"

const Gpt2Pat* = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
const R50kBasePat* = Gpt2Pat
const P50kBasePat* = Gpt2Pat

const Cl100kBasePat* = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

const O200kBasePat* = "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

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

proc renameHook*(v: var HFPretokenizerRegex, key: var string) =
  if key == "Regex":
    key = "regex"

proc deserializeHfTokenizer*(jsonContent: string): HFTokenizer =
  jsonContent.fromJson(HFTokenizer)

proc deserializeTiktokenizer*(content: string, patStr: string = DefaultPat): TiktokenFormat =
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
    patStr: patStr,
    specialTokens: initOrderedTable[string, int]()
  )

proc convertHfToTiktoken*(hf: HFTokenizer): TiktokenFormat =
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

  var patStr = DefaultPat
  if hf.model.pattern != "":
    patStr = hf.model.pattern
  elif hf.preTokenizer.`type` == "Sequence":
    for pretok in hf.preTokenizer.pretokenizers:
      if pretok.`type` == "Split" and pretok.pattern.regex != "":
        patStr = pretok.pattern.regex
        break

  TiktokenFormat(
    mergeableRanks: mergeableRanks,
    patStr: patStr,
    specialTokens: specialTokens
  )
