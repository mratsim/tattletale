# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Minimal tokenizer-deserialization surface for toktoktok, rank files
## and HF json checkpoints in, a byte-level BPE codec out.
##
## Scope, checkpoint data path only:
## - rank tables in tiktoken base64 format.
## - the HF tokenizer.json vocab conversion to byte ranks.
## - the special-token table and the id-to-bytes decode.
## - the GPT-2 byte-to-unicode remap, table and string procs.
## - the remap is a checkpoint-format artifact, served vocabularies
##   are byte rank tables so the encode path never maps, the remap
##   serves checkpoints whose vocabulary keys carry the remapped text
##   and the unmap proc serves detokenization.
##
## A config-driven HF loader (pre_tokenizer/added_tokens orchestration beyond the data conversion)
## is a separate future op and is deliberately absent here.
##
## No pattern compilation, no encode cache, the pipeline machines
## own the encode side, the family chains come from scan.nim's pre-tokenization step.
##
## A normalizer spec, if a checkpoint ever ships one, lives here.
## No staged checkpoint declares a normalizer and no normalization
## stage exists in src/.
##
## Codec shape, TiktokenCodec fields:
## - `ranks` maps the mergeable byte tokens to ids.
## - `specials` maps special-token strings to ids, empty for the plain
##   rank files, the special strings of a checkpoint live beside
##   the rank file, e.g. a tokenizer_config.json added_tokens block,
##   and are supplied via loadTiktokenCodec's specialTokens argument.
## - `decoder` is the id-to-bytes inverse over both tables.
##
##   import workspace/toktoktok/src/deserializers  # the codec API
##
##   let codec = loadTiktokenCodec("r50k_base.tiktoken")
##   let ids = @[(int 270), (int 24713)]  # 'Hel' 'lo' shapes
##   doAssert decodeToString(codec, ids) == "Hello"

import std/[base64, os, strutils, options, tables, unicode]
import pkg/jsony

type
  ## Raised for missing/empty files, malformed rank lines and invalid
  ## token ids on the decode path.
  TokenizerLoadError* = object of ValueError

  ## Flat pattern-string wrapper (the checkpoint's split pattern as a string, the family chain resolution happens in scan.nim).
  TokRegexp* = object
    regexp*: string

  ## HF json Split-step pattern record, the json spells the key
  ## "Regex", capital R.
  HFtokRegexp* = object
    Regex*: string

  ## Intermediate deserialized checkpoint (mergeable ranks plus the checkpoint split pattern plus the special tokens):
  ## the split pattern is the flat alternation string, later resolved
  ## to a scan.Family by the caller (the deserialization contract).
  TiktokenFormat* = object
    mergeableRanks*: OrderedTable[seq[byte], int]
    pattern*: string
    specialTokens*: OrderedTable[string, int]

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

  ## Deserialized HF model section (jsony target, BPE fields plus the pattern wrapper).
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

  ## Byte-level BPE codec ready for the pipeline machines (ranks plus specials plus the id-to-bytes decoder).
  TiktokenCodec* = object
    ranks*: Table[seq[byte], int]
    specials*: Table[string, int]
    decoder*: Table[int, seq[byte]]

template toBytes(str: string): seq[byte] =
  @(toOpenArrayByte(str, 0, str.len - 1))

proc gpt2ByteDecoder*(): Table[uint32, int] =
  ## GPT-2 bytes-to-unicode inverse map, lookup contract:
  ## - codepoint in, original byte out.
  ## - printable ASCII and the Latin-1 printable run map onto themselves,
  ##   the remaining 67 bytes shift to 256+n.
  ## - the HF json vocabularies spell their tokens in this remapped space,
  ##   so the conversion walks every vocab key through the table.
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

proc renameHook*(v: var HFSpecialToken, key: var string) =
  ## jsony rename hook, snake_case json keys map to the camelCase fields.
  if key == "single_word":
    key = "singleWord"
  elif key == "special_tokens":
    key = "specialTokens"

proc deserializeHfTokenizer*(jsonContent: string): HFTokenizer =
  ## Parses one HF tokenizer.json (jsony into the HF record types).
  jsonContent.fromJson(HFTokenizer)

proc deserializeTiktokenRanks*(content: string,
    pattern = ""): TiktokenFormat =
  ## Expected input:
  ## - one "base64 rank" pair per line, '#' comment lines and empties skipped.
  ## Output:
  ## - the parsed TiktokenFormat, the specialTokens table stays empty,
  ##   rank files carry mergeable ranks only.
  let lines = content.splitLines()
  var mergeableRanks = initOrderedTable[seq[byte], int]()

  for line in lines:
    if line.len == 0:
      continue
    if line.startsWith("#"):
      continue

    let parts = line.split(" ")
    if parts.len < 2:
      raise newException(TokenizerLoadError, "Invalid tiktoken line: " & line)

    let rankStr = parts[1]
    let rank = parseInt(rankStr)
    let decodedTokenBytes = toBytes(decode(parts[0]))
    mergeableRanks[decodedTokenBytes] = rank

  TiktokenFormat(
    mergeableRanks: mergeableRanks,
    pattern: pattern,
    specialTokens: initOrderedTable[string, int]()
  )

proc convertHfToTiktoken*(hf: HFTokenizer): TiktokenFormat =
  ## Converts one deserialized HF checkpoint into tiktoken shape, conversion contract:
  ## - vocab keys, spelled in the GPT-2 bytes-to-unicode space, walk
  ##   back to raw bytes through gpt2ByteDecoder, keys with non-decodable
  ##   codepoints are dropped.
  ## - every byte keeps a rank, missing bytes get 1000000+i, high above
  ##   any real vocabulary rank.
  ## - the pattern field carries the model-level Regex, else the Split
  ##   steps of the pre_tokenizer chain joined with '|', else the GPT-2
  ##   default pattern for ByteLevel pre-tokenizers.
  ## - a checkpoint with none of those is a load error.

  var pattern: string

  if hf.model.pattern.regexp.len > 0:
    pattern = hf.model.pattern.regexp
  else:
    var splitPatterns: seq[string] = @[]
    if hf.preTokenizer.pretokenizers.len > 0:
      for step in hf.preTokenizer.pretokenizers:
        if step.type == "Split" and step.pattern.Regex.len > 0:
          splitPatterns.add(step.pattern.Regex)

    if splitPatterns.len > 0:
      pattern = splitPatterns.join("|")
    elif hf.preTokenizer.type == "ByteLevel":
      let useByteLevelDefault = hf.preTokenizer.useRegex.get(true)
      if useByteLevelDefault:
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
      else:
        raise newException(ValueError, "Error: the HuggingFace tokenizer JSON file is missing regexp information.")
    elif hf.preTokenizer.pretokenizers.len > 0:
      var useByteLevelDefault = false
      for step in hf.preTokenizer.pretokenizers:
        if step.type == "ByteLevel":
          useByteLevelDefault = step.useRegex.get(true)
          break

      if useByteLevelDefault:
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
      else:
        raise newException(ValueError, "Error: the HuggingFace tokenizer JSON file is missing regexp information.")
    else:
      raise newException(ValueError, "Error: the HuggingFace tokenizer JSON file is missing regexp information.")

  var mergeableRanks = initOrderedTable[seq[byte], int]()

  if hf.model.vocab.len > 0:
    let byteDecoder = gpt2ByteDecoder()
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

proc codecFromTiktoken*(ttk: TiktokenFormat): TiktokenCodec =
  ## Builds the codec from a deserialized checkpoint:
  ## - the rank and special tables are captured.
  ## - the decoder is the id-to-bytes inverse over the two tables.
  for keyBytes, rank in ttk.mergeableRanks:
    result.ranks[keyBytes] = rank
  for token, id in ttk.specialTokens:
    result.specials[token] = id
  for k, v in result.ranks:
    result.decoder[v] = k
  for k, v in result.specials:
    result.decoder[v] = toBytes(k)

proc loadTiktokenCodec*(path: string,
    specialTokens: sink OrderedTable[string, int] = initOrderedTable[string, int]()):
    TiktokenCodec =
  ## Loads a `.tiktoken` rank file into a codec, the specialTokens
  ## contract comes from the caller argument:
  ## - base64 lines carry mergeable ranks only.
  ## - explicit specialTokens extend the codec when supplied (e.g. parsed from the checkpoint's tokenizer_config.json added_tokens block).
  if not fileExists(path):
    raise newException(TokenizerLoadError, "Tiktoken file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerLoadError, "Tiktoken file is empty: " & path)

  var ttk = deserializeTiktokenRanks(content)
  ttk.specialTokens = specialTokens
  result = codecFromTiktoken(ttk)

proc loadHfCodec*(path: string): TiktokenCodec =
  ## Loads an HF tokenizer.json into a codec through the byte-rank
  ## conversion contract:
  ## - deserializeHfTokenizer parses the json.
  ## - convertHfToTiktoken converts to tiktoken shape.
  ## - codecFromTiktoken builds the codec.
  ## The checkpoint's pattern, needed for family resolution, is
  ## available via deserializeHfTokenizer + convertHfToTiktoken.
  if not fileExists(path):
    raise newException(TokenizerLoadError, "HF tokenizer JSON file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerLoadError, "HF tokenizer JSON file is empty: " & path)

  codecFromTiktoken(convertHfToTiktoken(deserializeHfTokenizer(content)))

proc decodeToBytes*(c: TiktokenCodec, tokenIds: openArray[int]): seq[byte] =
  ## Decodes ids back to raw bytes, mergeable ranks first, special
  ## ids second, unknown ids raise.
  for id in tokenIds:
    let bytes = c.decoder.getOrDefault(id, @[])
    if bytes.len > 0:
      result.add(bytes)
    else:
      raise newException(TokenizerLoadError, "Invalid token: " & $id)

proc decodeToString*(c: TiktokenCodec, tokenIds: openArray[int]): string =
  ## decodeToBytes rendered as a string
  ## (the vocabularies are byte rank tables, no byte-to-unicode remap participates in the decode).
  let bytes = decodeToBytes(c, tokenIds)
  if bytes.len == 0:
    return ""
  result = newString(bytes.len)
  copyMem(result[0].addr, bytes[0].unsafeAddr, bytes.len)

proc tokenCount*(c: TiktokenCodec): int =
  ## Vocabulary size:
  ##   mergeable ranks plus special tokens.
  c.ranks.len + c.specials.len

# ------------------------------------------------------------------------
# GPT-2 byte-to-unicode remap (checkpoint-format artifact)
# ------------------------------------------------------------------------

const
  ByteLevelRemap*: array[256, uint32] = block:
    ## Mapped codepoint per input byte (the GPT-2 bytes_to_unicode table):
    ##
    ##   printable bytes 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF map
    ## to themselves, every other byte maps to 0x100 + its rank among
    ## the non-printable bytes in ascending byte order.
    var t: array[256, uint32]
    var n = 0
    for b in 0 ..< 256:
      if (b >= 0x21 and b <= 0x7E) or (b >= 0xA1 and b <= 0xAC) or
          (b >= 0xAE and b <= 0xFF):
        t[b] = uint32(b)
      else:
        t[b] = 0x100'u32 + uint32(n)
        inc n
    t

  ByteLevelMaxCp = 0x143
  ## Highest mapped codepoint, 0x100 + (256 - 188) - 1, where 188
  ## counts the printable bytes of the identity set.

proc bytesToUnicode*(input: string, addPrefixSpace = false): string =
  ## Byte-to-unicode remap rendered as a string, mapping contract:
  ## - every input byte is replaced by its mapped codepoint rendered
  ##   as UTF-8 bytes.
  ## - a mapped codepoint outside ASCII emits its 2-byte UTF-8 encoding,
  ##   so the output is at most 2x the input length.
  ## - the inverse is `unmap`.
  ##
  ## - addPrefixSpace=true prepends one space to the remapped stream
  ##   when the input does not already open with one.
  ## - that space insertion is the post-processor position (EXAONE-style),
  ##   prefix-space semantics live in POST-processing,
  ##   pre-tokenization never adds one and keeps the default remap value.
  if addPrefixSpace and (input.len == 0 or input[0] != ' '):
    let sp = ByteLevelRemap[uint8(' ')]
    result.add char(0xC0'u8 or uint8(sp shr 6))
    result.add char(0x80'u8 or uint8(sp and 0x3F'u32))
  for i in 0 ..< input.len:
    let cp = ByteLevelRemap[uint8(input[i])]
    if cp < 0x80'u32:
      result.add char(cp)
    else:
      result.add char(0xC0'u8 or uint8(cp shr 6))
      result.add char(0x80'u8 or uint8(cp and 0x3F'u32))

proc computeUnmapTable(): array[ByteLevelMaxCp + 1, int16] {.compileTime.} =
  ## Byte per mapped codepoint, -1 outside the image.
  for cp in 0 .. ByteLevelMaxCp:
    result[cp] = -1
  for b in 0 ..< 256:
    result[int(ByteLevelRemap[b])] = int16(b)

proc unmap*(mapped: openArray[char]): string =
  ## Inverse of the remap (the decoder position):
  ##   reads the UTF-8
  ## bytes of a remapped stream and returns the original bytes.
  ##
  ## Raises ValueError on a codepoint outside the remap image (never produced by the remap, the decoder contract rejects it).
  const unmapTable = computeUnmapTable()
  var i = 0
  while i < mapped.len:
    let b0 = uint8(mapped[i])
    var cp: int
    var width: int
    if b0 < 0x80:
      cp = int(b0)
      width = 1
    elif (b0 and 0xE0) == 0xC0:
      if i + 1 >= mapped.len:
        raise newException(ValueError, "unmap: truncated UTF-8 sequence")
      cp = (int(b0 and 0x1F) shl 6) or int(uint8(mapped[i + 1]) and 0x3F)
      width = 2
    else:
      raise newException(ValueError, "unmap: byte outside the remap image")
    if cp > ByteLevelMaxCp or unmapTable[cp] < 0:
      raise newException(ValueError, "unmap: codepoint outside the remap image")
    result.add char(unmapTable[cp])
    inc i, width
