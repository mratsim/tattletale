# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tokenizer codec surface for the pipeline machines:
## rank files and HF json checkpoints in, a byte-level BPE codec out.
##
## Checkpoint parsing (jsony HF records, tiktoken rank lines, HF conversion) is the `serialization.nim` surface, reused here.
##
## Members:
##
## | member          | surface                                                                                                     |
## | --------------- | ----------------------------------------------------------------------------------------------------------- |
## | `TiktokenCodec` | bundles the rank table, the special-token table and the id-to-bytes decoder                                 |
## | loaders         | `loadTiktokenCodec` / `loadHfCodec` / `codecFromTiktoken`                                                   |
## | decode          | `decodeToBytes` / `decodeToString` / `tokenCount`                                                           |
## | remap           | the GPT-2 byte-to-unicode remap (`ByteLevelRemap`, `bytesToUnicode`, `unmap`), a checkpoint-format artifact |
##
## Served vocabularies are byte rank tables, the encode path never maps,
## `unmap` serves detokenization of checkpoints whose vocabulary keys
## carry the remapped text.
##
## No pattern compilation and no encode cache:
## the pipeline machines own the encode side, the family chains come from `scan.nim`'s pre-tokenization steps.
## A config-driven HF loader (pre_tokenizer/added_tokens orchestration beyond the data conversion) is deliberately absent.
##
##
##   import workspace/toktoktok/src/deserializers  # the codec API
##
##   let codec = loadTiktokenCodec("r50k_base.tiktoken")
##   let ids = @[(int 270), (int 24713)]  # 'Hel' 'lo' shapes
##   doAssert decodeToString(codec, ids) == "Hello"

import std/[os, strutils, options, tables]
import ./serialization

type
  TokenizerLoadError* = object of ValueError
    ## Raised for missing/empty files and invalid token ids
    ## on the codec load and decode paths.

  TiktokenCodec* = object
    ## Byte-level BPE codec ready for the pipeline machines,
    ## holding ranks, specials and the id-to-bytes decoder.
    ranks*: Table[seq[byte], int]
    # empty for plain rank files. A checkpoint's special strings live
    # beside the rank file, supplied via loadTiktokenCodec's specialTokens.
    specials*: Table[string, int]
    decoder*: Table[int, seq[byte]]

template toBytes(str: string): seq[byte] =
  @(toOpenArrayByte(str, 0, str.len - 1))

proc codecFromTiktoken(ttk: TiktokenFormat): TiktokenCodec =
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
  ## - explicit specialTokens extend the codec when supplied, carrying
  ##   a checkpoint's tokenizer_config.json added_tokens block.
  if not fileExists(path):
    raise newException(TokenizerLoadError, "Tiktoken file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerLoadError, "Tiktoken file is empty: " & path)

  var ttk = deserializeTiktokenizer(content)
  ttk.specialTokens = specialTokens
  result = codecFromTiktoken(ttk)

proc loadHfCodec*(path: string): TiktokenCodec =
  ## Loads an HF tokenizer.json into a codec:
  ## deserializeHfTokenizer parses the json, convertHfToTiktoken
  ## converts to tiktoken shape, codecFromTiktoken builds the codec.
  ##
  ## Returns the codec only, without the checkpoint's split pattern:
  ## the split pattern is read separately through the serialization surface
  ## (deserializeHfTokenizer + convertHfToTiktoken).
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
  ## Byte-to-unicode remap rendered as a string, the inverse of `unmap`.
  ## `addPrefixSpace` prepends one leading space when the input does not open with one.
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
  ## Raises ValueError on a codepoint outside the remap image,
  ## a codepoint the remap never produces.
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
