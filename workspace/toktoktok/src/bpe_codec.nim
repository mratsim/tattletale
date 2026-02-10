# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/tables
import std/os
import std/strutils
import std/sequtils
import std/strformat
import std/math

import workspace/pcre2
import workspace/probes

import ./serialization
import ./tokenizers_regexps

const MaxInt = high(int)

type
  Pcre2Code* = object
    ## Wrapper for compiled PCRE2 pattern
    code*: ptr Code
    pattern*: string

  Pcre2Matcher* = object
    code*: ptr Code
    matchData*: ptr MatchData
    ovector*: ptr UncheckedArray[int]
    ovectorCount*: uint32

  BPETokenizer* = object
    encoder*: Table[seq[byte], int]
    decoder*: Table[int, seq[byte]]
    specialTokensEncoder*: Table[string, int]
    specialTokensDecoder*: Table[int, seq[byte]]
    pattern*: Pcre2Code
    patternMatcher*: Pcre2Matcher
    specialPattern*: Pcre2Code
    specialMatcher*: Pcre2Matcher
    cache*: Table[seq[byte], seq[int]]
    byteDecoder*: Table[string, int]

  TokenizerError* = object of ValueError

proc `=destroy`(code: Pcre2Code) =
  if code.code != nil:
    code_free(code.code)

proc `=destroy`(matcher: Pcre2Matcher) =
  if matcher.matchData != nil:
    match_data_free(matcher.matchData)

proc init*(_: type BPETokenizer): BPETokenizer =
  default(BPETokenizer)

################################################################################
#                                                                              #
#                          Pattern Matching                                    #
#                                                                              #
################################################################################

proc compilePcre2(pattern: string, utf8: bool = true): Pcre2Code {.meter.} =
  var errorCode: CompileError
  var errorOffset: csize_t

  let options: Flag[CompileOption] = if utf8: flag(UTF, UCP) else: Flag[CompileOption](0)
  let code = compile(pattern, options, errorCode, errorOffset)
  if code == nil:
    raise newException(TokenizerError, &"PCRE2 compile error {errorCode} at offset {errorOffset}")

  result.code = code
  result.pattern = pattern

proc createMatcher(code: Pcre2Code): Pcre2Matcher {.meter.} =
  result.code = code.code
  result.matchData = match_data_create_from_pattern(code.code, nil)
  if result.matchData == nil:
    raise newException(TokenizerError, "Failed to create match data from pattern")

  result.ovector = get_ovector_pointer(result.matchData)
  result.ovectorCount = get_ovector_count(result.matchData)

iterator findAllPcre2(matcher: Pcre2Matcher, text: string, startOffset: int = 0): (int, int) =
  let subjectLen = text.len.csize_t
  var offset = startOffset.csize_t

  let options: Flag[pcre2.MatchOption] = flag(NOTEMPTY, pcre2.MatchOption.NO_UTF_CHECK)

  while offset < subjectLen:
    let rc = match(
      matcher.code,
      text,
      offset.int,
      options,
      matcher.matchData,
      nil
    )

    if rc == -1:
      break

    if rc < 0:
      raise newException(TokenizerError, &"PCRE2 match error: {rc}")

    let matchStart = matcher.ovector[0].int
    let matchEnd = matcher.ovector[1].int

    if matchStart >= text.len or matchEnd > text.len:
      break

    yield (matchStart, matchEnd)

    offset = matchEnd.csize_t

    if matchStart == matchEnd:
       offset += 1

################################################################################
#                                                                              #
#                          Byte-Pair Encoding                                  #
#                                                                              #
################################################################################
#
# Benchmarking shows that repeatedly returning/concatenating sequences was too much overhead
# in `bytePairEncode` and in-place construction was necessary.
#
# Note: TTL_METER introduces significant overhead especially for small functions (cache misses + atomic increment on function in/out)
#
# ❯ nim c -r --hints:off --warnings:off --verbosity:0 -d:danger -d:TTL_METER --outdir:build workspace/toktoktok/bench/meter_tokenizer.nim
#
# ======================================================================
# PERFORMANCE METERING: BPETokenizer.encode
# ======================================================================
# [INFO] Loading KimiK2.5 tokenizer...
# [OK] Tokenizer loaded with 163584 tokens
# [INFO] Reading Verne text (limited to 10000 chars)...
# [OK] Read 10000 chars
#
# ============================================================
# Metering tokenizer.encode on Verne text (10000 chars)
# ============================================================
#
# CPU: Intel(R) Core(TM) Ultra 7 265K
# The CPU Cycle Count is indicative only. It cannot be used to compare across systems, works at your CPU nominal frequency and is sensitive to overclocking, throttling and frequency scaling (powersaving and Turbo Boost).
#
# **BEFORE**
#
# |                         Procedures                         |  # of Calls  | Throughput (ops/s) |   Time (10⁻⁶s)   | Avg Time (10⁻⁶s) | CPU 10³cycles | Avg 10³cycles |
# |------------------------------------------------------------|--------------|--------------------|------------------|------------------|---------------|---------------|
# |bytePairMerge*(piece: seq[byte]; ranks: Table[seq[byte], ...|           662|          777351.135|           851.610|             1.286|       3162.868|          4.778|
# |bytePairEncode*(piece: seq[byte]; ranks: Table[seq[byte] ...|           662|             257.659|       2569284.415|          3881.094|    9964170.880|      15051.618|
# |splitTextOrdinary(tokenizer: BPETokenizer; text: string) ...|             1|            3152.058|           317.253|           317.253|       1230.220|       1230.220|
# |encodeOrdinary*(tokenizer: BPETokenizer; text: string):  ...|             1|               0.389|       2569959.520|       2569959.520|    9967008.492|    9967008.492|
# |encodeWithSpecialTokens*(tokenizer: BPETokenizer; text:  ...|             1|               0.389|       2569969.828|       2569969.828|    9967048.497|    9967048.497|
# |encode*(tokenizer: BPETokenizer; text: string): seq[int]    |             1|               0.389|       2569969.950|       2569969.950|    9967049.001|    9967049.001|
#
# Result: 3124 tokens encoded
#
# **AFTER**
#
# |                         Procedures                         |  # of Calls  | Throughput (ops/s) |   Time (10⁻⁶s)   | Avg Time (10⁻⁶s) | CPU 10³cycles | Avg 10³cycles |
# |------------------------------------------------------------|--------------|--------------------|------------------|------------------|---------------|---------------|
# |bytePairMerge(piece: seq[byte]; ranks: Table[seq[byte],  ...|           662|         1076666.108|           614.861|             0.929|       2337.808|          3.531|
# |bytePairEncode*(encodedResult: var seq[int]; piece: seq[ ...|           662|          966358.464|           685.046|             1.035|       2610.018|          3.943|
# |splitTextOrdinary(tokenizer: BPETokenizer; text: string) ...|             1|            2934.221|           340.806|           340.806|       1321.500|       1321.500|
# |encodeOrdinaryImpl(encodedResult: var seq[int]; tokenize ...|             1|             832.752|          1200.837|          1200.837|       4656.916|       4656.916|
# |encodeWithSpecialTokens*(tokenizer: BPETokenizer; text:  ...|             1|             830.313|          1204.365|          1204.365|       4670.778|       4670.778|
# |encode*(tokenizer: BPETokenizer; text: string): seq[int]    |             1|             830.270|          1204.427|          1204.427|       4671.020|       4671.020|
#
# Result: 3124 tokens encoded

proc bytePairMerge(piece: seq[byte], ranks: Table[seq[byte], int]): seq[(int, int)] {.meter.} =
  var parts = newSeqOfCap[(int, int)](piece.len + 2)

  var minRank = MaxInt
  var minRankIdx = 0

  for i in 0..<piece.len - 1:
    let pair = @[piece[i], piece[i+1]]          # TODO: That seems like a wasteful allocation
    let rank = ranks.getOrDefault(pair, MaxInt)
    if rank < minRank:
      minRank = rank
      minRankIdx = i
    parts.add((i, rank))

  parts.add((piece.len - 1, MaxInt))
  parts.add((piece.len, MaxInt))

  template getRank(parts: seq[(int, int)], i: int): int =
    ## Get rank for pair starting at parts[i], spanning to parts[i+3] boundary.
    ## Captures `ranks` and `piece`
    ## Always inlined
    if i + 3 < parts.len:
      let startIdx = parts[i][0]
      let endIdx = parts[i+3][0]
      let pair = piece[startIdx..<endIdx]
      ranks.getOrDefault(pair, MaxInt)
    else:
      MaxInt

  while minRank != MaxInt:
    let i = minRankIdx

    if i > 0:
      parts[i-1] = (parts[i-1][0], getRank(parts, i-1))

    parts[i] = (parts[i][0], getRank(parts, i))
    parts.delete(i + 1)

    minRank = MaxInt
    minRankIdx = 0
    for idx in 0..<parts.len - 1:
      let (_, rank) = parts[idx]
      if rank < minRank:
        minRank = rank
        minRankIdx = idx

  parts

proc bytePairEncode*(
        encodedResult: var seq[int],
        piece: seq[byte],
        ranks: Table[seq[byte], int]) {.meter.} =

  if piece.len == 1:
    encodedResult.add(ranks[piece])

  let mergedParts = bytePairMerge(piece, ranks)

  for i in 0 ..< mergedParts.len-1:
    encodedResult.add(ranks[piece[mergedParts[i][0]..<mergedParts[i+1][0]]])

################################################################################
#                                                                              #
#                             Tokenizing                                       #
#                                                                              #
################################################################################

proc splitTextOrdinary(tokenizer: BPETokenizer, text: string): seq[string] {.meter.} =
  var lastPos = 0
  for (start, stop) in findAllPcre2(tokenizer.patternMatcher, text):
    if start > lastPos:
      result.add(text[lastPos..<start])
    result.add(text[start..<stop])
    lastPos = stop

  if lastPos < text.len:
    result.add(text[lastPos..<text.len])

proc encodeOrdinaryImpl(encodedResult: var seq[int], tokenizer: BPETokenizer, text: string) {.meter.} =
  # TODO: text should be a view to avoid alloc
  let pieces = tokenizer.splitTextOrdinary(text)
  for piece in pieces:
    # string and seq[byte] have the same internal repr in Nim, at leat Nim v0, v1 and v2
    # except string have also a terminating \0 (not counted in len)
    let pieceByte = cast[seq[byte]](piece)
    if pieceByte in tokenizer.encoder:
      encodedResult.add(tokenizer.encoder[pieceByte])
    else:
      encodedResult.bytePairEncode(pieceByte, tokenizer.encoder)

proc encodeOrdinary*(tokenizer: BPETokenizer, text: string): seq[int] =
  result.encodeOrdinaryImpl(tokenizer, text)

proc encodeWithSpecialTokens*(tokenizer: BPETokenizer, text: string): seq[int] {.meter.} =
  var pos = 0

  while pos < text.len:
    var foundSpecial = false
    var nextPos = text.len
    var specialToken = ""

    for token, tokenId in tokenizer.specialTokensEncoder:
      let foundPos = text.find(token, pos)
      if foundPos != -1 and (nextPos == text.len or foundPos < nextPos):
        nextPos = foundPos
        specialToken = token
        foundSpecial = true

    if foundSpecial and nextPos == pos:
      result.add(tokenizer.specialTokensEncoder[specialToken])
      pos = pos + specialToken.len
    elif foundSpecial:
      if pos < nextPos:
        result.encodeOrdinaryImpl(tokenizer, text[pos ..< nextPos]) # TODO: view slices
      pos = nextPos
    else:
      result.encodeOrdinaryImpl(tokenizer, text[pos ..< text.len]) # TODO: view slices
      break

proc encode*(tokenizer: BPETokenizer, text: string): seq[int] {.meter.} =
  tokenizer.encodeWithSpecialTokens(text)

proc decodeToBytes(tokenizer: BPETokenizer, tokenIds: seq[int]): seq[byte] {.meter.} =
  for id in tokenIds:
    let bytes = tokenizer.decoder.getOrDefault(id, @[])
    if bytes.len > 0:
      result.add(bytes)
    else:
      let specialBytes = tokenizer.specialTokensDecoder.getOrDefault(id, @[])
      if specialBytes.len > 0:
        result.add(specialBytes)
      else:
        raise newException(TokenizerError, "Invalid token: " & $id)

proc decodeToString*(tokenizer: BPETokenizer, tokenIds: seq[int]): string {.meter.} =
  let bytes = tokenizer.decodeToBytes(tokenIds)
  if bytes.len == 0:
    return ""
  result = newString(bytes.len)
  copyMem(result[0].addr, bytes[0].unsafeAddr, bytes.len)

################################################################################
#                                                                              #
#                        Vocabulary loaders                                    #
#                                                                              #
################################################################################

proc loadFromTiktoken(ttk: TiktokenFormat): BPETokenizer =
  var tokenizer = BPETokenizer()

  # Build byte decoder
  tokenizer.byteDecoder = initTable[string, int]()
  for i in 0..255:
    tokenizer.byteDecoder[$char(i)] = i

  # Load special tokens
  if ttk.specialTokens.len > 0:
    for token, id in ttk.specialTokens:
      tokenizer.specialTokensEncoder[token] = id
      tokenizer.specialTokensDecoder[id] = toBytes(token)

  # Build encoder/decoder tables
  var encoder = initTable[seq[byte], int]()

  for keyBytes, rank in ttk.mergeableRanks:
    encoder[keyBytes] = rank

  tokenizer.encoder = encoder

  # Build decoder (reverse mapping)
  tokenizer.decoder = initTable[int, seq[byte]]()
  for k, v in encoder:
    tokenizer.decoder[v] = k

  # Build special tokens decoder
  for k, v in tokenizer.specialTokensEncoder:
    tokenizer.specialTokensDecoder[v] = toBytes(k)

   # Compile regex pattern
  tokenizer.pattern = compilePcre2(ttk.pattern.regexp)
  tokenizer.patternMatcher = createMatcher(tokenizer.pattern)

   # Compile special tokens pattern
  if tokenizer.specialTokensEncoder.len > 0:
    let specialTokens = toSeq(tokenizer.specialTokensEncoder.keys)
    var escapedTokens: seq[string] = @[]
    for token in specialTokens:
      var escaped = ""
      for c in token:
        if c in ['\\', '[', ']', '(', ')', '{', '}', '^', '$', '|', '*', '+', '?', '.', '#', '~']:
          escaped.add('\\')
          escaped.add(c)
        else:
          escaped.add(c)
      escapedTokens.add(escaped)
    let specialPatternStr = escapedTokens.join("|")
    tokenizer.specialPattern = compilePcre2(specialPatternStr)
    tokenizer.specialMatcher = createMatcher(tokenizer.specialPattern)
  else:
    # Create a matcher that never matches
    tokenizer.specialPattern = compilePcre2("(?!)")  # Never matches
    tokenizer.specialMatcher = createMatcher(tokenizer.specialPattern)

  tokenizer

proc loadHFTokenizer*(path: string): BPETokenizer =
  if not fileExists(path):
    raise newException(TokenizerError, "HF tokenizer JSON file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerError, "HF tokenizer JSON file is empty: " & path)

  let hf = deserializeHfTokenizer(content)
  let ttk = convertHfToTiktoken(hf)
  loadFromTiktoken(ttk)

proc loadTiktokenizer*(path: string, regexp: TokRegexp): BPETokenizer =
  if not fileExists(path):
    raise newException(TokenizerError, "Tiktoken file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerError, "Tiktoken file is empty: " & path)

  let ttk = deserializeTiktokenizer(content, regexp)
  loadFromTiktoken(ttk)

proc tokenCount*(tokenizer: BPETokenizer): int =
  tokenizer.encoder.len + tokenizer.specialTokensEncoder.len