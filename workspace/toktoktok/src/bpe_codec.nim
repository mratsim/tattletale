# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Byte-level BPE codec over PCRE2 pattern matching
## (pattern wrappers, tokenizer object, merge cores, checkpoint loaders).

import std/tables
import std/os
import std/strutils
import std/sequtils
import std/strformat
import std/math

import workspace/pcre2
import workspace/bencher

import ./serialization
import ./tokenizers_regexps

const MaxInt = high(int)

type
  ## Wrapper for a compiled PCRE2 pattern, owner of the code pointer.
  Pcre2Code* = object
    code*: ptr Code
    pattern*: string

  ## PCRE2 match state, owner of the match data, borrower of the code.
  Pcre2Matcher* = object
    code*: ptr Code
    matchData*: ptr MatchData
    ovector*: ptr UncheckedArray[int]
    ovectorCount*: uint32

  ## Byte-level BPE codec over PCRE2 pattern matching (encoder, decoder, special-token tables, per-piece cache).
  BPETokenizer* = ref object
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

  ## Raised for missing/empty files, malformed checkpoints and invalid
  ## token ids on the decode path.
  TokenizerError* = object of ValueError

proc `=destroy`(code: Pcre2Code) {.inline.} =
  if code.code != nil:
    code_free(code.code)
  `=destroy`(code.pattern)

proc `=wasMoved`(code: var Pcre2Code) {.inline.}  =
  code.code = nil
  `=wasMoved`(code.pattern)

proc `=destroy`(matcher: Pcre2Matcher) {.inline.} =
# Pcre2Matcher does NOT own the ptr Code, it is borrowed from Pcre2Code.
  # Only Pcre2Code's destructor calls code_free. Freeing matchData is sufficient.
  if matcher.matchData != nil:
    match_data_free(matcher.matchData)

proc `=wasMoved`(matcher: var Pcre2Matcher) {.inline.} =
  matcher.code = nil
  matcher.matchData = nil
  matcher.ovector = nil
  matcher.ovectorCount = 0

proc init*(_: type BPETokenizer): BPETokenizer =
  ## Empty tokenizer, the tables are filled by the loaders.
  default(BPETokenizer)

#------------------------------------------------------------------------------
# Pattern Matching
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# Byte-Pair Encoding
#------------------------------------------------------------------------------
#
# Benchmarking shows that repeatedly returning/concatenating sequences was too much overhead
# in `bytePairEncode` and in-place construction was necessary.
#
# Note:
#   TTT_METER introduces significant overhead especially for small functions (cache misses + atomic increment on function in/out)
#
# Metering harness for BPETokenizer.encode, run with -d:TTT_METER:
#
# ❯ nim c -r --hints:off --warnings:off --verbosity:0 -d:danger -d:TTT_METER --outdir:build workspace/toktoktok/bench/meter_tokenizer.nim
#
# bench/meter_tokenizer.nim loads a tokenizer, encodes a fixture text in-process
# with BPETokenizer.encode, and reports per {.meter.}-tagged proc:
#
#   - number of calls
#   - throughput in ops/s, total time and average time per call
#
# Metering receipt, Apple M4 Max (the bencher prints no CPU cycle
# columns on this CPU family, the cycle counter is unavailable):
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
# **2026-02-10 6c85b537** - the value-returning bytePairEncode, the
# revision where metering was introduced
#
# |                         Procedures                         |  # of Calls  | Throughput (ops/s) |    Time (µs)     |  Avg Time (µs)   |
# |------------------------------------------------------------|--------------|--------------------|------------------|------------------|
# |bytePairMerge*(piece: seq[byte]; ranks: Table[seq[byte], ...|           662|         1318268.170|           502.174|             0.759|
# |bytePairEncode*(piece: seq[byte]; ranks: Table[seq[byte] ...|           662|             260.505|       2541218.584|          3838.699|
# |splitTextOrdinary(tokenizer: BPETokenizer; text: string) ...|             1|            2458.011|           406.833|           406.833|
# |encodeOrdinary*(tokenizer: BPETokenizer; text: string):  ...|             1|               0.393|       2541807.125|       2541807.125|
# |encodeWithSpecial*(tokenizer: BPETokenizer; text: string ...|             1|               0.393|       2541821.791|       2541821.791|
# |encode*(tokenizer: BPETokenizer; text: string): seq[int]    |             1|               0.393|       2541821.875|       2541821.875|
#
# Result: 3124 tokens encoded
#
# **2026-09-14, master implementation** - the in-place encodedResult
# construction (introduced 2026-02-10 5184fe48)
#
# |                         Procedures                         |  # of Calls  | Throughput (ops/s) |    Time (µs)     |  Avg Time (µs)   |
# |------------------------------------------------------------|--------------|--------------------|------------------|------------------|
# |bytePairMerge(piece: seq[byte]; ranks: Table[seq[byte],  ...|           662|         1013203.790|           653.373|             0.987|
# |bytePairEncode*(encodedResult: var seq[int]; piece: seq[ ...|           662|          916954.658|           721.955|             1.091|
# |splitTextOrdinary(tokenizer: BPETokenizer; text: string) ...|             1|            2406.982|           415.458|           415.458|
# |encodeOrdinaryImpl(encodedResult: var seq[int]; tokenize ...|             1|             740.238|          1350.917|          1350.917|
# |encodeWithSpecialTokens*(tokenizer: BPETokenizer; text:  ...|             1|             736.490|          1357.791|          1357.791|
# |encode*(tokenizer: BPETokenizer; text: string): seq[int]    |             1|             736.400|          1357.958|          1357.958|
#
# Result: 3124 tokens encoded
#
# **2026-09-14, commit ID pending** - the streaming pipeline
# (TokPipeline over the same kimik2.5 ranks and the same Verne text,
# tables warmed, 40 interleaved alternating-order reps against the
# in-place implementation, id streams asserted identical):
#
# |   input   | in-place min | in-place median | streaming min | streaming median |
# |-----------|--------------|-----------------|---------------|------------------|
# | 10K Verne |     0.822 ms |         0.836 ms |      0.283 ms |         0.294 ms |
# | 461KB     |    40.144 ms |        41.380 ms |     16.108 ms |        16.635 ms |
#
# Running the meter produces the report.

proc bytePairMerge(piece: seq[byte], ranks: Table[seq[byte], int]): seq[(int, int)] {.meter.} =
  var parts = newSeqOfCap[(int, int)](piece.len + 2)

  var minRank = MaxInt
  var minRankIdx = 0

  for i in 0..<piece.len - 1:
    let pair = @[piece[i], piece[i+1]]          # TODO drop the per-pair seq allocation
    let rank = ranks.getOrDefault(pair, MaxInt)
    if rank < minRank:
      minRank = rank
      minRankIdx = i
    parts.add((i, rank))

  parts.add((piece.len - 1, MaxInt))
  parts.add((piece.len, MaxInt))

  template getRank(parts: seq[(int, int)], i: int): int =
    ## Get rank for pair starting at parts[i], spanning to parts[i+3] boundary. Captures `ranks` and `piece` Always inlined
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
  ## Naive full-rescan BPE merge of one piece over the rank table,
  ## the emitted ids append to encodedResult.

  if piece.len == 1:
    encodedResult.add(ranks[piece])

  let mergedParts = bytePairMerge(piece, ranks)

  for i in 0 ..< mergedParts.len-1:
    encodedResult.add(ranks[piece[mergedParts[i][0]..<mergedParts[i+1][0]]])

#------------------------------------------------------------------------------
# Tokenizing
#------------------------------------------------------------------------------

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
  # TODO:
  #   text should be a view to avoid alloc
  let pieces = tokenizer.splitTextOrdinary(text)
  for piece in pieces:
    # string and seq[byte] have the same internal repr in Nim, at leat Nim v0, v1 and v2
    # except strings also carry a terminating zero byte (not counted in len)
    let pieceByte = cast[seq[byte]](piece)
    if pieceByte in tokenizer.encoder:
      encodedResult.add(tokenizer.encoder[pieceByte])
    else:
      encodedResult.bytePairEncode(pieceByte, tokenizer.encoder)

proc encodeOrdinary*(tokenizer: BPETokenizer, text: string): seq[int] =
  ## Encodes text with no special-token handling, the ordinary path.
  result.encodeOrdinaryImpl(tokenizer, text)

proc encodeWithSpecialTokens*(tokenizer: BPETokenizer, text: string): seq[int] {.meter.} =
  ## Encodes text with special-token segmentation, one id per special
  ## token and the ordinary path per ordinary slice.
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
        result.encodeOrdinaryImpl(tokenizer, text[pos ..< nextPos]) # TODO view slices
      pos = nextPos
    else:
      result.encodeOrdinaryImpl(tokenizer, text[pos ..< text.len]) # TODO view slices
      break

proc encode*(tokenizer: BPETokenizer, text: string): seq[int] {.meter.} =
  ## Encodes text, special tokens included.
  tokenizer.encodeWithSpecialTokens(text)

proc decodeToBytes(tokenizer: BPETokenizer, tokenIds: openArray[int]): seq[byte] {.meter.} =
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

proc decodeToString*(tokenizer: BPETokenizer, tokenIds: openArray[int]): string {.meter.} =
  ## Decodes ids back to the original text (mergeable ranks first, special ids second, unknown ids raise TokenizerError).
  let bytes = tokenizer.decodeToBytes(tokenIds)
  if bytes.len == 0:
    return ""
  result = newString(bytes.len)
  copyMem(result[0].addr, bytes[0].unsafeAddr, bytes.len)

#------------------------------------------------------------------------------
# Vocabulary loaders
#------------------------------------------------------------------------------

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
    tokenizer.specialPattern = compilePcre2("(?!)")  # the never-matching pattern
    tokenizer.specialMatcher = createMatcher(tokenizer.specialPattern)

  tokenizer

proc loadHFTokenizer*(path: string): BPETokenizer =
  ## Loads an HF tokenizer.json into a codec (deserialize, convert, load).
  if not fileExists(path):
    raise newException(TokenizerError, "HF tokenizer JSON file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerError, "HF tokenizer JSON file is empty: " & path)

  let hf = deserializeHfTokenizer(content)
  let ttk = convertHfToTiktoken(hf)
  loadFromTiktoken(ttk)

proc loadTiktokenizer*(path: string, regexp: TokRegexp): BPETokenizer =
  ## Loads a `.tiktoken` rank file with the given pattern regexp.
  if not fileExists(path):
    raise newException(TokenizerError, "Tiktoken file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerError, "Tiktoken file is empty: " & path)

  let ttk = deserializeTiktokenizer(content, regexp)
  loadFromTiktoken(ttk)

proc loadTiktokenizer*(path: string, regexp: TokRegexp,
    specialTokens: sink OrderedTable[string, int]): BPETokenizer =
  ## Tiktoken rank table with explicit special tokens, loader contract:
  ## - the base64 rank lines of a `.tiktoken` file carry mergeable ranks only.
  ## - the special strings of a checkpoint live beside it, for example the checkpoint tokenizer_config.json added_tokens_decoder block.
  ## - the explicit specialTokens mapping fixes the special-token alternation order.
  if not fileExists(path):
    raise newException(TokenizerError, "Tiktoken file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerError, "Tiktoken file is empty: " & path)

  var ttk = deserializeTiktokenizer(content, regexp)
  ttk.specialTokens = specialTokens
  loadFromTiktoken(ttk)

proc tokenCount*(tokenizer: BPETokenizer): int =
  ## Vocabulary size, mergeable ranks plus special tokens.
  tokenizer.encoder.len + tokenizer.specialTokensEncoder.len