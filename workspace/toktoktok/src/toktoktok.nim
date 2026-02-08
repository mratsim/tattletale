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
import std/base64
import workspace/pcre2/pcre2
import ./serialization

const MaxInt = high(int)

const DefaultPat* = "'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s0-9a-zA-Z]+|\\r?\\n|\\s+(?!\\S)|\\s+"

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

proc compilePcre2(pattern: string, utf8: bool = true): Pcre2Code =
  var errorCode: CompileError
  var errorOffset: csize_t

  let options: Flag[CompileOption] = if utf8: flag(UTF, UCP) else: Flag[CompileOption](0)
  let code = compile(pattern, options, errorCode, errorOffset)
  if code == nil:
    raise newException(TokenizerError, &"PCRE2 compile error {errorCode} at offset {errorOffset}")

  result.code = code
  result.pattern = pattern

proc createMatcher(code: Pcre2Code): Pcre2Matcher =
  result.code = code.code
  result.matchData = match_data_create_from_pattern(code.code, nil)
  if result.matchData == nil:
    raise newException(TokenizerError, "Failed to create match data from pattern")

  result.ovector = get_ovector_pointer(result.matchData)
  result.ovectorCount = get_ovector_count(result.matchData)

proc free(matcher: var Pcre2Matcher) =
  if matcher.matchData != nil:
    match_data_free(matcher.matchData)
    matcher.matchData = nil
    matcher.ovector = nil

proc free(code: var Pcre2Code) =
  if code.code != nil:
    code_free(code.code)
    code.code = nil

proc findAllPcre2(matcher: Pcre2Matcher, text: string, startOffset: int = 0): seq[(int, int)] =
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

    result.add((matchStart, matchEnd))

    offset = matchEnd.csize_t

    if matchStart == matchEnd:
       offset += 1

proc `[]`(t: Table[int, seq[byte]], key: int, default: seq[byte]): seq[byte] =
  if t.hasKey(key): t[key] else: default

proc loadTokenizerFromFormat(format: TiktokenFormat): BPETokenizer =
  var tokenizer = BPETokenizer()

  # Build byte decoder
  tokenizer.byteDecoder = initTable[string, int]()
  for i in 0..255:
    tokenizer.byteDecoder[$char(i)] = i

  # Load special tokens
  if format.specialTokens.len > 0:
    for token, id in format.specialTokens:
      tokenizer.specialTokensEncoder[token] = id
      tokenizer.specialTokensDecoder[id] = toBytes(token)

  # Build encoder/decoder tables
  var encoder = initTable[seq[byte], int]()

  for keyBytes, rank in format.mergeableRanks:
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
  tokenizer.pattern = compilePcre2(format.patStr)
  tokenizer.patternMatcher = createMatcher(tokenizer.pattern)

  # Compile special tokens pattern
  if tokenizer.specialTokensEncoder.len > 0:
    let specialTokens = toSeq(tokenizer.specialTokensEncoder.keys)
    let specialPatternStr = specialTokens.join("|")
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
  let format = convertHfToTiktoken(hf)
  loadTokenizerFromFormat(format)

proc loadTiktokenizer*(path: string): BPETokenizer =
  if not fileExists(path):
    raise newException(TokenizerError, "Tiktoken file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerError, "Tiktoken file is empty: " & path)

  let lines = content.splitLines()
  var mergeableRanks = initOrderedTable[seq[byte], int]()

  for line in lines:
    if line.len == 0:
      continue
    if line.startsWith("#"):
      continue

    let parts = line.split(" ")
    if parts.len >= 2:
      let encodedToken = parts[0]
      let rankStr = parts[1]
      try:
        let rank = parseInt(rankStr)
        let decodedTokenStr = decode(encodedToken)
        let decodedTokenBytes = toBytes(decodedTokenStr)
        if decodedTokenBytes.len > 0:
          mergeableRanks[decodedTokenBytes] = rank
      except:
        discard

  let format = TiktokenFormat(
    mergeableRanks: mergeableRanks,
    patStr: DefaultPat,
    specialTokens: initOrderedTable[string, int]()
  )
  loadTokenizerFromFormat(format)

proc bytePairMerge(piece: seq[byte], ranks: Table[seq[byte], int]): seq[(int, int)] =
  var parts = newSeqOfCap[(int, int)](piece.len + 2)

  var minRank = MaxInt
  var minRankIdx = 0

  for i in 0..<piece.len - 1:
    let pair = @[piece[i], piece[i+1]]
    var rank = MaxInt
    if ranks.hasKey(pair):
      rank = ranks[pair]

    if rank < minRank:
      minRank = rank
      minRankIdx = i

    parts.add((i, rank))

  parts.add((piece.len - 1, MaxInt))
  parts.add((piece.len, MaxInt))

  proc getRank(parts: seq[(int, int)], i: int, piece: seq[byte], ranks: Table[seq[byte], int]): int =
    if i + 3 < parts.len:
      let startIdx = parts[i][0]
      let endIdx = parts[i+3][0]
      var pair: seq[byte] = @[]
      for j in startIdx..<endIdx:
        pair.add(piece[j])
      if ranks.hasKey(pair):
        return ranks[pair]
    return MaxInt

  while minRank != MaxInt:
    let i = minRankIdx

    if i > 0:
      parts[i-1] = (parts[i-1][0], getRank(parts, i-1, piece, ranks))

    parts[i] = (parts[i][0], getRank(parts, i, piece, ranks))
    parts.delete(i + 1)

    minRank = MaxInt
    minRankIdx = 0
    for idx in 0..<parts.len - 1:
      let (_, rank) = parts[idx]
      if rank < minRank:
        minRank = rank
        minRankIdx = idx

  parts

proc bytePairEncode(piece: seq[byte], ranks: Table[seq[byte], int]): seq[int] =
  if piece.len == 1:
    if ranks.hasKey(piece):
      return @[ranks[piece]]
    else:
      return @[]

  let merged = bytePairMerge(piece, ranks)
  var bpeResult: seq[int] = @[]
  for i in 0..<merged.len - 1:
    let startIdx = merged[i][0]
    let endIdx = merged[i+1][0]
    var pair: seq[byte] = @[]
    for j in startIdx..<endIdx:
      pair.add(piece[j])
    if ranks.hasKey(pair):
      bpeResult.add(ranks[pair])

  bpeResult

proc splitTextOrdinary*(tokenizer: BPETokenizer, text: string): seq[string] =
  let pieces = findAllPcre2(tokenizer.patternMatcher, text)

  var lastPos = 0
  for (start, stop) in pieces:
    if start > lastPos:
      result.add(text[lastPos..<start])
    result.add(text[start..<stop])
    lastPos = stop

  if lastPos < text.len:
    result.add(text[lastPos..<text.len])

proc splitTextSpecial*(tokenizer: BPETokenizer, text: string, start: int): tuple[pieces: seq[string], specialPos: int, specialToken: string] =
  var pieces: seq[string] = @[]
  var specialPos = -1
  var specialToken = ""

  var pos = start
  while pos < text.len:
    var found = false
    var nextPos = text.len

    # Use direct string find for special tokens (faster than regex for literals)
    if tokenizer.specialTokensEncoder.len > 0:
      for token, _ in tokenizer.specialTokensEncoder:
        let foundPos = text.find(token, pos)
        if foundPos != -1 and (nextPos == text.len or foundPos < nextPos):
          nextPos = foundPos
          specialToken = token
          found = true

    if found and nextPos == pos:
      pieces.add(specialToken)
      specialPos = pos
      pos = pos + specialToken.len
    elif found:
      if pos < nextPos:
        let ordinary = text[pos ..< nextPos]
        let ordinaryPieces = tokenizer.splitTextOrdinary(ordinary)
        for p in ordinaryPieces:
          pieces.add(p)
      pos = nextPos
    else:
      let remaining = text[pos ..< text.len]
      let remainingPieces = tokenizer.splitTextOrdinary(remaining)
      for p in remainingPieces:
        pieces.add(p)
      break

  (pieces, specialPos, specialToken)

proc encodeOrdinary*(tokenizer: BPETokenizer, text: string): seq[int] =
  result = @[]
  let pieces = tokenizer.splitTextOrdinary(text)
  for piece in pieces:
    let pieceBytes = toBytes(piece)
    if tokenizer.encoder.hasKey(pieceBytes):
      result.add(tokenizer.encoder[pieceBytes])
    else:
      let bpeTokens = bytePairEncode(pieceBytes, tokenizer.encoder)
      for tok in bpeTokens:
        result.add(tok)

proc encodeWithSpecial*(tokenizer: BPETokenizer, text: string): seq[int] =
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
        let ordinary = text[pos ..< nextPos]
        let encoded = tokenizer.encodeOrdinary(ordinary)
        for tok in encoded:
          result.add(tok)
      pos = nextPos
    else:
      let remaining = text[pos ..< text.len]
      let encoded = tokenizer.encodeOrdinary(remaining)
      for tok in encoded:
        result.add(tok)
      break

proc encode*(tokenizer: BPETokenizer, text: string): seq[int] =
  tokenizer.encodeWithSpecial(text)

proc decodeToBytes(tokenizer: BPETokenizer, tokenIds: seq[int]): seq[byte] =
  for id in tokenIds:
    let bytes = tokenizer.decoder[id, @[]]
    if bytes.len > 0:
      result.add(bytes)
    else:
      let specialBytes = tokenizer.specialTokensDecoder[id, @[]]
      if specialBytes.len > 0:
        result.add(specialBytes)
      else:
        raise newException(TokenizerError, "Invalid token: " & $id)

proc decodeToString*(tokenizer: BPETokenizer, tokenIds: seq[int]): string =
  let bytes = tokenizer.decodeToBytes(tokenIds)
  if bytes.len == 0:
    return ""
  result = newString(bytes.len)
  copyMem(result[0].addr, bytes[0].unsafeAddr, bytes.len)

proc tokenCount*(tokenizer: BPETokenizer): int =
  tokenizer.encoder.len + tokenizer.specialTokensEncoder.len

proc isSpecialToken*(tokenizer: BPETokenizer, token: int): bool =
  tokenizer.specialTokensDecoder.hasKey(token)

proc decodeToken*(tokenizer: BPETokenizer, tokenId: int): seq[byte] =
  if tokenizer.decoder.hasKey(tokenId):
    return tokenizer.decoder[tokenId]
  elif tokenizer.specialTokensDecoder.hasKey(tokenId):
    return tokenizer.specialTokensDecoder[tokenId]
  else:
    raise newException(TokenizerError, "Invalid token: " & $tokenId)

proc init*(_: type BPETokenizer): BPETokenizer =
  BPETokenizer(
    encoder: initTable[seq[byte], int](),
    decoder: initTable[int, seq[byte]](),
    specialTokensEncoder: initTable[string, int](),
    specialTokensDecoder: initTable[int, seq[byte]](),
    pattern: Pcre2Code(code: nil, pattern: ""),
    patternMatcher: Pcre2Matcher(code: nil, matchData: nil, ovector: nil, ovectorCount: 0),
    specialPattern: Pcre2Code(code: nil, pattern: ""),
    specialMatcher: Pcre2Matcher(code: nil, matchData: nil, ovector: nil, ovectorCount: 0),
    cache: initTable[seq[byte], seq[int]](),
    byteDecoder: initTable[string, int]()
  )

proc free*(tokenizer: var BPETokenizer) =
  tokenizer.specialMatcher.free()
  tokenizer.specialPattern.free()
  tokenizer.patternMatcher.free()
  tokenizer.pattern.free()
