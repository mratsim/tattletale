# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/tables
import std/os
import pkg/regex
import std/strutils
import std/sequtils
import ./serialization

const MaxInt = high(int)

proc createByteDecoder(): Table[string, int] =
  result = initTable[string, int]()

  result["!"] = 33
  result["\""] = 34
  result["#"] = 35
  result["$"] = 36
  result["%"] = 37
  result["&"] = 38
  result["'"] = 39
  result["("] = 40
  result[")"] = 41
  result["*"] = 42
  result["+"] = 43
  result[","] = 44
  result["-"] = 45
  result["."] = 46
  result["/"] = 47
  result["0"] = 48
  result["1"] = 49
  result["2"] = 50
  result["3"] = 51
  result["4"] = 52
  result["5"] = 53
  result["6"] = 54
  result["7"] = 55
  result["8"] = 56
  result["9"] = 57
  result[":"] = 58
  result[";"] = 59
  result["<"] = 60
  result["="] = 61
  result[">"] = 62
  result["?"] = 63
  result["@"] = 64
  result["A"] = 65
  result["B"] = 66
  result["C"] = 67
  result["D"] = 68
  result["E"] = 69
  result["F"] = 70
  result["G"] = 71
  result["H"] = 72
  result["I"] = 73
  result["J"] = 74
  result["K"] = 75
  result["L"] = 76
  result["M"] = 77
  result["N"] = 78
  result["O"] = 79
  result["P"] = 80
  result["Q"] = 81
  result["R"] = 82
  result["S"] = 83
  result["T"] = 84
  result["U"] = 85
  result["V"] = 86
  result["W"] = 87
  result["X"] = 88
  result["Y"] = 89
  result["Z"] = 90
  result["["] = 91
  result["\\"] = 92
  result["]"] = 93
  result["^"] = 94
  result["_"] = 95
  result["`"] = 96
  result["a"] = 97
  result["b"] = 98
  result["c"] = 99
  result["d"] = 100
  result["e"] = 101
  result["f"] = 102
  result["g"] = 103
  result["h"] = 104
  result["i"] = 105
  result["j"] = 106
  result["k"] = 107
  result["l"] = 108
  result["m"] = 109
  result["n"] = 110
  result["o"] = 111
  result["p"] = 112
  result["q"] = 113
  result["r"] = 114
  result["s"] = 115
  result["t"] = 116
  result["u"] = 117
  result["v"] = 118
  result["w"] = 119
  result["x"] = 120
  result["y"] = 121
  result["z"] = 122
  result["{"] = 123
  result["|"] = 124
  result["}"] = 125
  result["~"] = 126
  result["¡"] = 161
  result["¢"] = 162
  result["£"] = 163
  result["¤"] = 164
  result["¥"] = 165
  result["¦"] = 166
  result["§"] = 167
  result["¨"] = 168
  result["©"] = 169
  result["ª"] = 170
  result["«"] = 171
  result["¬"] = 172
  result["®"] = 174
  result["¯"] = 175
  result["°"] = 176
  result["±"] = 177
  result["²"] = 178
  result["³"] = 179
  result["´"] = 180
  result["µ"] = 181
  result["¶"] = 182
  result["·"] = 183
  result["¸"] = 184
  result["¹"] = 185
  result["º"] = 186
  result["»"] = 187
  result["¼"] = 188
  result["½"] = 189
  result["¾"] = 190
  result["¿"] = 191
  result["À"] = 192
  result["Á"] = 193
  result["Â"] = 194
  result["Ã"] = 195
  result["Ä"] = 196
  result["Å"] = 197
  result["Æ"] = 198
  result["Ç"] = 199
  result["È"] = 200
  result["É"] = 201
  result["Ê"] = 202
  result["Ë"] = 203
  result["Ì"] = 204
  result["Í"] = 205
  result["Î"] = 206
  result["Ï"] = 207
  result["Ð"] = 208
  result["Ñ"] = 209
  result["Ò"] = 210
  result["Ó"] = 211
  result["Ô"] = 212
  result["Õ"] = 213
  result["Ö"] = 214
  result["×"] = 215
  result["Ø"] = 216
  result["Ù"] = 217
  result["Ú"] = 218
  result["Û"] = 219
  result["Ü"] = 220
  result["Ý"] = 221
  result["Þ"] = 222
  result["ß"] = 223
  result["à"] = 224
  result["á"] = 225
  result["â"] = 226
  result["ã"] = 227
  result["ä"] = 228
  result["å"] = 229
  result["æ"] = 230
  result["ç"] = 231
  result["è"] = 232
  result["é"] = 233
  result["ê"] = 234
  result["ë"] = 235
  result["ì"] = 236
  result["í"] = 237
  result["î"] = 238
  result["ï"] = 239
  result["ð"] = 240
  result["ñ"] = 241
  result["ò"] = 242
  result["ó"] = 243
  result["ô"] = 244
  result["õ"] = 245
  result["ö"] = 246
  result["÷"] = 247
  result["ø"] = 248
  result["ù"] = 249
  result["ú"] = 250
  result["û"] = 251
  result["ü"] = 252
  result["ý"] = 253
  result["þ"] = 254
  result["ÿ"] = 255

  result["Ā"] = 0
  result["ā"] = 1
  result["Ă"] = 2
  result["ă"] = 3
  result["Ą"] = 4
  result["ą"] = 5
  result["Ć"] = 6
  result["ć"] = 7
  result["Ĉ"] = 8
  result["ĉ"] = 9
  result["Ċ"] = 10
  result["ċ"] = 11
  result["Č"] = 12
  result["č"] = 13
  result["Ď"] = 14
  result["ď"] = 15
  result["Đ"] = 16
  result["đ"] = 17
  result["Ē"] = 18
  result["ē"] = 19
  result["Ĕ"] = 20
  result["ĕ"] = 21
  result["Ė"] = 22
  result["ė"] = 23
  result["Ę"] = 24
  result["ę"] = 25
  result["Ě"] = 26
  result["ě"] = 27
  result["Ĝ"] = 28
  result["ĝ"] = 29
  result["Ğ"] = 30
  result["ğ"] = 31
  result["Ġ"] = 32
  result["ġ"] = 33
  result["Ģ"] = 34
  result["ģ"] = 35
  result["Ĥ"] = 36
  result["ĥ"] = 37
  result["Ħ"] = 38
  result["ħ"] = 39
  result["Ĩ"] = 40
  result["ĩ"] = 41
  result["Ī"] = 42
  result["ī"] = 43
  result["Ĭ"] = 44
  result["ĭ"] = 45
  result["Į"] = 46
  result["į"] = 47
  result["İ"] = 48
  result["ı"] = 49
  result["Ĳ"] = 50
  result["ĳ"] = 51
  result["Ĵ"] = 52
  result["ĵ"] = 53
  result["Ķ"] = 54
  result["ķ"] = 55
  result["ĸ"] = 56
  result["Ĺ"] = 57
  result["ĺ"] = 58
  result["Ļ"] = 59
  result["ļ"] = 60
  result["Ľ"] = 61
  result["ľ"] = 62
  result["Ŀ"] = 63
  result["ŀ"] = 64
  result["Ł"] = 65
  result["ł"] = 66
  result["Ń"] = 67
  result["ń"] = 68
  result["Ņ"] = 69
  result["ņ"] = 70
  result["ň"] = 71
  result["ŉ"] = 72
  result["Ŋ"] = 73
  result["ŋ"] = 74
  result["Ō"] = 75
  result["ō"] = 76
  result["Ŏ"] = 77
  result["ŏ"] = 78
  result["Ő"] = 79
  result["ő"] = 80
  result["Œ"] = 81
  result["œ"] = 82
  result["Ŕ"] = 83
  result["ŕ"] = 84
  result["Ŗ"] = 85
  result["ŗ"] = 86
  result["Ř"] = 87
  result["ř"] = 88
  result["Ś"] = 89
  result["ś"] = 90
  result["Ŝ"] = 91
  result["ŝ"] = 92
  result["Ş"] = 93
  result["ş"] = 94
  result["Š"] = 95
  result["š"] = 96
  result["Ţ"] = 97
  result["ţ"] = 98
  result["Ť"] = 99
  result["ť"] = 100
  result["Ŧ"] = 101
  result["ŧ"] = 102
  result["Ũ"] = 103
  result["ũ"] = 104
  result["Ū"] = 105
  result["ū"] = 106
  result["Ŭ"] = 107
  result["ŭ"] = 108
  result["Ů"] = 109
  result["ů"] = 110
  result["Ű"] = 111
  result["ű"] = 112
  result["Ų"] = 113
  result["ų"] = 114
  result["Ŵ"] = 115
  result["ŵ"] = 116
  result["Ŷ"] = 117
  result["ŷ"] = 118
  result["Ÿ"] = 119
  result["Ź"] = 120
  result["ź"] = 121
  result["Ż"] = 122
  result["ż"] = 123
  result["Ž"] = 124
  result["ž"] = 125
  result["ŀ"] = 126
  result["ⱀ"] = 127
  result["ⱁ"] = 128
  result["ⱂ"] = 129
  result["ⱃ"] = 130
  result["ⱄ"] = 131
  result["ⱅ"] = 132
  result["ⱆ"] = 133
  result["ⱇ"] = 134
  result["ⱈ"] = 135
  result["ⱉ"] = 136
  result["ⱊ"] = 137
  result["ⱋ"] = 138
  result["ⱌ"] = 139
  result["ⱍ"] = 140
  result["ⱎ"] = 141
  result["ⱏ"] = 142
  result["ⱐ"] = 143
  result["ⱑ"] = 144
  result["ⱒ"] = 145
  result["ⱓ"] = 146
  result["ⱔ"] = 147
  result["ⱕ"] = 148
  result["ⱖ"] = 149
  result["ⱗ"] = 150
  result["ⱘ"] = 151
  result["ⱙ"] = 152
  result["ⱚ"] = 153
  result["ⱛ"] = 154
  result["ⱜ"] = 155
  result["ⱝ"] = 156
  result["ⱞ"] = 157
  result["ⱟ"] = 158
  result["Ⱡ"] = 159
  result["ⱡ"] = 160
  result["Ɫ"] = 161
  result["Ᵽ"] = 162
  result["Ɽ"] = 163
  result["ⱥ"] = 164
  result["ⱦ"] = 165
  result["Ⱨ"] = 166
  result["ⱨ"] = 167
  result["Ⱪ"] = 168
  result["ⱪ"] = 169
  result["Ⱬ"] = 170
  result["ⱬ"] = 171
  result["Ɑ"] = 172
  result["Ɱ"] = 173
  result["Ɐ"] = 174
  result["Ɒ"] = 175
  result["ⱱ"] = 176
  result["Ⱳ"] = 177
  result["ⱳ"] = 178
  result["ⱴ"] = 179
  result["Ⱶ"] = 180
  result["ⱶ"] = 181
  result["ⱷ"] = 182
  result["ⱸ"] = 183
  result["ⱹ"] = 184
  result["ⱺ"] = 185
  result["ⱻ"] = 186
  result["ⱼ"] = 187
  result["ⱽ"] = 188
  result["Ȿ"] = 189
  result["Ɀ"] = 190
  result["Ⲁ"] = 191
  result["ⲁ"] = 192
  result["Ⲃ"] = 193
  result["ⲃ"] = 194
  result["Ⲅ"] = 195
  result["ⲅ"] = 196
  result["Ⲇ"] = 197
  result["ⲇ"] = 198
  result["Ⲉ"] = 199
  result["ⲉ"] = 200
  result["Ⲋ"] = 201
  result["ⲋ"] = 202
  result["Ⲍ"] = 203
  result["ⲍ"] = 204
  result["Ⲏ"] = 205
  result["ⲏ"] = 206
  result["Ⲑ"] = 207
  result["ⲑ"] = 208
  result["Ⲓ"] = 209
  result["ⲓ"] = 210
  result["Ⲕ"] = 211
  result["ⲕ"] = 212
  result["Ⲗ"] = 213
  result["ⲗ"] = 214
  result["Ⲙ"] = 215
  result["ⲙ"] = 216
  result["Ⲛ"] = 217
  result["ⲛ"] = 218
  result["Ⲝ"] = 219
  result["ⲝ"] = 220
  result["Ⲟ"] = 221
  result["ⲟ"] = 222
  result["Ⲡ"] = 223
  result["ⲡ"] = 224
  result["Ⲣ"] = 225
  result["ⲣ"] = 226
  result["Ⲥ"] = 227
  result["ⲥ"] = 228
  result["Ⲧ"] = 229
  result["ⲧ"] = 230
  result["Ⲩ"] = 231
  result["ⲩ"] = 232
  result["Ⲫ"] = 233
  result["ⲫ"] = 234
  result["Ⲭ"] = 235
  result["ⲭ"] = 236
  result["Ⲯ"] = 237
  result["ⲯ"] = 238
  result["Ⲱ"] = 239
  result["ⲱ"] = 240
  result["Ⲳ"] = 241
  result["ⲳ"] = 242
  result["Ⲵ"] = 243
  result["ⲵ"] = 244
  result["Ⲷ"] = 245
  result["ⲷ"] = 246
  result["Ⲹ"] = 247
  result["ⲹ"] = 248
  result["Ⲻ"] = 249
  result["ⲻ"] = 250
  result["Ⲽ"] = 251
  result["ⲽ"] = 252
  result["Ⲿ"] = 253
  result["ⲿ"] = 254
  result["Ⳁ"] = 255

type
  BPETokenizer* = object
    encoder*: Table[seq[byte], int]
    decoder*: Table[int, seq[byte]]
    specialTokensEncoder*: Table[string, int]
    specialTokensDecoder*: Table[int, seq[byte]]
    pattern*: Regex2
    specialRegex*: Regex2
    cache*: Table[seq[byte], seq[int]]
    byteDecoder*: Table[string, int]

  TokenizerError* = object of ValueError

proc init*(_: type BPETokenizer): BPETokenizer =
  BPETokenizer(
    encoder: initTable[seq[byte], int](),
    decoder: initTable[int, seq[byte]](),
    specialTokensEncoder: initTable[string, int](),
    specialTokensDecoder: initTable[int, seq[byte]](),
    pattern: re2(""),
    specialRegex: re2(""),
    cache: initTable[seq[byte], seq[int]](),
    byteDecoder: initTable[string, int]()
  )

proc toBytes(str: string): seq[byte] =
  result = newSeq[byte](str.len)
  for i in 0..<str.len:
    result[i] = byte(ord(str[i]))

proc strKeyToBytes(keyStr: string): seq[byte] =
  result = newSeq[byte]()
  if keyStr.startsWith("[") and keyStr.endsWith("]"):
    let inner = keyStr[1..^2]
    for part in inner.split(", "):
      if part.len > 0:
        try:
          result.add(byte(parseInt(part)))
        except:
          return newSeq[byte]()
  else:
    for c in keyStr:
      result.add(byte(ord(c)))

proc loadTokenizerFromFormat*(format: TiktokenFormat): BPETokenizer =
  var tokenizer = BPETokenizer.init()

  tokenizer.byteDecoder = createByteDecoder()

  if format.specialTokens.len > 0:
    for token, id in format.specialTokens:
      tokenizer.specialTokensEncoder[token] = id

  var encoder = initTable[seq[byte], int]()

  for keyStr, rank in format.mergeableRanks:
    let rawBytes = strKeyToBytes(keyStr)
    if rawBytes.len > 0:
      encoder[rawBytes] = rank

  tokenizer.encoder = encoder

  for k, v in encoder:
    tokenizer.decoder[v] = k

  tokenizer.pattern = re2(format.patStr)

  if tokenizer.specialTokensEncoder.len > 0:
    let specialKeys = toSeq(tokenizer.specialTokensEncoder.keys)
    let specialPattern = specialKeys.join("|")
    tokenizer.specialRegex = re2(specialPattern)

  tokenizer

proc loadTokenizerJSON*(path: string): BPETokenizer =
  if not fileExists(path):
    raise newException(TokenizerError, "HF tokenizer JSON file not found: " & path)

  let content = readFile(path)
  if content.len == 0:
    raise newException(TokenizerError, "HF tokenizer JSON file is empty: " & path)

  let hf = deserializeHfTokenizer(content)
  let format = convertHfToTiktoken(hf)
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

proc splitTextOrdinary(tokenizer: BPETokenizer, text: string): seq[string] =
  result = @[]
  let matches = findAll(text, tokenizer.pattern)
  for match in matches:
    let piece = text[match.boundaries]
    if piece.len > 0:
      result.add(piece)

proc splitTextSpecial*(tokenizer: BPETokenizer, text: string, start: int): tuple[pieces: seq[string], specialPos: int, specialToken: string] =
  var pieces: seq[string] = @[]
  var specialPos = -1
  var specialToken = ""

  var pos = start
  while pos < text.len:
    var found = false
    var nextPos = text.len

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
  let pieces = tokenizer.splitTextOrdinary(text)
  for piece in pieces:
    let pieceBytes = toBytes(piece)
    if tokenizer.encoder.hasKey(pieceBytes):
      result.add(tokenizer.encoder[pieceBytes])
    else:
      let bpeTokens = bytePairEncode(pieceBytes, tokenizer.encoder)
      for tok in bpeTokens:
        result.add(tok)

proc encodeWithSpecial(tokenizer: BPETokenizer, text: string): seq[int] =
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
    if tokenizer.decoder.hasKey(id):
      let bytes = tokenizer.decoder[id]
      for b in bytes:
        result.add(b)
    elif tokenizer.specialTokensDecoder.hasKey(id):
      let bytes = tokenizer.specialTokensDecoder[id]
      for b in bytes:
        result.add(b)
    else:
      raise newException(TokenizerError, "Invalid token: " & $id)

proc decodeToString*(tokenizer: BPETokenizer, tokenIds: seq[int]): string =
  let bytes = tokenizer.decodeToBytes(tokenIds)
  result = newStringOfCap(bytes.len)
  for b in bytes:
    result.add(chr(int(b)))

proc tokenCount*(tokenizer: BPETokenizer): int = tokenizer.encoder.len + tokenizer.specialTokensEncoder.len

proc isSpecialToken*(tokenizer: BPETokenizer, token: int): bool =
  tokenizer.specialTokensDecoder.hasKey(token)

proc decodeToken*(tokenizer: BPETokenizer, tokenId: int): seq[byte] =
  if tokenizer.decoder.hasKey(tokenId):
    return tokenizer.decoder[tokenId]
  elif tokenizer.specialTokensDecoder.hasKey(tokenId):
    return tokenizer.specialTokensDecoder[tokenId]
  else:
    raise newException(TokenizerError, "Invalid token: " & $tokenId)
