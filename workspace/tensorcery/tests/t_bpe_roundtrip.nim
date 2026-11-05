# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## BPE codec round-trip de-risk for the microgpt tokenizer.
##
## Test that encode/decodeToString round-trips Shakespeare text through the pure-Nim BPE codec.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_bpe_roundtrip.nim

import std/[base64, json, os, sequtils, strutils, tables]

import workspace/toktoktok
import workspace/zstd/zstd_highlevel

const
  FixturePath = currentSourcePath().parentDir().parentDir().parentDir() /
    "toktoktok" / "tests" / "fixtures" / "bytepairmerge" / "bytepairmerge"

  # Shakespeare corpus excerpt (arraymancer ex06_shakespeare_input.txt, opening lines),
  # inlined so the de-risk runs without an external file.
  CorpusExcerpt = "First Citizen:\nBefore we proceed any further, hear me speak.\n\n" &
    "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n\n" &
    "All:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n"

type
  CodecFixture = object
    description: string
    inputBytes: seq[byte]
    ranks: Table[seq[byte], int]
    expectedTokens: seq[int]

proc parseFixture(node: JsonNode): CodecFixture =
  result.description = node["description"].getStr()
  for e in node["input_bytes"]:
    result.inputBytes.add byte(e.getInt())
  for key, value in node["ranks"]:
    let decoded = decode(key)
    var keyBytes: seq[byte]
    for c in decoded:
      keyBytes.add byte(c)
    result.ranks[keyBytes] = value.getInt()
  for e in node["expected_tokens"]:
    result.expectedTokens.add e.getInt()

proc writeTiktokenFile(ranks: Table[seq[byte], int], path: string) =
  ## Rank table in the tiktoken text format: one base64-encoded byte token and its rank per line.
  ## The 256 single-byte fillers make every span encodable.
  var lines: seq[string]
  for b in 0 ..< 256:
    lines.add(encode(@[byte(b)]) & " " & $(1000000 + b))
  for keyBytes, rank in ranks:
    if keyBytes.len > 1:
      var raw = ""
      for b in keyBytes:
        raw.add char(b)
      lines.add(encode(raw) & " " & $rank)
  writeFile(path, lines.join("\n"))

proc main =
  let content = readFile(FixturePath & ".json.zst").zstdDecompress(string)
  let fixtures = parseJson(content).getElems().mapIt(parseFixture(it))

  let ranksPath = getTempDir() / "t_bpe_roundtrip_ranks.tiktoken"
  writeTiktokenFile(fixtures[0].ranks, ranksPath)
  let tokenizer = loadTiktokenizer(ranksPath, R50kRegexp)

  # Recorded merge table: the fixture text encodes to the fixture's recorded ids.
  for fixture in fixtures:
    if fixture.description == "Simple ASCII text 'hello'":
      var text = ""
      for b in fixture.inputBytes:
        text.add char(b)
      doAssert tokenizer.encode(text) == fixture.expectedTokens,
          "recorded ids mismatch on: " & fixture.description

  # Corpus round-trip: decode(encode(text)) == text through the full codec path.
  let ids = tokenizer.encode(CorpusExcerpt)
  doAssert ids.len > 0, "encode produced no tokens"
  doAssert tokenizer.decodeToString(ids) == CorpusExcerpt,
      "decode(encode(text)) != text over the Shakespeare excerpt"
  removeFile(ranksPath)

main()
