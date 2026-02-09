import std/unittest
import std/base64
import std/tables
import std/os
import std/sequtils
import std/json
import std/strutils

import workspace/toktoktok {.all.}

const FixtureFilePath = currentSourcePath().parentDir() / "fixtures" / "bytepairmerge" / "bytepairmerge.json"

proc b64DecodeToBytes*(b64_str: string): seq[byte] =
  let decoded = decode(b64_str)
  result = newSeq[byte](decoded.len)
  for i in 0..<decoded.len:
    result[i] = byte(decoded[i])

type
  Fixture* = object
    description*: string
    inputBytes*: seq[byte]
    ranks*: Table[seq[byte], int]
    expectedTokens*: seq[int]

proc parseFixture*(node: JsonNode): Fixture =
  result.description = node["description"].getStr()
  result.inputBytes = node["input_bytes"].getElems().mapIt(it.getInt().byte)
  result.ranks = initTable[seq[byte], int]()
  for k, v in node["ranks"]:
    let keyBytes = b64DecodeToBytes(k)
    result.ranks[keyBytes] = v.getInt()
  result.expectedTokens = node["expected_tokens"].getElems().mapIt(it.getInt())

proc runBytePairMergeTests*() =
  suite "Byte Pair Merge Fixtures":
    let content = readFile(FixtureFilePath)
    let fixtureNodes = parseJson(content).getElems()

    for fixtureNode in fixtureNodes:
      let fixture = parseFixture(fixtureNode)

      test fixture.description:
        let ranks = fixture.ranks
        let piece = fixture.inputBytes
        let result = bytePairEncode(piece, ranks)
        check result == fixture.expectedTokens

when isMainModule:
  runBytePairMergeTests()
