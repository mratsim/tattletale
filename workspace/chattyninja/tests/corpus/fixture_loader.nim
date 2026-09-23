# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fixture loader for the integration suites. Reads the recorded corpus rows under
## tests/corpus/<suite>/ and turns each one into a ready ChatRenderRequest plus the recorded
## ground truth (rendered bytes, generation codepoint spans, expected error).
##
## A suite is exactly its directory:
## - the `*.json.zst` stems are the rows, the stem naming its row
## - the suite's `<suite>.jinja` is the template
##
## The zstd reader is the workspace/zstd high-level one-shot decompress. The JSON bridge
## preserves JSON object insertion order.
##
## Consumers observe dict order through tojson, items and plain iteration.
## Recorded bytes are read-only here, no loader path writes into the corpus tree.

import std/[json, os, options, tables, algorithm, strutils]
import workspace/zstd/zstd_highlevel
import jinja_data_model {.all.}

const
  FixturesDir* = currentSourcePath().parentDir()
    ## the corpus tree this module lives in, tests/corpus
  RowSuffix = ".json.zst"

type
  ChatRenderRequest* = object
    ## One recorded render request, the HF-standard inputs in the engine's value model.
    messages*: JinjaVal
    tools*: JinjaVal
      ## the none value when the recording passed no tools
    documents*: JinjaVal
      ## the none value when the recording passed no documents
    addGenerationPrompt*: bool
    kwargs*: seq[tuple[k: string, v: JinjaVal]]
      ## the template kwargs as key/value pairs in recording order, the renderer appending
      ## them after the standard keys
    clockEpoch*: float64
      ## the epoch `strftime_now` reads, 0 when the row carries none

  FixtureRow* = object
    ## One recorded golden row, ready to render and ready to compare.
    suite*: string
    row*: string
    request*: ChatRenderRequest
    rendered*: string
      ## the recorded ground truth, empty on expected-error rows
    generationSpans*: seq[tuple[start, stop: int]]
      ## recorded codepoint spans, empty when the template has none
    expectError*: bool
    errorMessage*: string
      ## recorded message, carried verbatim into the comparison

func jsonToValue(n: JsonNode): JinjaVal =
  ## Recorded JSON into engine values. Objects keep their insertion order,
  ## numbers keep the int/float split, null becomes the none value, exactly
  ## the tiers the two-tier value model distinguishes.
  case n.kind
  of JObject:
    var fields: seq[tuple[k: string, v: JinjaVal]]
    for key, child in n.fields:
      fields.add (key, jsonToValue(child))
    dictVal(fields)
  of JArray:
    var items = newSeq[JinjaVal](n.len)
    var i = 0
    for child in n.items:
      items[i] = jsonToValue(child)
      inc i
    seqVal(items)
  of JString: strVal(n.getStr())
  of JInt: intVal(n.getInt())
  of JFloat: floatVal(n.getFloat())
  of JBool: boolVal(n.getBool())
  of JNull: noneVal()

func asList(n: JsonNode): JinjaVal =
  ## An optional list input. Absent key or null renders as the none value, the way the HF render
  ## path passes a missing tools or documents argument through as None.
  if n.isNil or n.kind == JNull:
    noneVal()
  else:
    jsonToValue(n)

func spanPairs(n: JsonNode): seq[tuple[start, stop: int]] =
  result = newSeq[tuple[start, stop: int]](n.len)
  var i = 0
  for pair in n.items:
    result[i] = (pair[0].getInt(), pair[1].getInt())
    inc i

proc stemOf(path: string, suffix: string): string =
  ## File stem without the trailing suffix, so `default.json.zst` maps to `default`.
  let name = lastPathPart(path)
  doAssert name.endsWith(suffix)
  name[0 ..< name.len - suffix.len]

proc suiteRowNames*(suite: string): seq[string] =
  ## Row stems of one suite, the `*.json.zst` stems of the suite directory, sorted.
  for path in walkPattern(FixturesDir / suite / ("*" & RowSuffix)):
    result.add stemOf(path, RowSuffix)
  result.sort()

proc suiteTemplateSource*(suite: string): string =
  ## Byte-exact template copy recorded beside the suite, the file name
  ## carrying the family, read-only.
  readFile(FixturesDir / suite / suite & ".jinja")

proc loadRow*(suite: string, row: string): FixtureRow =
  ## One row payload in, one render-ready row out, the stem naming the row.
  ## Contract:
  ## - the row payload decompresses to its recorded JSON
  ## - kwargs travel as a separate dict, the renderer appends them after the standard keys
  let dir = FixturesDir / suite
  let payload = parseJson(zstdDecompress(readFile(
      dir / (row & RowSuffix)), string))
  result.suite = suite
  result.row = row
  var kwargsPairs: seq[tuple[k: string, v: JinjaVal]]
  for key, child in pairs(payload["kwargs"]):
    kwargsPairs.add (key, jsonToValue(child))
  result.request = ChatRenderRequest(
    messages: jsonToValue(payload["messages"]),
    tools: asList(payload.getOrDefault("tools")),
    documents: asList(payload.getOrDefault("documents")),
    addGenerationPrompt: payload["add_generation_prompt"].getBool(),
    kwargs: kwargsPairs,
    clockEpoch:
      if payload.hasKey("epoch"): float64(payload["epoch"].getFloat())
      else: 0.0)
  if payload.hasKey("expected_error"):
    result.expectError = true
    result.errorMessage = payload["expected_error"]["message"].getStr()
  else:
    result.rendered = payload["rendered"].getStr()
    result.generationSpans = spanPairs(payload["generation_spans"])

proc loadSuite*(suite: string): seq[FixtureRow] =
  ## Every recorded row of one suite, sorted row order.
  for row in suiteRowNames(suite):
    result.add loadRow(suite, row)
