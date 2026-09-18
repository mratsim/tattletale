# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fixture loader for the integration suites. Reads the recorded golden frames under
## tests/fixtures/<suite>/ and turns each one into a ready ChatRenderRequest plus the recorded
## ground truth (rendered bytes, generation codepoint spans, expected error).
##
## A suite is exactly its directory:
## - the `*.json.zst` stems are the rows
## - the per-row `.meta.json` sidecar carries the row id and template sha256
## - loadRow cross-checks both against the frame payload
##
## The zstd reader is the workspace/zstd high-level one-shot decompress. The JSON bridge keeps
## insertion order because dict order is observable through tojson, items and plain iteration.
## Recorded bytes are read-only here, no loader path writes into the fixtures tree.

import std/[json, os, options, tables, algorithm, strutils]
import workspace/zstd/zstd_highlevel
import workspace/chattyninja

const
  FixturesDir* = currentSourcePath().parentDir()
    ## the fixtures tree this module lives inside, one step from the source
  FrameSuffix = ".json.zst"
  SidecarSuffix = ".meta.json"
  ModelsDir* = joinPath(FixturesDir, "..", "..", "..", "..", "..", "..",
    "MODELS")
    ## the read-only roster copy the fixtures were recorded from. MODELS sits two levels above
    ## the project root. Exact path resolution only, never written

type
  FixtureRow* = object
    ## One recorded golden row, ready to render and ready to compare.
    suite*: string
    row*: string
    templateSha*: string
    request*: ChatRenderRequest
    rendered*: string
      ## the recorded ground truth, empty on expected-error rows
    generationSpans*: seq[tuple[start, stop: int]]
      ## recorded codepoint spans, empty when the template has none
    expectError*: bool
    errorKind*: string
      ## recorded exception class name, e.g. the jinja TemplateError
    errorMessage*: string
      ## recorded message, carried verbatim into the comparison

func jsonToValue(n: JsonNode): Value =
  ## Recorded JSON into engine values. Objects keep their insertion order,
  ## numbers keep the int/float split, null becomes the none value, exactly
  ## the tiers the two-tier value model distinguishes.
  case n.kind
  of JObject:
    var entries = newSeq[DictEntry]()
    for key, child in n.fields:
      entries.add (key, jsonToValue(child))
    dictVal(entries)
  of JArray:
    var items = newSeq[Value](n.len)
    var i = 0
    for child in n.items:
      items[i] = jsonToValue(child)
      inc i
    seqVal(items)
  of JString: strVal(n.getStr())
  of JInt: intVal(n.getInt())
  of JFloat: floatVal(n.getFloat())
  of JBool: boolVal(n.getBool())
  of JNull: nullVal()

func asList(n: JsonNode): Value =
  ## An optional list input. Absent key or null renders as the none value, the way the HF render
  ## path passes a missing tools or documents argument through as None.
  if n.isNil or n.kind == JNull:
    nullVal()
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
  ## Row stems of one suite, the `*.json.zst` stems of the suite directory, sorted. The sidecar list mirrors it with one `.meta.json` per frame.
  for path in walkPattern(FixturesDir / suite / ("*" & FrameSuffix)):
    result.add stemOf(path, FrameSuffix)
  result.sort()

proc firstSidecarSha(suite: string): string =
  ## template_sha256 off the suite's first sidecar (sorted stems), the one
  ## provenance field the sidecars keep.
  var names: seq[string]
  for path in walkPattern(FixturesDir / suite / ("*" & SidecarSuffix)):
    names.add stemOf(path, SidecarSuffix)
  if names.len == 0:
    raise ValueError.newException(
      "suite " & suite & ": no sidecar under " & (FixturesDir / suite))
  names.sort()
  parseFile(FixturesDir / suite / (names[0] & SidecarSuffix))["template_sha256"]
    .getStr()

proc suiteTemplateSha*(suite: string): string =
  ## sha256 of the template bytes the suite was recorded from, read
  ## from the row sidecars (every row of a suite carries the same one).
  firstSidecarSha(suite)

proc modelTokenizer*(modelDir: string): string =
  ## Exact path of one roster checkpoint tokenizer json under MODELS.
  joinPath(ModelsDir, modelDir, "tokenizer.json")

proc suiteTemplateSource*(suite: string): string =
  ## Byte-exact template copy recorded beside the suite, the file name
  ## carrying the family, read-only.
  readFile(FixturesDir / suite / suite & ".jinja")

proc loadRow*(suite: string, row: string): FixtureRow =
  ## One frame in, one render-ready row out.
  ## Contract:
  ## - the frame decompresses to its recorded JSON payload
  ## - the inputs build the request with the HF layering in mind, kwargs travel separately because
  ##   the renderer appends them after the standard keys
  ## - the sidecar is the row's provenance. Its row id and template sha256 must match the frame payload,
  ##   and a disagreement is a corrupted fixture that stops the load
  let dir = FixturesDir / suite
  let sidecar = parseFile(dir / (row & SidecarSuffix))
  let frame = parseJson(zstdDecompress(readFile(
      dir / (row & FrameSuffix)), string))
  if sidecar["row"].getStr() != row or frame["row"].getStr() != row:
    raise ValueError.newException(
      "fixture " & suite & "/" & row & ": sidecar/frame row id mismatch")
  let sha = sidecar["template_sha256"].getStr()
  if frame["template"]["sha256"].getStr() != sha:
    raise ValueError.newException(
      "fixture " & suite & "/" & row & ": template sha mismatch")
  result.suite = suite
  result.row = row
  result.templateSha = sha
  var kwargs = newSeq[DictEntry]()
  for key, child in pairs(frame["kwargs"]):
    kwargs.add (key, jsonToValue(child))
  result.request = ChatRenderRequest(
    messages: jsonToValue(frame["messages"]),
    tools: asList(frame.getOrDefault("tools")),
    documents: asList(frame.getOrDefault("documents")),
    addGenerationPrompt: frame["add_generation_prompt"].getBool(),
    kwargs: kwargs,
    clockEpoch:
      if frame.hasKey("epoch"): float64(frame["epoch"].getFloat())
      else: 0.0)
  if frame.hasKey("expected_error"):
    result.expectError = true
    result.errorKind = frame["expected_error"]["exception"].getStr()
    result.errorMessage = frame["expected_error"]["message"].getStr()
  else:
    result.rendered = frame["rendered"].getStr()
    result.generationSpans = spanPairs(frame["generation_spans"])

proc loadSuite*(suite: string): seq[FixtureRow] =
  ## Every recorded row of one suite, sorted row order.
  for row in suiteRowNames(suite):
    result.add loadRow(suite, row)
