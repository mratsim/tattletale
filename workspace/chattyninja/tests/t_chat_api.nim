# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. these terms.

## Typed-surface referee, every recorded corpus row rendered through the public
## chat API alone, `import chattyninja` naming the surface, over a real `ChatContext`
## built from the row's JSON, the read going through `std/json`, insertion order preserved
##
## - ok rows render byte-exact against the recording
## - err rows raise the recorded message, gap rows raising `ceUnimplemented`
##   naming the recorded construct
## - the ledger must equal the recorded 94 ok, 9 gap, 21 err, 18 err rows raising
##   message-exact, 3 content-type rows structurally unreachable over the typed records,
##   a synthetic kwarg row exercising the bool and int64 kwarg forms
##
## Run:
##   $ nim test_chattyninja

import std/[json, options, os, strutils]
import workspace/zstd/zstd_highlevel
import workspace/chattyninja
import corpus/fixture_loader

const
  CorpusRoot = currentSourcePath().parentDir / "corpus"
    ## the corpus tree this test reads, row payloads read-only
  RowSuffix = ".json.zst"

const ParseableSuites = ["deepseekv2lite", "gemma3", "gemma4", "glm47", "glm47flash",
    "glm53flash", "gptoss20b", "kimik26", "kimi", "lagunaxs21", "lfm25", "ling30",
    "mimo25", "minimaxm27", "minimaxm3", "mistral7bv01", "moonlight",
    "northminicode10", "qwen3", "qwen35", "qwen36", "qwen38", "qwen38flashnext"]
  ## the suite list `t_corpus` renders, the referee covering the same rows

# Row JSON into the typed records

func readSchemaValue(n: JsonNode): ParamValue =
  ## One recorded schema value into the typed form, objects keep insertion order.
  case n.kind
  of JString: pvVal(n.getStr())
  of JBool: pvVal(n.getBool())
  of JInt: pvVal(int64 n.getInt())
  of JFloat: pvVal(n.getFloat())
  of JArray:
    var xs: seq[ParamValue]
    for c in n.items:
      xs.add readSchemaValue(c)
    ParamValue(kind: pvList, items: xs)
  of JObject:
    var fs: seq[tuple[k: string, v: ParamValue]]
    for k, c in pairs(n):
      fs.add (k, readSchemaValue(c))
    ParamValue(kind: pvRec, fields: fs)
  of JNull:
    raise newException(ValueError, "schema leaf is null, the corpus never records one")

func readBlockValue(n: JsonNode): BlockValue
func readBlock(n: JsonNode): ContentBlock
  ## forward declarations, the two block forms recursing into each other

func readBlockValue(n: JsonNode): BlockValue =
  case n.kind
  of JString: bvVal(n.getStr())
  of JArray:
    var bs: seq[ContentBlock]
    for c in n.items:
      bs.add readBlock(c)
    bvList(bs)
  else:
    raise newException(ValueError, "block field is neither string nor block list")

func readBlock(n: JsonNode): ContentBlock =
  ## One recorded content block, key order kept.
  var fs: seq[tuple[k: string, v: BlockValue]]
  for k, x in pairs(n):
    fs.add (k, readBlockValue(x))
  blockOf(fs)

func readContent(n: JsonNode): Content =
  case n.kind
  of JString: textContent(n.getStr())
  of JArray:
    var bs: seq[ContentBlock]
    for c in n.items:
      bs.add readBlock(c)
    blockContent(bs)
  of JInt, JFloat, JBool, JNull, JObject:
    # Caller-data type errors are not expressible on the typed surface, templates
    # raising on the raw value's type. The mapping records the value as text,
    # the row's message parity proving the raise order.
    textContent($n)

func readToolCall(n: JsonNode): ToolCall =
  ## One recorded tool call, the recording always wrapping name and arguments
  ## in the `function` record.
  let fn = n["function"]
  result.name = fn["name"].getStr
  let args = fn["arguments"]
  if args.kind != JObject:
    raise newException(ValueError, "tool call arguments is not a record")
  for k, c in pairs(args):
    result.arguments.add (k, readSchemaValue(c))
  if n.hasKey("id"):
    result.id = some(n["id"].getStr)
  if n.hasKey("content_type"):
    result.content_type = some(n["content_type"].getStr)

func readTool(n: JsonNode): Tool =
  ## One recorded tool, the wrapper key optional and the JSON-schema parameters
  ## read in insertion order.
  if n.hasKey("type"):
    result.tool_type = some(n["type"].getStr)
  let fn = n["function"]
  result.name = fn["name"].getStr
  result.description = fn["description"].getStr
  for k, c in pairs(fn["parameters"]):
    result.parameters.add (k, readSchemaValue(c))

func readMessage(n: JsonNode): Message =
  ## One recorded message. A reasoning payload arrives under whichever spelling
  ## the recording passed, null staying absent like the template readers see it.
  result.role = n["role"].getStr
  result.content = readContent(n["content"])
  for spelling in ["reasoning_content", "reasoning", "thinking"]:
    let v = n.getOrDefault(spelling)
    if not v.isNil and v.kind == JString:
      result.reasoning_content = some(v.getStr)
      break
  if n.hasKey("name"):
    result.name = some(n["name"].getStr)
  if n.hasKey("tool_call_id"):
    result.tool_call_id = some(n["tool_call_id"].getStr)
  let tcs = n.getOrDefault("tool_calls")
  if not tcs.isNil:
    for tc in tcs.items:
      result.tool_calls.add readToolCall(tc)

func readContext(payload: JsonNode): ChatContext =
  ## One row payload into a real `ChatContext`, keys and order the recording passed.
  for m in payload["messages"].items:
    result.messages.add readMessage(m)
  let tools = payload.getOrDefault("tools")
  if not tools.isNil and tools.kind != JNull:
    for t in tools.items:
      result.tools.add readTool(t)
  let docs = payload.getOrDefault("documents")
  if not docs.isNil and docs.kind != JNull:
    for d in docs.items:
      result.documents.add readBlock(d)
  result.add_generation_prompt = payload["add_generation_prompt"].getBool
  for k, v in pairs(payload["kwargs"]):
    case v.kind
    of JString: result.kwargs.add (k, kwVal(v.getStr))
    of JBool: result.kwargs.add (k, kwVal(v.getBool))
    of JInt: result.kwargs.add (k, kwVal(int64 v.getInt))
    else:
      raise newException(ValueError, "kwarg `" & k & "` is neither string, bool nor int")

proc rowPayload(suite, row: string): JsonNode =
  ## One recorded row payload, decompressed and parsed, insertion order preserved.
  parseJson(zstdDecompress(readFile(CorpusRoot / suite / (row & RowSuffix)), string))

# Gap ledger the raw suite records, the typed surface gapping the same rows
# naming the same construct

const CallerTypeErrNames = [
  ("gemma3/err_content_type", "Invalid content type"),
  ("qwen38/err_content_type", "Unexpected content type."),
  ("qwen38flashnext/err_content_type", "Unexpected content type."),
]
  ## Caller-data type errors the typed records make unreachable, the message content
  ## carrying a non-string type, no `Content` form existing

const GapNames = [
  ("gemma4/channel_strip", "nkSetBlock"),
  ("gemma4/default", "nkSetBlock"),
  ("gemma4/enable_thinking_true", "nkSetBlock"),
  ("gemma4/tools_tool_response", "dictsort"),
  ("northminicode10/default", "nkSetBlock"),
  ("northminicode10/documents_grounding", "nkSetBlock"),
  ("northminicode10/reasoning_off", "nkSetBlock"),
  ("northminicode10/tool_break", "nkSetBlock"),
  ("northminicode10/tools_tool_response", "nkSetBlock"),
]

func expectedGap(suite, row: string): string =
  for g in GapNames:
    if g[0] == suite & "/" & row:
      return g[1]

proc fail(msg: string) =
  raise newException(AssertionError, msg)

# Referee sweep, ledger equality against the recorded counts

var okExact = 0
var gapRows = 0
var errRaised = 0
var typeEliminated = 0

for suite in ParseableSuites:
  let src = suiteTemplateSource(suite)
  let (tmpl, sym) = parseJinjaTemplate(src)
  for row in suiteRowNames(suite):
    let payload = rowPayload(suite, row)
    let ctx = readContext(payload)
    let expectError = payload.hasKey("expected_error")
    if expectError:
      var raised = ""
      try:
        discard renderToString(src, ctx, float64 payload.getOrDefault("epoch").getFloat(0.0))
      except JinjaError as e:
        raised = e.what
      var isTypeErr = false
      var wantTypeMsg = ""
      for t in CallerTypeErrNames:
        if t[0] == suite & "/" & row:
          isTypeErr = true
          wantTypeMsg = t[1]
      if isTypeErr:
        # Typed records cannot carry the non-string content this row records
        # and the template's type check passes. The render must complete
        # with the recorded message unreachable exactly as listed above.
        if raised.len > 0:
          fail(suite & "/" & row & ": the typed render raised `" & raised &
              "` where the type error is unreachable")
        inc typeEliminated
        continue
      if raised != payload["expected_error"]["message"].getStr:
        fail(suite & "/" & row & ": err message mismatch, got `" & raised & "`")
      inc errRaised
      continue
    var whole = ""
    try:
      whole = renderToString(tmpl, sym, ctx, float64 payload.getOrDefault("epoch").getFloat(0.0))
    except JinjaError as e:
      if e.cause != ceUnimplemented:
        fail(suite & "/" & row & ": the typed render raised `" & e.what & "`")
      let want = expectedGap(suite, row)
      if want.len == 0:
        fail(suite & "/" & row & ": a new gap row appeared, record its construct name")
      if want notin e.what:
        fail(suite & "/" & row & ": the gap raise does not name the recorded construct")
      inc gapRows
      continue
    let want = payload["rendered"].getStr
    if whole != want:
      fail(suite & "/" & row & ": the typed render differs from the recording")
    inc okExact

# Typed pull path, startJinjaRender over the artifact plus pullInto delivering
# bytes equal to the one-shot render

block:
  let src = suiteTemplateSource("gemma3")
  let (tmpl, sym) = parseJinjaTemplate(src)
  let ctx = readContext(rowPayload("gemma3", "default"))
  let c = startJinjaRender(tmpl, sym, ctx)
  var buf: array[256, char]
  var pulled = ""
  while true:
    let n = pullInto(c, buf)
    if n == 0:
      break
    for i in 0 ..< n:
      pulled.add buf[i]
  doAssert pulled == renderToString(src, ctx),
      "the pull path differs from the one-shot typed render"

# Synthetic kwarg row, the bool and int64 forms the corpus kwargs never pass

block:
  let src = "{{ bos_token }}|{{ max_pos }}"
  let ctx = ChatContext(
      messages: @[Message(role: "user", content: textContent("hi"))],
      kwargs: @[("bos_token", kwVal(true)), ("max_pos", kwVal(4096'i64))])
  doAssert renderToString(src, ctx) == "True|4096",
      "the bool and int64 kwarg forms did not render, Python stringification required"
  let (tmpl, sym) = parseJinjaTemplate(src)
  doAssert renderToString(tmpl, sym, ctx) == "True|4096",
      "the artifact overload differs from the one-shot render"

# Typed tools path through the `tojson` filter, one int64 schema leaf rendered

block:
  let src = "{{ tools | tojson }}"
  let ctx = ChatContext(
      tools: @[Tool(
          name: "get_weather",
          description: "Weather for one city",
          parameters: @[("type", pvVal("object")), ("max_pos", pvVal(4096'i64))],
          tool_type: some("function"))])
  let rendered = renderToString(src, ctx)
  doAssert rendered.contains("\"max_pos\": 4096") or rendered.contains("\"max_pos\":4096"),
      "the int64 schema leaf did not survive the tojson filter: `" & rendered & "`"
  doAssert rendered.contains("\"type\": \"function\"") or rendered.contains("\"type\":\"function\""),
      "the tool wrapper key did not survive the tojson filter"

# Suite discovery matches the const list, a dir without a list entry walking zero rows

var suiteDirs = 0
for entry in walkDir(CorpusRoot):
  if entry.kind == pcDir:
    inc suiteDirs
doAssert suiteDirs == ParseableSuites.len,
    "corpus suite dirs: " & $suiteDirs & ", list: " & $ParseableSuites.len

doAssert okExact == 94, "expected 94 ok rows byte-exact over the typed surface, got " & $okExact
doAssert gapRows == 9, "expected 9 gap rows, got " & $gapRows
doAssert errRaised == 18, "expected 18 message-exact err rows, got " & $errRaised
doAssert typeEliminated == 3,
    "expected 3 content-type rows unreachable over the typed records, got " & $typeEliminated

echo "t_chat_api: 94 ok rows byte-exact through the typed chat surface, 9 gap rows loud, " &
    "18 err rows message-exact, 3 content-type rows unreachable over the typed records, " &
    "pull path and synthetic kwarg rows green"
