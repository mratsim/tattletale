# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Corpus row reader for the chattyninja tests.
## One `corpus/<suite>/<row>.json` frame becomes one render-ready record, with the recorded ground truth carried alongside.
##
## JSON is parsed here rather than through `std/json`, because template output observes dict
## insertion order (`items`, `tojson`, `{% for %}` over a mapping) while `std/json` stores object
## members in a hash table.
##
## This parser keeps `DictVal.keys` in document order, the order the recording captured.
##
## The ground truth is self-contained. Every ok row embeds `rendered`, every `err_*` row
## embeds `expected_error {exception, message}`. No HF or Python call is made at test time.

import std/[algorithm, os, strutils, unicode]
import cjn_types, cjn_values

export cjn_types, cjn_values

type
  Row* = object
    ## One recorded frame:
    ##   the render inputs plus the expected outcome.
    suite*: string
    row*: string
    templateSha*: string
    context*: Value
      ## the render context (`messages`, `tools`, `documents`, `add_generation_prompt`, then kwargs)
    rendered*: string
      ## recorded bytes, empty on an expected-error row
    spans*: seq[tuple[start, stop: int]]
      ## recorded generation spans, codepoint `[start, end)` ranges into `rendered`
    expectError*: bool
    errorClass*: string
      ## recorded exception class name, `TemplateError` on every err row
    errorMessage*: string
      ## recorded message, compared verbatim
    clock*: float64
      ## the epoch `strftime_now` reads, absent rows carry 0

  JsonParseError = ref object of CatchableError

const CorpusRoot* = currentSourcePath().parentDir.parentDir / "corpus"
  ## the extracted corpus tree, read-only from a test's point of view

const RenderRowSchema* = "chattyninja-chat-render-row-1"
  ## the `schema` value every recorded render row carries. A suite also holds frames whose
  ## schema differs (its generator metadata), which carry no render inputs and are skipped.

# JSON reader
# ---------------------------------------------------------------------------

type J = object
  ## Cursor over one JSON document.
  s: string
  i: int

func jsonError(msg: string): JsonParseError =
  ## Returns an unraised parse error. Every caller raises it.
  new(result)
  result.msg = msg

template fail(j: J, msg: string): untyped =
  ## Raises a parse error naming the construct and the byte offset. `raise` sits at the call
  ## site so this template serves as a statement and as the value branch of a `case`.
  raise jsonError(msg & " at byte " & $j.i & ": " &
      j.s[0 ..< min(j.i + 24, j.s.len)].escape)

proc ws(j: var J) =
  while j.i < j.s.len and j.s[j.i] in {' ', '\t', '\n', '\r'}:
    inc j.i

proc peek(j: var J): char =
  ws(j)
  if j.i >= j.s.len:
    fail j, "JSON ends early"
  j.s[j.i]

proc expect(j: var J, c: char) =
  if peek(j) != c:
    fail j, "JSON expects `" & c & "`"
  inc j.i

proc lit(j: var J, w: string) =
  if j.s.len - j.i < w.len or j.s[j.i ..< j.i + w.len] != w:
    fail j, "JSON expects `" & w & "`"
  inc j.i, w.len

proc hex4(j: var J): int =
  var v = 0
  for k in 0 ..< 4:
    if j.i >= j.s.len:
      fail j, "JSON ends inside an escape"
    let c = j.s[j.i]
    let d =
      case c
      of '0' .. '9': ord(c) - ord('0')
      of 'a' .. 'f': ord(c) - ord('a') + 10
      of 'A' .. 'F': ord(c) - ord('A') + 10
      else: fail j, "JSON escape is not a hex digit"
    v = v * 16 + d
    inc j.i
  v

proc jsonStr(j: var J): string =
  ## Returns one decoded JSON string, with `\u` surrogate pairs folded to codepoints.
  expect(j, '"')
  while j.i < j.s.len:
    let c = j.s[j.i]
    inc j.i
    if c == '"':
      return result
    if c != '\\':
      result.add c
      continue
    if j.i >= j.s.len:
      fail j, "JSON ends inside an escape"
    let e = j.s[j.i]
    inc j.i
    case e
    of '"': result.add '"'
    of '\\': result.add '\\'
    of '/': result.add '/'
    of 'b': result.add '\b'
    of 'f': result.add '\f'
    of 'n': result.add '\n'
    of 'r': result.add '\r'
    of 't': result.add '\t'
    of 'u':
      var cp = hex4(j)
      if cp >= 0xD800 and cp <= 0xDBFF and j.s.len - j.i >= 6 and j.s[j.i] == '\\' and
          j.s[j.i + 1] == 'u':
        inc j.i, 2
        let lo = hex4(j)
        cp = 0x10000 + ((cp - 0xD800) shl 10) + (lo - 0xDC00)
      result.add $Rune(cp)
    else: fail j, "unknown JSON escape"
  fail j, "JSON string is not closed"

proc jsonValue(j: var J): Value =
  case peek(j)
  of '{':
    expect(j, '{')
    var d = DictVal()
    if peek(j) != '}':
      while true:
        let k = jsonStr(j)
        expect(j, ':')
        dictSet(d, k, jsonValue(j))
        case peek(j)
        of ',': expect(j, ',')
        of '}': break
        else: fail j, "JSON object expects `,` or `}`"
    expect(j, '}')
    dictVal(d)
  of '[':
    expect(j, '[')
    var xs = newSeq[Value]()
    if peek(j) != ']':
      while true:
        xs.add jsonValue(j)
        case peek(j)
        of ',': expect(j, ',')
        of ']': break
        else: fail j, "JSON array expects `,` or `]`"
    expect(j, ']')
    seqVal(xs)
  of '"': strVal(jsonStr(j))
  of 't':
    lit(j, "true")
    boolVal(true)
  of 'f':
    lit(j, "false")
    boolVal(false)
  of 'n':
    lit(j, "null")
    noneVal()
  else:
    let start = j.i
    if j.s[j.i] == '-':
      inc j.i
    while j.i < j.s.len and j.s[j.i] in '0' .. '9':
      inc j.i
    var isFloat = false
    if j.i < j.s.len and j.s[j.i] == '.':
      isFloat = true
      inc j.i
      while j.i < j.s.len and j.s[j.i] in '0' .. '9':
        inc j.i
    if j.i < j.s.len and j.s[j.i] in {'e', 'E'}:
      isFloat = true
      inc j.i
      if j.i < j.s.len and j.s[j.i] in {'+', '-'}:
        inc j.i
      while j.i < j.s.len and j.s[j.i] in '0' .. '9':
        inc j.i
    let text = j.s[start ..< j.i]
    if isFloat:
      floatVal(parseFloat(text))
    else:
      intVal(parseBiggestInt(text))

proc jsonDoc*(src: string): Value =
  ## Returns a whole JSON document as an engine value, mapping insertion order preserved.
  var j = J(s: src, i: 0)
  result = jsonValue(j)
  ws(j)
  if j.i != src.len:
    fail j, "JSON has trailing text"

# Row reading
# ---------------------------------------------------------------------------

func field(v: Value, name: string): Value =
  if v.kind != vkDict: undefinedVal() else: v.d.dictGet(name)

func optList(v: Value, name: string): Value =
  ## An optional list input. An absent or null key becomes none, the way the recording
  ## path passes a missing `tools` or `documents` through as `None`.
  let got = field(v, name)
  if got.kind == vkUndefined or got.kind == vkNone: noneVal() else: got

func spanList(v: Value): seq[tuple[start, stop: int]] =
  if v.kind != vkSeq:
    return
  for pair in v.xs.items:
    if pair.kind != vkSeq or pair.xs.items.len != 2:
      raise jsonError("generation_spans entry is not a [start, end] pair")
    result.add (pair.xs.items[0].i.int, pair.xs.items[1].i.int)

func clockOf(frame: Value): float64 =
  ## Returns the recorded epoch, 0 when the row carries none.
  let e = field(frame, "epoch")
  case e.kind
  of vkInt: float64 e.i
  of vkFloat: e.f
  else: 0.0

func contextOf(frame: Value): Value =
  ## Builds the template context:
  ##   the standard keys in recording order, then the row's kwargs.
  var d = DictVal()
  dictSet(d, "messages", field(frame, "messages"))
  dictSet(d, "tools", optList(frame, "tools"))
  dictSet(d, "documents", optList(frame, "documents"))
  dictSet(d, "add_generation_prompt", field(frame, "add_generation_prompt"))
  let kw = field(frame, "kwargs")
  if kw.kind == vkDict:
    for k, key in kw.d.keys:
      dictSet(d, key, kw.d.vals[k])
  dictVal(d)

proc loadRow*(suite, row: string): Row =
  ## Reads `corpus/<suite>/<row>.json` and returns the render inputs plus the recorded
  ## outcome. The frame's `suite` and `row` fields must agree with the path components
  ## naming the frame.
  let path = CorpusRoot / suite / (row & ".json")
  let frame = jsonDoc(readFile(path))
  if field(frame, "suite").kind == vkStr and field(frame, "suite").s != suite:
    raise jsonError(path & ": frame suite is " & field(frame, "suite").s)
  if field(frame, "row").kind == vkStr and field(frame, "row").s != row:
    raise jsonError(path & ": frame row is " & field(frame, "row").s)
  let tmpl = field(frame, "template")
  let err = field(frame, "expected_error")
  Row(
      suite: suite,
      row: row,
      templateSha:
        if tmpl.kind == vkDict: pyStr(field(tmpl, "sha256")) else: "",
      context: contextOf(frame),
      rendered: if err.kind == vkUndefined: pyStr(field(frame, "rendered")) else: "",
      spans: spanList(field(frame, "generation_spans")),
      expectError: err.kind != vkUndefined,
      errorClass: if err.kind == vkDict: pyStr(field(err, "exception")) else: "",
      errorMessage: if err.kind == vkDict: pyStr(field(err, "message")) else: "",
      clock: clockOf(frame))

proc rows*(suite: string): seq[Row] =
  ## Every recorded row of one suite, in sorted row order. Frames whose `schema` is not
  ## `RenderRowSchema` are skipped because a suite's generator metadata carries no render inputs.
  var stems = newSeq[string]()
  for path in walkPattern(CorpusRoot / suite / "*.json"):
    let name = lastPathPart(path)
    if not name.endsWith(".json") or name.endsWith(".meta.json"):
      continue
    let schema = field(jsonDoc(readFile(path)), "schema")
    if schema.kind == vkStr and schema.s != RenderRowSchema:
      continue
    stems.add name[0 ..< name.len - ".json".len]
  stems.sort
  for s in stems:
    result.add loadRow(suite, s)

proc templateSource*(suite: string): string =
  ## Returns the recorded template bytes, `<suite>/<suite>.jinja`.
  readFile(CorpusRoot / suite / (suite & ".jinja"))
