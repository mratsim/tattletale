# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Corpus-driven delivery proof for the chattyninja engine.
##
## Every recorded corpus row walks the delivery paths and the window contract:
##
## - byte-exact delivery of every ok row through a buffered `pullInto` drain loop,
##   windowed consumers, every err row raising the recorded error, declared gaps loud,
##   and every ok row's generation spans asserted verbatim on every delivery path
## - the window contract, the for-filter raise with repull-resume, the macro scope pop,
##   the ensure_ascii escape shapes, the compiled artifact layout
## - allocation counting under `-d:nimAllocStats`
##
## Run:
##   $ nim test_chattyninja

{.experimental: "views".}

import std/[algorithm, importutils, macros, os, strutils, unicode]
import cnj_types {.all.}
import jinja_data_model {.all.}
import jinja_serialize {.all.}
import cnj_parse {.all.}
import cnj_engine {.all.}
import workspace/data_structures/src/small_seqs
import workspace/zstd/zstd_highlevel

# Corpus row reader.
# Document-order contract:
# template output observes dict insertion order (`items`, `tojson`,
# `{% for %}` over a mapping). This parser keeps `DictVal.keys` in document order,
# the order the recording captured. `std/json` stores object members inside a hash
# hash table and cannot serve that order.
#
# Ground truth is self-contained. Every ok row embeds `rendered` and every
# `err_*` row embeds `expected_error {exception, message, offset, span}`,
# the raise call's name-token span into the suite template.

type
  Row* = object
    ## One recorded row:
    ##   the render inputs plus the expected outcome.
    suite*: string
    row*: string
    context*: JinjaVal
      ## the render context (`messages`, `tools`, `documents`, `add_generation_prompt`, then kwargs)
    rendered*: string
      ## recorded bytes, empty on an expected-error row
    spans*: seq[tuple[start, stop: int]]
      ## recorded generation spans, codepoint `[start, end)` ranges into `rendered`
    expectError*: bool
    errorMessage*: string
      ## recorded message, compared verbatim
    errorOffset*: int
      ## recorded byte offset of the offending construct into the suite template,
      ## compared against the raised `JinjaError.offset`
    errorSpan*: int
      ## recorded byte length of the offending construct, compared when nonzero
    clock*: float64
      ## the epoch `strftime_now` reads, absent rows carry 0

  JsonParseError = ref object of CatchableError

const CorpusRoot* = currentSourcePath().parentDir / "corpus"
  ## the extracted corpus tree, read-only from a test's point of view

const RowSuffix = ".json.zst"
  ## the recorded row suffix, the corpus tree carrying one zstd-compressed row per row id

# JSON reader over one document

type J = object
  ## Cursor over one JSON document.
  s: string
  i: int

type RenderMismatch = ref object of CatchableError

type PullChunks[N: static int] = object
  ## Bounded pull windows over one chattyninja render, a fixed `N`-byte window per pull.
  ##
  ## Contract:
  ## - the tail window carries the remainder when the render length is not a multiple of N
  ## - a full drain equals the whole-render `renderToString`, the next pull after it reports 0
  ## - partial consumption resumes from the driver's fields, no byte re-handed
  ##
  ## The compiled template stays at the consumer's scope, `CompiledTemplate.jinja` borrowing
  ## the template text. The template is a ref, so a `JinjaRenderContext` embeds in another object
  ## without losing the borrowed view.
  c: JinjaRenderContext
  buf: array[N, char]

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

proc jsonValue(j: var J): JinjaVal =
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
    var xs = newSeq[JinjaVal]()
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

proc jsonDoc*(src: string): JinjaVal =
  ## Returns a whole JSON document as an engine value, mapping insertion order preserved.
  var j = J(s: src, i: 0)
  result = jsonValue(j)
  ws(j)
  if j.i != src.len:
    fail j, "JSON has trailing text"

# Row reading from a parsed document

func field(v: JinjaVal, name: string): JinjaVal =
  if v.kind != vkDict: undefinedVal() else: v.d.dictGet(name)

func optList(v: JinjaVal, name: string): JinjaVal =
  ## An optional list input. An absent or null key becomes none, the way the recording
  ## path passes a missing `tools` or `documents` through as `None`.
  let got = field(v, name)
  if got.kind == vkUndefined or got.kind == vkNone: noneVal() else: got

func spanList(v: JinjaVal): seq[tuple[start, stop: int]] =
  if v.kind != vkSeq:
    return
  for pair in v.xs.items:
    if pair.kind != vkSeq or pair.xs.items.len != 2:
      raise jsonError("generation_spans entry is not a [start, end] pair")
    result.add (pair.xs.items[0].i.int, pair.xs.items[1].i.int)

func clockOf(payload: JinjaVal): float64 =
  ## Returns the recorded epoch, 0 when the row carries none.
  let e = field(payload, "epoch")
  case e.kind
  of vkInt: float64 e.i
  of vkFloat: e.f
  else: 0.0

func contextOf(payload: JinjaVal): JinjaVal =
  ## Builds the template context:
  ##   the standard keys in recording order, then the row's kwargs.
  var d = DictVal()
  dictSet(d, "messages", field(payload, "messages"))
  dictSet(d, "tools", optList(payload, "tools"))
  dictSet(d, "documents", optList(payload, "documents"))
  dictSet(d, "add_generation_prompt", field(payload, "add_generation_prompt"))
  let kw = field(payload, "kwargs")
  if kw.kind == vkDict:
    for k, key in kw.d.keys:
      dictSet(d, key, kw.d.vals[k])
  dictVal(d)

proc loadRow*(suite, row: string): Row =
  ## Decompresses `corpus/<suite>/<row>.json.zst` and returns the render inputs plus
  ## the recorded outcome. The stem is the row id. An err row's `expected_error`
  ## supplies the message, the byte offset and the span the raise must report.
  let path = CorpusRoot / suite / (row & RowSuffix)
  let payload = jsonDoc(zstdDecompress(readFile(path), string))
  let err = field(payload, "expected_error")
  Row(
      suite: suite,
      row: row,
      context: contextOf(payload),
      rendered: if err.kind == vkUndefined: pyStr(field(payload, "rendered")) else: "",
      spans: spanList(field(payload, "generation_spans")),
      expectError: err.kind != vkUndefined,
      errorMessage: if err.kind == vkDict: pyStr(field(err, "message")) else: "",
      errorOffset:
        if err.kind == vkDict and field(err, "offset").kind == vkInt:
          field(err, "offset").i.int
        else:
          NoOffset,
      errorSpan:
        if err.kind == vkDict and field(err, "span").kind == vkInt:
          field(err, "span").i.int
        else:
          0,
      clock: clockOf(payload))

proc rows*(suite: string): seq[Row] =
  ## Every recorded row of one suite, in sorted row order, the `*.json.zst` stems
  ## naming the rows.
  var stems = newSeq[string]()
  for path in walkPattern(CorpusRoot / suite / ("*" & RowSuffix)):
    let name = lastPathPart(path)
    stems.add name[0 ..< name.len - RowSuffix.len]
  stems.sort
  for s in stems:
    result.add loadRow(suite, s)

proc templateSource*(suite: string): string =
  ## Returns the recorded template bytes, `<suite>/<suite>.jinja`.
  readFile(CorpusRoot / suite / (suite & ".jinja"))


proc fail(msg: string): void {.noreturn.} =
  var e = RenderMismatch()
  e.msg = msg
  raise e

func firstDiff(got, want: string): int =
  ## Returns the index of the first differing byte, or the shared length when one is a prefix.
  let n = min(got.len, want.len)
  while result < n and got[result] == want[result]:
    inc result

func surround(s: string, at: int): string =
  ## Returns a repr of the 24 bytes around `at`, for a failure message.
  s[max(0, at - 24) ..< min(s.len, at + 24)].replace("\n", "\\n").replace("\t", "\\t")

func sameBytes*(got, want: string): bool =
  ## Reports whether two renderings are byte-identical.
  got.len == want.len and firstDiff(got, want) == got.len

func report(got, want: string): string =
  if got.len != want.len and firstDiff(got, want) == min(got.len, want.len):
    "length " & $got.len & " vs " & $want.len & ", common prefix; got tail " &
        surround(got, got.len) & " | want tail " & surround(want, want.len)
  else:
    let k = firstDiff(got, want)
    "first difference at byte " & $k & " (got " & $got.len & " B, want " & $want.len & " B)\n" &
        "   got  " & surround(got, k) & "\n   want " & surround(want, k)

func bytesOf(buf: openArray[char], n: int): string =
  ## Copies the first `n` bytes of a pull window into a string.
  for i in 0 ..< n:
    result.add buf[i]

func pieceRemaining(p: Piece): int =
  ## Bytes of a pending piece not yet delivered, lazy pieces carrying no counted length.
  case p.kind
  of pkNone: 0
  of pkSpan: int(p.hi - p.lo) - p.pos
  of pkStr: p.s.len - p.pos
  of pkCut: int(p.chi - p.clo) - p.pos
  of pkLazy: 0

proc renderPull(m: CompiledTemplate, sym: CompiledSymbols, ctx: JinjaVal, clock: float64, cap: int): tuple[
    text: string, spans: seq[tuple[start, stop: int]]] =
  ## Renders through `pullInto` with a `cap`-byte caller buffer, accumulating every fill.
  ## Returns the render bytes and the driver's recorded generation spans, byte coordinates
  ## into the bytes.
  var c = startJinjaRender(m, sym, ctx, clock)
  var buf = newSeq[char](cap)
  while true:
    let n = pullInto(c, buf)
    if n == 0:
      break
    for i in 0 ..< n:
      result.text.add buf[i]
  result.spans = c.state.spans

proc renderAllPull(m: CompiledTemplate, sym: CompiledSymbols, ctx: JinjaVal, clock: float64): string =
  ## Renders whole through the buffered `pullInto` drain loop, a fresh render context.
  var c = startJinjaRender(m, sym, ctx, clock)
  var buf: array[4096, char]
  while true:
    let n = pullInto(c, buf)
    if n == 0:
      break
    for i in 0 ..< n:
      result.add buf[i]


func pullChunks[N: static int](c: JinjaRenderContext): PullChunks[N] =
  ## Builds the windowed pull machine over the render context `c` of a compiled template.
  PullChunks[N](c: c)

iterator items[N: static int](p: var PullChunks[N]): openArray[char] =
  ## Yields the render as bounded windows:
  ## - one `pullInto` call fills the window, the view carries at most N bytes
  ## - the tail window closes the render, a 0 count ends the stream
  while true:
    let n = pullInto(p.c, p.buf)
    if n == 0:
      break
    yield p.buf.toOpenArray(0, n - 1)

proc renderAllSpans(m: CompiledTemplate, sym: CompiledSymbols, ctx: JinjaVal, clock: float64): tuple[
    text: string, spans: seq[tuple[start, stop: int]]] =
  ## Renders one row whole and returns the bytes plus the driver's recorded generation spans,
  ## byte coordinates into the bytes.
  var d = startJinjaRender(m, sym, ctx, clock)
  var buf: array[4096, char]
  while true:
    let n = pullInto(d, buf)
    if n == 0:
      break
    for i in 0 ..< n:
      result.text.add buf[i]
  result.spans = d.state.spans

func cpIndex(s: string, byteAt: int): int =
  ## Codepoint index of a byte offset, one lead-byte stride walk to the offset.
  var i = 0
  var cp = 0
  while i < byteAt:
    inc cp
    inc i, runeLenAt(s, i)
  cp

func asCodepointSpans(s: string, spans: seq[tuple[start, stop: int]]): seq[tuple[start, stop: int]] =
  ## Converts engine byte spans to the recording's codepoint ranges over the same bytes.
  for (a, b) in spans:
    result.add (cpIndex(s, a), cpIndex(s, b))

proc renderChunked[N: static int](m: CompiledTemplate, sym: CompiledSymbols, ctx: JinjaVal, clock: float64): tuple[
    text: string, spans: seq[tuple[start, stop: int]]] =
  ## Renders one row through `N`-byte windows, accumulating every window.
  ## Returns the render bytes and the driver's recorded generation spans, byte coordinates
  ## into the bytes.
  var pc = pullChunks[N](startJinjaRender(m, sym, ctx, clock))
  for w in pc.items():
    for ch in w:
      result.text.add ch
  result.spans = pc.c.state.spans

# Totality contract:
# every NodeKind has a non-nil step, checked over every kind. A nil entry
# would send a render through a null pointer.

proc testStepsTotality() =
  for k in NodeKind:
    doAssert not Steps[k].isNil, "steps has no entry for " & $k


# Equality must reject a one-byte change:
# a comparison that cannot fail leaves the corpus walk vacuous.

proc testEqualityRejectsChange() =

  doAssert sameBytes("abc", "abc")
  doAssert not sameBytes("abc", "abd"), "sameBytes accepted a changed byte"
  doAssert not sameBytes("abc", "abcd"), "sameBytes accepted a length change"
  doAssert not sameBytes("", "a"), "sameBytes accepted empty against non-empty"


# Every recorded corpus row through the three delivery paths.

proc testCorpusDelivery() =

  const parseable = ["deepseekv2lite", "gemma3", "gemma4", "glm47flash", "glm53flash",
      "gptoss20b", "kimi", "lagunaxs21", "lfm25", "ling30", "mimo25", "mistral7bv01",
      "moonlight", "northminicode10", "qwen3", "qwen35", "qwen36", "qwen38flashnext"]
  # Suite discovery is the const list, not the directory. A suite dir added under corpus/
  # would walk zero rows silently, so the dir count is checked against the list.
  var suiteDirs = 0
  for entry in walkDir(CorpusRoot):
    if entry.kind == pcDir:
      inc suiteDirs
  doAssert suiteDirs == parseable.len,
      "corpus suite dirs: " & $suiteDirs & ", parseable list: " & $parseable.len
  # Each gap row records the construct name its raise must carry. A gap naming
  # another construct fails the name check before the count.
  const gapNames = [
    ("gemma4/channel_strip", "nkSetBlock"),
    ("gemma4/default", "nkSetBlock"),
    ("gemma4/enable_thinking_true", "nkSetBlock"),
    ("gemma4/tools_tool_response", "dictsort"),
    ("mimo25/tools_tool_response", "items"),
    ("northminicode10/default", "nkSetBlock"),
    ("northminicode10/documents_grounding", "nkSetBlock"),
    ("northminicode10/reasoning_off", "nkSetBlock"),
    ("northminicode10/tool_break", "nkSetBlock"),
    ("northminicode10/tools_tool_response", "nkSetBlock"),
    ("qwen35/tools_tool_response", "items"),
    ("qwen36/tools_tool_response", "items"),
    ("qwen38flashnext/tools_tool_response", "items"),
  ]
  var okExact = 0
  var gapRows = 0
  var errRaised = 0
  for suite in parseable:
    let src = templateSource(suite)
    let (m, tables) = parseJinjaTemplate(src)
    for r in rows(suite):
      if r.expectError:
        # Error outcome, not a wrong success. The match compares the recorded
        # message verbatim, the recorded offset against the raised
        # `JinjaError.offset`, and the recorded span when nonzero.
        var raisedMsg = ""
        var raisedAt = NoOffset
        var raisedSpan = 0
        var wrongSuccess = ""
        try:
          wrongSuccess = renderAllPull(m, tables, r.context, r.clock)
        except JinjaError as e:
          raisedMsg = e.what
          raisedAt = e.offset
          raisedSpan = e.span
        if wrongSuccess.len > 0:
          fail(suite & "/" & r.row & ": expected `" & r.errorMessage &
              "` but the render produced " & $wrongSuccess.len & " bytes")
        if raisedMsg != r.errorMessage:
          fail(suite & "/" & r.row & ": message mismatch\n   got  " & raisedMsg &
              "\n   want " & r.errorMessage)
        if raisedAt != r.errorOffset:
          fail(suite & "/" & r.row & ": offset mismatch: raised at byte " & $raisedAt &
              ", recorded " & $r.errorOffset)
        if r.errorSpan != 0 and raisedSpan != r.errorSpan:
          fail(suite & "/" & r.row & ": span mismatch: raised " & $raisedSpan &
              ", recorded " & $r.errorSpan)
        inc errRaised
        continue

      # Ok row. The whole render, a 256-byte buffered pull loop and 7-byte and 1-byte
      # windowed consumers all deliver the recorded bytes, the recorded generation spans
      # exact on every path. A declared gap raises loud.
      var raised = ""
      var whole = ""
      var gotSpans: seq[tuple[start, stop: int]]
      try:
        (whole, gotSpans) = renderAllSpans(m, tables, r.context, r.clock)
      except JinjaError as e:
        # A declared gap surfaces as a gap, never as a wrong answer:
        # - declared constructs and unimplemented filter names raise with cause `ceUnimplemented`
        # - any other raise fails the row
        if e.cause == ceUnimplemented:
          var want = ""
          for g in gapNames:
            if g[0] == suite & "/" & r.row:
              want = g[1]
          if want.len == 0:
            fail(suite & "/" & r.row & ": a new gap row appeared, record its expected construct name")
          if want notin e.what:
            fail(suite & "/" & r.row & ": the gap raise does not name the recorded missing construct: got `" &
                e.what & "`, want `" & want & "`")
          inc gapRows
          continue
        raised = e.what
      if raised.len > 0:
        fail(suite & "/" & r.row & ": the whole render raised " & raised)
      if not sameBytes(whole, r.rendered):
        fail(suite & "/" & r.row & ": the whole render differs: " & report(whole, r.rendered))
      # Generation spans compare verbatim, engine byte coordinates mapped over the same
      # bytes to the recording's codepoint ranges, empty against empty.
      if asCodepointSpans(whole, gotSpans) != r.spans:
        fail(suite & "/" & r.row & ": generation spans differ: got " & $asCodepointSpans(whole,
            gotSpans) & ", want " & $r.spans)
      let (buffered, bufSpans) = renderPull(m, tables, r.context, r.clock, 256)
      if not sameBytes(buffered, r.rendered):
        fail(suite & "/" & r.row & ": the 256-byte pull render differs: " &
            report(buffered, r.rendered))
      if asCodepointSpans(buffered, bufSpans) != r.spans:
        fail(suite & "/" & r.row & ": the 256-byte pull render's generation spans differ: got " &
            $asCodepointSpans(buffered, bufSpans) & ", want " & $r.spans)
      let (chunked7, spans7) = renderChunked[7](m, tables, r.context, r.clock)
      if not sameBytes(chunked7, r.rendered):
        fail(suite & "/" & r.row & ": the 7-byte window render differs: " &
            report(chunked7, r.rendered))
      if asCodepointSpans(chunked7, spans7) != r.spans:
        fail(suite & "/" & r.row & ": the 7-byte window render's generation spans differ: got " &
            $asCodepointSpans(chunked7, spans7) & ", want " & $r.spans)
      let (chunked1, spans1) = renderChunked[1](m, tables, r.context, r.clock)
      if not sameBytes(chunked1, r.rendered):
        fail(suite & "/" & r.row & ": the 1-byte window render differs: " &
            report(chunked1, r.rendered))
      if asCodepointSpans(chunked1, spans1) != r.spans:
        fail(suite & "/" & r.row & ": the 1-byte window render's generation spans differ: got " &
            $asCodepointSpans(chunked1, spans1) & ", want " & $r.spans)
      inc okExact
  # Render invariance under fresh drivers:
  # the same compiled template rendered twice delivers byte-equal output.
  let srcKeep = templateSource("moonlight")
  let (mKeep, tablesKeep) = parseJinjaTemplate(srcKeep)
  let rowKeep = rows("moonlight")[0]
  doAssert renderAllPull(mKeep, tablesKeep, rowKeep.context, rowKeep.clock) ==
      renderAllPull(mKeep, tablesKeep, rowKeep.context, rowKeep.clock),
      "two renders of the same artifact through fresh drivers differed"

  doAssert okExact == 77, "expected 77 rendered ok rows across 18 suites, checked " & $okExact
  doAssert gapRows == 13, "expected 13 gap rows across 18 suites, skipped " & $gapRows
  doAssert errRaised == 16, "expected 16 err rows, checked " & $errRaised


# Window-capacity contract of the delivery `Cursor`:
#   an append the window cannot hold raises the located `ceWindow` error naming
#   capacity and shortfall, writing nothing
# the measuring cursor counts without touching storage, no raise past any size

proc testWindowContract() =

  var small: array[4, char]
  var sb = over(small)
  try:
    sb.add "hello"
    doAssert false, "an append past the window capacity did not raise"
  except JinjaError as e:
    doAssert "render window capacity 4 exceeded, 1 more bytes needed" in e.what, e.what
  doAssert sb.len == 0, "the overflowing append wrote nothing, the check preceding the copy"
  # 4 bytes fit exactly, the whole capacity usable.
  var sb2 = over(small)
  sb2.add "abcd"
  doAssert sb2.len == 4
  try:
    sb2.add 'e'
    doAssert false, "a one-byte append at full capacity did not raise"
  except JinjaError as e:
    doAssert "render window capacity 4 exceeded, 1 more bytes needed" in e.what, e.what
  # A measuring cursor answers the byte count without storage, no raise past any size.
  var measure = measureBuf()
  measure.add "hello"
  doAssert measure.len == 5


# Boundary shapes of the delivery window on one corpus row.
# Oversized, exact-size, 1-byte and stop-then-resume windows all deliver the recorded bytes.

proc testBoundaryShapes() =

  let src = templateSource("deepseekv2lite")
  let (m, tables) = parseJinjaTemplate(src)
  let row = loadRow("deepseekv2lite", "assistant_history")
  let want = row.rendered

  # A buffer larger than the whole render takes everything in one pull, then reports 0.
  var dBig = startJinjaRender(m, tables, row.context, row.clock)
  var big = newSeq[char](want.len + 1)
  let n1 = pullInto(dBig, big)
  doAssert n1 == want.len, "an oversized buffer took " & $n1 & " of " & $want.len & " bytes"
  doAssert bytesOf(big, n1) == want, "the one-pull render differs from the recorded bytes"
  doAssert pullInto(dBig, big) == 0, "a completed render kept returning bytes"

  # A buffer exactly the render size also drains in one pull.
  var dExact = startJinjaRender(m, tables, row.context, row.clock)
  var exact = newSeq[char](want.len)
  let n2 = pullInto(dExact, exact)
  doAssert n2 == want.len, "an exact-size buffer took " & $n2 & " of " & $want.len & " bytes"
  doAssert bytesOf(exact, n2) == want, "the exact-size render differs from the recorded bytes"
  doAssert pullInto(dExact, exact) == 0, "a completed render kept returning bytes"

  # A 1-byte buffer gives every byte its own pull, which forces mid-piece drains.
  var dOne = startJinjaRender(m, tables, row.context, row.clock)
  var one: array[1, char]
  var acc = ""
  while true:
    let n = pullInto(dOne, one)
    if n == 0:
      break
    doAssert n == 1, "a 1-byte buffer pull returned " & $n
    acc.add one[0]
  doAssert acc == want, "the 1-byte render differs from the recorded bytes"

  # A consumer that stops mid-drain and resumes never re-receives a byte.
  var dStop = startJinjaRender(m, tables, row.context, row.clock)
  var window = newSeq[char](16)
  var head = ""
  block stopEarly:
    for _ in 0 ..< 3:
      let n = pullInto(dStop, window)
      if n == 0:
        break
      head.add bytesOf(window, n)
  doAssert head.len > 0 and head.len < want.len, "the early stop covered the whole render"
  doAssert head == want[0 ..< head.len], "the bytes before the stop diverged from the recording"
  var tail = ""
  while true:
    let n = pullInto(dStop, one)
    if n == 0:
      break
    tail.add one[0]
  doAssert tail == want[head.len ..< want.len], "the resumed bytes overlapped or diverged"
  doAssert head & tail == want, "stop-then-resume is not the whole render"


# A zero-capacity buffer reports 0 without stepping the render.

proc testZeroCapacityBuffer() =

  let src = templateSource("deepseekv2lite")
  let (m, tables) = parseJinjaTemplate(src)
  let row = loadRow("deepseekv2lite", "assistant_history")

  var d = startJinjaRender(m, tables, row.context, row.clock)
  var empty: array[0, char]
  doAssert pullInto(d, empty) == 0, "a zero-capacity buffer did not report 0"

  # the untouched driver still delivers the whole render byte-exact
  var acc = ""
  var window = newSeq[char](256)
  while true:
    let n = pullInto(d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == row.rendered, "the render after a zero-capacity pull differs from the recorded bytes"


# Partial consumption stops mid-piece, and resumption from the same driver
# completes the render without losing or re-handing a byte.

proc testPartialConsumptionResumes() =

  for suite in ["moonlight", "qwen3"]:
    let src = templateSource(suite)
    let (m, tables) = parseJinjaTemplate(src)
    let r = rows(suite)[0]
    var pc = pullChunks[7](startJinjaRender(m, tables, r.context, r.clock))
    var head = ""
    var stoppedMidPiece = false
    for w in pc.items():
      for c in w:
        head.add c
      if pc.c.state.pend.kind != pkNone and pieceRemaining(pc.c.state.pend) > 0:
        stoppedMidPiece = true
        break
    doAssert stoppedMidPiece,
        suite & ": no window ended inside a pending piece, the mid-piece stop never happened"
    doAssert head.len > 0 and head.len < r.rendered.len,
        suite & ": the early stop covered the whole render"
    doAssert head == r.rendered[0 ..< head.len],
        suite & ": the bytes before the stop diverged from the recording"
    var tail = ""
    for w in pc.items():
      for c in w:
        tail.add c
    doAssert head & tail == r.rendered,
        suite & ": the resumed bytes overlapped or diverged from the recording"


func listCtx(): JinjaVal =
  ## One context holding `m`, a mixed container whose serialization exceeds a tiny window.
  var inner = DictVal()
  dictSet(inner, "alpha", strVal("one"))
  dictSet(inner, "beta", strVal("two"))
  dictSet(inner, "gamma", seqVal(@[strVal("x"), strVal("y"), strVal("z")]))
  var cd = DictVal()
  dictSet(cd, "m", dictVal(inner))
  dictVal(cd)


const listRepr = "{'alpha': 'one', 'beta': 'two', 'gamma': ['x', 'y', 'z']}"
  ## Python `repr()` of the `m` value above, the independent byte truth for the drains.


# A container emit drains as a lazy piece across pull calls byte-exact, a window far below
# the serialization forcing the drain mid-value.

proc testLazyWindowDrain() =

  let ctx = listCtx()
  let src = "{{ m }}"
  let (m, tables) = parseJinjaTemplate(src)

  var d = startJinjaRender(m, tables, ctx, 0.0)
  var window = newSeq[char](8)
  var acc = ""
  var lazyPieces = 0
  while true:
    if d.state.pend.kind == pkLazy:
      inc lazyPieces
    let n = pullInto(d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == listRepr, "the lazy-piece drain differs from the container repr"
  doAssert lazyPieces >= 3, "the lazy piece drained in fewer than three pulls, " &
      "the mid-piece drain is unobserved"


# A `~` concat emit accumulates its leaves into one string pending piece.
# An 8-byte window still forces the drain across several pulls mid-value.

proc testConcatWindowDrain() =

  let ctx = listCtx()
  let src = "{{ m ~ '::' ~ m }}"
  let (m, tables) = parseJinjaTemplate(src)
  let want = listRepr & "::" & listRepr

  var d = startJinjaRender(m, tables, ctx, 0.0)
  var window = newSeq[char](8)
  var acc = ""
  var lazyPulls = 0
  while true:
    if d.state.pend.kind == pkStr:
      inc lazyPulls
    let n = pullInto(d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == want, "the concat drain differs from the expected operand order"
  doAssert lazyPulls > 2, "the concat drained in fewer than three pulls, the " &
      "mid-value drain is unobserved"


# Resumption across a value boundary:
# an emit value longer than the buffer drains across pull calls through the pending piece.

proc testValueBoundary() =

  let longA = repeat("alpha-", 50)
  let longB = repeat("beta-", 40)
  let src = "{% for m in messages %}{{ m.content }}{% endfor %}"
  var msgs = newSeq[JinjaVal]()
  for c in [longA, longB]:
    var md = DictVal()
    dictSet(md, "content", strVal(c))
    msgs.add dictVal(md)
  var ctxd = DictVal()
  dictSet(ctxd, "messages", seqVal(msgs))
  let ctx = dictVal(ctxd)

  let (m, tables) = parseJinjaTemplate(src)
  let want = renderToString(src, ctx, 0.0)
  doAssert want == longA & longB, "the string render is not the contents concatenation"

  let (pulled, _) = renderPull(m, tables, ctx, 0.0, 8)
  doAssert pulled == want, "the 8-byte pull render differs across the value boundary"
  let whole = renderAllPull(m, tables, ctx, 0.0)
  doAssert whole == want, "the buffered pull render differs across the value boundary"


# Span pieces copy out of `CompiledTemplate.jinja` and drain across calls byte-exact.

proc testSpanDrain() =

  let verbatim = repeat("literal text ", 15)
  let src = verbatim & "{{ m }}"
  var ctxd = DictVal()
  dictSet(ctxd, "m", strVal("emit"))
  let ctx = dictVal(ctxd)
  let (m, tables) = parseJinjaTemplate(src)
  let want = renderToString(src, ctx, 0.0)
  doAssert want == verbatim & "emit", "the string render is not the expected bytes"

  # A window far below the verbatim run forces span pieces through several pulls.
  var d = startJinjaRender(m, tables, ctx, 0.0)
  var window = newSeq[char](8)
  var acc = ""
  var spanPulls = 0
  while true:
    if d.state.pend.kind == pkSpan:
      inc spanPulls
    let n = pullInto(d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == want, "the span-drain render differs from the string render"
  doAssert spanPulls > 0, "no pull drained a pending span piece"


# A raise inside a for-filter propagates per the pull contract. The loop cursor stays
# committed past the failed item and a repull resumes after it. One-byte window first,
# where every byte delivered before the failing call is already with the caller.

proc testFilterRaiseRepull() =

  var msgs = newSeq[JinjaVal]()
  msgs.add strVal("aa")
  msgs.add intVal(7)
  msgs.add strVal("ab")
  var cd = DictVal()
  dictSet(cd, "xs", seqVal(msgs))
  let ctx = dictVal(cd)
  # `x[0]` raises on the integer item and passes the strings through the filter comparison.
  let src = "pre{% for x in xs if x[0] == 'a' %}[{{ x }}]{% endfor %}post"
  let (m, tables) = parseJinjaTemplate(src)
  let want = "pre[aa][ab]post"
  # One-shot render propagates the same raise, the filtered strings never reaching it.
  try:
    discard renderToString(src, ctx, 0.0)
    doAssert false, "the one-shot render did not propagate the failing filter"
  except JinjaError as e:
    doAssert "not subscriptable" in e.what, e.what

  var d = startJinjaRender(m, tables, ctx, 0.0)
  var win1 = newSeq[char](1)
  var acc = ""
  var raised = false
  var message = ""
  try:
    while true:
      let n = pullInto(d, win1)
      if n == 0:
        break
      acc.add bytesOf(win1, n)
  except JinjaError as e:
    raised = true
    message = e.what
  doAssert raised, "the failing filter did not raise"
  doAssert "not subscriptable" in message,
      "the error did not name the failed operation: " & message
  doAssert acc == "pre[aa]", "the caller-held bytes at the raise are not exactly the prefix"

  # Repull skips nothing. The integer item stays consumed and the render completes.
  var rest = newSeq[char](64)
  while true:
    let n = pullInto(d, rest)
    if n == 0:
      break
    acc.add bytesOf(rest, n)
  doAssert acc == want, "the repull after the raise differs from the single-shot render"

  # Wide window. The prefix and the failed item's evaluation land in one call, whose
  # window bytes are discarded and never reach the caller. The repull resumes after them,
  # discarded prefix included.
  var dWide = startJinjaRender(m, tables, ctx, 0.0)
  var wide = newSeq[char](64)
  var wideAcc = ""
  var wideRaised = false
  try:
    while true:
      let n = pullInto(dWide, wide)
      if n == 0:
        break
      wideAcc.add bytesOf(wide, n)
  except CatchableError:
    wideRaised = true
  doAssert wideRaised, "the wide-window filter raise did not raise"
  doAssert wideAcc == "", "the failing call returned bytes: <" & wideAcc & ">"
  var wideRest = ""
  while true:
    let n = pullInto(dWide, wide)
    if n == 0:
      break
    wideRest.add bytesOf(wide, n)
  doAssert wideRest == "[ab]post",
      "the wide-window repull did not resume after the discarded bytes"


# A streamed macro call resolves names against the caller's scopes only before the call
# and against its own scopes only inside the body:
# the macro scope is popped on close.

proc testMacroScopePop() =

  let leakCaller = "{% macro mm(q) %}[{{ q }}]{% endmacro %}" &
      "{% set q = 'caller' %}{{ mm('inner') }}:{{ q }}"
  doAssert renderToString(leakCaller, listCtx()) == "[inner]:caller",
      "the macro body read a caller binding set after the call"

  let leakBody = "{% macro mm() %}{% set z = 'body' %}{{ z }}{% endmacro %}{{ mm() }}:{{ z }}"
  doAssert renderToString(leakBody, listCtx()) == "body:",
      "a macro body binding leaked into the caller's name resolution"


# Every close shape pops exactly the row's own scope range, the shapes being
# for exhaust, empty-body for, for break, macro body end, macro boundary stop
# and generation span.
# A binding set in an enclosing row's scope survives every inner close.
# A close popping one scope past the row's mark loses an outer binding, showing
# the undefined fallback in its place in the render.

proc testNestedRowClosesKeepOuterScope() =

  let src = "{% macro mm() %}{% set q = 'Q' %}{{ q }}{% endmacro %}" &
      "{% for i in items %}{% set x = 'X' ~ i %}" &
      "{% for j in inner %}{% endfor %}{{ x }}" &
      "{% for j in inner %}{% if j == 'b' %}{% break %}{% endif %}{{ x }}{% endfor %}" &
      "{{ mm() }}{{ x }}{% generation %}G{% endgeneration %}{{ x }}{% endfor %}"
  # Per outer item `a` the empty-body for close, the break close, the macro close
  # and the generation close each leave `x` intact, then the same for `b`.
  let want = "XaXaQXaGXaXbXbQXbGXb"
  var ctx = DictVal()
  dictSet(ctx, "items", seqVal(@[strVal("a"), strVal("b")]))
  dictSet(ctx, "inner", seqVal(@[strVal("p"), strVal("b"), strVal("q")]))
  doAssert renderToString(src, dictVal(ctx)) == want,
      "a row close popped past the row's own scope mark and lost an outer binding"


# `tojson` with `ensure_ascii` exercises every escape shape, control characters included,
# plus the UTF-16 surrogate pair for a code point beyond the Basic Multilingual Plane. The corpus records `ensure_ascii`-off output,
# so this suite checks the engine's escape set directly:
# uppercase hex digits and the surrogate pair.

proc testEnsureAsciiEscapes() =

  let raw = strVal("a\tb\rc\bd\x0Ce\x01f\"g\\h<i>j&k'lém😀n")
  doAssert toJson(raw, JsonOpts(ensureAscii: true)) ==
      "\"a\\tb\\rc\\bd\\fe\\u0001f\\\"g\\\\h\\u003ci\\u003ej\\u0026k\\u0027l\\u00E9m\\uD83D\\uDE00n\"",
      "the ensure_ascii rendering differs from the expected escapes"
  doAssert toJson(raw) ==
      "\"a\\tb\\rc\\bd\\fe\\u0001f\\\"g\\\\h\\u003ci\\u003ej\\u0026k\\u0027lém😀n\"",
      "the raw-utf8 rendering differs from the expected escapes"


# Allocation counting. Compiled only under `-d:nimAllocStats`. A failing doAssert
# there hangs the run with no output.

privateAccess(AllocStats)

template allocsOf(body: untyped): int =
  ## Counts `alloc` calls made by `body`, with allocator state warmed by the caller.
  let before = getAllocStats()
  body
  (getAllocStats() - before).allocCount


proc testAllocDrainWindow() =

  let src = templateSource("deepseekv2lite")
  let (m, tables) = parseJinjaTemplate(src)
  let row = loadRow("deepseekv2lite", "assistant_history")

  # Warm-up renders, uncounted:
  # first-touch allocator state settles here.
  for _ in 0 ..< 3:
    discard renderPull(m, tables, row.context, row.clock, 1)
    discard renderAllPull(m, tables, row.context, row.clock)
    discard renderToString(src, row.context, row.clock)

  # A pullInto call that enters on a pending piece with bytes left only drains it, no step runs,
  # so it must allocate nothing.
  var d = startJinjaRender(m, tables, row.context, row.clock)
  var one: array[1, char]
  var acc = ""
  var drainCalls = 0
  while true:
    let pending = d.state.pend.kind != pkNone and pieceRemaining(d.state.pend) > 0
    let before = getAllocStats()
    let n = pullInto(d, one)
    let used = (getAllocStats() - before).allocCount
    if n == 0:
      break
    acc.add one[0]
    if pending:
      inc drainCalls
      doAssert used == 0, "a pullInto that only drained pending bytes allocated " & $used
  doAssert acc == row.rendered, "the counted render disagrees with the recorded bytes"
  doAssert drainCalls > 0, "no pending-piece drain was counted"
  echo "t_corpus alloc: ", drainCalls, " pending-piece drain calls, all 0 allocs"

  # Whole-render comparison:
  # the pullInto path against the string path, whose count also covers parsing the template
  # and therefore bounds the pull total from above. The counted render
  # pulls into the test's own 7-byte buffer, exercising the smallest-window path.
  var dTotal = startJinjaRender(m, tables, row.context, row.clock)
  var seven: array[7, char]
  let pullAllocs = allocsOf:
    while true:
      let n = pullInto(dTotal, seven)
      if n == 0:
        break
  let strAllocs = allocsOf:
    discard renderToString(src, row.context, row.clock)
  doAssert pullAllocs <= strAllocs, "the pullInto render allocated " & $pullAllocs &
      " against the string render's " & $strAllocs
  echo "t_corpus alloc: full pull render ", pullAllocs, " allocs, string render ", strAllocs,
      " allocs"


# Micro attribution over a 10-message for-loop context:
# a warm-up render per template stays uncounted, then `getAllocStats()` deltas measure
# the counted renders. DictGet lookup floors are measured in the same run.
# Every assert below is an upper bound taken at this binary's measured value:
# allocation inflation fails the bound, a lower count passing it.
# - an emit-role render costs nothing beyond the loop machinery, its lookups included
# - an emit-content render costs at most one lookup copy per emit, the accepted residual
# - a punctuator evaluation and the pending-piece move of an emit string cost 0

proc testAllocMicro() =

  const msgCount = 10
  let iters = 50

  func msgVal(role, content: string): JinjaVal =
    ## Builds one chat message carrying the two keys the templates read.
    var d = DictVal()
    dictSet(d, "role", strVal(role))
    dictSet(d, "content", strVal(content))
    dictVal(d)

  var msgs = newSeq[JinjaVal]()
  for i in 0 ..< msgCount:
    msgs.add msgVal(if i mod 2 == 0: "user" else: "assistant",
        "Message " & $i & ": please continue the conversation and stay on topic.")
  var cd = DictVal()
  dictSet(cd, "messages", seqVal(msgs))
  let ctx = dictVal(cd)

  # Lookup floors over one message, each warmed by one uncounted call:
  # `role` strings are literal-backed so a lookup copies nothing, `content` strings are
  # runtime-built so a lookup copies once.
  let msg1 = ctx.d.dictGet("messages").xs.items[1]
  discard msg1.d.dictGet("role")
  discard msg1.d.dictGet("content")
  let dgRole = allocsOf:
    discard msg1.d.dictGet("role")
  let dgContent = allocsOf:
    discard msg1.d.dictGet("content")

  template countRenders(src: string, n: int): int =
    ## Warms one pull render uncounted, then totals `n` pull renders through `getAllocStats()` deltas.
    let (m, tables) = parseJinjaTemplate(src)
    let want = renderToString(src, ctx, 0.0)
    doAssert renderAllPull(m, tables, ctx, 0.0) == want,
        "the micro pullInto render differs from the string render for " & src
    allocsOf:
      for _ in 0 ..< n:
        discard renderAllPull(m, tables, ctx, 0.0)

  let loopOnly = countRenders("{% for m in messages %}x{% endfor %}", iters)
  let roleRenders = countRenders("{% for m in messages %}{{ m.role }}{% endfor %}", iters)
  let contentRenders = countRenders("{% for m in messages %}{{ m.content }}{% endfor %}", iters)
  let bothSrc = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
  let bothRenders = countRenders(bothSrc, iters)

  doAssert roleRenders <= loopOnly + iters * msgCount * dgRole,
      "the emit-role render cost " & $(roleRenders - loopOnly) &
      " allocs beyond the loop baseline"
  doAssert contentRenders <= loopOnly + iters * msgCount * dgContent,
      "the emit-content render cost " & $(contentRenders - loopOnly) &
      " allocs beyond the loop baseline"
  doAssert bothRenders <= loopOnly + iters * msgCount * (dgRole + dgContent),
      "the two-emit render cost " & $(bothRenders - loopOnly) &
      " allocs beyond the loop baseline"
  doAssert (roleRenders - loopOnly) div iters <= msgCount and
      (contentRenders - loopOnly) div iters <= msgCount,
      "an emit cost more than the accepted one-lookup residual"
  echo "t_corpus alloc: emit-role ", roleRenders, ", emit-content ", contentRenders,
      ", emit both ", bothRenders, ", loop baseline ", loopOnly,
      " allocs over ", iters, " renders each"


proc testAllocSerializer() =

  let iters = 50

  func toolsVal(): JinjaVal =
    ## One function-tool definition, the bench tool schema shape.
    var cityProp = DictVal()
    dictSet(cityProp, "type", strVal("string"))
    var props = DictVal()
    dictSet(props, "city", dictVal(cityProp))
    var params = DictVal()
    dictSet(params, "type", strVal("object"))
    dictSet(params, "properties", dictVal(props))
    var fn = DictVal()
    dictSet(fn, "name", strVal("get_weather"))
    dictSet(fn, "description", strVal("Current weather for one city"))
    dictSet(fn, "parameters", dictVal(params))
    var tool = DictVal()
    dictSet(tool, "type", strVal("function"))
    dictSet(tool, "function", dictVal(fn))
    seqVal(@[dictVal(tool)])

  # Direct tojson of the tool schema. The writer drains into a growable buffer with no
  # presize pass, so a call costs one allocation for the buffer, one for the literal queue's
  # grown capacity and one for the stack behind the schema's two nested containers.
  let tools = toolsVal()
  # warm-up call, excluded from the counted region
  discard toJson(tools)
  let tjAllocs = allocsOf:
    for _ in 0 ..< iters:
      discard toJson(tools)
  doAssert tjAllocs <= 3 * iters, "toJson of the tool schema cost " & $(tjAllocs div iters) &
      " allocations per call against the measured three"

  # Same schema through the pullInto render, driver setup uncounted:
  # the counted region holds only the pull loop, and the render costs
  # the filter's argument list plus the serializer's container stack.
  const tJson = "{{ tools|tojson }}"
  var cd = DictVal()
  dictSet(cd, "tools", tools)
  let ctx = dictVal(cd)
  let (m, tables) = parseJinjaTemplate(tJson)
  let want = renderToString(tJson, ctx, 0.0)

  var dWarm = startJinjaRender(m, tables, ctx, 0.0)
  var bufWarm = newSeq[char](256)
  var warm = ""
  while true:
    let n = pullInto(dWarm, bufWarm)
    if n == 0:
      break
    warm.add bytesOf(bufWarm, n)
  doAssert warm == want, "the pullInto render differs from the string render"

  var buf = newSeq[char](256)
  var renderAllocs = 0
  for _ in 0 ..< iters:
    var di = startJinjaRender(m, tables, ctx, 0.0)
    let renderCost = allocsOf:
      while true:
        let n = pullInto(di, buf)
        if n == 0:
          break
    renderAllocs += renderCost
  doAssert renderAllocs <= 4 * iters, "the tojson pullInto render cost " &
      $(renderAllocs div iters) & " allocations per render against the measured four"

  # A container emit costs one allocation per emit for the lookup copy plus one per
  # render for the serializer's container stack, over the loop machinery.
  var msgs = newSeq[JinjaVal]()
  for i in 0 ..< 10:
    var md = DictVal()
    dictSet(md, "n", strVal($i))
    msgs.add dictVal(md)
  var mcd = DictVal()
  dictSet(mcd, "messages", seqVal(msgs))
  let loopCtx = dictVal(mcd)

  template countRenders(src: string, n: int): int =
    ## Warms one pull render uncounted, then totals `n` renders through
    ## `getAllocStats()` deltas with one driver per render, as above.
    var (mm, ts) = parseJinjaTemplate(src)
    let wantLocal = renderToString(src, loopCtx, 0.0)
    var dWarm2 = startJinjaRender(mm, ts, loopCtx, 0.0)
    var bufWarm2 = newSeq[char](256)
    var accWarm = ""
    while true:
      let got = pullInto(dWarm2, bufWarm2)
      if got == 0:
        break
      accWarm.add bytesOf(bufWarm2, got)
    doAssert accWarm == wantLocal, "the micro render differs for " & src
    var total = 0
    for _ in 0 ..< n:
      var di = startJinjaRender(mm, ts, loopCtx, 0.0)
      var bi = newSeq[char](256)
      let renderCost = allocsOf:
        while true:
          let got = pullInto(di, bi)
          if got == 0:
            break
      total += renderCost
    total

  let loopOnly = countRenders("{% for m in messages %}x{% endfor %}", iters)
  let strEmits = countRenders("{% for m in messages %}{{ m.n }}{% endfor %}", iters)
  let dictEmits = countRenders("{% for m in messages %}{{ m }}{% endfor %}", iters)
  # Runtime-built message values keep the engine's one-lookup-copy residual per emit.
  doAssert strEmits <= loopOnly + iters * 10, "the string emit cost " &
      $(strEmits - loopOnly) & " allocations beyond the loop baseline"
  # A container emit through the lazy piece costs one allocation per emit over the string
  # emit and one per render for the serializer's container stack.
  doAssert dictEmits <= strEmits + iters * 12, "the container emit cost " &
      $(dictEmits - strEmits) & " allocations beyond the string emit"

  echo "t_corpus alloc: tojson direct ", tjAllocs div iters, "/call, tojson render ",
      renderAllocs div iters, "/render, string emit ", (strEmits - loopOnly) div iters,
      ", container emit ", (dictEmits - loopOnly) div iters,
      " allocs beyond the loop baseline over ", iters, " renders"


proc main() =
  testStepsTotality()
  testEqualityRejectsChange()
  testCorpusDelivery()
  testWindowContract()
  testBoundaryShapes()
  testZeroCapacityBuffer()
  testPartialConsumptionResumes()
  testLazyWindowDrain()
  testConcatWindowDrain()
  testValueBoundary()
  testSpanDrain()
  testFilterRaiseRepull()
  testMacroScopePop()
  testNestedRowClosesKeepOuterScope()
  testEnsureAsciiEscapes()
  when defined(nimAllocStats):
    testAllocDrainWindow()
    testAllocMicro()
    testAllocSerializer()
  echo "t_corpus: 77 ok rows byte-exact through pull, compose and render, 13 gap rows loud, " &
      "16 err rows raise the recorded error"

main()
