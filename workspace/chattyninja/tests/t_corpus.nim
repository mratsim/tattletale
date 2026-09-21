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
## - byte-exact delivery of every ok row through `pullAll`, a buffered pull loop,
##   windowed consumers, every err row raising the recorded error, declared gaps loud
## - the window contract, the for-filter raise with repull-resume, the macro scope pop,
##   the ensure_ascii escape shapes, the compiled artifact layout
## - allocation counting under `-d:nimAllocStats`
##
## Run:
##   $ nim test_chattyninja

{.experimental: "views".}

import std/[algorithm, importutils, macros, os, strutils, unicode]
import cnj_types, jinja_data_model, jinja_serialize, cnj_parse, cnj_engine
import workspace/data_structures/src/small_seqs
import workspace/zstd/zstd_highlevel

# Corpus row reader.
# ---------------------------------------------------------------------------
# JSON is parsed here rather than through `std/json`, because template output observes
# dict insertion order (`items`, `tojson`, `{% for %}` over a mapping) while `std/json`
# stores object members in a hash table. This parser keeps `DictVal.keys` in document order,
# the order the recording captured.
#
# The ground truth is self-contained.
# Every ok row embeds `rendered`. Every `err_*` row embeds `expected_error
# {exception, message, offset, span}`, the raise call's name-token span into the suite template.

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

# JSON reader
# ---------------------------------------------------------------------------

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
  ## - a full drain equals the whole-render `pullAll`, the next pull after it reports 0
  ## - partial consumption resumes from the driver's fields, no byte re-handed
  ##
  ## The compiled template stays at the consumer's scope, `CompiledTemplate.jinja` borrowing
  ## the template text. The template is a ref, so a `Context` embeds in another object
  ## without losing the borrowed view.
  c: Context
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

# Row reading
# ---------------------------------------------------------------------------

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

proc renderPull(m: CompiledTemplate, sym: var CompiledSymbols, ctx: JinjaVal, clock: float64, cap: int): string =
  ## Renders through `pull` with a `cap`-byte caller buffer, accumulating every fill.
  var c = startRender(m, sym, ctx, clock)
  var buf = newSeq[char](cap)
  while true:
    let n = pull(c, buf)
    if n == 0:
      break
    for i in 0 ..< n:
      result.add buf[i]

proc renderAllPull(m: CompiledTemplate, sym: var CompiledSymbols, ctx: JinjaVal, clock: float64): string =
  ## Renders through `pullAll` with a fresh render context.
  var c = startRender(m, sym, ctx, clock)
  pullAll(c)


func pullChunks[N: static int](c: Context): PullChunks[N] =
  ## Builds the windowed pull machine over the render context `c` of a compiled template.
  PullChunks[N](c: c)

iterator items[N: static int](p: var PullChunks[N]): openArray[char] =
  ## Yields the render as bounded windows:
  ## - one `pull` call fills the window, the view carries at most N bytes
  ## - the tail window closes the render, a 0 count ends the stream
  while true:
    let n = pull(p.c, p.buf)
    if n == 0:
      break
    yield p.buf.toOpenArray(0, n - 1)

proc renderChunked[N: static int](m: CompiledTemplate, sym: var CompiledSymbols, ctx: JinjaVal, clock: float64): string =
  ## Renders one row through `N`-byte windows, accumulating every window.
  var pc = pullChunks[N](startRender(m, sym, ctx, clock))
  for w in pc.items():
    for ch in w:
      result.add ch

# Compiled-form ABI over the node budget, the arena's POD status, the shared
# read-only `CompiledTemplate` and dispatch totality.
# ---------------------------------------------------------------------------
macro fieldNames(T: type): untyped =
  ## Returns the field names of an object type, in declaration order.
  var t = getTypeImpl(T)
  if t.kind == nnkBracketExpr:
    t = getTypeImpl(t[1])
  while t.kind == nnkRefTy or t.kind == nnkSym:
    t = getTypeImpl(t[0])
  expectKind t, nnkObjectTy
  let rec = t[2]
  expectKind rec, nnkRecList
  result = newTree(nnkBracket)
  for f in rec:
    case f.kind
    of nnkSym:
      result.add newLit(f.strVal)
    of nnkIdentDefs:
      for k in 0 ..< f.len - 2:
        result.add newLit(f[k].strVal)
    else:
      error("unexpected field node", f)

const
  nodeFields: array[2, string] = fieldNames(Node)
  tmplFields: array[2, string] = fieldNames(CompiledTemplate)
  contextFields: array[3, string] = fieldNames(Context)

static:
  # The node vocabulary is the corpus-derived ten, in declaration order.
  assert NodeKind.high.ord + 1 == 10, "NodeKind must hold the ten corpus-derived kinds"
  assert $NodeKind.low == "nkVerbatim", "nkVerbatim opens the enum"
  assert $NodeKind.high == "nkMacroDef", "nkMacroDef closes the enum"

  # Node is `{kind, slots}` and nothing else. `SmallSeq[5, int32]` measures 40 bytes,
  # the one-byte kind pads to the tail pointer's alignment, so the per-node budget is
  # 48 bytes. Only a proc, `string` or `seq` member would change `sizeof`, so this pair
  # of assertions checks the field set and the size, which together fix the layout.
  assert nodeFields == ["kind", "slots"], "Node must be exactly {kind, slots}"
  assert sizeof(SmallSeq[5, int32]) == 40, "SmallSeq[5, int32] layout: " & $sizeof(SmallSeq[5, int32])
  assert sizeof(Node) == 48, "Node layout: " & $sizeof(Node)
  assert alignof(Node) == 8, "the payload tail pointer aligns the node to 8 bytes"
  assert not (Node is ref), "nodes are POD in one seq, never a ref box"

  # CompiledTemplate is the borrowed text plus the arena, two fields, so nothing render-mutable
  # is reachable from the artifact and one artifact can serve several render instantiations.
  assert tmplFields == ["jinja", "nodes"], "CompiledTemplate must be exactly {jinja, nodes}"
  assert CompiledTemplate is ref, "CompiledTemplate is a shared ref"

  # Context bundles the shared artifact and symbols with the per-instantiation state.
  assert contextFields == ["tmpl", "symbols", "state"], "Context must be exactly {tmpl, symbols, state}"

  # Dispatch is one array total over the enum. The array's type makes an uncovered kind
  # a compile error, and the length check keeps the table total across an enum rename.
  assert Steps.len == NodeKind.high.ord + 1, "steps must be total over NodeKind"

# A nil step would be a hole in the table:
#   a render would jump through a null pointer rather than
# report the gap, so totality is checked over every kind, not merely counted.
for k in NodeKind:
  doAssert not Steps[k].isNil, "steps has no entry for " & $k

# A `Node` must move by assignment with no reference left behind:
#   this is the property that lets the arena be one allocation. An assignment deep-copies
# the spilled payload, so the copy stays valid after the source's block is freed.
#
# The check runs at runtime, the compile-time VM cannot run the payload's allocator.
var a = Node(kind: nkEmit)
for slot in 7'i32 .. 12'i32:
  a.slots.add slot
var arena = @[a, a]
arena[1] = arena[0]
arena[0].slots[0] = 99'i32
doAssert arena[1].slots[0] == 7'i32, "an assignment must deep-copy a spilled payload"
doAssert arena.len == 2, "the arena moved by assignment with no reference left behind"

# RenderState layout contract, size locked:
# 392 = 368 + 8 for the JinjaVal cut variant,
# 8 for the serializer's raw-body end bound, 8 for realignment,
# 8 for the pending cut piece, whose string descriptor reuses the string branch's slot
# and whose two byte bounds widen the variant payload to 24 bytes, +8 after realignment.
doAssert sizeof(RenderState) == 400, "RenderState is a plain value type"

# The equality must reject a one-byte change, since a comparison that cannot fail makes
# the corpus walk vacuous.
block equalityRejectsChange:
  doAssert sameBytes("abc", "abc")
  doAssert not sameBytes("abc", "abd"), "sameBytes accepted a changed byte"
  doAssert not sameBytes("abc", "abcd"), "sameBytes accepted a length change"
  doAssert not sameBytes("", "a"), "sameBytes accepted empty against non-empty"

# Every recorded corpus row through the three delivery paths.
# ---------------------------------------------------------------------------
block corpusDelivery:
  const parseable = ["deepseekv2lite", "gemma3", "gemma4", "glm47flash", "glm53flash",
      "gptoss20b", "kimi", "lagunaxs21", "lfm25", "ling30", "mimo25", "mistral7bv01",
      "moonlight", "northminicode10", "qwen3", "qwen35", "qwen36", "qwen38flashnext"]
  var okExact = 0
  var gapRows = 0
  var errRaised = 0
  for suite in parseable:
    let src = templateSource(suite)
    var (m, tables) = parseTemplate(src)
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
      # windowed consumers all deliver the recorded bytes. A declared gap raises loud.
      var raised = ""
      var whole = ""
      try:
        whole = renderAllPull(m, tables, r.context, r.clock)
      except JinjaError as e:
        # A declared gap surfaces as a gap, never as a wrong answer:
        # - declared constructs and unimplemented filter names raise with cause `ceUnimplemented`
        # - any other raise fails the row
        if e.cause == ceUnimplemented:
          inc gapRows
          continue
        raised = e.what
      if raised.len > 0:
        fail(suite & "/" & r.row & ": the whole render raised " & raised)
      if not sameBytes(whole, r.rendered):
        fail(suite & "/" & r.row & ": the whole render differs: " & report(whole, r.rendered))
      let buffered = renderPull(m, tables, r.context, r.clock, 256)
      if not sameBytes(buffered, r.rendered):
        fail(suite & "/" & r.row & ": the 256-byte pull render differs: " &
            report(buffered, r.rendered))
      let chunked7 = renderChunked[7](m, tables, r.context, r.clock)
      if not sameBytes(chunked7, r.rendered):
        fail(suite & "/" & r.row & ": the 7-byte window render differs: " &
            report(chunked7, r.rendered))
      let chunked1 = renderChunked[1](m, tables, r.context, r.clock)
      if not sameBytes(chunked1, r.rendered):
        fail(suite & "/" & r.row & ": the 1-byte window render differs: " &
            report(chunked1, r.rendered))
      inc okExact
  # Render invariance under fresh drivers:
  # the same compiled template rendered twice delivers byte-equal output.
  let srcKeep = templateSource("moonlight")
  var (mKeep, tablesKeep) = parseTemplate(srcKeep)
  let rowKeep = rows("moonlight")[0]
  doAssert renderAllPull(mKeep, tablesKeep, rowKeep.context, rowKeep.clock) ==
      renderAllPull(mKeep, tablesKeep, rowKeep.context, rowKeep.clock),
      "two renders of the same artifact through fresh drivers differed"

  doAssert okExact == 64, "expected 64 rendered ok rows across 18 suites, checked " & $okExact
  doAssert gapRows == 19, "expected 19 gap rows across 18 suites, skipped " & $gapRows
  doAssert errRaised == 16, "expected 16 err rows, checked " & $errRaised

# Boundary shapes of the delivery window on one corpus row.
# Oversized, exact-size, 1-byte and stop-then-resume windows all deliver the recorded bytes.
# ---------------------------------------------------------------------------
block boundaryShapes:
  let src = templateSource("deepseekv2lite")
  var (m, tables) = parseTemplate(src)
  let row = loadRow("deepseekv2lite", "assistant_history")
  let want = row.rendered

  # A buffer larger than the whole render takes everything in one pull, then reports 0.
  var dBig = startRender(m, tables, row.context, row.clock)
  var big = newSeq[char](want.len + 1)
  let n1 = pull(dBig, big)
  doAssert n1 == want.len, "an oversized buffer took " & $n1 & " of " & $want.len & " bytes"
  doAssert bytesOf(big, n1) == want, "the one-pull render differs from the recorded bytes"
  doAssert pull(dBig, big) == 0, "a completed render kept returning bytes"

  # A buffer exactly the render size also drains in one pull.
  var dExact = startRender(m, tables, row.context, row.clock)
  var exact = newSeq[char](want.len)
  let n2 = pull(dExact, exact)
  doAssert n2 == want.len, "an exact-size buffer took " & $n2 & " of " & $want.len & " bytes"
  doAssert bytesOf(exact, n2) == want, "the exact-size render differs from the recorded bytes"
  doAssert pull(dExact, exact) == 0, "a completed render kept returning bytes"

  # A 1-byte buffer gives every byte its own pull, which forces mid-piece drains.
  var dOne = startRender(m, tables, row.context, row.clock)
  var one: array[1, char]
  var acc = ""
  var calls = 0
  while true:
    let n = pull(dOne, one)
    if n == 0:
      break
    doAssert n == 1, "a 1-byte buffer pull returned " & $n
    acc.add one[0]
    inc calls
  doAssert acc == want, "the 1-byte render differs from the recorded bytes"
  doAssert calls == want.len, "expected one pull per byte, got " & $calls & " of " & $want.len

  # A consumer that stops mid-drain and resumes never re-receives a byte.
  var dStop = startRender(m, tables, row.context, row.clock)
  var window = newSeq[char](16)
  var head = ""
  block stopEarly:
    for _ in 0 ..< 3:
      let n = pull(dStop, window)
      if n == 0:
        break
      head.add bytesOf(window, n)
  doAssert head.len > 0 and head.len < want.len, "the early stop covered the whole render"
  doAssert head == want[0 ..< head.len], "the bytes before the stop diverged from the recording"
  doAssert dStop.state.cur == head.len, "cur is " & $dStop.state.cur & " but " & $head.len &
      " bytes were received"
  var tail = ""
  while true:
    let n = pull(dStop, one)
    if n == 0:
      break
    tail.add one[0]
  doAssert tail == want[head.len ..< want.len], "the resumed bytes overlapped or diverged"
  doAssert head & tail == want, "stop-then-resume is not the whole render"

# A zero-capacity buffer reports 0 without stepping the render.
# ---------------------------------------------------------------------------
block zeroCapacityBuffer:
  let src = templateSource("deepseekv2lite")
  var (m, tables) = parseTemplate(src)
  let row = loadRow("deepseekv2lite", "assistant_history")

  var d = startRender(m, tables, row.context, row.clock)
  var empty: array[0, char]
  doAssert pull(d, empty) == 0, "a zero-capacity buffer did not report 0"
  doAssert d.state.curNode != NoLink, "a zero-capacity pull stepped the render to the end"
  doAssert d.state.cur == 0, "a zero-capacity pull moved cur"
  doAssert d.state.pend.kind == pkNone, "a zero-capacity pull started a pending piece"

  # the untouched driver still delivers the whole render byte-exact
  var acc = ""
  var window = newSeq[char](256)
  while true:
    let n = pull(d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == row.rendered, "the render after a zero-capacity pull differs from the recorded bytes"

# Partial consumption stops mid-piece, and resumption from the same driver
# completes the render without losing or re-handing a byte.
# ---------------------------------------------------------------------------
block partialConsumptionResumes:
  for suite in ["moonlight", "qwen3"]:
    let src = templateSource(suite)
    var (m, tables) = parseTemplate(src)
    let r = rows(suite)[0]
    var pc = pullChunks[7](startRender(m, tables, r.context, r.clock))
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
    doAssert pc.c.state.cur == head.len,
        suite & ": cur is " & $pc.c.state.cur & " but " & $head.len & " bytes were received"
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
# ---------------------------------------------------------------------------
block lazyWindowDrain:
  let ctx = listCtx()
  let src = "{{ m }}"
  var (m, tables) = parseTemplate(src)

  var d = startRender(m, tables, ctx, 0.0)
  var window = newSeq[char](8)
  var acc = ""
  var lazyPieces = 0
  while true:
    if d.state.pend.kind == pkLazy:
      inc lazyPieces
    let n = pull(d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == listRepr, "the lazy-piece drain differs from the container repr"
  doAssert lazyPieces >= 3, "the lazy piece drained in fewer than three pulls, " &
      "the mid-piece drain is unobserved"

# A `~` concat emit streams its operands through the lazy-piece machinery, left to right,
# and an 8-byte window forces the drain across several pulls mid-value.
# ---------------------------------------------------------------------------
block concatWindowDrain:
  let ctx = listCtx()
  let src = "{{ m ~ '::' ~ m }}"
  var (m, tables) = parseTemplate(src)
  let want = listRepr & "::" & listRepr

  var d = startRender(m, tables, ctx, 0.0)
  var window = newSeq[char](8)
  var acc = ""
  var lazyPulls = 0
  while true:
    if d.state.pend.kind == pkLazy:
      inc lazyPulls
    let n = pull(d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == want, "the concat drain differs from the expected operand order"
  doAssert lazyPulls > 2, "the concat drained in fewer than three pulls, the " &
      "mid-value drain is unobserved"

# Resumption across a value boundary:
# an emit value longer than the buffer drains across pull calls through the pending piece.
# ---------------------------------------------------------------------------
block valueBoundary:
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

  var (m, tables) = parseTemplate(src)
  let want = renderToString(src, ctx, 0.0)
  doAssert want == longA & longB, "the string render is not the contents concatenation"

  let pulled = renderPull(m, tables, ctx, 0.0, 8)
  doAssert pulled == want, "the 8-byte pull render differs across the value boundary"
  let whole = renderAllPull(m, tables, ctx, 0.0)
  doAssert whole == want, "pullAll differs across the value boundary"

# Span pieces copy out of `CompiledTemplate.jinja` and drain across calls byte-exact.
# ---------------------------------------------------------------------------
block spanDrain:
  let verbatim = repeat("literal text ", 15)
  let src = verbatim & "{{ m }}"
  var ctxd = DictVal()
  dictSet(ctxd, "m", strVal("emit"))
  let ctx = dictVal(ctxd)
  var (m, tables) = parseTemplate(src)
  let want = renderToString(src, ctx, 0.0)
  doAssert want == verbatim & "emit", "the string render is not the expected bytes"

  # A window far below the verbatim run forces span pieces through several pulls.
  var d = startRender(m, tables, ctx, 0.0)
  var window = newSeq[char](8)
  var acc = ""
  var spanPulls = 0
  while true:
    if d.state.pend.kind == pkSpan:
      inc spanPulls
    let n = pull(d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == want, "the span-drain render differs from the string render"
  doAssert spanPulls > 0, "no pull drained a pending span piece"

# A raise inside a for-filter propagates per the pull contract. The loop cursor stays
# committed past the failed item and a repull resumes after it. One-byte window first,
# where every byte delivered before the failing call is already with the caller.
# ---------------------------------------------------------------------------
block filterRaiseRepull:
  var msgs = newSeq[JinjaVal]()
  msgs.add strVal("aa")
  msgs.add intVal(7)
  msgs.add strVal("ab")
  var cd = DictVal()
  dictSet(cd, "xs", seqVal(msgs))
  let ctx = dictVal(cd)
  # `x[0]` raises on the integer item and passes the strings through the filter comparison.
  let src = "pre{% for x in xs if x[0] == 'a' %}[{{ x }}]{% endfor %}post"
  var (m, tables) = parseTemplate(src)
  let want = "pre[aa][ab]post"
  # The one-shot render propagates the same raise, the filtered strings never reaching it.
  try:
    discard renderToString(src, ctx, 0.0)
    doAssert false, "the one-shot render did not propagate the failing filter"
  except JinjaError as e:
    doAssert "not subscriptable" in e.what, e.what

  var d = startRender(m, tables, ctx, 0.0)
  var win1 = newSeq[char](1)
  var acc = ""
  var raised = false
  var message = ""
  try:
    while true:
      let n = pull(d, win1)
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

  # The repull skips nothing. The integer item stays consumed and the render completes.
  var rest = newSeq[char](64)
  while true:
    let n = pull(d, rest)
    if n == 0:
      break
    acc.add bytesOf(rest, n)
  doAssert acc == want, "the repull after the raise differs from the single-shot render"

  # Wide window. The prefix and the failed item's evaluation land in one call, whose
  # window bytes are discarded and never reach the caller. The repull resumes after them,
  # discarded prefix included.
  var dWide = startRender(m, tables, ctx, 0.0)
  var wide = newSeq[char](64)
  var wideAcc = ""
  var wideRaised = false
  try:
    while true:
      let n = pull(dWide, wide)
      if n == 0:
        break
      wideAcc.add bytesOf(wide, n)
  except CatchableError:
    wideRaised = true
  doAssert wideRaised, "the wide-window filter raise did not raise"
  doAssert wideAcc == "", "the failing call returned bytes: <" & wideAcc & ">"
  var wideRest = ""
  while true:
    let n = pull(dWide, wide)
    if n == 0:
      break
    wideRest.add bytesOf(wide, n)
  doAssert wideRest == "[ab]post",
      "the wide-window repull did not resume after the discarded bytes"

# A macro call as a whole emit streams its body's pieces through the caller's window,
# and the streamed bytes match the string render.
# ---------------------------------------------------------------------------
block captureSink:
  let ctx = listCtx()
  let src = "{%- macro mm(v) -%}[{{ v }}]{%- endmacro -%}{{ mm(m) }}"
  var (m, tables) = parseTemplate(src)
  let want = renderToString(src, ctx, 0.0)

  var d = startRender(m, tables, ctx, 0.0)
  doAssert pullAll(d) == want,
      "the capture-sink emit differs from the string render"

# A streamed macro call resolves names against the caller's scopes only before the call
# and against its own scopes only inside the body: the macro scope is popped on close.
# ---------------------------------------------------------------------------
block macroScopePop:
  let leakCaller = "{% macro mm(q) %}[{{ q }}]{% endmacro %}" &
      "{% set q = 'caller' %}{{ mm('inner') }}:{{ q }}"
  doAssert renderToString(leakCaller, listCtx()) == "[inner]:caller",
      "the macro body read a caller binding set after the call"

  let leakBody = "{% macro mm() %}{% set z = 'body' %}{{ z }}{% endmacro %}{{ mm() }}:{{ z }}"
  doAssert renderToString(leakBody, listCtx()) == "body:",
      "a macro body binding leaked into the caller's name resolution"

# `tojson` with `ensure_ascii` exercises every escape shape, control characters included,
# plus the astral-codepoint surrogate pair. The corpus records `ensure_ascii`-off output,
# so this suite checks the engine's escape set directly: uppercase hex digits and the surrogate pair.
# ---------------------------------------------------------------------------
block ensureAsciiEscapes:
  let raw = strVal("a\tb\rc\bd\x0Ce\x01f\"g\\h<i>j&k'lém😀n")
  doAssert toJson(raw, JsonOpts(ensureAscii: true)) ==
      "\"a\\tb\\rc\\bd\\fe\\u0001f\\\"g\\\\h\\u003ci\\u003ej\\u0026k\\u0027l\\u00E9m\\uD83D\\uDE00n\"",
      "the ensure_ascii rendering differs from the expected escapes"
  doAssert toJson(raw) ==
      "\"a\\tb\\rc\\bd\\fe\\u0001f\\\"g\\\\h\\u003ci\\u003ej\\u0026k\\u0027lém😀n\"",
      "the raw-utf8 rendering differs from the expected escapes"

# Allocation counting. Compiled only under `-d:nimAllocStats`, and a failing doAssert
# there hangs the run with no output instead of failing it.
# ---------------------------------------------------------------------------
when defined(nimAllocStats):
  privateAccess(AllocStats)

  template allocsOf(body: untyped): int =
    ## Counts `alloc` calls made by `body`, with allocator state warmed by the caller.
    let before = getAllocStats()
    body
    (getAllocStats() - before).allocCount

  block allocDrainWindow:
    let src = templateSource("deepseekv2lite")
    var (m, tables) = parseTemplate(src)
    let row = loadRow("deepseekv2lite", "assistant_history")

    # Warm-up renders, uncounted:
    # first-touch allocator state settles here.
    for _ in 0 ..< 3:
      discard renderPull(m, tables, row.context, row.clock, 1)
      discard renderAllPull(m, tables, row.context, row.clock)
      discard renderToString(src, row.context, row.clock)

    # A pull that enters on a pending piece with bytes left only drains it, no step runs,
    # so it must allocate nothing.
    var d = startRender(m, tables, row.context, row.clock)
    var one: array[1, char]
    var acc = ""
    var drainCalls = 0
    while true:
      let pending = d.state.pend.kind != pkNone and pieceRemaining(d.state.pend) > 0
      let before = getAllocStats()
      let n = pull(d, one)
      let used = (getAllocStats() - before).allocCount
      if n == 0:
        break
      acc.add one[0]
      if pending:
        inc drainCalls
        doAssert used == 0, "a pull that only drained pending bytes allocated " & $used
    doAssert acc == row.rendered, "the counted render disagrees with the recorded bytes"
    doAssert drainCalls > 0, "no pending-piece drain was counted"
    echo "t_corpus alloc: ", drainCalls, " pending-piece drain calls, all 0 allocs"

    # Whole-render comparison:
    # the pull path against the string path, whose count also covers parsing the template
    # and therefore bounds the pull total from above.
    var dTotal = startRender(m, tables, row.context, row.clock)
    let pullAllocs = allocsOf:
      discard pullAll(dTotal)
    let strAllocs = allocsOf:
      discard renderToString(src, row.context, row.clock)
    doAssert pullAllocs <= strAllocs, "the pull render allocated " & $pullAllocs &
        " against the string render's " & $strAllocs
    echo "t_corpus alloc: full pull render ", pullAllocs, " allocs, string render ", strAllocs,
        " allocs"

  # Micro attribution over a 10-message for-loop context:
  # a warm-up render per template stays uncounted, then `getAllocStats()` deltas measure
  # the counted renders. DictGet lookup floors are measured in the same run.
  # Every assert below is an exact equality against a value this binary just measured:
  # - an emit-role render costs nothing beyond the loop machinery, its lookups included
  # - an emit-content render costs exactly one lookup copy per emit, the accepted residual
  # - a punctuator evaluation and the pending-piece move of an emit string cost 0
  block allocMicro:
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
      var (m, tables) = parseTemplate(src)
      let want = renderToString(src, ctx, 0.0)
      doAssert renderAllPull(m, tables, ctx, 0.0) == want,
          "the micro pull render differs from the string render for " & src
      allocsOf:
        for _ in 0 ..< n:
          discard renderAllPull(m, tables, ctx, 0.0)

    let loopOnly = countRenders("{% for m in messages %}x{% endfor %}", iters)
    let roleRenders = countRenders("{% for m in messages %}{{ m.role }}{% endfor %}", iters)
    let contentRenders = countRenders("{% for m in messages %}{{ m.content }}{% endfor %}", iters)
    let bothSrc = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    let bothRenders = countRenders(bothSrc, iters)

    doAssert roleRenders == loopOnly + iters * msgCount * dgRole,
        "the emit-role render cost " & $(roleRenders - loopOnly) &
        " allocs beyond the loop baseline"
    doAssert contentRenders == loopOnly + iters * msgCount * dgContent,
        "the emit-content render cost " & $(contentRenders - loopOnly) &
        " allocs beyond the loop baseline"
    doAssert bothRenders == loopOnly + iters * msgCount * (dgRole + dgContent),
        "the two-emit render cost " & $(bothRenders - loopOnly) &
        " allocs beyond the loop baseline"
    doAssert (roleRenders - loopOnly) div iters <= msgCount and
        (contentRenders - loopOnly) div iters <= msgCount,
        "an emit cost more than the accepted one-lookup residual"
    echo "t_corpus alloc: emit-role ", roleRenders, ", emit-content ", contentRenders,
        ", emit both ", bothRenders, ", loop baseline ", loopOnly,
        " allocs over ", iters, " renders each"

  block allocSerializer:
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
    # presize pass, so a call costs one allocation for the buffer plus one for the stack
    # behind the schema's two nested containers.
    let tools = toolsVal()
    # warm-up call, excluded from the counted region
    discard toJson(tools)
    let tjAllocs = allocsOf:
      for _ in 0 ..< iters:
        discard toJson(tools)
    doAssert tjAllocs == 2 * iters, "toJson of the tool schema cost " & $(tjAllocs div iters) &
        " allocations per call against the measured two"

    # The same schema through the pull render, driver setup uncounted. The counted region
    # holds only the pull loop, and the render costs the filter's argument list plus
    # the serializer's container stack.
    const tJson = "{{ tools|tojson }}"
    var cd = DictVal()
    dictSet(cd, "tools", tools)
    let ctx = dictVal(cd)
    var (m, tables) = parseTemplate(tJson)
    let want = renderToString(tJson, ctx, 0.0)

    var dWarm = startRender(m, tables, ctx, 0.0)
    var bufWarm = newSeq[char](256)
    var warm = ""
    while true:
      let n = pull(dWarm, bufWarm)
      if n == 0:
        break
      warm.add bytesOf(bufWarm, n)
    doAssert warm == want, "the pull render differs from the string render"

    var buf = newSeq[char](256)
    var renderAllocs = 0
    for _ in 0 ..< iters:
      var di = startRender(m, tables, ctx, 0.0)
      let renderCost = allocsOf:
        while true:
          let n = pull(di, buf)
          if n == 0:
            break
      renderAllocs += renderCost
    doAssert renderAllocs == 3 * iters, "the tojson pull render cost " &
        $(renderAllocs div iters) & " allocations per render against the measured three"

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
      var (mm, ts) = parseTemplate(src)
      let wantLocal = renderToString(src, loopCtx, 0.0)
      var dWarm2 = startRender(mm, ts, loopCtx, 0.0)
      var bufWarm2 = newSeq[char](256)
      var accWarm = ""
      while true:
        let got = pull(dWarm2, bufWarm2)
        if got == 0:
          break
        accWarm.add bytesOf(bufWarm2, got)
      doAssert accWarm == wantLocal, "the micro render differs for " & src
      var total = 0
      for _ in 0 ..< n:
        var di = startRender(mm, ts, loopCtx, 0.0)
        var bi = newSeq[char](256)
        let renderCost = allocsOf:
          while true:
            let got = pull(di, bi)
            if got == 0:
              break
        total += renderCost
      total

    let loopOnly = countRenders("{% for m in messages %}x{% endfor %}", iters)
    let strEmits = countRenders("{% for m in messages %}{{ m.n }}{% endfor %}", iters)
    let dictEmits = countRenders("{% for m in messages %}{{ m }}{% endfor %}", iters)
    # The runtime-built message values keep the engine's one-lookup-copy residual per emit.
    doAssert strEmits == loopOnly + iters * 10, "the string emit cost " &
        $(strEmits - loopOnly) & " allocations beyond the loop baseline"
    # A container emit through the lazy piece costs one allocation per emit over the string
    # emit and one per render for the serializer's container stack.
    doAssert dictEmits == strEmits + iters * 11, "the container emit cost " &
        $(dictEmits - strEmits) & " allocations beyond the string emit"

    echo "t_corpus alloc: tojson direct ", tjAllocs div iters, "/call, tojson render ",
        renderAllocs div iters, "/render, string emit ", (strEmits - loopOnly) div iters,
        ", container emit ", (dictEmits - loopOnly) div iters,
        " allocs beyond the loop baseline over ", iters, " renders"

echo "t_corpus: 64 ok rows byte-exact through pull, compose and render, 19 gap rows loud, " &
    "16 err rows raise the recorded error"
