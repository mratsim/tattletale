# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Fused expression walker of the chattyninja engine.
# One `lo..hi` span of template text is parsed and evaluated in a single pass, no expression
# becomes a node. The walker reads the render only through the injected `Ports` and never
# touches the render state, which stays at the step tier. Lookup, clock and macro forcer
# arrive as ports. A dry pass advances tokens without evaluating, one mechanism three
# places use:
#
# - `and` and `or` skip the operand they do not evaluate
# - a ternary is located by a dry scan, so the condition runs first and exactly one branch
#   then runs. Re-running a branch holding `strftime_now` or `raise_exception` would change
#   bytes or raise from the branch not taken
# - skipped branches never reach the registries, so an unimplemented construct cannot fail
#   a render from a branch Jinja would not have entered

import std/[math, parseutils, unicode]
import cnj_types, jinja_data_model, jinja_serialize, jinja_builtins

type
  ExKind = enum
    exEof, exName, exInt, exFloat, exStr, exPunct, exIntLow

  ExTok = object
    kind: ExKind
    lo, hi: int # span into CompiledTemplate.jinja
    i: int64 # payload of the exInt token
    f: float64 # payload of the exFloat token
    s: string # decoded string literal
    p0, p1: char # punctuator bytes, `p1 == '\0'` for a one-byte punctuator

  Cx = object
    ## Walker cursor:
    ##   the half-open span it owns, a one-token lookahead, the dry flag, the recursion depth,
    ##   the render ports handed to nested calls.
    pos, stop: int
    tok: ExTok
    dry: bool
    depth: int
    ports: Ports

  GlobalProc = proc (tmpl: CompiledTemplate, lo, hi: int, args: Args, ports: Ports): JinjaVal {.nimcall, noSideEffect.}
    ## A call to a template global. `namespace` and `dict` store a keyword name as a dict key.
    ## - `lo` and `hi` bound the global's name token, the location the raise sites report
    ## - globals read the template text and the injected ports, never the render state

  GlobalName = enum
    gNamespace, gRange, gStrftimeNow, gRaiseException, gDict, gLipsum, gCycler, gJoiner

  Op = enum
    ## Infix operator an expression token spells, word operators and punctuator spellings alike,
    ## `opNone` a token that opens no infix and the field's default.
    opNone, opAnd, opOr, opIn, opNotIn, opEq, opNe, opLt, opGt, opLe, opGe,
    opConcat, opAdd, opSub, opMul, opDiv, opFloorDiv, opMod, opPow
  Ternary = object
    ## Spans of `A if C else B`, measured by a dry walk before any of them runs.
    aHi, cLo, cHi, bLo, bHi: int
    hasElse: bool
    isTernary: bool

# Lexer:

func digitVal(c: char): int =
  let d = ord(c) - ord('0')
  if d >= 0 and d <= 9: d else: (ord(c) or 32) - ord('a') + 10

func decodeEscapesInto(sb: var Cursor, s: openArray[char], lo, hi: int) =
  ## Writes the string literal's bytes with Python's escape set resolved into `sb`,
  ## raising when `sb` cannot hold the decoding.
  var i = lo
  while i < hi:
    if s[i] != '\\':
      sb.add s[i]
      inc i
      continue
    inc i
    if i >= hi:
      raise jinjaErr("truncated escape in a string literal", i - 1)
    case s[i]
    of 'n': sb.add '\n'
    of 't': sb.add '\t'
    of 'r': sb.add '\r'
    of '0': sb.add '\0'
    of 'a': sb.add '\a'
    of 'b': sb.add '\b'
    of 'f': sb.add '\f'
    of 'v': sb.add '\v'
    of 'e': sb.add '\e'
    of '\\': sb.add '\\'
    of '\'': sb.add '\''
    of '"': sb.add '"'
    of 'x', 'u':
      let digits = if s[i] == 'x': 2 else: 4
      var code = 0
      var k = 1
      while k <= digits:
        if i + k >= hi or s[i + k] notin {'0' .. '9', 'a' .. 'f', 'A' .. 'F'}:
          raise jinjaErr("bad \\" & s[i] & " escape in a string literal", i - 1)
        code = code * 16 + digitVal(s[i + k])
        inc k
      inc i, digits
      sb.addRune(Rune(code))
    else:
      sb.add s[i]
    inc i

func decodeEscapes(s: openArray[char], lo, hi: int): string =
  ## Returns the string literal with Python's escape set resolved as one fresh string,
  ## sized exactly by the measuring cursor, never grown past its allocation.
  var measure = measureBuf()
  measure.decodeEscapesInto(s, lo, hi)
  result = newString(measure.len)
  var sb = over(result)
  sb.decodeEscapesInto(s, lo, hi)

const
  IntLowLit = "9223372036854775808"
    ## Only digit spelling past int64 naming a representable value, the value int64.low
    ## under a unary minus. The lexer emits a marker token for it, any other past-range
    ## magnitude still raising in `parseIntToken`.

func checkArgOrder(a: Args, lo, hi: int) =
  ## Raises located at the filter, method or test call when a positional argument follows
  ## a keyword one, the order `bindMacroArgs` already rejects for macros, a positional
  ## past a keyword otherwise binding under no parameter and silently dropping.
  var keywordSeen = false
  for x in a.argItems:
    if x.nameLo == NoLink:
      if keywordSeen:
        raise jinjaErr("a positional argument follows a keyword argument", lo, hi - lo)
    else:
      keywordSeen = true

func parseIntToken(s: openArray[char], at: int): int64 =
  ## Returns the integer the token bytes spell, an int64-range magnitude breach
  ## raising a located `JinjaError` at the literal.
  ## - Python renders such a literal as an unbounded integer, a value kind this tier
  ##   has no slot for, so the raise is the contract here
  ## - `parseutils.parseBiggestInt` itself raises on that magnitude, so the digits
  ##   accumulate in checked arithmetic instead, never crossing int64
  var n = 0'i64
  for c in s:
    let d = c.ord - '0'.ord
    if n > (int64.high - d) div 10:
      raise jinjaErr("integer literal `" & spanString(s) & "` is outside the int64 range",
          at, s.len)
    n = n * 10 + d
  n

func parseFloatToken(s: openArray[char]): float64 =
  ## Returns the float the token bytes spell, ValueError on a malformed token.
  var f: float64
  if s.len == 0 or parseutils.parseFloat(s, f) != s.len:
    raise newException(ValueError, "invalid float: " & spanString(s))
  f

func lexNumber(s: openArray[char], i: var int, hi: int): ExTok =
  let start = i
  var isFloat = false
  while i < hi and s[i] in {'0' .. '9'}:
    inc i
  if i + 1 < hi and s[i] == '.' and s[i + 1] in {'0' .. '9'}:
    isFloat = true
    inc i
    while i < hi and s[i] in {'0' .. '9'}:
      inc i
  if i < hi and s[i] in {'e', 'E'}:
    var j = i + 1
    if j < hi and s[j] in {'+', '-'}:
      inc j
    if j < hi and s[j] in {'0' .. '9'}:
      while j < hi and s[j] in {'0' .. '9'}:
        inc j
      isFloat = true
      i = j
  let text = s.toOpenArray(start, i - 1)
  if isFloat:
    ExTok(kind: exFloat, lo: start, hi: i, f: parseFloatToken(text))
  elif text == IntLowLit:
    ExTok(kind: exIntLow, lo: start, hi: i)
  else:
    ExTok(kind: exInt, lo: start, hi: i, i: parseIntToken(text, start))

func lexString(s: openArray[char], i: var int, hi: int): ExTok =
  let q = s[i]
  let start = i
  inc i
  let body = i
  while i < hi and s[i] != q:
    if s[i] == '\\':
      inc i
    inc i
  if i >= hi:
    raise jinjaErr("unterminated string literal at byte " & $start, start)
  let endBody = i
  inc i
  ExTok(kind: exStr, lo: start, hi: i, s: decodeEscapes(s, body, endBody))

func punctAt(s: openArray[char], i, hi: int): tuple[c0, c1: char, len: int] =
  ## Returns the punctuator matching at `i` as its two bytes and its byte length,
  ## `c1 == '\0'` marking a one-byte punctuator.
  ## - `//`, `**`, `<=`, `>=`, `==` and `!=` are lexed whole, each reported
  ##   as one construct
  if i + 1 < hi:
    case s[i]
    of '=':
      if s[i + 1] == '=': return ('=', '=', 2)
    of '!':
      if s[i + 1] == '=': return ('!', '=', 2)
      raise jinjaErr("unexpected `!` in an expression at byte " & $i, i, 1)
    of '<':
      if s[i + 1] == '=': return ('<', '=', 2)
    of '>':
      if s[i + 1] == '=': return ('>', '=', 2)
    of '/':
      if s[i + 1] == '/': return ('/', '/', 2)
    of '*':
      if s[i + 1] == '*': return ('*', '*', 2)
    else:
      discard
  case s[i]
  of '+', '-', '*', '/', '%', '~', '(', ')', '[', ']', '{', '}', ',', ':', '.', '|', '<', '>':
    (s[i], '\0', 1)
  of '=':
    ('=', '\0', 1)
  else:
    raise jinjaErr("unexpected character '" & s[i] & "' in an expression at byte " & $i, i, 1)

func advance(tmpl: CompiledTemplate, cx: var Cx) =
  ## Loads the next token into the cursor, stopping at the cursor's own `stop`.
  let s = tmpl.jinja
  var i = cx.pos
  while i < cx.stop and s[i] in Whitespace:
    inc i
  if i >= cx.stop:
    cx.tok = ExTok(kind: exEof, lo: i, hi: i)
    cx.pos = i
    return
  cx.pos = i
  cx.tok =
    case s[i]
    of '\'', '"':
      lexString(s, i, cx.stop)
    of '0' .. '9':
      lexNumber(s, i, cx.stop)
    of 'a' .. 'z', 'A' .. 'Z', '_':
      let start = i
      while i < cx.stop and (s[i] in {'a' .. 'z', 'A' .. 'Z', '0' .. '9'} or s[i] == '_'):
        inc i
      ExTok(kind: exName, lo: start, hi: i)
    else:
      let (c0, c1, len) = punctAt(s, i, cx.stop)
      inc i, len
      ExTok(kind: exPunct, lo: i - len, hi: i, p0: c0, p1: c1)
  cx.pos = i

func isPunct(cx: Cx, p: string): bool =
  ## Reports whether the lookahead is the punctuator `p`, matched byte against byte so no
  ## string is built per comparison.
  cx.tok.kind == exPunct and cx.tok.p0 == p[0] and
    ((p.len == 1 and cx.tok.p1 == '\0') or (p.len == 2 and cx.tok.p1 == p[1]))

func isWord(tmpl: CompiledTemplate, cx: Cx, w: string): bool =
  ## Reports whether the lookahead is the bare identifier `w`, keywords matched without
  ## copying text out of the template.
  if cx.tok.kind != exName or cx.tok.hi - cx.tok.lo != w.len:
    return false
  for k in 0 ..< w.len:
    if tmpl.jinja[cx.tok.lo + k] != w[k]:
      return false
  true

template wordSpan(tmpl: CompiledTemplate, lo, hi: int): openArray[char] =
  ## Returns the template text in `lo ..< hi` as a view, so neither an identifier nor
  ## a registry lookup allocates. `lo ..< hi` is the half-open span the token carries.
  ##
  ## The template wording is required for the compiler's view analysis under
  ## `--experimental:views`, which rejects a helper returning a view over a field
  ## of the `ref` type.
  tmpl.jinja.toOpenArray(lo, hi - 1)

template wordSpan(tmpl: CompiledTemplate, cx: Cx): openArray[char] =
  ## Returns the identifier of the lookahead token as a view into the template text.
  tmpl.wordSpan(cx.tok.lo, cx.tok.hi)

func argName(tmpl: CompiledTemplate, a: Arg): openArray[char] =
  ## Returns an argument's keyword name as a view into the template text, `nameLo == NoLink`
  ## marking a positional argument, which has no name to read.
  tmpl.jinja.toOpenArray(a.nameLo.int, a.nameHi.int - 1)

func forceCall(ports: Ports, cx: var Cx, v: JinjaVal): JinjaVal =
  ## Renders one pending macro call to its output value, the primitive `forceOperand`
  ## routes every value-position forcing through, the forcer itself carried by the ports.
  ## Raises at `cx.tok.lo` when no macro forcer was supplied.
  if cx.ports.force.isNil:
    raise jinjaErr("a macro call result was consumed where no macro forcer was supplied",
        cx.tok.lo)
  cx.ports.force(cx.ports.env, v.pc.mc, v.pc.args)

func forceOperand(ports: Ports, cx: var Cx, v: JinjaVal): JinjaVal =
  ## Returns `v` with a pending macro call rendered to its output value, a concat rendered
  ## to the text it emits, the value-position forcing contract held in one proc.
  ## - reached from every truth test, `and`/`or` left operand, ternary condition,
  ##   call argument, postfix operator operand and binary-operator boundary
  ## - a consumed call with no macro forcer supplied raises at `cx.tok.lo`, a dry walk
  ##   returning `v` unevaluated, the skipped branch never running a body
  ## - the emit step alone captures, a whole-expression macro call streaming its body there,
  ##   a forced call's output drained from the engine's transient capture buffer into the returned string,
  ##   the copy then discarded
  if cx.dry:
    return v
  case v.kind
  of vkCall: forceCall(ports, cx, v)
  of vkConcat: strVal(pyStr(v))
  else: v

func evalItem(ports: Ports, cx: var Cx, v: JinjaVal): JinjaVal =
  ## Returns `v` for the operators that read a plain value:
  ## - a concat raises here, the argument list being the one plain-value reader
  ## - everything else routes through the forcing contract, a dry walk returning `v`
  if cx.dry:
    return v
  if v.kind == vkConcat:
    raise jinjaErr("a concat must be rendered in emit position", cx.tok.lo)
  forceOperand(ports, cx, v)

func argKey(tmpl: CompiledTemplate, a: Arg): string =
  ## Returns the dict key one argument supplies to `namespace` or `dict`, a keyword-bound argument
  ## giving the keyword text, a positional one its stringified value. A `DictVal` key is a string,
  ## so this is where a keyword name becomes one.
  if a.nameLo == NoLink:
    pyStr(a.val)
  else:
    spanString(argName(tmpl, a))

func runeOffset(s: string, k: int): int =
  ## Returns the byte offset of the codepoint at index `k`, advancing by UTF-8
  ## lead-byte strides from the scan start, the same walk `runeLen` takes.
  var j = 0
  for _ in 0 ..< k:
    inc j, runeLenAt(s, j)
  j

func runeOffsets(s: string, a, b: int): tuple[lo, hi: int] =
  ## Returns the byte offsets bracketing the codepoint index span `a ..< b`, one forward
  ## stride walk answering both ends. An empty span (`a >= b`) reports `hi == lo`.
  var j = 0
  result.lo = -1
  var k = 0
  while k < b and j < s.len:
    if k == a:
      result.lo = j
    inc j, runeLenAt(s, j)
    inc k
  if result.lo < 0:
    result.lo = j
  result.hi = j

func runeSub(s: string, i: int): Rune =
  ## Returns the codepoint at Python index `i`, a negative `i` counting from the end.
  ## One stride walk answers the read:
  ## - `i >= 0` walks forward from the scan start
  ## - `i < 0` walks backward over continuation bytes from the end
  ## Never a full-string length scan. An ASCII codepoint answers by one byte read,
  ## a multibyte one decodes in place.
  var j: int
  if i >= 0:
    j = 0
    var left = i
    while left > 0 and j < s.len:
      inc j, runeLenAt(s, j)
      dec left
    if left > 0 or j >= s.len:
      raise jinjaErr("string subscript " & $i & " is out of range")
  else:
    j = s.len
    var left = -i
    while left > 0 and j > 0:
      # A rune starts where the backward continuation-byte scan stops.
      dec j
      while j > 0 and (s[j].ord and 0xC0) == 0x80:
        dec j
      dec left
    if left > 0:
      raise jinjaErr("string subscript " & $i & " is out of range")
  if s[j].ord < 0x80: Rune(s[j].ord) else: runeAt(s, j)

func steppedSliceInto(sb: var Cursor, s: string, a, b, by: int) =
  ## Writes the stride-`by` codepoint slice into `sb`, visiting `a, a + by, ...`
  ## while the stride keeps the walk inside the clamped bounds.
  ## One byte walk answers the whole slice per pass:
  ## - `by > 0` advances by lead-byte strides
  ## - `by < 0` steps backward over continuation bytes
  ## Every visited codepoint reads in place at its own offset, never a per-index rescan.
  ## A stride beyond the walk span visits the start element only, so the stride clamps
  ## to one step past the span and the index arithmetic stays inside int64 however
  ## extreme the step value is.
  if by > 0:
    if a >= b:
      return
    let by = min(by, b - a + 1)
    var j = runeOffset(s, a)
    var k = a
    while k < b:
      let l = runeLenAt(s, j)
      sb.add s.toOpenArray(j, j + l - 1)
      inc k, by
      if k < b:
        # Step to the next visited codepoint, `by` runes ahead, `k + by < b` keeping
        # every skipped stride in bounds.
        inc j, l
        for _ in 1 ..< by:
          inc j, runeLenAt(s, j)
  else:
    if a <= b:
      return
    let by = max(by, b - a - 1)
    var j = runeOffset(s, a)
    var k = a
    while k > b:
      let l = runeLenAt(s, j)
      sb.add s.toOpenArray(j, j + l - 1)
      inc k, by
      if k > b:
        # Step to the next visited codepoint, `|by|` runes back, `k + by > b >= -1`
        # keeping every visited index non-negative.
        for _ in 1 .. -by:
          dec j
          while j > 0 and (s[j].ord and 0xC0) == 0x80:
            dec j

# Registries:
#
# Filters, tests, methods and globals dispatch by name. `tojson` is one filter name exactly like
# `trim`, never a construct. A nil entry is a declared name no template in the corpus uses,
# reaching it raising `JinjaError` with cause `ceUnimplemented`, never answering wrongly.


const
  GlobalNames: array[GlobalName, string] = [
    "namespace", "range", "strftime_now", "raise_exception", "dict", "lipsum", "cycler", "joiner"
  ]

func loopAttr(v: JinjaVal, name: openArray[char]): JinjaVal =
  ## Returns a `loop.*` attribute, read through the for-row's shared cursor. The attribute is
  ## selected by span compare, so an attribute inside a `{% for %}` body costs no string.
  let lp = v.lp
  let n = lp.loopLen
  let i = lp.idx
  if name == "index": intVal(i + 1)
  elif name == "index0": intVal(i)
  elif name == "first": boolVal(i == 0)
  elif name == "last": boolVal(i == n - 1)
  elif name == "length": intVal(n)
  elif name == "previtem":
    if i > 0: lp.loopItem(i - 1) else: noneVal()
  elif name == "nextitem":
    if i + 1 < n: lp.loopItem(i + 1) else: noneVal()
  else:
    gapWhat("`loop` attribute", name)

func sliceIndices(n: int, lo, hi, step: JinjaVal, hasLo, hasHi, hasStep: bool):
    tuple[start, stop, by: int] =
  ## Returns Python's `slice.indices(n)` for one slice:
  ##   the walk bounds and the stride, direction-dependent defaults and clamps applied.
  ## - defaults follow the stride's direction, not the range's ends, which is what
  ##   makes `x[::-1]` visit every element and `x[:2:-1]` stop at the head
  ## - forward (`by > 0`):
  ##   bounds clamp into `[0, n]`, defaults `0` and `n`
  ## - backward (`by < 0`):
  ##   bounds clamp into `[-1, n - 1]`, defaults `n - 1` and `-1`
  ## A backward stop of `-1` means "one past the head", so the walk includes index 0.
  var by = 1
  if hasStep:
    if step.kind != vkInt:
      raise jinjaErr("slice step needs an integer")
    by = step.i.int
    if by == 0:
      raise jinjaErr("slice step must not be zero")
  if (hasLo and lo.kind != vkInt) or (hasHi and hi.kind != vkInt):
    raise jinjaErr("slice bounds need integers")
  let low = if by > 0: 0 else: -1
  let high = if by > 0: n else: n - 1
  var a = if hasLo: lo.i.int else: (if by > 0: low else: high)
  var b = if hasHi: hi.i.int else: (if by > 0: high else: low)
  if hasLo and a < 0:
    a += n
  if hasHi and b < 0:
    b += n
  (clamp(a, low, high), clamp(b, low, high), by)

func subslice(v, lo, hi, step: JinjaVal, hasLo, hasHi, hasStep, isSlice: bool): JinjaVal =
  ## Returns a subscript or a slice, `x[1:]` and `x[::-1]` the slice shapes the corpus uses.
  let v = if v.kind == vkCut: materializeVal(v) else: v
  if not isSlice:
    return case v.kind
    of vkSeq:
      if lo.kind != vkInt:
        raise jinjaErr("sequence subscript needs an integer")
      let n = v.xs.items.len
      let idx = if lo.i < 0: n + lo.i.int else: lo.i.int
      if idx < 0 or idx >= n:
        raise jinjaErr("subscript " & $lo.i & " is out of range for a length-" & $n & " sequence")
      v.xs.items[idx]
    of vkDict, vkNs:
      v.d.dictGet(pyStr(lo))
    of vkStr:
      if lo.kind != vkInt:
        raise jinjaErr("string subscript needs an integer")
      strVal($runeSub(v.s, lo.i.int))
    of vkUndefined:
      undefinedVal()
    else:
      raise jinjaErr("a " & $v.kind & " is not subscriptable")
  let n =
    case v.kind
    of vkSeq: v.xs.items.len
    of vkStr: runeLen(v.s)
    else: raise jinjaErr("a " & $v.kind & " is not sliceable")
  let (a, b, by) = sliceIndices(n, lo, hi, step, hasLo, hasHi, hasStep)
  return case v.kind
  of vkSeq:
    var acc = newSeq[JinjaVal]()
    var k = a
    while (by > 0 and k < b) or (by < 0 and k > b):
      acc.add v.xs.items[k]
      inc k, by
    seqVal(acc)
  else:
    if by == 1:
      let (lo, hi) = runeOffsets(v.s, a, b)
      if hi > lo:
        strVal(spanString(v.s.toOpenArray(lo, hi - 1)))
      else:
        strVal("")
    else:
      var sb = measureBuf()
      sb.steppedSliceInto(v.s, a, b, by)
      var win = newString(sb.len)
      var dst = over(win)
      dst.steppedSliceInto(v.s, a, b, by)
      strVal(win)

func argDict(tmpl: CompiledTemplate, args: Args): DictVal =
  ## Returns one mapping holding the call's arguments, keyword names as dict keys.
  var dv = DictVal()
  for a in args.argItems:
    dv.dictSet(argKey(tmpl, a), a.val)
  dv

func namespaceGlobal(tmpl: CompiledTemplate, lo, hi: int, args: Args, ports: Ports): JinjaVal =
  ## `namespace(field=init, ...)`:
  ##   the mutable mapping `{% set ns.field = ... %}` mutates in place.
  nsVal(argDict(tmpl, args))

func dictGlobal(tmpl: CompiledTemplate, lo, hi: int, args: Args, ports: Ports): JinjaVal =
  dictVal(argDict(tmpl, args))

func rangeGlobal(tmpl: CompiledTemplate, lo, hi: int, args: Args, ports: Ports): JinjaVal =
  ## `range(a, b, step)`:
  ##   the lazy bounds value. Elements compute per index, the serializer rendering the list
  ##   form arithmetically and a `for` walking the same arithmetic, so a range never materializes.
  var a = 0'i64
  var b = 0'i64
  var step = 1'i64
  for i in 0 ..< args.n:
    let x = args.vals[i]
    if x.val.kind != vkInt:
      raise jinjaErr("`range` needs integer bounds", lo, hi - lo)
    case i
    of 0: b = x.val.i
    of 1:
      a = b
      b = x.val.i
    of 2: step = x.val.i
    else: raise jinjaErr("`range` takes at most three arguments", lo, hi - lo)
  if step == 0:
    raise jinjaErr("`range` step must not be zero", lo, hi - lo)
  let v = rangeVal(a, b, step)
  # One element count check at construction bounds every consumer, the count
  # answering through the same arithmetic each consumer reads:
  # - a range past `RangeElemCap` raises located at the range expression here
  # - no loop, materialization, serializer or comparison ever sees one
  discard rangeLen(v.r, lo, hi)
  v

func civilFromDays(z: int): tuple[y, m, d: int] =
  ## Returns the civil date of `z` days since 1970-01-01, proleptic Gregorian.
  let z2 = z + 719468
  let era = floorDiv(z2, 146097)
  let doe = z2 - era * 146097
  let yoe = (doe - doe div 1460 + doe div 36524 - doe div 146096) div 365
  let doy = doe - (365 * yoe + yoe div 4 - yoe div 100)
  let mp = (5 * doy + 2) div 153
  let mon = mp + (if mp < 10: 3 else: -9)
  (yoe + era * 400 + ord(mon <= 2), mon, doy - (153 * mp + 2) div 5 + 1)

const MonthStart = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

func dayOfYear(y, m, d: int): int =
  ## Returns the 1-based day of year, Gregorian leap rule with century handling.
  MonthStart[m - 1] + d + ord(m > 2 and y mod 4 == 0 and
      (y mod 100 != 0 or y mod 400 == 0))

func yearField(y: int): string =
  ## Returns the `%Y` field, four digits zero-padded, non-positive years counting up
  ## from one and years past four digits carrying a leading `+`.
  result = $(if y <= 0: 1 - y else: y)
  if y > 9999: return "+" & result
  while result.len < 4:
    result = '0' & result

func twoDigits(n: int): string =
  ## Returns `n` zero-padded to two digits.
  if n < 10: "0" & $n else: $n

func threeDigits(n: int): string =
  ## Returns `n` zero-padded to three digits.
  result = $n
  while result.len < 3:
    result = '0' & result

func strftimeGlobal(tmpl: CompiledTemplate, lo, hi: int, args: Args, ports: Ports): JinjaVal =
  ## Renders the format against the epoch read through the clock port, never the wall clock,
  ## which is what keeps two render instantiations over one artifact byte-identical.
  let fmt = pyStr(getArg(args, 0, akNone, strVal("")))
  let secs = ports.clock(ports.env).int64
  let tod = floorMod(secs.int, 86400)
  let (yr, mo, dy) = civilFromDays(floorDiv(secs.int, 86400))
  var acc = ""
  var i = 0
  while i < fmt.len:
    if fmt[i] != '%':
      acc.add fmt[i]
      inc i
      continue
    if i + 1 >= fmt.len:
      raise jinjaErr("`strftime_now` format ends on a `%`", lo, hi - lo)
    case fmt[i + 1]
    of 'Y': acc.add yearField(yr)
    of 'm': acc.add twoDigits(mo)
    of 'd': acc.add twoDigits(dy)
    of 'H': acc.add twoDigits(tod div 3600)
    of 'M': acc.add twoDigits((tod mod 3600) div 60)
    of 'S': acc.add twoDigits(tod mod 60)
    of 'j': acc.add threeDigits(dayOfYear(yr, mo, dy))
    of '%': acc.add '%'
    else: gapWhat("`strftime_now` directive", "%" & fmt[i + 1])
    inc i, 2
  strVal(acc)

func raiseExceptionGlobal(tmpl: CompiledTemplate, lo, hi: int, args: Args, ports: Ports): JinjaVal =
  ## Corpus `err_*` rows record exactly this raise:
  ##   the message verbatim, the raise call's name-token span as `offset` and `span`, cause `ceRaiseCall`.
  raise jinjaErr(pyStr(getArg(args, 0, akNone, strVal(""))), lo, hi - lo, cause = ceRaiseCall)

const
  GlobalProcs: array[GlobalName, GlobalProc] = [
    namespaceGlobal, rangeGlobal, strftimeGlobal, raiseExceptionGlobal, dictGlobal, nil, nil, nil
  ]

# Lookup:

# Walker:

func evalRange(tmpl: CompiledTemplate, ports: Ports, lo, hi: int, depth = 0): JinjaVal
func expr(tmpl: CompiledTemplate, ports: Ports, cx: var Cx, minPrec: int): JinjaVal


const OpSpelling: array[Op, string] = [
  "", "and", "or", "in", "not in", "==", "!=", "<", ">", "<=", ">=", "~", "+", "-", "*",
  "/", "//", "%", "**"
]
  ## Operator spellings for error text, indexed by `Op`. Error reporting is the only reader.

func punctOp(c0, c1: char): Op =
  ## Returns the infix operator the punctuator `(c0, c1)` spells, `opNone` when it is none. A lone `=` is no infix in Jinja, so it yields
  ## `opNone` and the expression ends before it:
  ##   keyword arguments are detected in `argList` by `isPunct`, which never consults this map.
  if c1 == '\0':
    case c0
    of '<': opLt
    of '>': opGt
    of '=': opNone
    of '~': opConcat
    of '+': opAdd
    of '-': opSub
    of '*': opMul
    of '/': opDiv
    of '%': opMod
    else: opNone
  else:
    case c0
    of '=': opEq
    of '!': opNe
    of '<': opLe
    of '>': opGe
    of '/': opFloorDiv
    of '*': opPow
    else: opNone

func binPrec(op: Op): int =
  ## Returns the left binding power of an infix operator, 0 when `op` is not infix. Jinja orders operators ternary-lowest, then `or`,
  ## `and`, comparison and tests, `~`, `+ -`, `* / // %`, `**`. A ternary never enters this table, `scanTernary` claiming it before
  ## precedence climbing runs.
  case op
  of opOr: 2
  of opAnd: 3
  of opEq, opNe, opLt, opGt, opLe, opGe, opIn, opNotIn: 5
  of opConcat: 6
  of opAdd, opSub: 7
  of opMul, opDiv, opFloorDiv, opMod: 8
  of opPow: 9
  of opNone: 0

template enterDepth(cx: var Cx) =
  ## Counts one recursion level of the expression walker toward `ExprDepthCap`, a breach
  ## raising located, dry walks included. `expr` counts at its entry, and so does
  ## every recursion leg that bypasses `expr`, the paired exit a `dec cx.depth`.
  inc cx.depth
  if cx.depth > ExprDepthCap:
    raise jinjaErr("expression nests deeper than ExprDepthCap = " & $ExprDepthCap,
        cx.tok.lo)

func skipExpr(tmpl: CompiledTemplate, ports: Ports, cx: var Cx, minPrec: int) =
  ## Advances the cursor over an expression without evaluating it, how `and`, `or` and the ternary skip the text they do not run.
  let wasDry = cx.dry
  cx.dry = true
  discard expr(tmpl, ports, cx, minPrec)
  cx.dry = wasDry

func argList(tmpl: CompiledTemplate, ports: Ports, cx: var Cx): Args =
  ## Parses a parenthesised argument list with `name = expr` keyword arguments, the opening paren the lookahead. A keyword keeps its span
  ## and gains a builtin keyword slot, so binding one costs no string. Arguments fill the fixed-capacity
  ## carrier in call order, no per-call sequence.
  advance(tmpl, cx)
  while not isPunct(cx, ")"):
    var nameLo = NoLink
    var nameHi = NoLink
    var kw = akNone
    if cx.tok.kind == exName:
      let save = cx
      advance(tmpl, cx)
      if isPunct(cx, "="):
        nameLo = int32 save.tok.lo
        nameHi = int32 save.tok.hi
        kw = argKeyword(tmpl.wordSpan(save))
        advance(tmpl, cx)
      else:
        cx = save
    let v = forceOperand(ports, cx, expr(tmpl, ports, cx, 1))
    result.addArg(Arg(nameLo: nameLo, nameHi: nameHi, kw: kw, val: v))
    if isPunct(cx, ","):
      advance(tmpl, cx)
      if isPunct(cx, ")"):
        break
      continue
    break
  if not isPunct(cx, ")"):
    raise jinjaErr("argument list is not closed", cx.tok.lo)
  advance(tmpl, cx)

func postfix(tmpl: CompiledTemplate, ports: Ports, cx: var Cx, v: JinjaVal): JinjaVal =
  ## Applies attr, subscript, call, filter and test chains, which bind tighter than any operator.
  ## An integer constant after a dot is a subscript, `m.content.0` spelling
  ## `m.content[0]` the way upstream Jinja does.
  var v = v
  while true:
    # An operator reading the chained value renders a pending macro call first, through
    # the forcing contract. A call produced by the call operator inside the chain is
    # re-forced here the same way, its pending form never surviving past the next operator.
    if v.kind == vkCall and (isPunct(cx, ".") or isPunct(cx, "[") or isPunct(cx, "|") or
        isWord(tmpl, cx, "is")):
      v = forceOperand(ports, cx, v)
    if isPunct(cx, "."):
      advance(tmpl, cx)
      if cx.tok.kind == exInt:
        # `x.0` is upstream Jinja's spelling of `x[0]`, an integer-constant subscript.
        let n = cx.tok.i
        advance(tmpl, cx)
        v = if cx.dry: undefinedVal() else:
          subslice(v, intVal(n), undefinedVal(), undefinedVal(), true, false, false, false)
        continue
      if cx.tok.kind != exName:
        raise jinjaErr("expected a name after `.`", cx.tok.lo)
      let (lo, hi) = (cx.tok.lo, cx.tok.hi)
      advance(tmpl, cx)
      if isPunct(cx, "("):
        let a = argList(tmpl, ports, cx)
        if cx.dry:
          v = undefinedVal()
          continue
        let mi = findIn(MethodNames, tmpl.wordSpan(lo, hi))
        if mi < 0:
          raise jinjaErr("unknown method `" & spanString(tmpl.wordSpan(lo, hi)) & "`", lo, hi - lo)
        checkArgOrder(a, lo, hi)
        let mp = MethodProcs[MethodName mi]
        if mp.isNil:
          gapWhat("method", tmpl.wordSpan(lo, hi))
        v = mp(v, a)
      else:
        v =
          if cx.dry:
            undefinedVal()
          else:
            case v.kind
            of vkDict, vkNs: v.d.dictGet(tmpl.wordSpan(lo, hi))
            of vkLoop: loopAttr(v, tmpl.wordSpan(lo, hi))
            else: undefinedVal()
    elif isPunct(cx, "["):
      advance(tmpl, cx)
      var lo, hi, step = undefinedVal()
      var hasLo, hasHi, hasStep, isSlice = false
      if not isPunct(cx, ":"):
        lo = expr(tmpl, ports, cx, 1)
        hasLo = true
      if isPunct(cx, ":"):
        isSlice = true
        advance(tmpl, cx)
        if not (isPunct(cx, ":") or isPunct(cx, "]")):
          hi = expr(tmpl, ports, cx, 1)
          hasHi = true
        if isPunct(cx, ":"):
          advance(tmpl, cx)
          if not isPunct(cx, "]"):
            step = expr(tmpl, ports, cx, 1)
            hasStep = true
      if not isPunct(cx, "]"):
        raise jinjaErr("subscript is not closed", cx.tok.lo)
      advance(tmpl, cx)
      v = if cx.dry: undefinedVal() else:
        subslice(v, lo, hi, step, hasLo, hasHi, hasStep, isSlice)
    elif isPunct(cx, "("):
      let callLo = cx.tok.lo
      let a = argList(tmpl, ports, cx)
      v =
        if cx.dry:
          undefinedVal()
        elif v.kind == vkMacro:
          callVal(PendingCallVal(mc: v.mc, args: a))
        else:
          raise jinjaErr("only a macro is callable, this is a " & $v.kind, callLo)
    elif isPunct(cx, "|"):
      advance(tmpl, cx)
      if cx.tok.kind != exName:
        raise jinjaErr("expected a filter name after `|`", cx.tok.lo)
      let (lo, hi) = (cx.tok.lo, cx.tok.hi)
      advance(tmpl, cx)
      var a: Args
      if isPunct(cx, "("):
        a = argList(tmpl, ports, cx)
      if cx.dry:
        v = undefinedVal()
        continue
      let fi = findIn(FilterNames, tmpl.wordSpan(lo, hi))
      if fi < 0:
        raise jinjaErr("unknown filter `" & spanString(tmpl.wordSpan(lo, hi)) & "`", lo, hi - lo, cause = ceUnimplemented)
      checkArgOrder(a, lo, hi)
      let fp = FilterProcs[FilterName fi]
      if fp.isNil:
        gapWhat("filter", tmpl.wordSpan(lo, hi))
      v = fp(v, a)
    elif isWord(tmpl, cx, "is"):
      advance(tmpl, cx)
      var negated = false
      if isWord(tmpl, cx, "not"):
        negated = true
        advance(tmpl, cx)
      if cx.tok.kind != exName:
        raise jinjaErr("expected a test name after `is`", cx.tok.lo)
      let (lo, hi) = (cx.tok.lo, cx.tok.hi)
      advance(tmpl, cx)
      var a: Args
      if isPunct(cx, "("):
        a = argList(tmpl, ports, cx)
      if cx.dry:
        v = undefinedVal()
        continue
      let ti = findIn(TestNames, tmpl.wordSpan(lo, hi))
      if ti < 0:
        raise jinjaErr("unknown test `" & spanString(tmpl.wordSpan(lo, hi)) & "`", lo, hi - lo)
      checkArgOrder(a, lo, hi)
      let tp = TestProcs[TestName ti]
      if tp.isNil:
        gapWhat("test", tmpl.wordSpan(lo, hi))
      v = boolVal(if negated: not tp(v, a) else: tp(v, a))
    else:
      break
  v

func primary(tmpl: CompiledTemplate, ports: Ports, cx: var Cx): JinjaVal =
  ## Parses a literal, a name, a parenthesised group, an array literal or a dict literal,
  ## then the postfix chain.
  var v: JinjaVal
  case cx.tok.kind
  of exEof:
    raise jinjaErr("expression ends early at byte " & $cx.pos, cx.pos)
  of exInt:
    v = intVal(cx.tok.i)
    advance(tmpl, cx)
  of exIntLow:
    raise jinjaErr("integer literal `" & IntLowLit & "` is outside the int64 range",
        cx.tok.lo, cx.tok.hi - cx.tok.lo)
  of exFloat:
    v = floatVal(cx.tok.f)
    advance(tmpl, cx)
  of exStr:
    v = strVal(cx.tok.s)
    advance(tmpl, cx)
  of exName:
    let (lo, hi) = (cx.tok.lo, cx.tok.hi)
    advance(tmpl, cx)
    let name = tmpl.wordSpan(lo, hi)
    # Literal spellings are compared as spans, the same test `case` applied to a copied string.
    if name == "true" or name == "True":
      v = boolVal(true)
    elif name == "false" or name == "False":
      v = boolVal(false)
    elif name == "none" or name == "None":
      v = noneVal()
    else:
      var bound = if cx.dry: undefinedVal() else: cx.ports.lookup(cx.ports.env, name)
      let gi = findIn(GlobalNames, name)
      if bound.kind == vkUndefined and gi >= 0 and isPunct(cx, "("):
        # A global is reached only when the name is unbound, Jinja's own precedence:
        #   context shadows globals, and a skipped branch never gets here. A dry walk still
        #   consumes the argument list, so the skipped text is never left behind as trailing text.
        let a = argList(tmpl, ports, cx)
        if cx.dry:
          v = undefinedVal()
        else:
          let gp = GlobalProcs[GlobalName gi]
          if gp.isNil:
            gapWhat("global", tmpl.wordSpan(lo, hi))
          v = gp(tmpl, lo, hi, a, ports)
      else:
        v = bound
  of exPunct:
    case cx.tok.p0
    of '(':
      advance(tmpl, cx)
      var parts = newSeq[JinjaVal]()
      var isTuple = false
      while not isPunct(cx, ")"):
        parts.add expr(tmpl, ports, cx, 1)
        if isPunct(cx, ","):
          isTuple = true
          advance(tmpl, cx)
          if isPunct(cx, ")"):
            break
          continue
        break
      if not isPunct(cx, ")"):
        raise jinjaErr("parenthesised expression is not closed", cx.tok.lo)
      advance(tmpl, cx)
      v = if parts.len == 0: seqVal(parts) elif isTuple: seqVal(parts) else: parts[0]
    of '[':
      advance(tmpl, cx)
      var parts = newSeq[JinjaVal]()
      while not isPunct(cx, "]"):
        parts.add evalItem(ports, cx, expr(tmpl, ports, cx, 1))
        if isPunct(cx, ","):
          advance(tmpl, cx)
          if isPunct(cx, "]"):
            break
          continue
        break
      if not isPunct(cx, "]"):
        raise jinjaErr("array literal is not closed", cx.tok.lo)
      advance(tmpl, cx)
      v = seqVal(parts)
    of '{':
      advance(tmpl, cx)
      var dv = DictVal()
      while not isPunct(cx, "}"):
        let k = evalItem(ports, cx, expr(tmpl, ports, cx, 1))
        if not isPunct(cx, ":"):
          raise jinjaErr("dict literal entry needs a `:`", cx.tok.lo)
        advance(tmpl, cx)
        let val = evalItem(ports, cx, expr(tmpl, ports, cx, 1))
        if not cx.dry:
          dv.dictSet(pyStr(k), val)
        if isPunct(cx, ","):
          advance(tmpl, cx)
          if isPunct(cx, "}"):
            break
          continue
        break
      if not isPunct(cx, "}"):
        raise jinjaErr("dict literal is not closed", cx.tok.lo)
      advance(tmpl, cx)
      v = dictVal(dv)
    else:
      let spelled = $cx.tok.p0 & (if cx.tok.p1 != '\0': $cx.tok.p1 else: "")
      raise jinjaErr("unexpected `" & spelled & "` starting an expression at byte " & $cx.tok.lo, cx.tok.lo, cx.tok.hi - cx.tok.lo)
  postfix(tmpl, ports, cx, v)

func unary(tmpl: CompiledTemplate, ports: Ports, cx: var Cx): JinjaVal =
  ## Parses `not`, unary `-` and `+`, then a primary.
  ## - `not` binds looser than the comparisons, its operand parsing at comparison
  ##   binding power through `expr`, whose entry counts the operand walk toward
  ##   `ExprDepthCap` however deep the chain, so `not a == 5` tests `a == 5`
  ## - unary `-` and `+` bind tighter than any comparison, their chain recursing here
  ##   without re-entering `expr`, so each recursion counts one level itself,
  ##   `enterDepth` at the leg's entry and a `dec` once the operand is evaluated
  if isWord(tmpl, cx, "not"):
    advance(tmpl, cx)
    let operandLo = cx.tok.lo
    let v = forceOperand(ports, cx, expr(tmpl, ports, cx, 5))
    return boolVal(if cx.dry: false else: not isTruthy(v, operandLo))
  if isPunct(cx, "-") or isPunct(cx, "+"):
    let neg = isPunct(cx, "-")
    advance(tmpl, cx)
    if neg and cx.tok.kind == exIntLow:
      # Digits spelling exactly 2^63 render negated as int64.low, Python's rendering.
      # A re-negation or `+` reaches the raise through value or primary.
      advance(tmpl, cx)
      return intVal(int64.low)
    let operandLo = cx.tok.lo
    enterDepth(cx)
    let v = evalItem(ports, cx, unary(tmpl, ports, cx))
    dec cx.depth
    if cx.dry:
      return undefinedVal()
    case v.kind
    of vkInt:
      if neg and v.i == int64.low:
        raise jinjaErr("integer overflow in unary `-`", operandLo, cx.tok.lo - operandLo)
      intVal(if neg: -v.i else: v.i)
    of vkFloat: floatVal(if neg: -v.f else: v.f)
    else: raise jinjaErr("arithmetic needs a number, this is a " & $v.kind,
        operandLo, cx.tok.lo - operandLo)
  else:
    primary(tmpl, ports, cx)

func arith(op: Op, a, b: JinjaVal, opLo: int): JinjaVal =
  ## Combines two numbers, or two strings and two sequences under `+`.
  ## `%` follows Python's floor rule, the result taking the divisor's sign, so `-3 % 2` is `1`.
  ##
  ## Integer `+` and `-` check their result, an overflow raising a located
  ## `JinjaError` at the operator, never an uncatchable `OverflowDefect`.
  case op
  of opAdd:
    if a.kind == vkInt and b.kind == vkInt:
      if (b.i > 0 and a.i > int64.high - b.i) or (b.i < 0 and a.i < int64.low - b.i):
        raise jinjaErr("integer overflow in `+`", opLo)
      intVal(a.i + b.i)
    elif a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
      floatVal((if a.kind == vkInt: float64 a.i else: a.f) +
          (if b.kind == vkInt: float64 b.i else: b.f))
    elif a.kind == vkStr and b.kind == vkStr:
      strVal(a.s & b.s)
    elif a.kind in {vkStr, vkCut} and b.kind in {vkStr, vkCut}:
      # A cut operand materializes once here, the re-computed position of `+`.
      strVal(materializeVal(a).s & materializeVal(b).s)
    elif a.kind == vkSeq and b.kind == vkSeq:
      seqVal(a.xs.items & b.xs.items)
    else:
      raise jinjaErr("`+` cannot combine a " & $a.kind & " with a " & $b.kind)
  of opSub:
    if a.kind == vkInt and b.kind == vkInt:
      if (b.i > 0 and a.i < int64.low + b.i) or (b.i < 0 and a.i > int64.high + b.i):
        raise jinjaErr("integer overflow in `-`", opLo)
      intVal(a.i - b.i)
    elif a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
      floatVal((if a.kind == vkInt: float64 a.i else: a.f) -
          (if b.kind == vkInt: float64 b.i else: b.f))
    else:
      raise jinjaErr("`-` needs numbers")
  of opMod:
    if a.kind == vkInt and b.kind == vkInt:
      if b.i == 0:
        raise jinjaErr("`%` needs a non-zero divisor")
      var r = a.i mod b.i
      if r != 0 and ((r < 0) != (b.i < 0)):
        r += b.i
      intVal(r)
    elif a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
      let x = if a.kind == vkInt: float64 a.i else: a.f
      let y = if b.kind == vkInt: float64 b.i else: b.f
      if y == 0:
        raise jinjaErr("`%` needs a non-zero divisor")
      floatVal(math.floorMod(x, y))
    else:
      raise jinjaErr("`%` needs numbers")
  else:
    raise jinjaErr("unknown arithmetic `" & OpSpelling[op] & "`")

func cmpOne(op: Op, a, b: JinjaVal, at: int): JinjaVal =
  ## `at` locates the depth-cap raise a container-heavy comparison can carry.
  let r =
    case op
    of opEq: eqVal(a, b, at)
    of opNe: not eqVal(a, b, at)
    of opLt: cmpVal(a, b) < 0
    of opGt: cmpVal(a, b) > 0
    of opLe: cmpVal(a, b) <= 0
    of opGe: cmpVal(a, b) >= 0
    else: raise jinjaErr("unknown comparison `" & OpSpelling[op] & "`")
  boolVal(r)

func binOp(tmpl: CompiledTemplate, ports: Ports, cx: var Cx, lhs: JinjaVal, op: Op, opLo: int): JinjaVal =
  ## Evaluates the right operand of `op` and combines it with `lhs`. `and` and `or` skip the operand they do not evaluate, every other
  ## infix evaluating both sides. `and` and `or` render a pending macro call on the left
  ## before the truth test, a boolean position reading the output's bytes.
  case op
  of opAnd:
    let l = forceOperand(ports, cx, lhs)
    if not cx.dry and not isTruthy(l, opLo):
      skipExpr(tmpl, ports, cx, 4)
      return l
    expr(tmpl, ports, cx, 4)
  of opOr:
    let l = forceOperand(ports, cx, lhs)
    if not cx.dry and isTruthy(l, opLo):
      skipExpr(tmpl, ports, cx, 3)
      return l
    expr(tmpl, ports, cx, 3)
  of opIn, opNotIn:
    let rhs = evalItem(ports, cx, expr(tmpl, ports, cx, 6))
    if cx.dry:
      undefinedVal()
    else:
      let r = containsVal(rhs, lhs, opLo)
      boolVal(if op == opIn: r else: not r)
  of opConcat:
    let rhs = expr(tmpl, ports, cx, 7)
    if cx.dry: undefinedVal()
    else: concatVal(lhs, if rhs.kind == vkCall: forceOperand(ports, cx, rhs) else: rhs)
  of opAdd, opSub, opMod:
    let rhs = evalItem(ports, cx, expr(tmpl, ports, cx, binPrec(op) + 1))
    if cx.dry: undefinedVal() else: arith(op, lhs, rhs, opLo)
  of opMul, opDiv, opFloorDiv, opPow:
    skipExpr(tmpl, ports, cx, binPrec(op) + 1)
    # A dry walk is only mapping the skipped-branch spans, no value is read,
    # so the unimplemented operators surface only on a live evaluation.
    if cx.dry:
      undefinedVal()
    else:
      gapWhat("operator", OpSpelling[op])
  else:
    let rhs = evalItem(ports, cx, expr(tmpl, ports, cx, binPrec(op) + 1))
    if cx.dry: undefinedVal() else: cmpOne(op, lhs, rhs, opLo)

func ifWordAhead(tmpl: CompiledTemplate, at, stop: int): bool =
  ## Reports whether a depth-zero `if` survives in `tmpl.jinja[at..<stop)`. Byte scan, not a parse.
  ## Guarantees:
  ## - quoted text and bracketed subexpressions are skipped, so the scan never misses a ternary
  ## - the scan stops at the unit's own delimiters, a depth-zero `,`, `:`,
  ##   or an unmatched closer, so an `if` past them belongs to an enclosing
  ##   expression or to a sibling dict entry, never to this one
  ## - it can only over-report, costing one extra walk
  ##   while keeping the answer correct
  ## An expression holding no `if`, the corpus majority, is walked exactly once.
  var i = at
  var depth = 0
  var q: char = '\0'
  while i < stop:
    if q != '\0':
      if tmpl.jinja[i] == '\\':
        inc i
      elif tmpl.jinja[i] == q:
        q = '\0'
    elif tmpl.jinja[i] in {'\'', '"'}:
      q = tmpl.jinja[i]
    elif depth == 0 and tmpl.jinja[i] in {')', ']', '}', ',', ':'}:
      return false
    elif tmpl.jinja[i] in {'(', '[', '{'}:
      inc depth
    elif tmpl.jinja[i] in {')', ']', '}'}:
      dec depth
    elif depth == 0 and tmpl.jinja[i] == 'i' and stop - i >= 2 and tmpl.jinja[i + 1] == 'f' and
        (i + 2 >= stop or tmpl.jinja[i + 2] notin WsNameChars) and
        (i == at or tmpl.jinja[i - 1] notin WsNameChars):
      return true
    inc i
  false

func scanTernary(tmpl: CompiledTemplate, ports: Ports, cx: var Cx, headLo: int): Ternary =
  ## Measures the ternary spans starting at `headLo`, leaving the cursor past the whole ternary. A walk that lands on no ternary restores
  ## the cursor to the head, so the caller evaluates the expression normally. Nothing is evaluated here.
  let head = cx
  cx.pos = headLo
  advance(tmpl, cx)
  let dry = cx.dry
  cx.dry = true
  discard expr(tmpl, ports, cx, 2) # the then-branch text, unevaluated
  result.aHi = cx.tok.lo
  if not isWord(tmpl, cx, "if"):
    cx = head
    return
  advance(tmpl, cx)
  result.cLo = cx.tok.lo
  discard expr(tmpl, ports, cx, 2) # the condition, unevaluated
  result.cHi = cx.tok.lo
  if isWord(tmpl, cx, "else"):
    result.hasElse = true
    advance(tmpl, cx)
    result.bLo = cx.tok.lo
    discard expr(tmpl, ports, cx, 1) # the else-branch text, unevaluated
    result.bHi = cx.tok.lo
  else:
    result.bLo = result.cHi
    result.bHi = result.cHi
  let endPos = cx.tok.lo
  cx = head
  cx.pos = endPos
  advance(tmpl, cx)
  result.isTernary = true
  cx.dry = dry

func expr(tmpl: CompiledTemplate, ports: Ports, cx: var Cx, minPrec: int): JinjaVal =
  ## Parses and evaluates one expression, Pratt-style:
  ##   a prefix, then infix while the operator binds at least `minPrec`.
  ## - a ternary binds loosest, and no other operator holds binding power 1, so the ternary is
  ##   resolved before its head runs
  ## - the condition is evaluated once and exactly one branch is, which keeps a branch holding
  ##   `raise_exception` or `strftime_now` from acting while unelected
  ##
  ## Depth contract:
  ## - every entry counts one level toward `ExprDepthCap`, dry walks included, so skipped
  ##   operands and the ternary scan are bounded like evaluated ones
  ## - one exit decrements, the ternary legs included, so an entry is never left counted
  ## A stray `if` ends the expression, and `evalRange` reports the tail text.
  enterDepth(cx)
  var v: JinjaVal
  var ranTernary = false
  let headLo = cx.tok.lo
  if minPrec <= 1 and cx.tok.kind != exEof and ifWordAhead(tmpl, headLo, cx.stop):
    let shape = scanTernary(tmpl, ports, cx, headLo)
    if shape.isTernary:
      ranTernary = true
      if cx.dry:
        v = undefinedVal()
      else:
        let cond = evalRange(tmpl, cx.ports, shape.cLo, shape.cHi, cx.depth)
        let tested = forceOperand(cx.ports, cx, cond)
        if isTruthy(tested):
          v = evalRange(tmpl, cx.ports, headLo, shape.aHi, cx.depth)
        elif shape.hasElse:
          v = evalRange(tmpl, cx.ports, shape.bLo, shape.bHi, cx.depth)
        else:
          v = undefinedVal()
  if not ranTernary:
    v = unary(tmpl, ports, cx)
    while true:
      var op = opNone
      if cx.tok.kind == exName:
        if isWord(tmpl, cx, "and"):
          op = opAnd
        elif isWord(tmpl, cx, "or"):
          op = opOr
        elif isWord(tmpl, cx, "in"):
          op = opIn
        elif isWord(tmpl, cx, "not"):
          let save = cx
          advance(tmpl, cx)
          if isWord(tmpl, cx, "in"):
            op = opNotIn
          else:
            cx = save
            break
        else:
          break
      elif cx.tok.kind == exPunct:
        op = punctOp(cx.tok.p0, cx.tok.p1)
      let prec = binPrec(op)
      if prec == 0 or prec < minPrec:
        break
      if v.kind == vkCall:
        v = forceOperand(ports, cx, v)
      let opLo = cx.tok.lo # the operator token, still current here
      advance(tmpl, cx)
      v = binOp(tmpl, ports, cx, v, op, opLo)
  dec cx.depth
  v

func evalRange(tmpl: CompiledTemplate, ports: Ports, lo, hi: int, depth = 0): JinjaVal =
  ## Evaluates the expression held in `tmpl.jinja[lo..<hi]` in its own cursor.
  ## - `depth` seeds the nesting counter, so a sub-span reached through a ternary still counts toward `ExprDepthCap`
  ## - `ports` carries the lookup, clock and macro-forcer services, so a nested call can still run
  var cx = Cx(pos: lo, stop: hi, dry: false, depth: depth, ports: ports)
  advance(tmpl, cx)
  result = expr(tmpl, ports, cx, 1)
  if cx.tok.kind != exEof:
    raise jinjaErr("expression has trailing text at byte " & $cx.tok.lo, cx.tok.lo)

func evalSpan*(tmpl: CompiledTemplate, ports: Ports, lo, hi: int32): JinjaVal =
  ## Evaluates the expression held in `tmpl.jinja[lo..<hi]`, the entry every expression-bearing step uses.
  ## Contract:
  ## - `ports` carries the render services, a nil forcer in them making a consumed macro
  ##   call a reported gap
  ## - a macro call that is a whole expression returns pending, for the emit step to stream
  evalRange(tmpl, ports, lo.int, hi.int, 0)
