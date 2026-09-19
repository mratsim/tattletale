# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Fused expression walker of the chattyninja engine.
# One `lo..hi` span of template text is parsed and evaluated in a single pass, no expression
# becomes a node. A dry pass advances tokens without evaluating, one mechanism three places use:
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
    exEof, exName, exInt, exFloat, exStr, exPunct

  ExTok = object
    kind: ExKind
    lo, hi: int # span into Machine.jinja
    i: int64 # payload of the exInt token
    f: float64 # payload of the exFloat token
    s: string # decoded string literal
    p0, p1: char # punctuator bytes, `p1 == '\0'` for a one-byte punctuator

  Cx = object
    ## Walker cursor:
    ##   the half-open span it owns, a one-token lookahead, the dry flag, the recursion depth, the macro body runner handed to nested calls.
    pos, stop: int
    tok: ExTok
    dry: bool
    depth: int
    force: MacroForcer

  GlobalProc* = proc (m: Machine, args: seq[Arg], d: var Driver): JinjaVal {.nimcall.}
    ## A call to a template global, `namespace` and `dict` storing a keyword name as a dict key.

  MacroForcer* = proc (m: Machine, t: Tables, d: var Driver, mc: MacroVal,
      args: seq[CallArg]): string {.nimcall.}
    ## Runs one macro body to completion and returns the captured text. The statement tier
    ## injects the forcer, so `cnj_engine` and `jinja_interpolation` stay free of an import cycle.

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

func parseIntToken(s: openArray[char]): int64 =
  ## Returns the integer the token bytes spell, ValueError on a malformed token.
  var n: int64
  if s.len == 0 or parseutils.parseBiggestInt(s, n) != s.len:
    raise newException(ValueError, "invalid integer: " & spanString(s))
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
  else:
    ExTok(kind: exInt, lo: start, hi: i, i: parseIntToken(text))

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

func advance(m: Machine, cx: var Cx) =
  ## Loads the next token into the cursor, stopping at the cursor's own `stop`.
  let s = m.jinja
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

func isWord(m: Machine, cx: Cx, w: string): bool =
  ## Reports whether the lookahead is the bare identifier `w`, keywords matched without
  ## copying text out of the template.
  if cx.tok.kind != exName or cx.tok.hi - cx.tok.lo != w.len:
    return false
  for k in 0 ..< w.len:
    if m.jinja[cx.tok.lo + k] != w[k]:
      return false
  true

func wordSpan(m: Machine, lo, hi: int): openArray[char] =
  ## Returns the template text in `lo ..< hi` as a view, so neither an identifier nor a registry lookup allocates. `lo ..< hi`
  ## is the half-open span the token carries.
  m.jinja.toOpenArray(lo, hi - 1)

func wordSpan(m: Machine, cx: Cx): openArray[char] =
  ## Returns the identifier of the lookahead token as a view into the template text.
  wordSpan(m, cx.tok.lo, cx.tok.hi)

func argName*(m: Machine, a: Arg): openArray[char] =
  ## Returns an argument's keyword name as a view into the template text, `nameLo == NoLink`
  ## marking a positional argument, which has no name to read.
  m.jinja.toOpenArray(a.nameLo.int, a.nameHi.int - 1)

proc forceCall(m: Machine, t: Tables, d: var Driver, cx: var Cx, v: JinjaVal): JinjaVal =
  ## Returns `v` rendered to its macro output text. Every expression consumer other than
  ## the emit step reads a pending macro call in this form.
  if cx.force.isNil:
    raise jinjaErr("a macro call result was consumed where no macro forcer was supplied")
  strVal(cx.force(m, t, d, v.pc.mc, v.pc.args))

proc evalItem(m: Machine, t: Tables, d: var Driver, cx: var Cx, v: JinjaVal): JinjaVal =
  ## Returns `v`, rendering a pending macro call to text for the value containers and operators
  ## that read a plain value. A concat raises here, the argument list being its one
  ## plain-value reader. A dry walk returns `v` unevaluated.
  if not cx.dry and v.kind == vkCall:
    forceCall(m, t, d, cx, v)
  elif not cx.dry and v.kind == vkConcat:
    raise jinjaErr("a concat must be rendered in emit position")
  else:
    v

proc argVal(m: Machine, t: Tables, d: var Driver, cx: var Cx, v: JinjaVal): JinjaVal =
  ## Returns one call argument's value, a pending macro call rendering to text and a concat
  ## materializing through the serializer's drain-and-grow form, arguments reading plain
  ## values only. A dry walk returns `v` unevaluated.
  if cx.dry:
    return v
  if v.kind == vkCall:
    return forceCall(m, t, d, cx, v)
  if v.kind == vkConcat:
    return strVal(pyStr(v))
  v

func argKey(m: Machine, a: Arg): string =
  ## Returns the dict key one argument supplies to `namespace` or `dict`, a keyword-bound argument
  ## giving the keyword text, a positional one its stringified value. A `DictVal` key is a string,
  ## so this is where a keyword name becomes one.
  if a.nameLo == NoLink:
    pyStr(a.val)
  else:
    spanString(argName(m, a))

func runeOffset(s: string, k: int): int =
  ## Returns the byte offset of the codepoint at index `k`, advancing by UTF-8 lead-byte strides, the same walk `runeLen`
  ## and the `runes` iterator take.
  var j = 0
  for _ in 0 ..< k:
    inc j, runeLenAt(s, j)
  j

func runeSub(s: string, i: int): Rune =
  ## Returns the codepoint at Python index `i`, a negative `i` counting from the end.
  ## An ASCII codepoint answers by one byte read, a multibyte one decodes in place.
  let n = runeLen(s)
  let idx = if i < 0: n + i else: i
  if idx < 0 or idx >= n:
    raise jinjaErr("string subscript " & $i & " is out of range")
  let j = runeOffset(s, idx)
  if s[j].ord < 0x80: Rune(s[j].ord) else: runeAt(s, j)

func steppedSliceInto(sb: var Cursor, s: string, a, b, by: int) =
  ## Writes the stride-`by` codepoint slice into `sb`, visiting `a, a + by, ...`
  ## while the stride keeps the walk inside the clamped bounds.
  var k = a
  while (by > 0 and k < b) or (by < 0 and k > b):
    sb.addRune runeSub(s, k)
    inc k, by

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
  ## Returns a `loop.*` attribute, read through the driver frame's shared cursor. The attribute is
  ## selected by span compare, so an attribute inside a `{% for %}` body costs no string.
  let lp = v.lp
  let n = lp.items.len
  let i = lp.idx
  if name == "index": intVal(i + 1)
  elif name == "index0": intVal(i)
  elif name == "first": boolVal(i == 0)
  elif name == "last": boolVal(i == n - 1)
  elif name == "length": intVal(n)
  elif name == "previtem":
    if i > 0: lp.items[i - 1] else: noneVal()
  elif name == "nextitem":
    if i + 1 < n: lp.items[i + 1] else: noneVal()
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
      strVal(spanString(v.s.toOpenArray(runeOffset(v.s, a), runeOffset(v.s, b) - 1)))
    else:
      var sb = measureBuf()
      sb.steppedSliceInto(v.s, a, b, by)
      var win = newString(sb.len)
      var dst = over(win)
      dst.steppedSliceInto(v.s, a, b, by)
      strVal(win)

func argDict(m: Machine, args: seq[Arg]): DictVal =
  ## Returns one mapping holding the call's arguments, keyword names as dict keys.
  var dv = DictVal()
  for a in args:
    dv.dictSet(argKey(m, a), a.val)
  dv

func namespaceGlobal(m: Machine, args: seq[Arg], d: var Driver): JinjaVal =
  ## `namespace(field=init, ...)`:
  ##   the mutable mapping `{% set ns.field = ... %}` mutates in place.
  nsVal(argDict(m, args))

func dictGlobal(m: Machine, args: seq[Arg], d: var Driver): JinjaVal =
  dictVal(argDict(m, args))

func rangeGlobal(m: Machine, args: seq[Arg], d: var Driver): JinjaVal =
  var a = 0'i64
  var b = 0'i64
  var step = 1'i64
  for i, x in args:
    if x.val.kind != vkInt:
      raise jinjaErr("`range` needs integer bounds")
    case i
    of 0: b = x.val.i
    of 1:
      a = b
      b = x.val.i
    of 2: step = x.val.i
    else: raise jinjaErr("`range` takes at most three arguments")
  if step == 0:
    raise jinjaErr("`range` step must not be zero")
  var acc = newSeq[JinjaVal]()
  var k = a
  while (step > 0 and k < b) or (step < 0 and k > b):
    acc.add intVal(k)
    inc k, step
  seqVal(acc)

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

func strftimeGlobal(m: Machine, args: seq[Arg], d: var Driver): JinjaVal =
  ## Renders the format against the driver's injected epoch, never the wall clock, which is
  ## what keeps two drivers over one `Machine` byte-identical.
  let fmt = pyStr(getArg(args, 0, akNone, strVal("")))
  let secs = d.clock.int64
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
      raise jinjaErr("`strftime_now` format ends on a `%`")
    case fmt[i + 1]
    of 'Y': acc.add yearField(yr)
    of 'm': acc.add twoDigits(mo)
    of 'd': acc.add twoDigits(dy)
    of 'H': acc.add twoDigits(tod div 3600)
    of 'M': acc.add twoDigits((tod mod 3600) div 60)
    of 'S': acc.add twoDigits(tod mod 60)
    of 'j': acc.add $dayOfYear(yr, mo, dy)
    of '%': acc.add '%'
    else: gapWhat("`strftime_now` directive", "%" & fmt[i + 1])
    inc i, 2
  strVal(acc)

func raiseExceptionGlobal(m: Machine, args: seq[Arg], d: var Driver): JinjaVal =
  ## Corpus `err_*` rows record exactly this raise:
  ##   the message verbatim, cause `ceRaiseCall`.
  raise jinjaErr(pyStr(getArg(args, 0, akNone, strVal(""))), cause = ceRaiseCall)

const
  GlobalProcs: array[GlobalName, GlobalProc] = [
    namespaceGlobal, rangeGlobal, strftimeGlobal, raiseExceptionGlobal, dictGlobal, nil, nil, nil
  ]

# Lookup:

func lookupName(t: Tables, d: var Driver, name: openArray[char]): JinjaVal =
  ## Returns the binding of `name`, undefined when absent. Absence is a value, never an error:
  ##   that is what `is defined` tests, what makes a missing dict key a missing name, scope scan
  ##   and root lookup reading the caller's bytes in place.
  let id = findName(t, name)
  if id != NoLink:
    for si in countdown(d.scopes.len - 1, 0):
      for b in d.scopes[si]:
        if b.name == id:
          return b.val
  if d.root.kind == vkDict:
    return d.root.d.dictGet(name)
  undefinedVal()

func lookupNameById*(t: Tables, d: var Driver, id: int32): JinjaVal =
  ## Returns the binding of an interned name, undefined when absent. The scope key is the id, so no
  ## string is rebuilt per lookup.
  if id == NoLink:
    return undefinedVal()
  for si in countdown(d.scopes.len - 1, 0):
    for b in d.scopes[si]:
      if b.name == id:
        return b.val
  if d.root.kind == vkDict and id < t.names.len.int32:
    return d.root.d.dictGet(t.names[id])
  undefinedVal()

# Walker:

proc evalRange(m: Machine, t: Tables, d: var Driver, lo, hi: int, depth = 0, force: MacroForcer = nil): JinjaVal
proc expr(m: Machine, t: Tables, d: var Driver, cx: var Cx, minPrec: int): JinjaVal


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

proc skipExpr(m: Machine, t: Tables, d: var Driver, cx: var Cx, minPrec: int) =
  ## Advances the cursor over an expression without evaluating it, how `and`, `or` and the ternary skip the text they do not run.
  let wasDry = cx.dry
  cx.dry = true
  discard expr(m, t, d, cx, minPrec)
  cx.dry = wasDry

proc argList(m: Machine, t: Tables, d: var Driver, cx: var Cx): seq[Arg] =
  ## Parses a parenthesised argument list with `name = expr` keyword arguments, the opening paren the lookahead. A keyword keeps its span
  ## and gains a builtin keyword slot, so binding one costs no string.
  advance(m, cx)
  while not isPunct(cx, ")"):
    var nameLo = NoLink
    var nameHi = NoLink
    var kw = akNone
    if cx.tok.kind == exName:
      let save = cx
      advance(m, cx)
      if isPunct(cx, "="):
        nameLo = int32 save.tok.lo
        nameHi = int32 save.tok.hi
        kw = argKeyword(wordSpan(m, save))
        advance(m, cx)
      else:
        cx = save
    let v = argVal(m, t, d, cx, expr(m, t, d, cx, 1))
    result.add Arg(nameLo: nameLo, nameHi: nameHi, kw: kw, val: v)
    if isPunct(cx, ","):
      advance(m, cx)
      if isPunct(cx, ")"):
        break
      continue
    break
  if not isPunct(cx, ")"):
    raise jinjaErr("argument list is not closed", cx.tok.lo)
  advance(m, cx)

proc postfix(m: Machine, t: Tables, d: var Driver, cx: var Cx, v: JinjaVal): JinjaVal =
  ## Applies attr, subscript, call, filter and test chains, which bind tighter than any operator.
  var v = v
  while true:
    # An operator reading the chained value renders a pending macro call first.
    # A call keeps the natural not-callable error for the rendered text.
    if v.kind == vkCall and (isPunct(cx, ".") or isPunct(cx, "[") or isPunct(cx, "|") or
        isWord(m, cx, "is")):
      v = forceCall(m, t, d, cx, v)
    if isPunct(cx, "."):
      advance(m, cx)
      if cx.tok.kind != exName:
        raise jinjaErr("expected a name after `.`", cx.tok.lo)
      let (lo, hi) = (cx.tok.lo, cx.tok.hi)
      advance(m, cx)
      if isPunct(cx, "("):
        let a = argList(m, t, d, cx)
        if cx.dry:
          v = undefinedVal()
          continue
        let mi = findIn(MethodNames, wordSpan(m, lo, hi))
        if mi < 0:
          raise jinjaErr("unknown method `" & spanString(wordSpan(m, lo, hi)) & "`", lo, hi - lo)
        let mp = MethodProcs[MethodName mi]
        if mp.isNil:
          gapWhat("method", wordSpan(m, lo, hi))
        v = mp(v, a)
      else:
        v =
          if cx.dry:
            undefinedVal()
          else:
            case v.kind
            of vkDict, vkNs: v.d.dictGet(wordSpan(m, lo, hi))
            of vkLoop: loopAttr(v, wordSpan(m, lo, hi))
            of vkUndefined: undefinedVal()
            else: raise jinjaErr("`" & spanString(wordSpan(m, lo, hi)) &
                "` is not an attribute of a " & $v.kind, lo, hi - lo)
    elif isPunct(cx, "["):
      advance(m, cx)
      var lo, hi, step = undefinedVal()
      var hasLo, hasHi, hasStep, isSlice = false
      if not isPunct(cx, ":"):
        lo = expr(m, t, d, cx, 1)
        hasLo = true
      if isPunct(cx, ":"):
        isSlice = true
        advance(m, cx)
        if not (isPunct(cx, ":") or isPunct(cx, "]")):
          hi = expr(m, t, d, cx, 1)
          hasHi = true
        if isPunct(cx, ":"):
          advance(m, cx)
          if not isPunct(cx, "]"):
            step = expr(m, t, d, cx, 1)
            hasStep = true
      if not isPunct(cx, "]"):
        raise jinjaErr("subscript is not closed", cx.tok.lo)
      advance(m, cx)
      v = if cx.dry: undefinedVal() else:
        subslice(v, lo, hi, step, hasLo, hasHi, hasStep, isSlice)
    elif isPunct(cx, "("):
      let a = argList(m, t, d, cx)
      v =
        if cx.dry:
          undefinedVal()
        elif v.kind == vkMacro:
          var cargs = newSeq[CallArg](a.len)
          for i, arg in a:
            cargs[i] = CallArg(nameLo: arg.nameLo, nameHi: arg.nameHi, val: arg.val)
          callVal(PendingCallVal(mc: v.mc, args: cargs))
        else:
          raise jinjaErr("only a macro is callable, this is a " & $v.kind)
    elif isPunct(cx, "|"):
      advance(m, cx)
      if cx.tok.kind != exName:
        raise jinjaErr("expected a filter name after `|`", cx.tok.lo)
      let (lo, hi) = (cx.tok.lo, cx.tok.hi)
      advance(m, cx)
      var a = newSeq[Arg]()
      if isPunct(cx, "("):
        a = argList(m, t, d, cx)
      if cx.dry:
        v = undefinedVal()
        continue
      let fi = findIn(FilterNames, wordSpan(m, lo, hi))
      if fi < 0:
        raise jinjaErr("unknown filter `" & spanString(wordSpan(m, lo, hi)) & "`", lo, hi - lo, cause = ceUnimplemented)
      let fp = FilterProcs[FilterName fi]
      if fp.isNil:
        gapWhat("filter", wordSpan(m, lo, hi))
      v = fp(v, a)
    elif isWord(m, cx, "is"):
      advance(m, cx)
      var negated = false
      if isWord(m, cx, "not"):
        negated = true
        advance(m, cx)
      if cx.tok.kind != exName:
        raise jinjaErr("expected a test name after `is`", cx.tok.lo)
      let (lo, hi) = (cx.tok.lo, cx.tok.hi)
      advance(m, cx)
      var a = newSeq[Arg]()
      if isPunct(cx, "("):
        a = argList(m, t, d, cx)
      if cx.dry:
        v = undefinedVal()
        continue
      let ti = findIn(TestNames, wordSpan(m, lo, hi))
      if ti < 0:
        raise jinjaErr("unknown test `" & spanString(wordSpan(m, lo, hi)) & "`", lo, hi - lo)
      let tp = TestProcs[TestName ti]
      if tp.isNil:
        gapWhat("test", wordSpan(m, lo, hi))
      v = boolVal(if negated: not tp(v, a) else: tp(v, a))
    else:
      break
  v

proc primary(m: Machine, t: Tables, d: var Driver, cx: var Cx): JinjaVal =
  ## Parses a literal, a name, a parenthesised group, an array literal or a dict literal, then
  ## the postfix chain.
  var v: JinjaVal
  case cx.tok.kind
  of exEof:
    raise jinjaErr("expression ends early at byte " & $cx.pos, cx.pos)
  of exInt:
    v = intVal(cx.tok.i)
    advance(m, cx)
  of exFloat:
    v = floatVal(cx.tok.f)
    advance(m, cx)
  of exStr:
    v = strVal(cx.tok.s)
    advance(m, cx)
  of exName:
    let (lo, hi) = (cx.tok.lo, cx.tok.hi)
    advance(m, cx)
    let name = wordSpan(m, lo, hi)
    # Literal spellings are compared as spans, the same test `case` applied to a copied string.
    if name == "true" or name == "True":
      v = boolVal(true)
    elif name == "false" or name == "False":
      v = boolVal(false)
    elif name == "none" or name == "None":
      v = noneVal()
    else:
      var bound = if cx.dry: undefinedVal() else: lookupName(t, d, name)
      let gi = findIn(GlobalNames, name)
      if bound.kind == vkUndefined and gi >= 0 and isPunct(cx, "("):
        # A global is reached only when the name is unbound, Jinja's own precedence:
        #   context shadows globals, and a skipped branch never gets here. A dry walk still
        #   consumes the argument list, so the skipped text is never left behind as trailing text.
        let a = argList(m, t, d, cx)
        if cx.dry:
          v = undefinedVal()
        else:
          let gp = GlobalProcs[GlobalName gi]
          if gp.isNil:
            gapWhat("global", wordSpan(m, lo, hi))
          v = gp(m, a, d)
      else:
        v = bound
  of exPunct:
    case cx.tok.p0
    of '(':
      advance(m, cx)
      var parts = newSeq[JinjaVal]()
      var isTuple = false
      while not isPunct(cx, ")"):
        parts.add expr(m, t, d, cx, 1)
        if isPunct(cx, ","):
          isTuple = true
          advance(m, cx)
          if isPunct(cx, ")"):
            break
          continue
        break
      if not isPunct(cx, ")"):
        raise jinjaErr("parenthesised expression is not closed", cx.tok.lo)
      advance(m, cx)
      v = if parts.len == 0: seqVal(parts) elif isTuple: seqVal(parts) else: parts[0]
    of '[':
      advance(m, cx)
      var parts = newSeq[JinjaVal]()
      while not isPunct(cx, "]"):
        parts.add evalItem(m, t, d, cx, expr(m, t, d, cx, 1))
        if isPunct(cx, ","):
          advance(m, cx)
          if isPunct(cx, "]"):
            break
          continue
        break
      if not isPunct(cx, "]"):
        raise jinjaErr("array literal is not closed", cx.tok.lo)
      advance(m, cx)
      v = seqVal(parts)
    of '{':
      advance(m, cx)
      var dv = DictVal()
      while not isPunct(cx, "}"):
        let k = evalItem(m, t, d, cx, expr(m, t, d, cx, 1))
        if not isPunct(cx, ":"):
          raise jinjaErr("dict literal entry needs a `:`", cx.tok.lo)
        advance(m, cx)
        let val = evalItem(m, t, d, cx, expr(m, t, d, cx, 1))
        if not cx.dry:
          dv.dictSet(pyStr(k), val)
        if isPunct(cx, ","):
          advance(m, cx)
          if isPunct(cx, "}"):
            break
          continue
        break
      if not isPunct(cx, "}"):
        raise jinjaErr("dict literal is not closed", cx.tok.lo)
      advance(m, cx)
      v = dictVal(dv)
    else:
      let spelled = $cx.tok.p0 & (if cx.tok.p1 != '\0': $cx.tok.p1 else: "")
      raise jinjaErr("unexpected `" & spelled & "` starting an expression at byte " & $cx.tok.lo, cx.tok.lo, cx.tok.hi - cx.tok.lo)
  postfix(m, t, d, cx, v)

proc unary(m: Machine, t: Tables, d: var Driver, cx: var Cx): JinjaVal =
  ## Parses `not`, unary `-` and `+`, then a primary.
  if isWord(m, cx, "not"):
    advance(m, cx)
    let v = evalItem(m, t, d, cx, unary(m, t, d, cx))
    return boolVal(if cx.dry: false else: not isTruthy(v))
  if isPunct(cx, "-") or isPunct(cx, "+"):
    let neg = isPunct(cx, "-")
    advance(m, cx)
    let v = evalItem(m, t, d, cx, unary(m, t, d, cx))
    if cx.dry:
      return undefinedVal()
    case v.kind
    of vkInt: intVal(if neg: -v.i else: v.i)
    of vkFloat: floatVal(if neg: -v.f else: v.f)
    else: raise jinjaErr("arithmetic needs a number, this is a " & $v.kind)
  else:
    primary(m, t, d, cx)

func arith(op: Op, a, b: JinjaVal): JinjaVal =
  ## Combines two numbers, or two strings and two sequences under `+`.
  ## `%` follows Python's floor rule, the result taking the divisor's sign, so `-3 % 2` is `1`.
  case op
  of opAdd:
    if a.kind == vkInt and b.kind == vkInt:
      intVal(a.i + b.i)
    elif a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
      floatVal((if a.kind == vkInt: float64 a.i else: a.f) +
          (if b.kind == vkInt: float64 b.i else: b.f))
    elif a.kind == vkStr and b.kind == vkStr:
      strVal(a.s & b.s)
    elif a.kind == vkSeq and b.kind == vkSeq:
      seqVal(a.xs.items & b.xs.items)
    else:
      raise jinjaErr("`+` cannot combine a " & $a.kind & " with a " & $b.kind)
  of opSub:
    if a.kind == vkInt and b.kind == vkInt:
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

func cmpOne(op: Op, a, b: JinjaVal): JinjaVal =
  let r =
    case op
    of opEq: eqVal(a, b)
    of opNe: not eqVal(a, b)
    of opLt: cmpVal(a, b) < 0
    of opGt: cmpVal(a, b) > 0
    of opLe: cmpVal(a, b) <= 0
    of opGe: cmpVal(a, b) >= 0
    else: raise jinjaErr("unknown comparison `" & OpSpelling[op] & "`")
  boolVal(r)

proc binOp(m: Machine, t: Tables, d: var Driver, cx: var Cx, lhs: JinjaVal, op: Op): JinjaVal =
  ## Evaluates the right operand of `op` and combines it with `lhs`. `and` and `or` skip the operand they do not evaluate, every other
  ## infix evaluating both sides.
  case op
  of opAnd:
    if not cx.dry and not isTruthy(lhs):
      skipExpr(m, t, d, cx, 4)
      return lhs
    expr(m, t, d, cx, 4)
  of opOr:
    if not cx.dry and isTruthy(lhs):
      skipExpr(m, t, d, cx, 3)
      return lhs
    expr(m, t, d, cx, 3)
  of opIn, opNotIn:
    let rhs = evalItem(m, t, d, cx, expr(m, t, d, cx, 6))
    if cx.dry:
      undefinedVal()
    else:
      let r = containsVal(rhs, lhs)
      boolVal(if op == opIn: r else: not r)
  of opConcat:
    let rhs = expr(m, t, d, cx, 7)
    if cx.dry: undefinedVal()
    else: concatVal(lhs, if rhs.kind == vkCall: forceCall(m, t, d, cx, rhs) else: rhs)
  of opAdd, opSub, opMod:
    let rhs = evalItem(m, t, d, cx, expr(m, t, d, cx, binPrec(op) + 1))
    if cx.dry: undefinedVal() else: arith(op, lhs, rhs)
  of opMul, opDiv, opFloorDiv, opPow:
    skipExpr(m, t, d, cx, binPrec(op) + 1)
    gapWhat("operator", OpSpelling[op])
  else:
    let rhs = evalItem(m, t, d, cx, expr(m, t, d, cx, binPrec(op) + 1))
    if cx.dry: undefinedVal() else: cmpOne(op, lhs, rhs)

func ifWordAhead(m: Machine, at, stop: int): bool =
  ## Reports whether a depth-zero `if` survives in `m.jinja[at..<stop)`. Byte scan, not a parse.
  ## Guarantees:
  ## - quoted text and bracketed subexpressions are skipped, so the scan never misses a ternary
  ## - it can only over-report, costing one extra walk
  ##   while keeping the answer correct
  ## An expression holding no `if`, the corpus majority, is walked exactly once.
  var i = at
  var depth = 0
  var q: char = '\0'
  while i < stop:
    if q != '\0':
      if m.jinja[i] == '\\':
        inc i
      elif m.jinja[i] == q:
        q = '\0'
    elif m.jinja[i] in {'\'', '"'}:
      q = m.jinja[i]
    elif m.jinja[i] in {'(', '[', '{'}:
      inc depth
    elif m.jinja[i] in {')', ']', '}'}:
      dec depth
    elif depth == 0 and m.jinja[i] == 'i' and stop - i >= 2 and m.jinja[i + 1] == 'f' and
        (i + 2 >= stop or m.jinja[i + 2] notin wsNameChars) and
        (i == at or m.jinja[i - 1] notin wsNameChars):
      return true
    inc i
  false

proc scanTernary(m: Machine, t: Tables, d: var Driver, cx: var Cx, headLo: int): Ternary =
  ## Measures the ternary spans starting at `headLo`, leaving the cursor past the whole ternary. A walk that lands on no ternary restores
  ## the cursor to the head, so the caller evaluates the expression normally. Nothing is evaluated here.
  let head = cx
  cx.pos = headLo
  advance(m, cx)
  let dry = cx.dry
  cx.dry = true
  discard expr(m, t, d, cx, 2) # the then-branch text, unevaluated
  result.aHi = cx.tok.lo
  if not isWord(m, cx, "if"):
    cx = head
    return
  advance(m, cx)
  result.cLo = cx.tok.lo
  discard expr(m, t, d, cx, 2) # the condition, unevaluated
  result.cHi = cx.tok.lo
  if isWord(m, cx, "else"):
    result.hasElse = true
    advance(m, cx)
    result.bLo = cx.tok.lo
    discard expr(m, t, d, cx, 1) # the else-branch text, unevaluated
    result.bHi = cx.tok.lo
  else:
    result.bLo = result.cHi
    result.bHi = result.cHi
  let endPos = cx.tok.lo
  cx = head
  cx.pos = endPos
  advance(m, cx)
  result.isTernary = true
  cx.dry = dry

proc expr(m: Machine, t: Tables, d: var Driver, cx: var Cx, minPrec: int): JinjaVal =
  ## Parses and evaluates one expression, Pratt-style:
  ##   a prefix, then infix while the operator binds at least `minPrec`.
  ## - a ternary binds loosest, and no other operator holds binding power 1, so the ternary is
  ##   resolved before its head runs
  ## - the condition is evaluated once and exactly one branch is, which keeps a branch holding
  ##   `raise_exception` or `strftime_now` from acting while unelected
  ## A stray `if` ends the expression, and `evalRange` reports the tail text.
  if not cx.dry:
    inc cx.depth
    if cx.depth > ExprDepthCap:
      raise jinjaErr("expression nests deeper than ExprDepthCap = " & $ExprDepthCap)
  let headLo = cx.tok.lo
  if minPrec <= 1 and cx.tok.kind != exEof and ifWordAhead(m, headLo, cx.stop):
    let shape = scanTernary(m, t, d, cx, headLo)
    if shape.isTernary:
      if cx.dry:
        return undefinedVal()
      let cond = evalRange(m, t, d, shape.cLo, shape.cHi, cx.depth, cx.force)
      if isTruthy(cond):
        return evalRange(m, t, d, headLo, shape.aHi, cx.depth, cx.force)
      if shape.hasElse:
        return evalRange(m, t, d, shape.bLo, shape.bHi, cx.depth, cx.force)
      return undefinedVal()
  var v = unary(m, t, d, cx)
  while true:
    var op = opNone
    if cx.tok.kind == exName:
      if isWord(m, cx, "and"):
        op = opAnd
      elif isWord(m, cx, "or"):
        op = opOr
      elif isWord(m, cx, "in"):
        op = opIn
      elif isWord(m, cx, "not"):
        let save = cx
        advance(m, cx)
        if isWord(m, cx, "in"):
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
      v = forceCall(m, t, d, cx, v)
    advance(m, cx)
    v = binOp(m, t, d, cx, v, op)
  if not cx.dry:
    dec cx.depth
  v

proc evalRange(m: Machine, t: Tables, d: var Driver, lo, hi: int, depth = 0, force: MacroForcer = nil): JinjaVal =
  ## Evaluates the expression held in `m.jinja[lo..<hi]` in its own cursor.
  ## - `depth` seeds the nesting counter, so a sub-span reached through a ternary still counts toward `ExprDepthCap`
  ## - `force` carries the macro forcer, so a nested call can still run
  var cx = Cx(pos: lo, stop: hi, dry: false, depth: depth, force: force)
  advance(m, cx)
  result = expr(m, t, d, cx, 1)
  if cx.tok.kind != exEof:
    raise jinjaErr("expression has trailing text at byte " & $cx.tok.lo, cx.tok.lo)

proc evalSpan*(m: Machine, t: Tables, d: var Driver, lo, hi: int32, force: MacroForcer = nil): JinjaVal =
  ## Evaluates the expression held in `m.jinja[lo..<hi]`, the entry every expression-bearing step uses.
  ## Contract:
  ## - `force` is the macro forcer, passed by every step that can meet a macro call
  ## - a nil `force` makes a consumed macro call a reported gap
  ## - a macro call that is a whole expression returns pending, for the emit step to stream
  evalRange(m, t, d, lo.int, hi.int, 0, force)
