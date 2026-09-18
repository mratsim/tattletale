# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Fused expression walker of the chattyninja engine.
# An expression is template text, and one `lo..hi` span of it is parsed and evaluated
# in a single pass. No expression becomes a node.
#
# A dry pass advances tokens without evaluating. Three places need it, and all three are the same
# mechanism:
#
# - `and` and `or` skip the operand they do not evaluate
# - a ternary is located by a dry scan so the condition can be evaluated first and exactly one
#   branch then runs. Re-running a branch that holds `strftime_now` or `raise_exception` would
#   change bytes or raise from the branch not taken
# - skipped branches never reach the registries, so an unimplemented construct cannot fail
#   a render from a branch Jinja would not have entered

import std/[math, strbasics, strutils, times, unicode]
import cjn_errors, cjn_types, cjn_values

type
  ExKind = enum
    exEof
    exName
    exInt
    exFloat
    exStr
    exPunct

  ExTok = object
    kind: ExKind
    lo, hi: int # span into Machine.jinja
    i: int64 # payload of the exInt token
    f: float64 # payload of the exFloat token
    s: string # decoded string literal, or punctuator text

  Cx = object
    ## Walker cursor:
    ##   the half-open span it owns, one token of lookahead, the dry flag, the recursion depth,
    ## and the macro body runner handed to nested calls.
    pos, stop: int
    tok: ExTok
    dry: bool
    depth: int
    run: MacroRunner

  ArgKeyword = enum
    ## A keyword name a builtin reads out of an argument list.
    ## `akNone` is a positional argument or a keyword no builtin reads, and is the field's default.
    akNone
    akChars
    akDefault
    akEnsureAscii
    akSeparators

  Arg* = object
    ## One call or filter argument, keyword-bound when `nameLo` is not `noLink`.
    ## A keyword keeps its template span rather than becoming an interned id, because a call may name
    ## anything and only a name the parser binds reaches `Tables.names`.
    nameLo*, nameHi*: int32
      ## keyword name span into `Machine.jinja`, `noLink` in `nameLo` for a positional argument
    kw*: ArgKeyword
      ## keyword slot named by that span, `akNone` when no builtin reads that keyword
    val*: Value

  FilterProc* = proc (v: Value, args: seq[Arg]): Value {.nimcall.}
  TestProc* = proc (v: Value, args: seq[Arg]): bool {.nimcall.}
  MethodProc* = proc (v: Value, args: seq[Arg]): Value {.nimcall.}
  GlobalProc* = proc (m: Machine, args: seq[Arg], d: var Driver): Value {.nimcall.}
    ## A call to a template global.
    ## `namespace` and `dict` store a keyword name as a dict key, so the proc reads the artifact.

  MacroRunner* = proc (m: Machine, t: Tables, d: var Driver, mc: MacroVal,
      args: seq[Arg]): string {.nimcall.}
    ## Runs one macro body to completion and returns the captured text. The statement tier owns
    ## the arena, so eval reaches it through this injected runner rather than by importing it.
    ## `chattyninja` and `cjn_expr` cannot import each other, keeping the artifact proc-free.

const wsChars = {' ', '\t', '\n', '\r', '\v', '\f'}

# Lexer
# ---------------------------------------------------------------------------

func digitVal(c: char): int =
  let d = ord(c) - ord('0')
  if d >= 0 and d <= 9: d else: (ord(c) or 32) - ord('a') + 10

func decodeEscapes(s: openArray[char], lo, hi: int): string =
  ## Returns a string literal with Python's escape set resolved.
  result = newStringOfCap(hi - lo)
  var i = lo
  while i < hi:
    if s[i] != '\\':
      result.add s[i]
      inc i
      continue
    inc i
    if i >= hi:
      raise err("truncated escape in a string literal")
    case s[i]
    of 'n': result.add '\n'
    of 't': result.add '\t'
    of 'r': result.add '\r'
    of '0': result.add '\0'
    of 'a': result.add '\a'
    of 'b': result.add '\b'
    of 'f': result.add '\f'
    of 'v': result.add '\v'
    of 'e': result.add '\e'
    of '\\': result.add '\\'
    of '\'': result.add '\''
    of '"': result.add '"'
    of 'x', 'u':
      let digits = if s[i] == 'x': 2 else: 4
      var code = 0
      var k = 1
      while k <= digits:
        if i + k >= hi or s[i + k] notin {'0' .. '9', 'a' .. 'f', 'A' .. 'F'}:
          raise err("bad \\" & s[i] & " escape in a string literal")
        code = code * 16 + digitVal(s[i + k])
        inc k
      inc i, digits
      result.add(Rune(code))
    else:
      result.add s[i]
    inc i

func lexNumber(s: openArray[char], i: var int, hi: int): ExTok =
  let start = i
  var isFloat = false
  while i < hi and s[i].isDigit():
    inc i
  if i + 1 < hi and s[i] == '.' and s[i + 1].isDigit():
    isFloat = true
    inc i
    while i < hi and s[i].isDigit():
      inc i
  if i < hi and s[i] in {'e', 'E'}:
    var j = i + 1
    if j < hi and s[j] in {'+', '-'}:
      inc j
    if j < hi and s[j].isDigit():
      while j < hi and s[j].isDigit():
        inc j
      isFloat = true
      i = j
  var text = ""
  for k in start ..< i:
    text.add s[k]
  if isFloat:
    ExTok(kind: exFloat, lo: start, hi: i, f: parseFloat(text))
  else:
    ExTok(kind: exInt, lo: start, hi: i, i: parseBiggestInt(text).int64)

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
    raise err("unterminated string literal at byte " & $start)
  let endBody = i
  inc i
  ExTok(kind: exStr, lo: start, hi: i, s: decodeEscapes(s, body, endBody))

func punctAt(s: openArray[char], i, hi: int): (string, int) =
  ## Returns the longest punctuator matching at `i` and its byte length. `//`, `**`, `<=`, `>=`,
  ## `==` and `!=` are lexed so they are reported as rejected constructs rather than as two tokens.
  if i + 1 < hi:
    case s[i]
    of '=':
      if s[i + 1] == '=': return ("==", 2)
    of '!':
      if s[i + 1] == '=': return ("!=", 2)
      raise err("unexpected `!` in an expression at byte " & $i)
    of '<':
      if s[i + 1] == '=': return ("<=", 2)
    of '>':
      if s[i + 1] == '=': return (">=", 2)
    of '/':
      if s[i + 1] == '/': return ("//", 2)
    of '*':
      if s[i + 1] == '*': return ("**", 2)
    else:
      discard
  case s[i]
  of '+', '-', '*', '/', '%', '~', '(', ')', '[', ']', '{', '}', ',', ':', '.', '|', '<', '>':
    ($s[i], 1)
  of '=':
    ("=", 1)
  else:
    raise err("unexpected character '" & s[i] & "' in an expression at byte " & $i)

func advance(m: Machine, cx: var Cx) =
  ## Loads the next token into the cursor, stopping at the cursor's own `stop`.
  let s = m.jinja
  var i = cx.pos
  while i < cx.stop and s[i] in wsChars:
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
      while i < cx.stop and (s[i].isAlphaNumeric() or s[i] == '_'):
        inc i
      ExTok(kind: exName, lo: start, hi: i)
    else:
      let (text, len) = punctAt(s, i, cx.stop)
      inc i, len
      ExTok(kind: exPunct, lo: i - len, hi: i, s: text)
  cx.pos = i

func isPunct(cx: Cx, p: string): bool =
  cx.tok.kind == exPunct and cx.tok.s == p

func isWord(m: Machine, cx: Cx, w: string): bool =
  ## Reports whether the lookahead is the bare identifier `w`, so keywords are matched without
  ## copying text out of the template.
  if cx.tok.kind != exName or cx.tok.hi - cx.tok.lo != w.len:
    return false
  for k in 0 ..< w.len:
    if m.jinja[cx.tok.lo + k] != w[k]:
      return false
  true

func wordSpan(m: Machine, lo, hi: int): openArray[char] =
  ## Returns the template text in `lo ..< hi` as a view, so neither an identifier nor a registry
  ## lookup allocates. `lo ..< hi` is the half-open span the token carries.
  m.jinja.toOpenArray(lo, hi - 1)

func wordSpan(m: Machine, cx: Cx): openArray[char] =
  ## Returns the identifier of the lookahead token as a view into the template text.
  wordSpan(m, cx.tok.lo, cx.tok.hi)

func nameText(m: Machine, lo, hi: int): string =
  ## Returns the template text in `lo ..< hi` as a fresh string.
  ## Only an error message or a gap report copies, because the render path compares spans in place.
  result = newString(hi - lo)
  for i in lo ..< hi:
    result[i - lo] = m.jinja[i]

func findIn[N: enum](names: array[N, string], n: openArray[char]): int =
  ## Returns the index of `n` in a registry's name column, or -1 when the name is unknown.
  ## Each entry is compared against the caller's bytes without building a string.
  for k, v in names:
    if v == n:
      return k.ord
  -1

const
  argKeywordNames: array[akChars .. akSeparators, string] = [
    "chars", "default", "ensure_ascii", "separators"]
    ## Keyword names the builtins read, indexed by `ArgKeyword`.

func argName*(m: Machine, a: Arg): openArray[char] =
  ## Returns an argument's keyword name as a view into the template text.
  ## `nameLo == noLink` marks a positional argument, which has no name to read.
  m.jinja.toOpenArray(a.nameLo.int, a.nameHi.int - 1)

func argKey(m: Machine, a: Arg): string =
  ## Returns the dict key one argument supplies to `namespace` or `dict`.
  ## A keyword-bound argument gives the keyword text, a positional one gives its stringified value.
  ## A key is stored in a `DictVal`, so this is where a keyword name becomes a string.
  if a.nameLo == noLink:
    pyStr(a.val)
  else:
    var k: string
    k.add argName(m, a)
    k

func argKeyword(name: openArray[char]): ArgKeyword =
  ## Returns the builtin keyword `name` selects, `akNone` when no builtin reads that keyword.
  for k in akChars .. akSeparators:
    if name == argKeywordNames[k]:
      return k
  akNone

# Argument helpers
# ---------------------------------------------------------------------------

func getArg(args: seq[Arg], pos: int, kw: ArgKeyword, default: Value): Value =
  ## Returns the argument bound under `kw`, else the positional slot `pos`, else `default`.
  if kw != akNone:
    for a in args:
      if a.kw == kw:
        return a.val
  if pos < args.len and args[pos].nameLo == noLink:
    return args[pos].val
  default

func toRunes(s: string): seq[Rune] =
  var r: seq[Rune]
  for x in s.runes:
    r.add x
  r

func runeCount(s: string): int =
  ## Returns the codepoint length, which is what `length` and slicing count.
  var n = 0
  for _ in s.runes:
    inc n
  n

# Registries
# ---------------------------------------------------------------------------
#
# Filters, tests, methods and globals dispatch by name. `tojson` is one filter name exactly like
# `trim`, never a construct. A nil entry is a declared name no template in the corpus uses,
# and reaching it raises `NotImplementedError` rather than answering wrongly.

type
  FilterName = enum
    fTojson, fLength, fTrim, fDefault, fJoin, fLower, fUpper, fCapitalize, fList, fSafe, fDictsort,
    fMap, fSelect, fReject, fReplace, fIndent, fTruncate, fReverse, fWordcount, fSum, fMin, fMax,
    fAbs, fRound, fBatch, fSlice, fUnique, fGroupby, fAttr, fCenter, fEscape, fTitle
  TestName = enum
    tString, tDefined, tUndefined, tMapping, tSequence, tIterable, tNone, tBoolean, tTrue, tFalse,
    tNumber, tInteger, tFloat, tFilter, tTest, tSameas, tIn, tEqualTo, tDivisibleby, tEscaped,
    tEven, tOdd, tLower, tUpper, tCallable
  MethodName = enum
    mGet, mItems, mKeys, mValues, mSplit, mStrip, mLstrip, mRstrip, mStartswith, mEndswith, mLower,
    mUpper, mTitle, mReplace, mFind, mCount, mFormat, mPop, mUpdate
  GlobalName = enum
    gNamespace, gRange, gStrftimeNow, gRaiseException, gDict, gLipsum, gCycler, gJoiner

const
  filterNames: array[FilterName, string] = [
    "tojson", "length", "trim", "default", "join", "lower", "upper", "capitalize", "list", "safe",
    "dictsort", "map", "select", "reject", "replace", "indent", "truncate", "reverse", "wordcount",
    "sum", "min", "max", "abs", "round", "batch", "slice", "unique", "groupby", "attr", "center",
    "escape", "title"
  ]
  testNames: array[TestName, string] = [
    "string", "defined", "undefined", "mapping", "sequence", "iterable", "none", "boolean", "true",
    "false", "number", "integer", "float", "filter", "test", "sameas", "in", "equalto",
    "divisibleby", "escaped", "even", "odd", "lower", "upper", "callable"
  ]
  methodNames: array[MethodName, string] = [
    "get", "items", "keys", "values", "split", "strip", "lstrip", "rstrip", "startswith",
    "endswith", "lower", "upper", "title", "replace", "find", "count", "format", "pop", "update"
  ]
  globalNames: array[GlobalName, string] = [
    "namespace", "range", "strftime_now", "raise_exception", "dict", "lipsum", "cycler", "joiner"
  ]

func gapWhat(what: string, name: openArray[char]): void {.noreturn.} =
  ## Reports a declared registry name that no template in the corpus uses, so a gap is never
  ## mistaken for a wrong answer.
  ## Only this report quotes `name`, so the span is copied here and nowhere else.
  var quoted = ""
  quoted.add name
  raise newImplementError(what & " `" & quoted & "` is not implemented; no template in the corpus uses it")

proc tojsonFilter(v: Value, args: seq[Arg]): Value =
  ## Renders JSON. `ensure_ascii` and `separators` are the only kwargs the corpus passes.
  var opts = JsonOpts()
  for a in args:
    case a.kw
    of akEnsureAscii:
      if a.val.kind == vkBool:
        opts.ensureAscii = a.val.b
    of akSeparators:
      if a.val.kind == vkSeq and a.val.xs.items.len == 2:
        opts.itemSep = a.val.xs.items[0].s
        opts.kvSep = a.val.xs.items[1].s
    of akNone, akChars, akDefault:
      # Any argument outside the two keywords above is a gap, positional ones included, as before.
      # A filter has no access to the template text, so the report names a keyword span by its bounds.
      gapWhat("tojson kwarg", $a.nameLo)
  strVal(toJson(v, opts))

proc lengthFilter(v: Value, args: seq[Arg]): Value =
  case v.kind
  of vkStr: intVal(runeCount(v.s))
  of vkSeq: intVal(v.xs.items.len)
  of vkDict, vkNs: intVal(v.d.keys.len)
  else: raise err("`length` needs a string, sequence or mapping")

proc trimFilter(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`trim` needs a string")
  var chars = ""
  let a = getArg(args, 0, akChars, undefinedVal())
  if a.kind == vkStr:
    chars = a.s
  strVal(pyStrip(v.s, chars, true, true))

proc defaultFilter(v: Value, args: seq[Arg]): Value =
  if v.kind == vkUndefined: getArg(args, 0, akDefault, noneVal()) else: v

proc lowerFilter(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`lower` needs a string")
  strVal(v.s.toLowerAscii())

proc upperFilter(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`upper` needs a string")
  strVal(v.s.toUpperAscii())

proc capitalizeFilter(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`capitalize` needs a string")
  var acc = v.s.toLowerAscii()
  if acc.len > 0 and acc[0] in 'a' .. 'z':
    acc[0] = chr(ord(acc[0]) - 32)
  strVal(acc)

proc listFilter(v: Value, args: seq[Arg]): Value =
  case v.kind
  of vkSeq: v
  of vkStr:
    var acc = newSeq[Value]()
    for r in v.s.runes:
      acc.add strVal($r)
    seqVal(acc)
  else: raise err("`list` needs a string or sequence")

proc safeFilter(v: Value, args: seq[Arg]): Value =
  ## Autoescape is off in the upstream environment, so marking output safe changes no bytes.
  v

proc joinMethod(v: Value, args: seq[Arg]): Value =
  ## `x | join(sep)` and `x.join(sep)`:
  ##   concatenates the values of a sequence or mapping.
  let parts =
    case v.kind
    of vkSeq: v.xs.items
    of vkDict, vkNs: v.d.vals
    else: raise err("`join` needs a sequence")
  let sep = pyStr(getArg(args, 0, akNone, strVal("")))
  var acc = ""
  for i, x in parts:
    if i > 0:
      acc.add sep
    acc.add pyStr(x)
  strVal(acc)

proc joinFilter(v: Value, args: seq[Arg]): Value = joinMethod(v, args)

proc stringTest(v: Value, args: seq[Arg]): bool = v.kind == vkStr
proc definedTest(v: Value, args: seq[Arg]): bool = v.kind != vkUndefined
proc undefinedTest(v: Value, args: seq[Arg]): bool = v.kind == vkUndefined
proc mappingTest(v: Value, args: seq[Arg]): bool = v.kind in {vkDict, vkNs}
proc sequenceTest(v: Value, args: seq[Arg]): bool = v.kind == vkSeq
proc iterableTest(v: Value, args: seq[Arg]): bool = v.kind in {vkSeq, vkDict, vkNs, vkStr}
proc noneTest(v: Value, args: seq[Arg]): bool = v.kind == vkNone
proc booleanTest(v: Value, args: seq[Arg]): bool = v.kind == vkBool
proc trueTest(v: Value, args: seq[Arg]): bool = v.kind == vkBool and v.b
proc falseTest(v: Value, args: seq[Arg]): bool = v.kind == vkBool and not v.b
proc numberTest(v: Value, args: seq[Arg]): bool = v.kind in {vkInt, vkFloat}
proc integerTest(v: Value, args: seq[Arg]): bool = v.kind == vkInt
proc floatTest(v: Value, args: seq[Arg]): bool = v.kind == vkFloat

func loopAttr(v: Value, name: openArray[char]): Value =
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

func sliceIndices(n: int, lo, hi, step: Value, hasLo, hasHi, hasStep: bool):
    tuple[start, stop, by: int] =
  ## Returns Python's `slice.indices(n)` for one slice:
  ##   the walk bounds and the stride, plus the direction-dependent defaults and clamps applied.
  ##
  ## Defaults follow the stride's direction rather than sitting at the ends of the range, which is
  ## what makes `x[::-1]` visit every element and `x[:2:-1]` stop at the head.
  ##
  ## - forward (`by > 0`):
  ##   bounds clamp into `[0, n]`, defaults are `0` and `n`
  ## - backward (`by < 0`):
  ##   bounds clamp into `[-1, n - 1]`, defaults are `n - 1` and `-1`
  ##
  ## A backward stop of `-1` means "one past the head", so the walk includes index 0.
  var by = 1
  if hasStep:
    if step.kind != vkInt:
      raise err("slice step needs an integer")
    by = step.i.int
    if by == 0:
      raise err("slice step must not be zero")
  if (hasLo and lo.kind != vkInt) or (hasHi and hi.kind != vkInt):
    raise err("slice bounds need integers")
  let low = if by > 0: 0 else: -1
  let high = if by > 0: n else: n - 1
  var a = if hasLo: lo.i.int else: (if by > 0: low else: high)
  var b = if hasHi: hi.i.int else: (if by > 0: high else: low)
  if hasLo and a < 0:
    a += n
  if hasHi and b < 0:
    b += n
  (clamp(a, low, high), clamp(b, low, high), by)

func subslice(v, lo, hi, step: Value, hasLo, hasHi, hasStep, isSlice: bool): Value =
  ## Returns a subscript or a slice. `x[1:]` and `x[::-1]` are the slice shapes the corpus uses.
  if not isSlice:
    return case v.kind
    of vkSeq:
      if lo.kind != vkInt:
        raise err("sequence subscript needs an integer")
      let n = v.xs.items.len
      let idx = if lo.i < 0: n + lo.i.int else: lo.i.int
      if idx < 0 or idx >= n:
        raise err("subscript " & $lo.i & " is acc of range for a length-" & $n & " sequence")
      v.xs.items[idx]
    of vkDict, vkNs:
      v.d.dictGet(pyStr(lo))
    of vkStr:
      if lo.kind != vkInt:
        raise err("string subscript needs an integer")
      let runes = toRunes(v.s)
      let idx = if lo.i < 0: runes.len + lo.i.int else: lo.i.int
      if idx < 0 or idx >= runes.len:
        raise err("string subscript " & $lo.i & " is acc of range")
      strVal($runes[idx])
    of vkUndefined:
      undefinedVal()
    else:
      raise err("a " & $v.kind & " is not subscriptable")
  let n =
    case v.kind
    of vkSeq: v.xs.items.len
    of vkStr: runeCount(v.s)
    else: raise err("a " & $v.kind & " is not sliceable")
  let (a, b, by) = sliceIndices(n, lo, hi, step, hasLo, hasHi, hasStep)
  return case v.kind
  of vkSeq:
    var acc = newSeq[Value]()
    var k = a
    while (by > 0 and k < b) or (by < 0 and k > b):
      acc.add v.xs.items[k]
      inc k, by
    seqVal(acc)
  else:
    let runes = toRunes(v.s)
    var acc = ""
    var k = a
    while (by > 0 and k < b) or (by < 0 and k > b):
      acc.add $runes[k]
      inc k, by
    strVal(acc)

proc getMethod(v: Value, args: seq[Arg]): Value =
  ## `d.get(key, default)`. Absence yields the default, itself undefined when unsupplied.
  if v.kind notin {vkDict, vkNs}:
    raise err("`get` needs a mapping")
  let got = v.d.dictGet(pyStr(getArg(args, 0, akNone, undefinedVal())))
  if got.kind == vkUndefined: getArg(args, 1, akDefault, undefinedVal()) else: got

proc itemsMethod(v: Value, args: seq[Arg]): Value =
  ## `[key, value]` pairs in insertion order, the form `{% for k, v in x.items() %}` iterates.
  case v.kind
  of vkDict, vkNs:
    var acc = newSeq[Value](v.d.keys.len)
    for i, k in v.d.keys:
      acc[i] = seqVal(@[strVal(k), v.d.vals[i]])
    seqVal(acc)
  of vkSeq:
    var acc = newSeq[Value](v.xs.items.len)
    for i, x in v.xs.items:
      acc[i] = seqVal(@[intVal(i), x])
    seqVal(acc)
  else:
    raise err("`items` needs a mapping or a sequence")

proc keysMethod(v: Value, args: seq[Arg]): Value =
  if v.kind notin {vkDict, vkNs}:
    raise err("`keys` needs a mapping")
  var acc = newSeq[Value](v.d.keys.len)
  for i, k in v.d.keys:
    acc[i] = strVal(k)
  seqVal(acc)

proc valuesMethod(v: Value, args: seq[Arg]): Value =
  if v.kind notin {vkDict, vkNs}:
    raise err("`values` needs a mapping")
  seqVal(v.d.vals)

proc splitMethod(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`split` needs a string")
  let sep = pyStr(getArg(args, 0, akNone, strVal(" ")))
  var acc = newSeq[Value]()
  if sep.len == 0:
    for r in v.s.runes:
      acc.add strVal($r)
  else:
    for piece in v.s.split(sep):
      acc.add strVal(piece)
  seqVal(acc)

proc stripMethod(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`strip` needs a string")
  strVal(pyStrip(v.s, pyStr(getArg(args, 0, akNone, strVal(""))), true, true))

proc lstripMethod(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`lstrip` needs a string")
  strVal(pyStrip(v.s, pyStr(getArg(args, 0, akNone, strVal(""))), true, false))

proc rstripMethod(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`rstrip` needs a string")
  strVal(pyStrip(v.s, pyStr(getArg(args, 0, akNone, strVal(""))), false, true))

proc startswithMethod(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`startswith` needs a string")
  boolVal(v.s.startsWith(pyStr(getArg(args, 0, akNone, strVal("")))))

proc endswithMethod(v: Value, args: seq[Arg]): Value =
  if v.kind != vkStr:
    raise err("`endswith` needs a string")
  boolVal(v.s.endsWith(pyStr(getArg(args, 0, akNone, strVal("")))))

proc namespaceGlobal(m: Machine, args: seq[Arg], d: var Driver): Value =
  ## `namespace(field=init, ...)`:
  ##   the mutable mapping `{% set ns.field = ... %}` mutates in place.
  var dv = DictVal()
  for a in args:
    dv.dictSet(argKey(m, a), a.val)
  nsVal(dv)

proc dictGlobal(m: Machine, args: seq[Arg], d: var Driver): Value =
  var dv = DictVal()
  for a in args:
    dv.dictSet(argKey(m, a), a.val)
  dictVal(dv)

proc rangeGlobal(m: Machine, args: seq[Arg], d: var Driver): Value =
  var a = 0'i64
  var b = 0'i64
  var step = 1'i64
  for i, x in args:
    if x.val.kind != vkInt:
      raise err("`range` needs integer bounds")
    case i
    of 0: b = x.val.i
    of 1:
      a = b
      b = x.val.i
    of 2: step = x.val.i
    else: raise err("`range` takes at most three arguments")
  if step == 0:
    raise err("`range` step must not be zero")
  var acc = newSeq[Value]()
  var k = a
  while (step > 0 and k < b) or (step < 0 and k > b):
    acc.add intVal(k)
    inc k, step
  seqVal(acc)

proc strftimeGlobal(m: Machine, args: seq[Arg], d: var Driver): Value =
  ## Renders the format against the driver's injected epoch. It never reads the wall clock, which is
  ## what keeps two drivers over one `Machine` byte-identical.
  let fmt = pyStr(getArg(args, 0, akNone, strVal("")))
  let dt = initTime(d.clock.int64, 0).inZone(utc())
  var acc = ""
  var i = 0
  while i < fmt.len:
    if fmt[i] != '%':
      acc.add fmt[i]
      inc i
      continue
    if i + 1 >= fmt.len:
      raise err("`strftime_now` format ends on a `%`")
    case fmt[i + 1]
    of 'Y': acc.add dt.format("yyyy")
    of 'm': acc.add dt.format("MM")
    of 'd': acc.add dt.format("dd")
    of 'H': acc.add dt.format("HH")
    of 'M': acc.add dt.format("mm")
    of 'S': acc.add dt.format("ss")
    of 'j':
      # Day of year, 1-based:
      #   days in the completed months of this year, plus the monthday.
      var doy = dt.monthday.int
      for mo in 1 ..< ord(dt.month):
        doy += getDaysInMonth(Month(mo), dt.year)
      acc.add $doy
    of '%': acc.add '%'
    else: gapWhat("`strftime_now` directive", "%" & fmt[i + 1])
    inc i, 2
  strVal(acc)

proc raiseExceptionGlobal(m: Machine, args: seq[Arg], d: var Driver): Value =
  ## Corpus `err_*` rows record exactly this raise. The class is `TemplateError` and the message passes through verbatim.
  raise err(pyStr(getArg(args, 0, akNone, strVal(""))))

const
  filterProcs: array[FilterName, FilterProc] = [
    tojsonFilter, lengthFilter, trimFilter, defaultFilter, joinFilter, lowerFilter, upperFilter,
    capitalizeFilter, listFilter, safeFilter, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
    nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil
  ]
  testProcs: array[TestName, TestProc] = [
    stringTest, definedTest, undefinedTest, mappingTest, sequenceTest, iterableTest, noneTest,
    booleanTest, trueTest, falseTest, numberTest, integerTest, floatTest, nil, nil, nil, nil, nil,
    nil, nil, nil, nil, nil, nil, nil
  ]
  methodProcs: array[MethodName, MethodProc] = [
    getMethod, itemsMethod, keysMethod, valuesMethod, splitMethod, stripMethod, lstripMethod,
    rstripMethod, startswithMethod, endswithMethod, nil, nil, nil, nil, nil, nil, nil, nil, nil
  ]
  globalProcs: array[GlobalName, GlobalProc] = [
    namespaceGlobal, rangeGlobal, strftimeGlobal, raiseExceptionGlobal, dictGlobal, nil, nil, nil
  ]

# Lookup
# ---------------------------------------------------------------------------

func lookupName(t: Tables, d: var Driver, name: openArray[char]): Value =
  ## Returns the binding of `name`, undefined when absent. Absence is a value, never an error:
  ## that is what `is defined` tests and what makes a missing dict key behave like a missing name.
  ## Interned-name scan and context lookup both read the caller's bytes in place.
  let id = findName(t, name)
  if id != noLink:
    for si in countdown(d.scopes.len - 1, 0):
      for b in d.scopes[si]:
        if b.name == id:
          return b.val
  if d.root.kind == vkDict:
    return d.root.d.dictGet(name)
  undefinedVal()

func lookupNameById*(t: Tables, d: var Driver, id: int32): Value =
  ## Returns the binding of an interned name, undefined when absent. The scope key is the id, so no
  ## string is rebuilt per lookup.
  if id == noLink:
    return undefinedVal()
  for si in countdown(d.scopes.len - 1, 0):
    for b in d.scopes[si]:
      if b.name == id:
        return b.val
  if d.root.kind == vkDict and id < t.names.len.int32:
    return d.root.d.dictGet(t.names[id])
  undefinedVal()

# The walker
# ---------------------------------------------------------------------------

proc evalRange(m: Machine, t: Tables, d: var Driver, lo, hi: int, depth = 0,
  run: MacroRunner = nil): Value
proc expr(m: Machine, t: Tables, d: var Driver, cx: var Cx, minPrec: int): Value

func binPrec(op: string): int =
  ## Returns the left binding power of an infix operator, 0 when `op` is not infix.
  ## Jinja orders operators ternary-lowest, then `or`, `and`, comparison and tests, `~`,
  ## `+ -`, `* / // %`, `**`.
  case op
  of "if": 1
  of "or": 2
  of "and": 3
  of "==", "!=", "<", ">=", "<=", ">", "in", "not in": 5
  of "~": 6
  of "+", "-": 7
  of "*", "/", "//", "%": 8
  of "**": 9
  else: 0

proc skipExpr(m: Machine, t: Tables, d: var Driver, cx: var Cx, minPrec: int) =
  ## Advances the cursor over an expression without evaluating it, which is how `and`, `or`
  ## and the ternary skip the text they do not run.
  let wasDry = cx.dry
  cx.dry = true
  discard expr(m, t, d, cx, minPrec)
  cx.dry = wasDry

proc argList(m: Machine, t: Tables, d: var Driver, cx: var Cx): seq[Arg] =
  ## Parses a parenthesised argument list with `name = expr` keyword arguments. The opening paren is the lookahead.
  ## A keyword keeps its span and gains a builtin keyword slot, so binding one costs no string.
  advance(m, cx)
  while not isPunct(cx, ")"):
    var nameLo = noLink
    var nameHi = noLink
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
    let v = expr(m, t, d, cx, 1)
    result.add Arg(nameLo: nameLo, nameHi: nameHi, kw: kw, val: v)
    if isPunct(cx, ","):
      advance(m, cx)
      if isPunct(cx, ")"):
        break
      continue
    break
  if not isPunct(cx, ")"):
    raise err("argument list is not closed")
  advance(m, cx)

proc postfix(m: Machine, t: Tables, d: var Driver, cx: var Cx, v: Value): Value =
  ## Applies attr, subscript, call, filter and test chains, which bind tighter than any operator.
  var v = v
  while true:
    if isPunct(cx, "."):
      advance(m, cx)
      if cx.tok.kind != exName:
        raise err("expected a name after `.`")
      let (lo, hi) = (cx.tok.lo, cx.tok.hi)
      advance(m, cx)
      if isPunct(cx, "("):
        let a = argList(m, t, d, cx)
        if cx.dry:
          v = undefinedVal()
          continue
        let mi = findIn(methodNames, wordSpan(m, lo, hi))
        if mi < 0:
          raise err("unknown method `" & nameText(m, lo, hi) & "`")
        let mp = methodProcs[MethodName mi]
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
            else: raise err("`" & nameText(m, lo, hi) &
                "` is not an attribute of a " & $v.kind)
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
        raise err("subscript is not closed")
      advance(m, cx)
      v = if cx.dry: undefinedVal() else:
        subslice(v, lo, hi, step, hasLo, hasHi, hasStep, isSlice)
    elif isPunct(cx, "("):
      let a = argList(m, t, d, cx)
      v =
        if cx.dry:
          undefinedVal()
        elif v.kind == vkMacro:
          if cx.run.isNil:
            raise err("a macro was called where no macro body runner was supplied")
          strVal(cx.run(m, t, d, v.mc, a))
        else:
          raise err("only a macro is callable, this is a " & $v.kind)
    elif isPunct(cx, "|"):
      advance(m, cx)
      if cx.tok.kind != exName:
        raise err("expected a filter name after `|`")
      let (lo, hi) = (cx.tok.lo, cx.tok.hi)
      advance(m, cx)
      var a = newSeq[Arg]()
      if isPunct(cx, "("):
        a = argList(m, t, d, cx)
      if cx.dry:
        v = undefinedVal()
        continue
      let fi = findIn(filterNames, wordSpan(m, lo, hi))
      if fi < 0:
        raise err("unknown filter `" & nameText(m, lo, hi) & "`")
      let fp = filterProcs[FilterName fi]
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
        raise err("expected a test name after `is`")
      let (lo, hi) = (cx.tok.lo, cx.tok.hi)
      advance(m, cx)
      var a = newSeq[Arg]()
      if isPunct(cx, "("):
        a = argList(m, t, d, cx)
      if cx.dry:
        v = undefinedVal()
        continue
      let ti = findIn(testNames, wordSpan(m, lo, hi))
      if ti < 0:
        raise err("unknown test `" & nameText(m, lo, hi) & "`")
      let tp = testProcs[TestName ti]
      if tp.isNil:
        gapWhat("test", wordSpan(m, lo, hi))
      v = boolVal(if negated: not tp(v, a) else: tp(v, a))
    else:
      break
  v

proc primary(m: Machine, t: Tables, d: var Driver, cx: var Cx): Value =
  ## Parses a literal, a name, a parenthesised group, an array literal or a dict literal, then
  ## the postfix chain.
  var v: Value
  case cx.tok.kind
  of exEof:
    raise err("expression ends early at byte " & $cx.pos)
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
    # The literal spellings are compared as spans, the same test `case` applied to a copied string.
    if name == "true" or name == "True":
      v = boolVal(true)
    elif name == "false" or name == "False":
      v = boolVal(false)
    elif name == "none" or name == "None":
      v = noneVal()
    else:
      var bound = if cx.dry: undefinedVal() else: lookupName(t, d, name)
      let gi = findIn(globalNames, name)
      if bound.kind == vkUndefined and gi >= 0 and isPunct(cx, "("):
        # A global is reached only when the name is unbound, which is Jinja's own precedence:
        # context shadows globals, and a skipped branch never gets here. A dry walk still consumes
        # the argument list, or the skipped text is left behind as trailing text.
        let a = argList(m, t, d, cx)
        if cx.dry:
          v = undefinedVal()
        else:
          let gp = globalProcs[GlobalName gi]
          if gp.isNil:
            gapWhat("global", wordSpan(m, lo, hi))
          v = gp(m, a, d)
      else:
        v = bound
  of exPunct:
    case cx.tok.s
    of "(":
      advance(m, cx)
      var parts = newSeq[Value]()
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
        raise err("parenthesised expression is not closed")
      advance(m, cx)
      v = if parts.len == 0: seqVal(parts) elif isTuple: seqVal(parts) else: parts[0]
    of "[":
      advance(m, cx)
      var parts = newSeq[Value]()
      while not isPunct(cx, "]"):
        parts.add expr(m, t, d, cx, 1)
        if isPunct(cx, ","):
          advance(m, cx)
          if isPunct(cx, "]"):
            break
          continue
        break
      if not isPunct(cx, "]"):
        raise err("array literal is not closed")
      advance(m, cx)
      v = seqVal(parts)
    of "{":
      advance(m, cx)
      var dv = DictVal()
      while not isPunct(cx, "}"):
        let k = expr(m, t, d, cx, 1)
        if not isPunct(cx, ":"):
          raise err("dict literal entry needs a `:`")
        advance(m, cx)
        let val = expr(m, t, d, cx, 1)
        if not cx.dry:
          dv.dictSet(pyStr(k), val)
        if isPunct(cx, ","):
          advance(m, cx)
          if isPunct(cx, "}"):
            break
          continue
        break
      if not isPunct(cx, "}"):
        raise err("dict literal is not closed")
      advance(m, cx)
      v = dictVal(dv)
    else:
      raise err("unexpected `" & cx.tok.s & "` starting an expression at byte " & $cx.tok.lo)
  postfix(m, t, d, cx, v)

proc unary(m: Machine, t: Tables, d: var Driver, cx: var Cx): Value =
  ## Parses `not`, unary `-` and `+`, then a primary.
  if isWord(m, cx, "not"):
    advance(m, cx)
    let v = unary(m, t, d, cx)
    return boolVal(if cx.dry: false else: not isTruthy(v))
  if isPunct(cx, "-") or isPunct(cx, "+"):
    let neg = isPunct(cx, "-")
    advance(m, cx)
    let v = unary(m, t, d, cx)
    if cx.dry:
      return undefinedVal()
    case v.kind
    of vkInt: intVal(if neg: -v.i else: v.i)
    of vkFloat: floatVal(if neg: -v.f else: v.f)
    else: raise err("arithmetic needs a number, this is a " & $v.kind)
  else:
    primary(m, t, d, cx)

func arith(op: string, a, b: Value): Value =
  ## Combines two numbers, or two strings and two sequences under `+`. `%` follows Python's floor rule,
  ## where the result takes the sign of the divisor, so `-3 % 2` is `1`.
  case op
  of "+":
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
      raise err("`+` cannot combine a " & $a.kind & " with a " & $b.kind)
  of "-":
    if a.kind == vkInt and b.kind == vkInt:
      intVal(a.i - b.i)
    elif a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
      floatVal((if a.kind == vkInt: float64 a.i else: a.f) -
          (if b.kind == vkInt: float64 b.i else: b.f))
    else:
      raise err("`-` needs numbers")
  of "%":
    if a.kind == vkInt and b.kind == vkInt:
      if b.i == 0:
        raise err("`%` needs a non-zero divisor")
      var r = a.i mod b.i
      if r != 0 and ((r < 0) != (b.i < 0)):
        r += b.i
      intVal(r)
    elif a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
      let x = if a.kind == vkInt: float64 a.i else: a.f
      let y = if b.kind == vkInt: float64 b.i else: b.f
      if y == 0:
        raise err("`%` needs a non-zero divisor")
      floatVal(math.floorMod(x, y))
    else:
      raise err("`%` needs numbers")
  else:
    raise err("unknown arithmetic `" & op & "`")

func cmpOne(op: string, a, b: Value): Value =
  let r =
    case op
    of "==": eqVal(a, b)
    of "!=": not eqVal(a, b)
    else:
      let c = cmpVal(a, b)
      case op
      of "<": c < 0
      of ">": c > 0
      of "<=": c <= 0
      of ">=": c >= 0
      else: raise err("unknown comparison `" & op & "`")
  boolVal(r)

proc binOp(m: Machine, t: Tables, d: var Driver, cx: var Cx, lhs: Value, op: string): Value =
  ## Evaluates the right operand of `op` and combines it with `lhs`. `and` and `or` skip the operand
  ## they do not evaluate. Every other infix evaluates both sides.
  case op
  of "and":
    if not cx.dry and not isTruthy(lhs):
      skipExpr(m, t, d, cx, 4)
      return lhs
    expr(m, t, d, cx, 4)
  of "or":
    if not cx.dry and isTruthy(lhs):
      skipExpr(m, t, d, cx, 3)
      return lhs
    expr(m, t, d, cx, 3)
  of "in", "not in":
    let rhs = expr(m, t, d, cx, 6)
    if cx.dry:
      undefinedVal()
    else:
      let r = containsVal(rhs, lhs)
      boolVal(if op == "in": r else: not r)
  of "~":
    let rhs = expr(m, t, d, cx, 7)
    if cx.dry: undefinedVal() else: strVal(pyStr(lhs) & pyStr(rhs))
  of "+", "-", "%":
    let rhs = expr(m, t, d, cx, binPrec(op) + 1)
    if cx.dry: undefinedVal() else: arith(op, lhs, rhs)
  of "*", "/", "//", "**":
    skipExpr(m, t, d, cx, binPrec(op) + 1)
    gapWhat("operator", op)
  else:
    let rhs = expr(m, t, d, cx, binPrec(op) + 1)
    if cx.dry: undefinedVal() else: cmpOne(op, lhs, rhs)

func ifWordAhead(m: Machine, at, stop: int): bool =
  ## Reports whether a depth-zero `if` survives in `m.jinja[at..<stop)`. Byte scan, not a parse.
  ## Guarantees:
  ## - quoted text and bracketed subexpressions are skipped, so the scan never misses a ternary
  ## - it can only over-report, which costs one extra walk rather than a wrong answer
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

type Ternary = object
  ## Spans of `A if C else B`, measured by a dry walk before any of them runs.
  aHi, cLo, cHi, bLo, bHi: int
  hasElse: bool
  isTernary: bool

proc scanTernary(m: Machine, t: Tables, d: var Driver, cx: var Cx, headLo: int): Ternary =
  ## Measures the ternary spans starting at `headLo` and leaves the cursor on the token past
  ## the whole ternary. When the walk does not land on a ternary the cursor is restored to the head,
  ## so the caller evaluates the expression normally. Nothing is evaluated here.
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

proc expr(m: Machine, t: Tables, d: var Driver, cx: var Cx, minPrec: int): Value =
  ## Parses and evaluates one expression, Pratt-style:
  ##   a prefix, then infix while the operator binds at least `minPrec`.
  ## Ternary handling:
  ## - a ternary binds loosest, and no other operator holds binding power 1, so the ternary is
  ##   resolved before its head runs
  ## - the condition is evaluated once and exactly one branch is, which keeps a branch holding
  ##   `raise_exception` or `strftime_now` from acting while unelected
  ## A stray `if` ends the expression, and `evalRange` reports the tail text.
  if not cx.dry:
    inc cx.depth
    if cx.depth > ExprDepthCap:
      raise err("expression nests deeper than ExprDepthCap = " & $ExprDepthCap)
  let headLo = cx.tok.lo
  if minPrec <= 1 and cx.tok.kind != exEof and ifWordAhead(m, headLo, cx.stop):
    let shape = scanTernary(m, t, d, cx, headLo)
    if shape.isTernary:
      if cx.dry:
        return undefinedVal()
      let cond = evalRange(m, t, d, shape.cLo, shape.cHi, cx.depth)
      if isTruthy(cond):
        return evalRange(m, t, d, headLo, shape.aHi, cx.depth)
      if shape.hasElse:
        return evalRange(m, t, d, shape.bLo, shape.bHi, cx.depth)
      return undefinedVal()
  var v = unary(m, t, d, cx)
  while true:
    var op = ""
    if cx.tok.kind == exName:
      if isWord(m, cx, "and"):
        op = "and"
      elif isWord(m, cx, "or"):
        op = "or"
      elif isWord(m, cx, "in"):
        op = "in"
      elif isWord(m, cx, "not"):
        let save = cx
        advance(m, cx)
        if isWord(m, cx, "in"):
          op = "not in"
        else:
          cx = save
          break
      else:
        break
    elif cx.tok.kind == exPunct:
      op = cx.tok.s
    let prec = binPrec(op)
    if prec == 0 or prec < minPrec:
      break
    advance(m, cx)
    v = binOp(m, t, d, cx, v, op)
  if not cx.dry:
    dec cx.depth
  v

proc evalRange(m: Machine, t: Tables, d: var Driver, lo, hi: int, depth = 0,
    run: MacroRunner = nil): Value =
  ## Evaluates the expression held in `m.jinja[lo..<hi]` in its own cursor. `depth` seeds the nesting
  ## counter so a sub-span reached through a ternary still counts toward `ExprDepthCap`, and `run`
  ## carries the macro runner so a nested call can still run.
  var cx = Cx(pos: lo, stop: hi, dry: false, depth: depth, run: run)
  advance(m, cx)
  result = expr(m, t, d, cx, 1)
  if cx.tok.kind != exEof:
    raise err("expression has trailing text at byte " & $cx.tok.lo)

proc evalSpan*(m: Machine, t: Tables, d: var Driver, lo, hi: int32,
    run: MacroRunner = nil): Value =
  ## Evaluates the expression held in `m.jinja[lo..<hi]`, the entry every expression-bearing step uses.
  ## Args:
  ## - `run` is the macro body runner. A step that can meet a macro call passes its own runner,
  ##   and a caller with no arena passes nil, which makes a macro call a reported gap.
  evalRange(m, t, d, lo.int, hi.int, 0, run)
