# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Builtin library of the expression language.
##
## - the corpus's Jinja filters, tests and methods as pure `(JinjaVal, args) -> JinjaVal` functions
## - name registries with the proc tables they index, dispatched by name in jinja_interpolation
## - a nil table entry raising `gapWhat`, the declared-gap contract
##
## Globals stay in jinja_interpolation.

import std/unicode
import cnj_types {.all.}
import jinja_data_model {.all.}
import jinja_serialize {.all.}

type
  FilterProc = proc (v: JinjaVal, args: var Args): JinjaVal {.nimcall, noSideEffect.}
  TestProc = proc (v: JinjaVal, args: var Args): bool {.nimcall, noSideEffect.}
  MethodProc = proc (v: JinjaVal, args: var Args): JinjaVal {.nimcall, noSideEffect.}
  FilterName* = enum
    fTojson, fLength, fTrim, fDefault, fJoin, fLower, fUpper, fCapitalize, fList, fSafe, fItems, fString, fDictsort,
    fMap
  TestName* = enum
    tString, tDefined, tUndefined, tMapping, tSequence, tIterable, tNone, tBoolean, tTrue, tFalse,

  MethodName* = enum
    mGet, mItems, mSplit, mStrip, mLstrip, mRstrip, mStartswith, mEndswith

const
  ArgKeywordNames: array[akChars .. akSeparators, string] = [
    "chars", "default", "ensure_ascii", "separators"]
    ## Keyword names the builtins read, indexed by `ArgKeyword`.

func findIn[N: enum](names: array[N, string], n: openArray[char]): int =
  ## Returns the index of `n` in a registry's name column, or -1 when the name is unknown,
  ## each entry compared against the caller's bytes without building a string.
  for k, v in names:
    if v == n:
      return k.ord
  -1
func gapWhat(what: string, name: openArray[char]): void {.noreturn.} =
  ## Reports a declared registry name that no template in the corpus uses.
  ## Only this report quotes `name`, so the span copies here alone.
  let quoted = spanString(name)
  raise jinjaErr(what & " `" & quoted & "` is not implemented; no template in the corpus uses it", cause = ceUnimplemented)

func argKeyword(name: openArray[char]): ArgKeyword =
  ## Returns the builtin keyword `name` selects, `akNone` when no builtin reads that keyword.
  for k in akChars .. akSeparators:
    if name == ArgKeywordNames[k]:
      return k
  akNone

# Argument helpers:

func getArg(args: var Args, pos: int, kw: ArgKeyword, default: JinjaVal): JinjaVal =
  ## Returns the argument bound under `kw`, else the `pos`-th positional in call order,
  ## else `default`. Positionals bind by their own count.
  if kw != akNone:
    for a in args.argItems:
      if a.kw == kw:
        return move a.val
  var seen = 0
  for a in args.argItems:
    if a.nameLo == NoLink:
      if seen == pos:
        return move a.val
      inc seen
  default

const
  FilterNames*: array[FilterName, string] = [
    "tojson", "length", "trim", "default", "join", "lower", "upper", "capitalize", "list", "safe",
    "items", "string", "dictsort", "map"
  ]
  TestNames*: array[TestName, string] = [
    "string", "defined", "undefined", "mapping", "sequence", "iterable", "none", "boolean", "true",
    "false"
  ]
  MethodNames*: array[MethodName, string] = [
    "get", "items", "split", "strip", "lstrip", "rstrip", "startswith", "endswith"
  ]

func tojsonFilter(v: JinjaVal, args: var Args): JinjaVal =
  ## Renders JSON, `ensure_ascii` and `separators` the kwargs the corpus passes,
  ## non-ASCII emitted as raw UTF-8 by default.
  var opts = JsonOpts()
  for a in args.argItems:
    case a.kw
    of akEnsureAscii:
      if a.val.kind == vkBool:
        opts.ensureAscii = a.val.b
    of akSeparators:
      if a.val.kind == vkSeq and a.val.xs.items.len == 2:
        opts.itemSep = pyStr(a.val.xs.items[0])
        opts.kvSep = pyStr(a.val.xs.items[1])
    of akNone, akChars, akDefault:
      # Any argument outside the two keywords above is a gap, positional ones included, as before.
      # A filter has no access to the template text, so the report names the keyword's start offset.
      gapWhat("tojson kwarg", $a.nameLo)
  strVal(toJson(v, opts))

func lengthFilter(v: JinjaVal, args: var Args): JinjaVal =
  case v.kind
  of vkStr: intVal(runeLen(v.s))
  of vkCut:
    # Python's `len` counts codepoints, so the span walks per rune like a string's.
    var n = 0
    var i = v.lo.int
    while i < v.hi.int:
      var r: Rune
      fastRuneAt(v.raw, i, r, true)
      inc n
    intVal(n)
  of vkSeq: intVal(v.xs.items.len)
  of vkDict, vkNs: intVal(v.d.keys.len)
  of vkRange: intVal(rangeLen(v.r))
  else: raise jinjaErr("`length` needs a string, sequence or mapping")

func trimFilter(v: JinjaVal, args: var Args): JinjaVal =
  let v = if v.kind == vkCut: materializeVal(v) else: v
  if v.kind != vkStr:
    raise jinjaErr("`trim` needs a string")
  let a = getArg(args, 0, akChars, undefinedVal())
  let chars = if a.kind == vkStr: a.s
              elif a.kind == vkCut: materializeVal(a).s
              else: ""
  let (lo, hi) = stripSpan(v.s, chars, true, true)
  cutVal(v.s, lo.int32, hi.int32)

func defaultFilter(v: JinjaVal, args: var Args): JinjaVal =
  if v.kind == vkUndefined: getArg(args, 0, akDefault, noneVal()) else: v

func asciiCased(s: openArray[char], upper: bool): string =
  ## Returns the bytes of `s` with ASCII letters cased per `upper`, other bytes verbatim,
  ## as one fresh string.
  result = newString(s.len)
  for i in 0 ..< s.len:
    let c = s[i]
    let flip = upper and c in {'a' .. 'z'} or not upper and c in {'A' .. 'Z'}
    result[i] = if flip: chr(ord(c) xor 0x20) else: c

func lowerFilter(v: JinjaVal, args: var Args): JinjaVal =
  let v = if v.kind == vkCut: materializeVal(v) else: v
  if v.kind != vkStr:
    raise jinjaErr("`lower` needs a string")
  strVal(asciiCased(v.s, false))

func upperFilter(v: JinjaVal, args: var Args): JinjaVal =
  let v = if v.kind == vkCut: materializeVal(v) else: v
  if v.kind != vkStr:
    raise jinjaErr("`upper` needs a string")
  strVal(asciiCased(v.s, true))

func capitalizeFilter(v: JinjaVal, args: var Args): JinjaVal =
  let v = if v.kind == vkCut: materializeVal(v) else: v
  if v.kind != vkStr:
    raise jinjaErr("`capitalize` needs a string")
  var acc = asciiCased(v.s, false)
  if acc.len > 0 and acc[0] in 'a' .. 'z':
    acc[0] = chr(ord(acc[0]) - 32)
  strVal(acc)

func listFilter(v: JinjaVal, args: var Args): JinjaVal =
  case v.kind
  of vkSeq: v
  of vkStr, vkCut:
    let v = if v.kind == vkCut: materializeVal(v) else: v
    seqVal(codepointVals(v.s))
  of vkRange:
    var acc = newSeq[JinjaVal](rangeLen(v.r))
    for i in 0 ..< rangeLen(v.r):
      acc[i] = rangeAt(v.r, i)
    seqVal(acc)
  else: raise jinjaErr("`list` needs a string, sequence or range")

func safeFilter(v: JinjaVal, args: var Args): JinjaVal =
  ## Autoescape is off in the upstream environment, so marking output safe changes no bytes.
  v

func stringFilter(v: JinjaVal, args: var Args): JinjaVal =
  ## Returns the value as template output text, the emit path's `pyStr` form
  ## carried into a value. Strings pass through unchanged, everything else takes
  ## its Python `str()` form, undefined rendering empty.
  strVal(pyStr(v))

func joinMethod(v: JinjaVal, args: var Args): JinjaVal =
  ## `x | join(sep)` and `x.join(sep)` concatenate a sequence's values, a mapping's keys.
  let sep = pyStr(getArg(args, 0, akNone, strVal("")))
  var acc = ""
  case v.kind
  of vkSeq:
    var first = true
    for x in v.xs.items:
      if first:
        first = false
      else:
        acc.add sep
      acc.add pyStr(x)
  of vkDict, vkNs:
    var first = true
    for k in v.d.keys:
      if first:
        first = false
      else:
        acc.add sep
      acc.add k
  else:
    raise jinjaErr("`join` needs a sequence")
  strVal(acc)

func stringTest(v: JinjaVal, args: var Args): bool = v.kind in {vkStr, vkCut}
func definedTest(v: JinjaVal, args: var Args): bool = v.kind != vkUndefined
func undefinedTest(v: JinjaVal, args: var Args): bool = v.kind == vkUndefined
func mappingTest(v: JinjaVal, args: var Args): bool = v.kind in {vkDict, vkNs}
func sequenceTest(v: JinjaVal, args: var Args): bool = v.kind == vkSeq
func iterableTest(v: JinjaVal, args: var Args): bool = v.kind in {vkSeq, vkDict, vkNs, vkStr, vkCut}
func noneTest(v: JinjaVal, args: var Args): bool = v.kind == vkNone
func booleanTest(v: JinjaVal, args: var Args): bool = v.kind == vkBool
func trueTest(v: JinjaVal, args: var Args): bool = v.kind == vkBool and v.b
func falseTest(v: JinjaVal, args: var Args): bool = v.kind == vkBool and not v.b
func getMethod(v: JinjaVal, args: var Args): JinjaVal =
  ## `d.get(key, default)`. Absence yields the default, itself undefined when unsupplied.
  if v.kind notin {vkDict, vkNs}:
    raise jinjaErr("`get` needs a mapping")
  let got = v.d.dictGet(pyStr(getArg(args, 0, akNone, undefinedVal())))
  if got.kind == vkUndefined: getArg(args, 1, akDefault, undefinedVal()) else: got

func itemsMethod(v: JinjaVal, args: var Args): JinjaVal =
  ## `[key, value]` pairs in insertion order, the form `{% for k, v in x.items() %}` iterates.
  case v.kind
  of vkDict, vkNs:
    var acc = newSeq[JinjaVal](v.d.keys.len)
    for i, k in v.d.keys:
      acc[i] = seqVal(@[strVal(k), v.d.vals[i]])
    seqVal(acc)
  of vkSeq:
    var acc = newSeq[JinjaVal](v.xs.items.len)
    for i, x in v.xs.items:
      acc[i] = seqVal(@[intVal(i), x])
    seqVal(acc)
  else:
    raise jinjaErr("`items` needs a mapping or a sequence")

func itemsFilter(v: JinjaVal, args: var Args): JinjaVal =
  ## `[key, value]` pairs in insertion order, the filter form the mapping methods mirror.
  ## Undefined yields no pairs, jinja2 3.1's `do_items` rule.
  ## Any other non-mapping kind raises, jinja2's own items filter
  ## rejecting a sequence.
  if v.kind == vkUndefined:
    return seqVal(newSeq[JinjaVal]())
  if v.kind != vkDict:
    raise jinjaErr("`items` needs a mapping")
  var acc = newSeq[JinjaVal](v.d.keys.len)
  for i, k in v.d.keys:
    acc[i] = seqVal(@[strVal(k), v.d.vals[i]])
  seqVal(acc)

func splitMethod(v: JinjaVal, args: var Args): JinjaVal =
  ## `s.split(sep)` over non-overlapping occurrences, an empty separator splitting per codepoint,
  ## a missing separator splitting on whitespace runs and dropping the empties.
  let v = if v.kind == vkCut: materializeVal(v) else: v
  if v.kind != vkStr:
    raise jinjaErr("`split` needs a string")
  let sepArg = getArg(args, 0, akNone, undefinedVal())
  var acc: seq[JinjaVal]
  if sepArg.kind == vkUndefined:
    # No separator given, split on whitespace runs, dropping the empties,
    # leading and trailing whitespace yielding no part.
    var i = 0
    var start = 0
    while i <= v.s.len:
      if i == v.s.len or v.s[i] in Whitespace:
        if i > start:
          acc.add strVal(spanString(v.s.toOpenArray(start, i - 1)))
        start = i + 1
      inc i
  else:
    let sep = pyStr(sepArg)
    if sep.len == 0:
      acc = codepointVals(v.s)
    else:
      var pos = 0
      var i = 0
      while i + sep.len <= v.s.len:
      # A non-empty separator is guaranteed here, the empty one splitting per codepoint above.
        if v.s[i] == sep[0] and sep == v.s.toOpenArray(i, i + sep.len - 1):
          acc.add strVal(spanString(v.s.toOpenArray(pos, i - 1)))
          pos = i + sep.len
          i = pos
        else:
          inc i
      acc.add strVal(spanString(v.s.toOpenArray(pos, v.s.len - 1)))
  seqVal(acc)

func sideStrip(v: JinjaVal, args: var Args, name: string, left, right: bool): JinjaVal =
  ## `s.strip(chars)`, `s.lstrip(chars)` and `s.rstrip(chars)`:
  ##   one body, the reported name and the stripped sides carried by the wrappers.
  let v = if v.kind == vkCut: materializeVal(v) else: v
  if v.kind != vkStr:
    raise jinjaErr("`" & name & "` needs a string")
  let (lo, hi) = stripSpan(v.s, pyStr(args.getArg(0, akNone, strVal(""))), left, right)
  cutVal(v.s, lo.int32, hi.int32)

func stripMethod(v: JinjaVal, args: var Args): JinjaVal = sideStrip(v, args, "strip", true, true)
func lstripMethod(v: JinjaVal, args: var Args): JinjaVal = sideStrip(v, args, "lstrip", true, false)
func rstripMethod(v: JinjaVal, args: var Args): JinjaVal = sideStrip(v, args, "rstrip", false, true)

func edgeWith(v: JinjaVal, args: var Args, name: string, tail: bool): JinjaVal =
  ## `s.startswith(p)` and `s.endswith(p)`:
  ##   one body, the reported name and the compared edge carried by the wrappers.
  let v = if v.kind == vkCut: materializeVal(v) else: v
  if v.kind != vkStr:
    raise jinjaErr("`" & name & "` needs a string")
  let p = pyStr(getArg(args, 0, akNone, strVal("")))
  boolVal(p.len == 0 or (p.len <= v.s.len and
      (if tail: p == v.s.toOpenArray(v.s.len - p.len, v.s.len - 1)
       else: p == v.s.toOpenArray(0, p.len - 1))))

func startswithMethod(v: JinjaVal, args: var Args): JinjaVal = edgeWith(v, args, "startswith", false)
func endswithMethod(v: JinjaVal, args: var Args): JinjaVal = edgeWith(v, args, "endswith", true)

const
  ## Builtin dispatch tables, one line per name, unlisted names nil, the unknown-name
  ## raise at the registry lookup serving as the gap contract. A keyed array literal pairs
  ## keys only when it names every index and the first pair shares the bracket line (Nim 2.2.10).
  FilterProcs*: array[FilterName, FilterProc] = [fTojson: tojsonFilter,
    fLength: lengthFilter,
    fTrim: trimFilter,
    fDefault: defaultFilter,
    fJoin: joinMethod,
    fLower: lowerFilter,
    fUpper: upperFilter,
    fCapitalize: capitalizeFilter,
    fList: listFilter,
    fSafe: safeFilter,
    fItems: itemsFilter,
    fString: stringFilter,
    fDictsort: nil,
    fMap: nil
  ]

  TestProcs*: array[TestName, TestProc] = [tString: stringTest,
    tDefined: definedTest,
    tUndefined: undefinedTest,
    tMapping: mappingTest,
    tSequence: sequenceTest,
    tIterable: iterableTest,
    tNone: noneTest,
    tBoolean: booleanTest,
    tTrue: trueTest,
    tFalse: falseTest
  ]

  MethodProcs*: array[MethodName, MethodProc] = [mGet: getMethod,
    mItems: itemsMethod,
    mSplit: splitMethod,
    mStrip: stripMethod,
    mLstrip: lstripMethod,
    mRstrip: rstripMethod,
    mStartswith: startswithMethod,
    mEndswith: endswithMethod
  ]