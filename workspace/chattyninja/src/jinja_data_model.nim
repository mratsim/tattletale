# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Data model of chat-template inputs and outputs, the Python-object subset Jinja templates observe.
## JSON shapes arrive as values, renderings leave as Python text or `tojson` JSON bytes.
## - values carry truthiness, equality, ordering and the stringification a template reads
## - one call's arguments travel in a fixed-capacity inline carrier (`Arg`, `Args`),
##   never a per-call sequence
## - renderings stream through a byte cursor over a caller-owned window, every failure
##   raising `JinjaError` with message, cause and template byte location

import std/unicode

const
  ArgsCap* = 8
    ## Inline capacity of one call's argument carrier. The most arguments one corpus
    ## call passes is 2. A call past the cap is a template error, reported at the call.

const
  NoOffset* = -1
    ## `JinjaError.offset` marker for a raise site with no template location in scope
  NoLink* = -1'i32
    ## Marks an absent link or absent span in every node payload slot.
  EmptyWindow: array[0, char] = []
    ## empty span backing the measuring cursor

type
  JinjaCause* = enum
    ## Kind of failure a `JinjaError` reports. Suite gap detection reads `ceUnimplemented`
    ## to classify a declared construct, registry name or filter demand.
    ceNone, ceRaiseCall, ceUnimplemented, ceScratch

  JinjaError* = ref object of CatchableError
    ## Template-level failure, the one error type the engine raises.
    ## The message lives in `what`, the inherited `msg` staying empty.
    offset*: int
      ## byte offset into the template text, `NoOffset` when the raise site has no location in scope
    span*: int
      ## byte length of the offending construct, `0` at a point location or an unknown length
    what*: string
      ## the failure message, the corpus `err_*` rows recording it verbatim
    cause*: JinjaCause
      ## kind of failure, `ceRaiseCall` a `raise_exception` call, `ceUnimplemented` a declared gap, `ceScratch` a render-scratch overflow

  Cursor* = object
    ## Appends bytes into a borrowed window. An append that does not fit raises `JinjaError`
    ## with cause `ceScratch`, naming capacity and shortfall, never growing the window.
    ## Measuring mode counts bytes without writing, the presize pass of a two-pass render.
    buf*: openArray[char]
      ## borrowed window, `buf.len` the writable capacity in bytes
    len*: int
      ## bytes appended so far, the measured length in measuring mode
    measuring*: bool
      ## count-only mode, appends advancing `len` and touching no byte

type
  ValueKind* = enum
    ## Jinja value tiers the corpus reaches, float carried for JSON fidelity only,
    ## no template in the corpus doing float arithmetic, `vkCall` holding a macro call
    ## whose body has not run, `vkConcat` holding a `~` tree awaiting its emit
    vkUndefined, vkNone, vkBool, vkInt, vkFloat, vkStr, vkSeq, vkDict, vkNs, vkLoop, vkMacro,
    vkCall, vkConcat, vkRange

  SeqVal* = ref object
    ## Shared sequence of values, the `vkSeq` payload.
    items*: seq[JinjaVal]

  DictVal* = ref object
    ## Insertion-ordered mapping, `vkNs` reusing it. All holders share
    ## one reference, so changes made through any holder reach the others.
    keys*: seq[string]
    vals*: seq[JinjaVal]

  RangeVal* = ref object
    ## Lazy `range(start, stop, step)` bounds. Elements compute per index, the range never
    ## materializing. The serializer renders the list form arithmetically and a `for` over it
    ## walks the same arithmetic through `LoopState.r`.
    start*, stop*, step*: int64

  LoopState* = ref object
    ## Cursor over the iterable a `for` walks, one cursor shared by the driver frame
    ## and the `loop` value bound in the loop scope:
    ## - `xs` borrows the sequence payload of a `vkSeq` iterable, or holds a materialized
    ##   one for mappings and strings, whose elements are derived per index
    ## - `r` holds a lazy range, nil unless the iterable is one, elements computing
    ##   per index and never materializing
    xs*: SeqVal
    r*: RangeVal
      ## lazy range bounds, nil unless the iterable is a range
    idx*: int

  ArgKeyword* = enum
    ## A keyword name a builtin reads out of an argument list, `akNone` the field's default:
    ## a positional argument or a keyword no builtin reads.
    akNone, akChars, akDefault, akEnsureAscii, akSeparators

  Arg* = object
    ## One call or filter argument, keyword-bound when `nameLo` is not `NoLink`.
    ## A keyword keeps its template span into `CompiledTemplate.jinja`, a keyword name
    ## carrying no interned `CompiledSymbols.names` entry.
    nameLo*, nameHi*: int32
      ## keyword name span into `CompiledTemplate.jinja`, `NoLink` in `nameLo` for a positional argument
    kw*: ArgKeyword
      ## keyword slot named by that span, `akNone` when no builtin reads that keyword
    val*: JinjaVal

  Args* = object
    ## Fixed-capacity inline carrier of one call's arguments in call order. `argList` fills it and every
    ## callee reads it. No per-call sequence, the carrier living on the stack
    ## at the call site and moving whole into a pending macro call.
    n*: int
      ## arguments carried, at most `ArgsCap`
    vals*: array[ArgsCap, Arg]

  PendingCallVal* = ref object
    ## A macro call whose body has not run. Holds the bound macro plus its evaluated arguments.
    ## The emit step streams the body into the drain window. An expression consumer renders
    ## the body to completion and reads the text.
    mc*: MacroVal
      ## the bound macro
    args*: Args
      ## evaluated arguments in call order

  MacroVal* = ref object
    ## A bound macro. `node` is the `nkMacroDef` arena index and the body's terminators land on it, so a call detects its end by arriving
    ## back there, name and parameters read from the definition node's payload slots at call time.
    name*: int32
    body*: int32
    node*: int32

  JinjaVal* = object
    ## One Jinja value, discriminated by `kind`.
    case kind*: ValueKind
    of vkUndefined, vkNone: nil
    of vkBool: b*: bool
    of vkInt: i*: int64
    of vkFloat: f*: float64
    of vkStr: s*: string
    of vkSeq, vkConcat: xs*: SeqVal
    of vkDict, vkNs: d*: DictVal
    of vkLoop: lp*: LoopState
    of vkMacro: mc*: MacroVal
    of vkCall: pc*: PendingCallVal
    of vkRange: r*: RangeVal

  JsonOpts* = object
    ## `tojson` knobs the corpus passes, `ensure_ascii` and `separators`.
    ## `ensureAscii` defaults to false, non-ASCII emitted as raw UTF-8.
    ensureAscii*: bool = false
    itemSep*: string = ", "
    kvSep*: string = ": "


func jinjaErr*(what: string, offset = NoOffset, span = 0, cause = ceNone): JinjaError =
  ## Returns an unraised template error. Raise sites with a template location in scope pass
  ## `offset`, plus `span` when the offending construct's length is known:
  ##   raise jinjaErr("unclosed `{% raw %}` opened at byte " & $openAt, openAt)
  JinjaError(what: what, offset: offset, span: span, cause: cause)

func over*(s: var string): Cursor =
  ## Returns a cursor over the whole byte span of `s`, capacity the string's length.
  Cursor(buf: toOpenArray(s, 0, s.len - 1))

func measureBuf*(): Cursor =
  ## Returns a measuring cursor, appends advancing `len` and touching no byte.
  Cursor(buf: toOpenArray(EmptyWindow, 0, -1), measuring: true)

func spanString*(s: openArray[char]): string =
  ## Returns a fresh string holding the bytes of `s`, one allocation bounded by the span.
  result = newString(s.len)
  if s.len > 0:
    copyMem(addr result[0], unsafeAddr s[0], s.len)

func scratchShort(sb: Cursor, need: int) {.noreturn.} =
  ## Raises the overflow an unfitting append reports, the capacity and shortfall names in the message.
  let shortfall = max(0, need - (sb.buf.len - sb.len))
  raise jinjaErr("render scratch capacity " & $sb.buf.len & " exceeded, " & $shortfall &
      " more bytes needed", cause = ceScratch)

func add*(sb: var Cursor, c: char) =
  ## Appends one byte, raising when the window cannot hold it.
  if sb.measuring:
    inc sb.len
    return
  if sb.len >= sb.buf.len:
    scratchShort(sb, 1)
  sb.buf[sb.len] = c
  inc sb.len

func add*(sb: var Cursor, s: openArray[char]) =
  ## Appends a byte span, reading `s` in place, raising when it does not fit.
  if s.len == 0:
    return
  if sb.measuring:
    sb.len += s.len
    return
  if sb.len + s.len > sb.buf.len:
    scratchShort(sb, s.len)
  copyMem(addr sb.buf[sb.len], unsafeAddr s[0], s.len)
  sb.len += s.len

func addInt*(sb: var Cursor, i: int64) =
  ## Appends the decimal form of `i`, matching `$i`.
  var digits: array[20, char]
  var n = 0
  let neg = i < 0
  # Two's-complement negation in unsigned space, so int64.low negates without overflow.
  var u = if neg: 0'u64 - cast[uint64](i) else: cast[uint64](i)
  while true:
    digits[n] = char(ord('0') + int(u mod 10'u64))
    inc n
    u = u div 10'u64
    if u == 0:
      break
  if neg:
    sb.add '-'
  for j in countdown(n - 1, 0):
    sb.add digits[j]

func addFloat*(sb: var Cursor, f: float64) =
  ## Appends Python's `str()` for a float, integral values keeping one decimal place,
  ## the shortest float repr coming from `$f`, one allocation per append.
  let s = $f
  sb.add s
  if '.' notin s and 'e' notin s and 'E' notin s and 'n' notin s and 'i' notin s:
    sb.add ".0"

func addRune*(sb: var Cursor, r: Rune) =
  ## Appends `r` as its UTF-8 bytes, matching `toUTF8` for every reachable codepoint.
  let c = ord(r)
  if c > 0x10FFFF:
    # Invalid UTF-8 decodes to out-of-range codepoints, whose `$` round-trip is not
    # standard UTF-8, so the bytes come from `toUTF8`.
    sb.add toUTF8(r)
  elif c < 0x80:
    sb.add char(c)
  elif c < 0x800:
    sb.add char(0xC0 or (c shr 6))
    sb.add char(0x80 or (c and 0x3F))
  elif c < 0x10000:
    sb.add char(0xE0 or (c shr 12))
    sb.add char(0x80 or ((c shr 6) and 0x3F))
    sb.add char(0x80 or (c and 0x3F))
  else:
    sb.add char(0xF0 or (c shr 18))
    sb.add char(0x80 or ((c shr 12) and 0x3F))
    sb.add char(0x80 or ((c shr 6) and 0x3F))
    sb.add char(0x80 or (c and 0x3F))

func undefinedVal*(): JinjaVal =
  ## Returns the absent-binding value, rendering empty, failing truthiness, equaling only itself.
  JinjaVal(kind: vkUndefined)

func noneVal*(): JinjaVal =
  ## Returns Python's `None`, rendering `None`, failing truthiness, distinct from undefined.
  JinjaVal(kind: vkNone)

func boolVal*(b: bool): JinjaVal = JinjaVal(kind: vkBool, b: b)
func intVal*(i: int64): JinjaVal = JinjaVal(kind: vkInt, i: i)
func intVal*(i: int): JinjaVal = JinjaVal(kind: vkInt, i: int64 i)
func floatVal*(f: float64): JinjaVal = JinjaVal(kind: vkFloat, f: f)
func strVal*(s: string): JinjaVal = JinjaVal(kind: vkStr, s: s)
func seqVal*(xs: seq[JinjaVal]): JinjaVal = JinjaVal(kind: vkSeq, xs: SeqVal(items: xs))
func dictVal*(d: DictVal): JinjaVal = JinjaVal(kind: vkDict, d: d)
func nsVal*(d: DictVal): JinjaVal = JinjaVal(kind: vkNs, d: d)
func loopVal*(lp: LoopState): JinjaVal = JinjaVal(kind: vkLoop, lp: lp)
func macroVal*(mc: MacroVal): JinjaVal = JinjaVal(kind: vkMacro, mc: mc)
func callVal*(pc: PendingCallVal): JinjaVal = JinjaVal(kind: vkCall, pc: pc)
func rangeVal*(start, stop, step: int64): JinjaVal =
  ## Returns the lazy range value over `start`, `stop` and `step`.
  JinjaVal(kind: vkRange, r: RangeVal(start: start, stop: stop, step: step))

func rangeLen*(r: RangeVal): int =
  ## Returns the element count of the range, Python's `len(range(start, stop, step))`:
  ## a step against the span's direction answers 0.
  let span = r.stop - r.start
  if r.step > 0:
    int(max(0'i64, (span + r.step - 1) div r.step))
  elif r.step < 0:
    int(max(0'i64, (span + r.step + 1) div r.step))
  else:
    0

func rangeAt*(r: RangeVal, i: int): JinjaVal =
  ## Returns element `i` of the range, `i` in `0 ..< rangeLen(r)`.
  intVal(r.start + i.int64 * r.step)

func rangesEqual*(a, b: RangeVal): bool =
  ## Returns Python's range equality, same length and the same element per index,
  ## not the same bounds, so `range(0, 6, 2)` equals `range(0, 5, 2)`.
  let n = rangeLen(a)
  if n != rangeLen(b):
    return false
  for i in 0 ..< n:
    if a.start + i.int64 * a.step != b.start + i.int64 * b.step:
      return false
  true

func loopLen*(lp: LoopState): int =
  ## Returns the element count the cursor walks, the lazy range's arithmetic count
  ## or the borrowed-or-materialized sequence's length.
  if lp.r != nil: lp.r.rangeLen else: lp.xs.items.len

func loopItem*(lp: LoopState, i: int): JinjaVal =
  ## Returns element `i` of the cursor's iterable, `i` in `0 ..< loopLen`,
  ## computed from the bounds for a lazy range.
  if lp.r != nil: lp.r.rangeAt(i) else: lp.xs.items[i]

func addArg*(a: var Args, v: sink Arg) =
  ## Appends one argument to the carrier, raising when the call would exceed `ArgsCap`.
  if a.n >= ArgsCap:
    raise jinjaErr("a call carries more than " & $ArgsCap & " arguments")
  a.vals[a.n] = v
  inc a.n

iterator argItems*(a: Args): lent Arg =
  ## Iterates the carrier's arguments in call order.
  for i in 0 ..< a.n:
    yield a.vals[i]

func concatVal*(cl, cr: JinjaVal): JinjaVal =
  ## Returns the `~` of two values. The operands flatten into one list in render order,
  ## so the serializer streams them one after another and the parse never re-walks a tree.
  var items: seq[JinjaVal]
  if cl.kind == vkConcat:
    items = cl.xs.items
  else:
    items = @[cl]
  if cr.kind == vkConcat:
    items.add cr.xs.items
  else:
    items.add cr
  JinjaVal(kind: vkConcat, xs: SeqVal(items: items))

func codepointVals*(s: string): seq[JinjaVal] =
  ## Returns one single-codepoint string value per codepoint of `s`, in order.
  var acc = newSeq[JinjaVal]()
  for r in s.runes:
    acc.add strVal($r)
  acc

func isTruthy*(v: JinjaVal): bool =
  ## Returns Jinja truthiness. Undefined, none, zero, empty text and empty containers are false.
  result = case v.kind
  of vkUndefined, vkNone: false
  of vkBool: v.b
  of vkInt: v.i != 0
  of vkFloat: v.f != 0
  of vkStr: v.s.len != 0
  of vkSeq: v.xs.items.len != 0
  of vkDict, vkNs: v.d.keys.len != 0
  of vkLoop: v.lp.loopLen != 0
  of vkRange: rangeLen(v.r) != 0
  of vkMacro: true
  of vkCall: raise jinjaErr("a macro call result must be rendered before a truthiness test")
  of vkConcat: raise jinjaErr("a concat must be rendered in emit position before a truthiness test")

func dictGet*(d: DictVal, key: openArray[char]): JinjaVal =
  ## Returns the value under `key`, undefined when absent. Absence is a value, never an error:
  ##   that is what makes `is defined` and `.get` fall back work. Comparison reads `key` in place,
  ##   a span lookup allocating nothing.
  for i, k in d.keys:
    if k == key:
      return d.vals[i]
  undefinedVal()

func dictSet*(d: DictVal, key: string, val: JinjaVal) =
  ## Binds `key`, replacing in place so every holder of the same DictVal observes the change.
  for i, k in d.keys:
    if k == key:
      d.vals[i] = val
      return
  d.keys.add key
  d.vals.add val

func eqVal*(a, b: JinjaVal): bool =
  ## Returns Jinja `==`:
  ##   numbers compare across tiers, containers element-wise, undefined equals only undefined.
  if a.kind == vkUndefined or b.kind == vkUndefined:
    return a.kind == vkUndefined and b.kind == vkUndefined
  if a.kind == vkBool and b.kind == vkBool:
    return a.b == b.b
  if a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
    let ai = if a.kind == vkInt: float64 a.i else: a.f
    let bi = if b.kind == vkInt: float64 b.i else: b.f
    return ai == bi
  if a.kind != b.kind:
    return false
  result = case a.kind
  of vkNone: true
  of vkStr: a.s == b.s
  of vkSeq:
    if a.xs.items.len != b.xs.items.len:
      return false
    for i, x in a.xs.items:
      if not eqVal(x, b.xs.items[i]):
        return false
    true
  of vkDict, vkNs:
    if a.d.keys.len != b.d.keys.len:
      return false
    for i, k in a.d.keys:
      if not eqVal(a.d.vals[i], b.d.dictGet(k)):
        return false
    true
  of vkLoop: a.lp == b.lp
  of vkMacro: a.mc == b.mc
  of vkRange: a.r.rangesEqual(b.r)
  of vkCall: raise jinjaErr("a macro call result must be rendered before an equality test")
  of vkConcat: raise jinjaErr("a concat must be rendered in emit position before an equality test")
  of vkUndefined, vkBool, vkInt, vkFloat: false

func cmpVal*(a, b: JinjaVal): int =
  ## Returns -1, 0 or 1 for an ordering comparison, numbers ordering numerically, text ordering
  ## by codepoint, anything else a template error, matching Jinja.
  if a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
    let ai = if a.kind == vkInt: float64 a.i else: a.f
    let bi = if b.kind == vkInt: float64 b.i else: b.f
    return if ai < bi: -1 elif ai > bi: 1 else: 0
  if a.kind == vkStr and b.kind == vkStr:
    return cmp(a.s, b.s)
  raise jinjaErr("`<` and `>` need two numbers or two strings, got " & $a.kind & " and " & $b.kind)

func substringOf(needle, haystack: string): bool =
  ## Returns whether `needle` occurs in `haystack`, the empty needle always matching.
  ## A needle longer than `haystack` leaves the scan range empty. The first-byte guard
  ## holds every non-matching position to one compare.
  if needle.len == 0:
    return true
  for i in 0 .. haystack.len - needle.len:
    if haystack[i] == needle[0] and haystack.toOpenArray(i, i + needle.len - 1) == needle:
      return true
  false

func containsVal*(haystack, needle: JinjaVal): bool =
  ## Returns Jinja `in`:
  ##   membership for sequences, keys for mappings, substring for strings.
  result = case haystack.kind
  of vkSeq:
    for x in haystack.xs.items:
      if eqVal(x, needle):
        return true
    false
  of vkDict, vkNs:
    needle.kind == vkStr and haystack.d.dictGet(needle.s).kind != vkUndefined
  of vkStr:
    needle.kind == vkStr and substringOf(needle.s, haystack.s)
  of vkRange:
    for i in 0 ..< haystack.r.rangeLen:
      if haystack.r.rangeAt(i).eqVal(needle):
        return true
    false
  else:
    raise jinjaErr("`in` needs a sequence, mapping or string on the right, got " & $haystack.kind)

func stripSpan*(s, chars: openArray[char], left, right: bool): tuple[a, b: int] =
  ## Returns the byte range of `s` that survives stripping the leading and/or trailing
  ## characters of `chars`, the way Python's `str.strip`, `lstrip` and `rstrip` cut.
  ##
  ## Contract:
  ## - empty `chars` selects the whitespace set
  ## - no byte is copied, the caller materializes the cut only where the result is stored
  var cut: set[char]
  if chars.len == 0:
    cut = {' ', '\t', '\n', '\r', '\v', '\f'}
  else:
    for c in chars:
      cut.incl c
  var a = 0
  var b = s.len
  if left:
    while a < b and s[a] in cut:
      inc a
  if right:
    while b > a and s[b - 1] in cut:
      dec b
  (a, b)

func stripMaterialized*(s, chars: openArray[char], left, right: bool): JinjaVal =
  ## Returns the stripped cut of `s` as a string value, the storage-boundary materialization
  ## of `stripSpan`: one string for the kept bytes, none for an empty cut.
  let (a, b) = stripSpan(s, chars, left, right)
  strVal(if a == b: "" else: spanString(s.toOpenArray(a, b - 1)))
