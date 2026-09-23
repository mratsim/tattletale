# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Data model of chat-template inputs and outputs, the Python-object subset Jinja templates observe.
## JSON shapes arrive as values, renderings leave as Python text or `tojson` JSON bytes.
## - values carry truthiness, equality, ordering and the stringification a template reads
## - one call's arguments travel in an inline array of three arguments (`Arg`, `Args`),
##   never a per-call sequence
## - renderings stream through a byte cursor over a caller-owned window, every failure
##   raising `JinjaError` with message, cause and template byte location

# Public API:
#   JinjaVal with ValueKind, the value constructors, eqVal, dictSet and dictGet over DictVal,
#   JinjaError with cause and location, JsonOpts and the value caps. Everything else is
#   value plumbing, cross-module code importing it through `import x {.all.}`.

import std/unicode
import workspace/data_structures/src/small_seqs
export small_seqs

const
  ArgsCap = 3
    ## Argument capacity of one call's `Args`, past it `SmallSeq` spills to the heap.

const
  TTT_CNJ_RangeElemCap {.intdefine.} = 1_000_000
    ## Element bound of one lazy `range`, checked eagerly at construction and at every
    ## count or drain, a breach raising a located `JinjaError`.
  TTT_CNJ_ValueDepthCap {.intdefine.} = 1000
    ## Recursion bound over the value graph fed host-provided data, corpus nesting
    ## in the single digits. Every walker raises a `JinjaError` on breach, located
    ## where a template position is in scope.
    ## - cyclic graphs raise the same way, their nesting unbounded

const
  NoOffset = -1
    ## `JinjaError.offset` marker for a raise site with no template location in scope
  NoLink = -1'i32
    ## Marks an absent link or span in a node payload slot.
  EmptyWindow: array[0, char] = []
    ## empty span backing the measuring cursor

type
  JinjaCause* = enum
    ## Kind of failure a `JinjaError` reports. Suite gap detection reads `ceUnimplemented`
    ## to classify a declared construct, registry name or filter demand.
    ceNone, ceRaiseCall, ceUnimplemented, ceWindow

  JinjaError = ref object of CatchableError
    ## Template-level failure, the one error type the engine raises.
    ## The message lives in `what`, the inherited `msg` staying empty.
    offset*: int
      ## byte offset into the template text, `NoOffset` when the raise site has no location in scope
    span*: int
      ## byte length of the offending construct, `0` at a point location or an unknown length
    what*: string
      ## the failure message, the corpus `err_*` rows recording it verbatim
    cause*: JinjaCause
      ## kind of failure, `ceRaiseCall` a `raise_exception` call, `ceUnimplemented` a declared gap, `ceWindow` a window-capacity overflow

  Cursor = object
    ## Appends bytes into a borrowed window, overflow raising `ceWindow`, measuring mode only counting.
    buf*: openArray[char]
      ## borrowed window, `buf.len` the writable capacity in bytes
    len*: int
      ## bytes appended so far, the measured length in measuring mode
    measuring: bool
      ## count-only mode, appends advancing `len` and touching no byte

type
  ValueKind* = enum
    ## Runtime type a `JinjaVal` carries, discriminated by `kind`.
    vkUndefined, vkNone, vkBool, vkInt, vkFloat, vkStr, vkSeq, vkDict, vkNs, vkLoop, vkMacro,
    vkCall, vkRange, vkCut

  SeqVal = ref object
    ## Shared sequence of values, the `vkSeq` payload.
    items*: seq[JinjaVal]

  DictVal = ref object
    ## Insertion-ordered mapping, `vkNs` reusing it, every holder sharing one reference so changes reach all.
    keys*: seq[string]
    vals*: seq[JinjaVal]

  RangeVal = object
    ## Lazy `range(start, stop, step)` bounds, elements computed per index and never materialized, an immutable value inline in the range value.
    start*, stop*, step*: int64

  LoopState = ref object
    ## Cursor over the iterable a `for` walks, `xs` borrowing the payload or holding
    ## the materialized copy, `r` the lazy range, `idx` the cursor position.
    xs*: SeqVal
    r*: RangeVal
      ## lazy range bounds, meaningful only when `isRange`
    isRange*: bool
      ## the iterable is a lazy range, `r` then carrying its bounds
    idx*: int

  ArgKeyword* = enum
    ## A keyword name a builtin reads out of an argument list, `akNone` otherwise.
    akNone, akChars, akDefault, akEnsureAscii, akSeparators

  Arg = object
    ## One call or filter argument, keyword-bound when `nameLo` is not `NoLink`.
    nameLo*, nameHi*: int32
      ## keyword name span into `CompiledTemplate.jinja`, `NoLink` in `nameLo` for a positional argument
    kw*: ArgKeyword
      ## keyword slot named by that span, `akNone` when no builtin reads that keyword
    val*: JinjaVal

  Args = SmallSeq[ArgsCap, Arg]
    ## One call's arguments in call order, filled by moves, read by borrows, `ArgsCap` inline.

  DeferredMacroCall = ref object
    ## A macro call whose body has not run, holding the bound macro and its evaluated arguments.
    ## Rendering the value runs the body.
    mc*: MacroVal
      ## the bound macro
    args*: Args
      ## evaluated arguments in call order

  MacroVal = object
    ## Bound macro over three immutable `nkMacroDef` node indexes (name, body, node), nothing shared.
    name*: int32
    body*: int32
    node*: int32

  JinjaVal = object
    ## One Jinja value, discriminated by `kind`.
    case kind*: ValueKind
    of vkUndefined, vkNone: nil
    of vkBool: b*: bool
    of vkInt: i*: int64
    of vkFloat: f*: float64
    of vkStr: s*: string
    of vkSeq: xs*: SeqVal
    of vkDict, vkNs: d*: DictVal
    of vkLoop: lp*: LoopState
    of vkMacro: mc*: MacroVal
    of vkCall: pc*: DeferredMacroCall
    of vkRange: r*: RangeVal
    of vkCut:
      raw*: string
        ## the unstripped input, moved in, its buffer shared with the source value
      lo*, hi*: int32
        ## byte bounds of the surviving sub-span, rendering as `raw[lo ..< hi]`

  JsonOpts = object
    ## `tojson` knobs the corpus passes, `ensure_ascii` and `separators`.
    ## `ensureAscii` defaults to false, non-ASCII emitted as raw UTF-8.
    ensureAscii*: bool = false
    itemSep*: string = ", "
    kvSep*: string = ": "



func jinjaErr(what: string, offset = NoOffset, span = 0, cause = ceNone): JinjaError =
  ## Returns an unraised template error. Raise sites with a template location in scope pass
  ## `offset`, plus `span` when the offending construct's length is known:
  ##   raise jinjaErr("unclosed `{% raw %}` opened at byte " & $openAt, openAt)
  JinjaError(what: what, offset: offset, span: span, cause: cause)

func over(s: var openArray[char]): Cursor =
  ## Returns a cursor over the whole byte span of `s`, capacity the span's length.
  Cursor(buf: toOpenArray(s, 0, s.len - 1))

func measureBuf(): Cursor =
  ## Returns a measuring cursor, appends advancing `len` and touching no byte.
  Cursor(buf: toOpenArray(EmptyWindow, 0, -1), measuring: true)

func spanString(s: openArray[char]): string =
  ## Returns a fresh string holding the bytes of `s`, one allocation bounded by the span.
  result = newString(s.len)
  if s.len > 0:
    copyMem(addr result[0], unsafeAddr s[0], s.len)

func addView(outp: var string, view: openArray[char]) =
  ## Appends the bytes of `view` to `outp`, one grow and one copy, no intermediate
  ## string. The string-side counterpart of `Cursor.add(openArray[char])`.
  ##
  ## Args:
  ## - `view` is read where it lives, nothing materialized
  ## - `outp` grows once, by `view.len`
  ##
  ## ``addView(acc, s.toOpenArray(lo, hi))`` appends the span `s` holds
  ## between `lo` and `hi` without building a slice.
  if view.len == 0:
    return
  let at = outp.len
  outp.setLen(at + view.len)
  copyMem(addr outp[at], unsafeAddr view[0], view.len)

func windowOverflow(sb: Cursor, need: int) {.noreturn.} =
  ## Raises the overflow an append reports when the window cannot hold the bytes,
  ## the capacity and shortfall named in the message.
  let shortfall = max(0, need - (sb.buf.len - sb.len))
  raise jinjaErr("render window capacity " & $sb.buf.len & " exceeded, " & $shortfall &
      " more bytes needed", cause = ceWindow)

func add(sb: var Cursor, c: char) =
  ## Appends one byte, raising when the window cannot hold it.
  if sb.measuring:
    inc sb.len
    return
  if sb.len >= sb.buf.len:
    windowOverflow(sb, 1)
  sb.buf[sb.len] = c
  inc sb.len

func add(sb: var Cursor, s: openArray[char]) =
  ## Appends a byte span, reading `s` in place, raising when it does not fit.
  if s.len == 0:
    return
  if sb.measuring:
    sb.len += s.len
    return
  if sb.len + s.len > sb.buf.len:
    windowOverflow(sb, s.len)
  copyMem(addr sb.buf[sb.len], unsafeAddr s[0], s.len)
  sb.len += s.len

func addInt(sb: var Cursor, i: int64) =
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

func addFloat(sb: var Cursor, f: float64) =
  ## Appends Python's `str()` for a float, integral values keeping one decimal place,
  ## the shortest float repr coming from `$f`, one allocation per append.
  let s = $f
  sb.add s
  if '.' notin s and 'e' notin s and 'E' notin s and 'n' notin s and 'i' notin s:
    sb.add ".0"

func addRune(sb: var Cursor, r: Rune) =
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

func undefinedVal(): JinjaVal =
  ## Returns the absent-binding value, rendering empty, failing truthiness, equaling only itself.
  JinjaVal(kind: vkUndefined)

func noneVal(): JinjaVal =
  ## Returns Python's `None`, rendering `None`, failing truthiness, distinct from undefined.
  JinjaVal(kind: vkNone)

func boolVal(b: bool): JinjaVal = JinjaVal(kind: vkBool, b: b)
func intVal(i: int64): JinjaVal = JinjaVal(kind: vkInt, i: i)
func intVal(i: int): JinjaVal = JinjaVal(kind: vkInt, i: int64 i)
func floatVal(f: float64): JinjaVal = JinjaVal(kind: vkFloat, f: f)
func strVal(s: string): JinjaVal = JinjaVal(kind: vkStr, s: s)
func seqVal(xs: seq[JinjaVal]): JinjaVal = JinjaVal(kind: vkSeq, xs: SeqVal(items: xs))
func dictVal(d: DictVal): JinjaVal = JinjaVal(kind: vkDict, d: d)
func nsVal(d: DictVal): JinjaVal = JinjaVal(kind: vkNs, d: d)
func loopVal(lp: LoopState): JinjaVal = JinjaVal(kind: vkLoop, lp: lp)
func macroVal(mc: MacroVal): JinjaVal = JinjaVal(kind: vkMacro, mc: mc)
func callVal(pc: DeferredMacroCall): JinjaVal = JinjaVal(kind: vkCall, pc: pc)
func rangeVal(start, stop, step: int64): JinjaVal =
  ## Returns the lazy range value over `start`, `stop` and `step`.
  JinjaVal(kind: vkRange, r: RangeVal(start: start, stop: stop, step: step))

func cutVal(s: sink string, lo, hi: int32): JinjaVal =
  ## Returns the stripped cut of `s`, rendering as the bytes of `s[lo ..< hi]`:
  ## - `s` moves in, so the cut shares the source's buffer
  ## - the cut materializes its string only where a consumer stores or re-computes it
  ##
  ##   cutVal(text, 4'i32, 9'i32)
  ##
  ## renders the 5 bytes `text[4 ..< 9]`, the strip span the caller's to compute.
  JinjaVal(kind: vkCut, raw: s, lo: lo, hi: hi)

func materializeVal(v: JinjaVal): JinjaVal =
  ## Returns the materialized form of `v`:
  ## - a cut becomes its string value, one copy of the surviving bytes
  ## - every other kind passes through unchanged
  ##
  ## Consumers that store or re-compute a value call this. The serializer streams
  ## a cut in emit position without copying.
  if v.kind == vkCut:
    strVal(if v.lo == v.hi: "" else: spanString(v.raw.toOpenArray(v.lo.int, v.hi.int - 1)))
  else:
    v

func rangeLen(r: RangeVal, lo = NoOffset, hi = 0): int =
  ## Returns the element count of the range, Python's `len(range(start, stop, step))`:
  ## a step against the span's direction answers 0.
  ##
  ## A count past `TTT_CNJ_RangeElemCap` raises a `JinjaError` located at the range
  ## expression `lo ..< hi`. A caller with no template location in scope passes
  ## `NoOffset` and the raise carries none.
  ##
  ## The walk distance and the stride magnitude compute in unsigned space, where each
  ## is exact, so the span arithmetic never wraps however extreme the bounds:
  ##
  ##   rangeLen(rangeVal(0, 9223372036854775807, 1)) computes the walk distance exactly
  ##   and the raise fires past `TTT_CNJ_RangeElemCap`, before any consumer sees the count
  if r.step == 0:
    return 0
  let fwd = r.step > 0
  let d = if fwd:
      if r.stop <= r.start: return 0
      cast[uint64](r.stop) - cast[uint64](r.start)
    else:
      if r.start <= r.stop: return 0
      cast[uint64](r.start) - cast[uint64](r.stop)
  let s = if fwd: cast[uint64](r.step) else: 0'u64 - cast[uint64](r.step)
  let n = (d - 1) div s + 1
  if n > cast[uint64](TTT_CNJ_RangeElemCap):
    let what = "range of " & $n & " elements exceeds TTT_CNJ_RangeElemCap = " & $TTT_CNJ_RangeElemCap
    if lo == NoOffset:
      raise jinjaErr(what)
    raise jinjaErr(what, lo, hi - lo)
  int(n)

func rangeAt(r: RangeVal, i: int): JinjaVal =
  ## Returns element `i` of the range, `i` in `0 ..< rangeLen(r)`.
  ## The element `start + i * step` computes in unsigned space, where the arithmetic
  ## is exact, every element of a bounded, direction-consistent range lying in int64:
  ## - nothing wraps, no overflow check, however extreme the bounds
  let s = if r.step > 0: cast[uint64](r.step) else: 0'u64 - cast[uint64](r.step)
  let e = if r.step > 0: cast[uint64](r.start) + cast[uint64](i) * s
      else: cast[uint64](r.start) - cast[uint64](i) * s
  intVal(cast[int64](e))

func rangesEqual(a, b: RangeVal): bool =
  ## Returns Python's range equality from the bounds and strides alone, no scan:
  ## - same length, the first element and the stride coinciding
  ##   (`range(0, 6, 2)` equals `range(0, 5, 2)`)
  ## - empty ranges always compare equal
  ## - singletons compare equal through the shared element alone
  let n = rangeLen(a)
  if n != rangeLen(b):
    return false
  if n == 0:
    return true
  if a.start != b.start:
    return false
  n == 1 or a.step == b.step

func rangeContains(r: RangeVal, needle: JinjaVal): bool =
  ## Returns Python's range membership from the bounds and stride alone, no scan:
  ## - an int needle lies inside the walked span on the stride's direction,
  ##   splitting the stride exactly, the offset and the stride magnitude
  ##   comparing in unsigned space so extreme bounds wrap nothing
  ## - a float needle matches the element whose rounding equals it, the equality
  ##   of the float cross-tier `==`, the match found by a direction-aware binary
  ##   search over the monotone element walk
  ## - every other needle kind fails the element `==`, elements being ints
  if needle.kind == vkInt:
    if r.step == 0:
      return false
    let d = if r.step > 0:
        if r.start <= needle.i and needle.i < r.stop:
          cast[uint64](needle.i) - cast[uint64](r.start)
        else:
          return false
      else:
        if r.stop < needle.i and needle.i <= r.start:
          cast[uint64](r.start) - cast[uint64](needle.i)
        else:
          return false
    let s = if r.step > 0: cast[uint64](r.step) else: 0'u64 - cast[uint64](r.step)
    result = d mod s == 0 and d div s < cast[uint64](rangeLen(r))
  elif needle.kind == vkFloat:
    let n = rangeLen(r)
    if n == 0:
      return false
    # Leftmost index whose element rounds onto the needle's side of the walk.
    # Elements move monotonically with the step's direction, a distinct int per
    # element with a nonzero step, so the search compares in the step's direction:
    # - forward, the first element rounded at or above the needle
    # - backward, the first element rounded at or below the needle
    var a = 0
    var b = n
    while a < b:
      let m = (a + b) div 2
      let e = rangeAt(r, m).i.float64
      if (e >= needle.f and r.step > 0) or (e <= needle.f and r.step < 0):
        b = m
      else:
        a = m + 1
    result = a < n and rangeAt(r, a).i.float64 == needle.f
  else:
    result = false

func loopLen(lp: LoopState): int =
  ## Returns the element count the cursor walks, the lazy range's arithmetic count
  ## or the borrowed-or-materialized sequence's length.
  if lp.isRange: lp.r.rangeLen else: lp.xs.items.len

func loopItem(lp: LoopState, i: int): JinjaVal =
  ## Returns element `i` of the cursor's iterable, `i` in `0 ..< loopLen`,
  ## computed from the bounds for a lazy range.
  if lp.isRange: lp.r.rangeAt(i) else: lp.xs.items[i]

iterator argItems(a: var Args): var Arg =
  ## Iterates `Args` in call order, each yielded by borrow.
  for i in 0 ..< a.len:
    yield a[i]

func codepointVals(s: string): seq[JinjaVal] =
  ## Returns one single-codepoint string value per codepoint of `s`, in order.
  var acc = newSeq[JinjaVal]()
  for r in s.runes:
    acc.add strVal($r)
  acc

func isTruthy(v: JinjaVal, at: int = NoOffset): bool =
  ## Returns Jinja truthiness, undefined, none, zero, empty text and empty containers false.
  ## `at` locates the raise a held call carries when a caller tests one without rendering it
  ## first:
  ## - every boolean-position consumer coerces a call to its rendered bytes before the test,
  ##   so this raise is the contract guard, never the render path
  result = case v.kind
  of vkUndefined, vkNone: false
  of vkBool: v.b
  of vkInt: v.i != 0
  of vkFloat: v.f != 0
  of vkStr: v.s.len != 0
  of vkCut: v.lo != v.hi
  of vkSeq: v.xs.items.len != 0
  of vkDict, vkNs: v.d.keys.len != 0
  of vkLoop: v.lp.loopLen != 0
  of vkRange: rangeLen(v.r) != 0
  of vkMacro: true
  of vkCall: raise jinjaErr("a macro call result must be rendered before a truthiness test", at)

func dictGet(d: DictVal, key: openArray[char]): JinjaVal =
  ## Returns the value under `key`, undefined when absent. Absence is a value, never an error:
  ##   that is what makes `is defined` and `.get` fall back work. Comparison reads `key` in place,
  ##   a span lookup allocating nothing.
  for i, k in d.keys:
    if k == key:
      return d.vals[i]
  undefinedVal()

func dictSet(d: DictVal, key: string, val: JinjaVal) =
  ## Binds `key`, replacing in place so every holder of the same DictVal observes the change.
  for i, k in d.keys:
    if k == key:
      d.vals[i] = val
      return
  d.keys.add key
  d.vals.add val

func dictFind(d: DictVal, key: openArray[char]): int =
  ## Returns the index of `key`, -1 when absent. Presence-aware lookups build
  ## on the scan, `dictGet` layers undefined-for-absence over the same walk.
  for i, k in d.keys:
    if k == key:
      return i
  -1

func sameBytes(x, y: openArray[char]): bool =
  ## Returns whether two byte spans hold the same bytes, comparing in place.
  if x.len != y.len:
    return false
  for i in 0 ..< x.len:
    if x[i] != y[i]:
      return false
  true

func eqValAt(a, b: JinjaVal, depth: int, offset: int): bool =
  ## `eqVal` recursion core, `depth` the container levels entered so far, counting
  ## toward `TTT_CNJ_ValueDepthCap`, the graph nesting past it raising a located
  ## `JinjaError` at `offset`:
  ## - data deeper than the cap
  ## - a value-graph cycle, whose comparison otherwise runs off the C stack
  if depth > TTT_CNJ_ValueDepthCap:
    raise jinjaErr("value nesting deeper than TTT_CNJ_ValueDepthCap = " & $TTT_CNJ_ValueDepthCap &
        " cannot be compared", offset)
  if a.kind == vkUndefined or b.kind == vkUndefined:
    return a.kind == vkUndefined and b.kind == vkUndefined
  if a.kind == vkBool and b.kind == vkBool:
    return a.b == b.b
  if a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
    let ai = if a.kind == vkInt: float64 a.i else: a.f
    let bi = if b.kind == vkInt: float64 b.i else: b.f
    return ai == bi
  if a.kind == vkCut or b.kind == vkCut:
    # Text equality spans a cut's sub-span against the other side, comparing in place.
    if a.kind == vkCut and b.kind == vkCut:
      return a.hi - a.lo == b.hi - b.lo and
          sameBytes(a.raw.toOpenArray(a.lo, a.hi - 1), b.raw.toOpenArray(b.lo, b.hi - 1))
    if a.kind == vkCut:
      return b.kind == vkStr and sameBytes(a.raw.toOpenArray(a.lo, a.hi - 1), b.s)
    return a.kind == vkStr and sameBytes(a.s, b.raw.toOpenArray(b.lo, b.hi - 1))
  if a.kind != b.kind:
    return false
  result = case a.kind
  of vkNone: true
  of vkStr: a.s == b.s
  of vkSeq:
    # One shared sequence payload always equals itself, whatever it nests.
    if a.xs == b.xs:
      return true
    if a.xs.items.len != b.xs.items.len:
      return false
    for i, x in a.xs.items:
      if not eqValAt(x, b.xs.items[i], depth + 1, offset):
        return false
    true
  of vkDict, vkNs:
    # One shared mapping payload always equals itself, cycles included.
    if a.d == b.d:
      return true
    if a.d.keys.len != b.d.keys.len:
      return false
    # Key presence decides before the values, a stored undefined value is not
    # equal to a key the other mapping lacks, dictGet handing undefined back
    # for absence otherwise equating `undefined == undefined`
    for i, k in a.d.keys:
      let j = b.d.dictFind(k)
      if j < 0 or not eqValAt(a.d.vals[i], b.d.vals[j], depth + 1, offset):
        return false
    true
  of vkLoop: a.lp == b.lp
  of vkMacro: a.mc == b.mc
  of vkRange: a.r.rangesEqual(b.r)
  of vkCall: raise jinjaErr("a macro call result must be rendered before an equality test")
  # Unreachable leg, a cut returns above against the other side's sub-span compare.
  of vkCut: false
  of vkUndefined, vkBool, vkInt, vkFloat: false

func eqVal(a, b: JinjaVal, offset = NoOffset): bool =
  ## Returns Jinja `==`:
  ## - numbers compare across tiers, containers element-wise, undefined
  ##   equaling only undefined
  ## - a shared container payload equals itself
  ## - a value graph nesting past `TTT_CNJ_ValueDepthCap` raises at `offset`,
  ##   never running off the C stack
  eqValAt(a, b, 0, offset)

func cmpVal(a, b: JinjaVal): int =
  ## Returns -1, 0 or 1 for an ordering comparison, numbers ordering numerically, text ordering
  ## by codepoint, anything else a template error, matching Jinja.
  if a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
    let ai = if a.kind == vkInt: float64 a.i else: a.f
    let bi = if b.kind == vkInt: float64 b.i else: b.f
    return if ai < bi: -1 elif ai > bi: 1 else: 0
  if a.kind in {vkStr, vkCut} and b.kind in {vkStr, vkCut}:
    if a.kind == vkCut or b.kind == vkCut:
      return cmp(materializeVal(a).s, materializeVal(b).s)
    return cmp(a.s, b.s)
  raise jinjaErr("`<` and `>` need two numbers or two strings, got " & $a.kind & " and " & $b.kind)

func substringOf(needle, haystack: openArray[char]): bool =
  ## Returns whether `needle` occurs in `haystack`, the empty needle always matching.
  ## A needle longer than `haystack` leaves the scan range empty. The first-byte guard
  ## holds every non-matching position to one compare.
  if needle.len == 0:
    return true
  for i in 0 .. haystack.len - needle.len:
    if haystack[i] == needle[0] and haystack.toOpenArray(i, i + needle.len - 1) == needle:
      return true
  false

func containsValAt(haystack, needle: JinjaVal, depth: int, offset: int): bool =
  ## `containsVal` recursion core, `depth` counting toward `TTT_CNJ_ValueDepthCap` exactly
  ## as `eqValAt` does, the breach raising a `JinjaError` located at `offset`.
  ## Each container level of the haystack is entered through the element `==`.
  if depth > TTT_CNJ_ValueDepthCap:
    raise jinjaErr("value nesting deeper than TTT_CNJ_ValueDepthCap = " & $TTT_CNJ_ValueDepthCap &
        " cannot be scanned", offset)
  let haystack = if haystack.kind == vkCut: materializeVal(haystack) else: haystack
  let needle = if needle.kind == vkCut: materializeVal(needle) else: needle
  result = case haystack.kind
  of vkSeq:
    for x in haystack.xs.items:
      if eqValAt(x, needle, depth + 1, offset):
        return true
    false
  of vkDict, vkNs:
    needle.kind == vkStr and haystack.d.dictGet(needle.s).kind != vkUndefined
  of vkStr:
    needle.kind == vkStr and substringOf(needle.s, haystack.s)
  of vkRange: rangeContains(haystack.r, needle)
  else:
    raise jinjaErr("`in` needs a sequence, mapping or string on the right, got " & $haystack.kind)

func containsVal(haystack, needle: JinjaVal, offset = NoOffset): bool =
  ## Returns Jinja `in`:
  ## - membership for sequences, keys for mappings, substring for strings
  ## - arithmetic membership for a lazy range
  ## - a value graph nesting past `TTT_CNJ_ValueDepthCap` raises at `offset`,
  ##   never running off the C stack
  containsValAt(haystack, needle, 0, offset)

func stripSpan(s, chars: openArray[char], left, right: bool): tuple[a, b: int] =
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
