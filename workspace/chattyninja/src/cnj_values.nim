# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Value model of the chattyninja engine. Carries Jinja-tier values with truthiness and equality,
# plus ordering, stringification, and the Python-repr/JSON renderings a template observes.
# No dependency on the node arena, so the value tier is testable without a compiled template.

import std/unicode
import cnj_errors

# Byte sink over a caller-owned window. A derived value renders into an existing buffer, so rendering costs no per-part allocation.

const emptyWindow: array[0, char] = []

type
  Cursor* = object
    ## Byte sink over a borrowed byte window. An append that does not fit raises
    ## `ScratchError` carrying the capacity and the shortfall, never growing the window.
    ## Measuring mode counts bytes without writing, the presize pass of a two-pass render.
    buf*: openArray[char]
      ## borrowed window, `buf.len` the writable capacity in bytes
    len*: int
      ## bytes appended so far, the measured length in measuring mode
    measuring*: bool
      ## count-only mode, appends advancing `len` and touching no byte

func over*(s: var string): Cursor =
  ## Returns a cursor over the whole byte span of `s`, capacity the string's length.
  Cursor(buf: toOpenArray(s, 0, s.len - 1))

func measureBuf*(): Cursor =
  ## Returns a measuring cursor, appends advancing `len` and touching no byte.
  Cursor(buf: toOpenArray(emptyWindow, 0, -1), measuring: true)

func spanString*(s: openArray[char]): string =
  ## Returns a fresh string holding the bytes of `s`, one allocation bounded by the span.
  result = newString(s.len)
  if s.len > 0:
    copyMem(addr result[0], unsafeAddr s[0], s.len)

proc scratchShort(sb: Cursor, need: int) {.noreturn.} =
  ## Raises the typed overflow an unfitting append reports, naming capacity and shortfall.
  var e = ScratchError(capacity: sb.buf.len, shortfall: max(0, need - (sb.buf.len - sb.len)))
  e.msg = "render scratch capacity " & $sb.buf.len & " exceeded, " & $e.shortfall &
      " more bytes needed"
  raise e

proc add*(sb: var Cursor, c: char) =
  ## Appends one byte, raising when the window cannot hold it.
  if sb.measuring:
    inc sb.len
    return
  if sb.len >= sb.buf.len:
    scratchShort(sb, 1)
  sb.buf[sb.len] = c
  inc sb.len

proc add*(sb: var Cursor, s: openArray[char]) =
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

proc addInt*(sb: var Cursor, i: int64) =
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

proc addFloat*(sb: var Cursor, f: float64) =
  ## Appends Python's `str()` for a float, integral values keeping one decimal place,
  ## the shortest float repr coming from `$f`, one allocation per append.
  let s = $f
  sb.add s
  if '.' notin s and 'e' notin s and 'E' notin s and 'n' notin s and 'i' notin s:
    sb.add ".0"

proc addRune*(sb: var Cursor, r: Rune) =
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

type
  ValueKind* = enum
    ## Jinja value tiers the corpus reaches, float carried for JSON fidelity only,
    ## no template in the corpus doing float arithmetic
    vkUndefined, vkNone, vkBool, vkInt, vkFloat, vkStr, vkSeq, vkDict, vkNs, vkLoop, vkMacro

  SeqVal* = ref object
    ## Shared sequence of values, the `vkSeq` payload.
    items*: seq[Value]

  DictVal* = ref object
    ## Insertion-ordered mapping, `vkNs` reusing it. All holders share
    ## one reference, so changes made through any holder reach the others.
    keys*: seq[string]
    vals*: seq[Value]

  LoopState* = ref object
    ## Materialized iterable plus the cursor `loop.*` reads, shared between the driver's frame
    ## and the `loop` value bound in the loop scope, both readers seeing one cursor.
    items*: seq[Value]
    idx*: int

  MacroVal* = ref object
    ## A bound macro. `node` is the `nkMacroDef` arena index and the body's terminators land on it, so a call detects its end by arriving
    ## back there, name and parameters read from the definition node's payload slots at call time.
    name*: int32
    body*: int32
    node*: int32

  Value* = object
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

  JsonOpts* = object
    ## `tojson` knobs the corpus passes, `ensure_ascii` and `separators`.
    ## `ensureAscii` defaults to false, non-ASCII emitted as raw UTF-8.
    ensureAscii*: bool = false
    itemSep*: string = ", "
    kvSep*: string = ": "

func undefinedVal*(): Value =
  ## Returns the absent-binding value, rendering empty, failing truthiness, equaling only itself.
  Value(kind: vkUndefined)

func noneVal*(): Value =
  ## Returns Python's `None`, rendering `None`, failing truthiness, distinct from undefined.
  Value(kind: vkNone)

func boolVal*(b: bool): Value = Value(kind: vkBool, b: b)
func intVal*(i: int64): Value = Value(kind: vkInt, i: i)
func intVal*(i: int): Value = Value(kind: vkInt, i: int64 i)
func floatVal*(f: float64): Value = Value(kind: vkFloat, f: f)
func strVal*(s: string): Value = Value(kind: vkStr, s: s)
func seqVal*(xs: seq[Value]): Value = Value(kind: vkSeq, xs: SeqVal(items: xs))
func dictVal*(d: DictVal): Value = Value(kind: vkDict, d: d)
func nsVal*(d: DictVal): Value = Value(kind: vkNs, d: d)
func loopVal*(lp: LoopState): Value = Value(kind: vkLoop, lp: lp)
func macroVal*(mc: MacroVal): Value = Value(kind: vkMacro, mc: mc)

func codepointVals*(s: string): seq[Value] =
  ## Returns one single-codepoint string value per codepoint of `s`, in order.
  var acc = newSeq[Value]()
  for r in s.runes:
    acc.add strVal($r)
  acc

func isTruthy*(v: Value): bool =
  ## Returns Jinja truthiness. Undefined, none, zero, empty text and empty containers are false.
  result = case v.kind
  of vkUndefined, vkNone: false
  of vkBool: v.b
  of vkInt: v.i != 0
  of vkFloat: v.f != 0
  of vkStr: v.s.len != 0
  of vkSeq: v.xs.items.len != 0
  of vkDict, vkNs: v.d.keys.len != 0
  of vkLoop: v.lp.items.len != 0
  of vkMacro: true

func dictGet*(d: DictVal, key: openArray[char]): Value =
  ## Returns the value under `key`, undefined when absent. Absence is a value, never an error:
  ##   that is what makes `is defined` and `.get` fall back work. Comparison reads `key` in place,
  ##   a span lookup allocating nothing.
  for i, k in d.keys:
    if k == key:
      return d.vals[i]
  undefinedVal()

func dictSet*(d: DictVal, key: string, val: Value) =
  ## Binds `key`, replacing in place so every holder of the same DictVal observes the change.
  for i, k in d.keys:
    if k == key:
      d.vals[i] = val
      return
  d.keys.add key
  d.vals.add val

func eqVal*(a, b: Value): bool =
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
  of vkUndefined, vkBool, vkInt, vkFloat: false

func cmpVal*(a, b: Value): int =
  ## Returns -1, 0 or 1 for an ordering comparison, numbers ordering numerically, text ordering
  ## by codepoint, anything else a template error, matching Jinja.
  if a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
    let ai = if a.kind == vkInt: float64 a.i else: a.f
    let bi = if b.kind == vkInt: float64 b.i else: b.f
    return if ai < bi: -1 elif ai > bi: 1 else: 0
  if a.kind == vkStr and b.kind == vkStr:
    return cmp(a.s, b.s)
  raise err("`<` and `>` need two numbers or two strings, got " & $a.kind & " and " & $b.kind)

func substringOf(needle, haystack: string): bool =
  ## Returns whether `needle` occurs in `haystack`, the empty needle always matching.
  ## A needle longer than `haystack` leaves the scan range empty.
  for i in 0 .. haystack.len - needle.len:
    if haystack.toOpenArray(i, i + needle.len - 1) == needle:
      return true
  false

func containsVal*(haystack, needle: Value): bool =
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
  else:
    raise err("`in` needs a sequence, mapping or string on the right, got " & $haystack.kind)

func pyStrip*(s, chars: string, left, right: bool): string =
  ## Returns `s` with leading and/or trailing characters in `chars` removed, the way Python's
  ## `str.strip`, `lstrip` and `rstrip` do, empty `chars` selecting the whitespace set.
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
  if a == b: "" else: spanString(s.toOpenArray(a, b - 1))

proc pyRepr*(v: Value): string

proc pyStrInto*(v: Value, sb: var Cursor)
proc pyReprInto(v: Value, sb: var Cursor)

proc materializeStr(write: proc (v: Value, sb: var Cursor) {.nimcall.}, v: Value): string =
  ## Returns one fresh string holding `write`'s rendering of `v`, a measuring pass presizing
  ## and the render pass filling, one allocation bounded by the size.
  var sb = measureBuf()
  write(v, sb)
  result = newString(sb.len)
  var dst = over(result)
  write(v, dst)

proc pyStrInto*(v: Value, sb: var Cursor) =
  ## Writes the value as template output text into `sb`:
  ## - strings pass through, scalars format in place, undefined renders empty
  ## - containers take their Python `repr()` form
  ## Raises when `sb` cannot hold the rendering, never growing it.
  case v.kind
  of vkUndefined: discard
  of vkNone: sb.add "None"
  of vkBool: sb.add(if v.b: "True" else: "False")
  of vkInt: sb.addInt v.i
  of vkFloat: sb.addFloat v.f
  of vkStr: sb.add v.s
  of vkSeq, vkDict, vkNs, vkLoop, vkMacro: pyReprInto(v, sb)

proc pyStr*(v: Value): string =
  ## Returns the value as template output text. Strings pass through unchanged, everything else
  ## takes its Python `str()` form, undefined rendering empty. One string is materialized, presized by a measuring pass.
  if v.kind == vkStr:
    return v.s
  materializeStr(pyStrInto, v)

func reprQuoted(sb: var Cursor, s: string) =
  ## Writes Python's single-quoted repr of `s`, the form container reprs use for keys
  ## and string items, non-escaped bytes passing through raw.
  sb.add '\''
  for c in s:
    case c
    of '\\': sb.add "\\\\"
    of '\'': sb.add "\\'"
    of '\n': sb.add "\\n"
    of '\r': sb.add "\\r"
    of '\t': sb.add "\\t"
    else: sb.add c
  sb.add '\''

proc pyReprInto(v: Value, sb: var Cursor) =
  ## Writes Python's `repr()` of `v` into `sb`, recursively.
  case v.kind
  of vkStr: reprQuoted(sb, v.s)
  of vkSeq:
    sb.add '['
    for i in 0 ..< v.xs.items.len:
      if i > 0:
        sb.add ", "
      pyReprInto(v.xs.items[i], sb)
    sb.add ']'
  of vkDict, vkNs:
    sb.add '{'
    for i in 0 ..< v.d.keys.len:
      if i > 0:
        sb.add ", "
      reprQuoted(sb, v.d.keys[i])
      sb.add ": "
      pyReprInto(v.d.vals[i], sb)
    sb.add '}'
  of vkLoop: sb.add "<LoopContext>"
  of vkMacro:
    sb.add "<macro "
    sb.addInt v.mc.name.int64
    sb.add '>'
  else:
    pyStrInto(v, sb)

proc pyRepr*(v: Value): string =
  ## Returns Python's `repr()`, the rendering a template sees when it stringifies a container.
  materializeStr(pyReprInto, v)

func hex4(sb: var Cursor, c: int) =
  ## Appends `c` as four uppercase hex digits, the payload a `\uXXXX` escape carries.
  const digits = "0123456789ABCDEF"
  for sh in countdown(12, 0, 4):
    sb.add digits[(c shr sh) and 0xF]

proc jsonEscapeInto(sb: var Cursor, s: string, ensureAscii, html: bool) =
  ## Writes the JSON string body of `s`, mirroring `json.dumps` escaping, ASCII-escaped when
  ## `ensureAscii` is set, `html` additionally escaping `<`, `>`, `&` and `'`.
  for r in s.runes:
    let c = ord(r)
    if c == ord('"'):
      sb.add "\\\""
    elif c == ord('\\'):
      sb.add "\\\\"
    elif c == 10:
      sb.add "\\n"
    elif c == 13:
      sb.add "\\r"
    elif c == 9:
      sb.add "\\t"
    elif c == 8:
      sb.add "\\b"
    elif c == 12:
      sb.add "\\f"
    elif c < 32:
      sb.add "\\u"
      hex4(sb, c)
    elif c > 127 and ensureAscii:
      if c > 0xFFFF:
        let u = c - 0x10000
        sb.add "\\u"
        hex4(sb, 0xD800 + (u shr 10))
        sb.add "\\u"
        hex4(sb, 0xDC00 + (u and 0x3FF))
      else:
        sb.add "\\u"
        hex4(sb, c)
    elif html:
      case c
      of ord('<'): sb.add "\\u003c"
      of ord('>'): sb.add "\\u003e"
      of ord('&'): sb.add "\\u0026"
      of ord('\''): sb.add "\\u0027"
      else: sb.addRune r
    else:
      sb.addRune r

proc toJsonBody(v: Value, sb: var Cursor, opts: JsonOpts) =
  ## Writes the `tojson` rendering of `v` into `sb`, recursively.
  case v.kind
  of vkUndefined, vkNone: sb.add "null"
  of vkBool: sb.add(if v.b: "true" else: "false")
  of vkInt: sb.addInt v.i
  of vkFloat: sb.addFloat v.f
  of vkStr:
    sb.add '"'
    jsonEscapeInto(sb, v.s, opts.ensureAscii, true)
    sb.add '"'
  of vkSeq:
    if v.xs.items.len == 0:
      sb.add "[]"
      return
    sb.add '['
    for i in 0 ..< v.xs.items.len:
      if i > 0:
        sb.add opts.itemSep
      toJsonBody(v.xs.items[i], sb, opts)
    sb.add ']'
  of vkDict, vkNs:
    if v.d.keys.len == 0:
      sb.add "{}"
      return
    sb.add '{'
    for i in 0 ..< v.d.keys.len:
      if i > 0:
        sb.add opts.itemSep
      sb.add '"'
      jsonEscapeInto(sb, v.d.keys[i], opts.ensureAscii, true)
      sb.add '"'
      sb.add opts.kvSep
      toJsonBody(v.d.vals[i], sb, opts)
    sb.add '}'
  of vkLoop, vkMacro:
    sb.add '"'
    jsonEscapeInto(sb, pyStr(v), opts.ensureAscii, true)
    sb.add '"'

proc toJson*(v: Value, opts = JsonOpts()): string =
  ## Returns the `tojson` filter rendering:
  ##   non-ASCII raw UTF-8 unless the template passes `ensure_ascii`, Jinja's HTML escaping
  ##   applied as the filter's post-pass, one string materialized bounded by the size.
  var sb = measureBuf()
  toJsonBody(v, sb, opts)
  result = newString(sb.len)
  var dst = over(result)
  toJsonBody(v, dst, opts)
