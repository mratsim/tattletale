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

proc pyStrInto*(v: Value, sb: var Cursor)
proc pyReprInto(v: Value, sb: var Cursor)

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

func hex4(sb: var Cursor, c: int) =
  ## Appends `c` as four uppercase hex digits, the payload a `\uXXXX` escape carries.
  const digits = "0123456789ABCDEF"
  for sh in countdown(12, 0, 4):
    sb.add digits[(c shr sh) and 0xF]

type
  SerMode* = enum
    ## Mode of a serializer, `smStr` rendering Python `str()` and `repr()` output text,
    ## `smJson` rendering the `tojson` filter form.
    smStr, smJson

  SerWalk* = enum
    ## Character unit a quoted string body advances by, whole runes for the `tojson` escaping
    ## rules and single bytes for the Python repr escaping rules.
    wkRune, wkByte

  SerAfter* = enum
    ## Activity following the string body being written, closing the value's quote or closing
    ## a mapping key's quote before the separator and the entry value.
    saValue, saKey

  SerNext* = enum
    ## Activity a finished separator hands to, rendering the value in `v` or the mapping key
    ## in `s` as a quoted string body.
    nxDispatch, nxKey

  SerPhase* = enum
    ## Pending activity of a serializer. Queued chunk bytes and the pending separator drain
    ## before the phase advances.
    spDispatch, spStr, spRaw, spSep, spClose, spDone

  SerFrame* = object
    ## One open container on the serializer's stack.
    val*: Value
      ## the container being rendered
    idx*: int
      ## entry the serializer writes next

  Ser* = object
    ## Defunctional serializer for one `Value`, rendering byte by byte into the caller's
    ## window with every pause point in the fields below, so a drain resumed through the same
    ## `Ser` never re-emits a byte:
    ## - queued literals and separators drain from the chunk buffer and `sep`
    ## - unquoted string bodies copy straight from `s`
    ## - quoted string bodies, container brackets and entries advance through the phases
    mode*: SerMode
    opts*: JsonOpts
      ## `tojson` knobs, read in `smJson` mode only
    phase*: SerPhase
    v*: Value
      ## value the `spDispatch` phase renders
    s*: string
      ## string body the `spStr` and `spRaw` phases write
    spos*: int
      ## bytes of `s` already written
    walk*: SerWalk
    quoted*: bool
      ## the `spStr` body sits between quotes the serializer itself writes
    after*: SerAfter
    nxt*: SerNext
    sep*: string
      ## separator the `spSep` phase writes
    sepos*: int
      ## bytes of `sep` already written
    buf*: array[40, char]
      ## literal queue draining byte by byte
    blen*, bpos*: int
      ## queued bytes in `buf` and the read position
    stack*: seq[SerFrame]
      ## open containers, outermost first
    closeSeq*: bool
      ## the `spClose` phase writes a sequence bracket, else a mapping bracket

const serChunkCap = 40
  ## Capacity of the literal queue, bounding every queued literal. The longest `tojson`
  ## escape is the astral surrogate pair (12 bytes), the longest atom rendering a float repr
  ## (26 bytes), the macro form `<macro ` plus one int64 plus `>` (30 bytes with quotes).

proc serQueue(js: var Ser, s: string) =
  ## Queues literal bytes for draining, `s` at most `serChunkCap` bytes long.
  js.blen = s.len
  js.bpos = 0
  if s.len > 0:
    copyMem(addr js.buf[0], unsafeAddr s[0], s.len)

proc serQueueRune(js: var Ser, r: Rune) =
  ## Queues one rune per the `tojson` escaping rules, the filter's HTML post-pass included.
  var c = Cursor(buf: toOpenArray(js.buf, 0, js.buf.high), len: 0)
  let x = ord(r)
  if x == ord('"'):
    c.add "\\\""
  elif x == ord('\\'):
    c.add "\\\\"
  elif x == 10:
    c.add "\\n"
  elif x == 13:
    c.add "\\r"
  elif x == 9:
    c.add "\\t"
  elif x == 8:
    c.add "\\b"
  elif x == 12:
    c.add "\\f"
  elif x < 32:
    c.add "\\u"
    hex4(c, x)
  elif x > 127 and js.opts.ensureAscii:
    if x > 0xFFFF:
      let u = x - 0x10000
      c.add "\\u"
      hex4(c, 0xD800 + (u shr 10))
      c.add "\\u"
      hex4(c, 0xDC00 + (u and 0x3FF))
    else:
      c.add "\\u"
      hex4(c, x)
  elif x == ord('<'):
    c.add "\\u003c"
  elif x == ord('>'):
    c.add "\\u003e"
  elif x == ord('&'):
    c.add "\\u0026"
  elif x == ord('\''):
    c.add "\\u0027"
  else:
    c.addRune r
  js.blen = c.len
  js.bpos = 0

proc serQueueByte(js: var Ser, c: char) =
  ## Queues one byte of a Python-repr string body, escaping the five characters Python's `repr()` escapes.
  case c
  of '\\': serQueue(js, "\\\\")
  of '\'': serQueue(js, "\\'")
  of '\n': serQueue(js, "\\n")
  of '\r': serQueue(js, "\\r")
  of '\t': serQueue(js, "\\t")
  else:
    js.buf[0] = c
    js.blen = 1
    js.bpos = 0

proc serFinish(js: var Ser) =
  ## Closes the value just rendered. The enclosing container advances to its next entry,
  ## nested containers closing outward, the rendering completing once the stack empties.
  if js.stack.len == 0:
    js.phase = spDone
    return
  inc js.stack[^1].idx
  let f = js.stack[^1]
  let n = if f.val.kind == vkSeq: f.val.xs.items.len else: f.val.d.keys.len
  if f.idx < n:
    js.sep = if js.mode == smJson: js.opts.itemSep else: ", "
    js.sepos = 0
    if f.val.kind == vkSeq:
      js.v = f.val.xs.items[f.idx]
      js.nxt = nxDispatch
    else:
      js.s = f.val.d.keys[f.idx]
      js.nxt = nxKey
    js.phase = spSep
  else:
    js.closeSeq = f.val.kind == vkSeq
    discard js.stack.pop()
    js.phase = spClose

proc serDispatch(js: var Ser) =
  ## Renders the value in `v`, one literal or string body at a time.
  let v = js.v
  case v.kind
  of vkUndefined:
    if js.mode == smJson:
      serQueue(js, "null")
    serFinish(js)
  of vkNone:
    serQueue(js, if js.mode == smJson: "null" else: "None")
    serFinish(js)
  of vkBool:
    serQueue(js, if js.mode == smJson: (if v.b: "true" else: "false")
        else: (if v.b: "True" else: "False"))
    serFinish(js)
  of vkInt, vkFloat:
    var c = Cursor(buf: toOpenArray(js.buf, 0, js.buf.high), len: 0)
    if v.kind == vkInt:
      c.addInt v.i
    else:
      c.addFloat v.f
    js.blen = c.len
    js.bpos = 0
    serFinish(js)
  of vkStr:
    if js.mode == smStr and js.stack.len == 0:
      js.s = v.s
      js.spos = 0
      js.blen = 0
      js.bpos = 0
      js.phase = spRaw
    else:
      js.s = v.s
      js.spos = 0
      js.quoted = true
      js.walk = if js.mode == smJson: wkRune else: wkByte
      js.after = saValue
      serQueue(js, if js.mode == smJson: "\"" else: "'")
      js.phase = spStr
  of vkLoop:
    # The `<` and `>` of the context form carry the tojson filter's HTML escaping.
    serQueue(js, if js.mode == smJson: "\"\\u003cLoopContext\\u003e\"" else: "<LoopContext>")
    serFinish(js)
  of vkMacro:
    var c = Cursor(buf: toOpenArray(js.buf, 0, js.buf.high), len: 0)
    if js.mode == smJson:
      c.add "\"\\u003cmacro "
    else:
      c.add "<macro "
    c.addInt v.mc.name.int64
    if js.mode == smJson:
      c.add "\\u003e\""
    else:
      c.add '>'
    js.blen = c.len
    js.bpos = 0
    serFinish(js)
  of vkSeq:
    if v.xs.items.len == 0:
      serQueue(js, "[]")
      serFinish(js)
    else:
      js.stack.add(SerFrame(val: v, idx: 0))
      serQueue(js, "[")
      js.v = v.xs.items[0]
  of vkDict, vkNs:
    if v.d.keys.len == 0:
      serQueue(js, "{}")
      serFinish(js)
    else:
      js.stack.add(SerFrame(val: v, idx: 0))
      serQueue(js, if js.mode == smJson: "{\"" else: "{'")
      js.s = v.d.keys[0]
      js.spos = 0
      js.quoted = true
      js.walk = if js.mode == smJson: wkRune else: wkByte
      js.after = saKey
      js.phase = spStr

proc serStep(js: var Ser) =
  ## Advances one logical unit once the chunk buffer and the separator are drained.
  case js.phase
  of spDispatch:
    serDispatch(js)
  of spStr:
    if js.spos < js.s.len:
      if js.walk == wkByte:
        let c = js.s[js.spos]
        inc js.spos
        serQueueByte(js, c)
      else:
        var r: Rune
        fastRuneAt(js.s, js.spos, r, true)
        serQueueRune(js, r)
    else:
      case js.after
      of saValue:
        if js.quoted:
          serQueue(js, if js.mode == smJson: "\"" else: "'")
        serFinish(js)
      of saKey:
        serQueue(js, if js.mode == smJson: "\"" else: "'")
        js.sep = if js.mode == smJson: js.opts.kvSep else: ": "
        js.sepos = 0
        js.v = js.stack[^1].val.d.vals[js.stack[^1].idx]
        js.nxt = nxDispatch
        js.phase = spSep
  of spClose:
    serQueue(js, if js.closeSeq: "]" else: "}")
    serFinish(js)
  of spSep:
    case js.nxt
    of nxDispatch:
      js.phase = spDispatch
    of nxKey:
      js.quoted = true
      js.walk = if js.mode == smJson: wkRune else: wkByte
      js.after = saKey
      js.spos = 0
      serQueue(js, if js.mode == smJson: "\"" else: "'")
      js.phase = spStr
  of spRaw, spDone:
    discard

proc serValue*(v: Value, mode: SerMode, opts = JsonOpts()): Ser =
  ## Returns a serializer positioned before the first byte of `v`'s rendering.
  Ser(mode: mode, opts: opts, phase: spDispatch, v: v)

proc serReset*(js: var Ser, v: Value, mode: SerMode, opts = JsonOpts()) =
  ## Repositions `js` before the first byte of `v`'s rendering, keeping the container
  ## stack's capacity for the next derived value rendered through it.
  js.mode = mode
  js.opts = opts
  js.phase = spDispatch
  js.v = v
  js.s = ""
  js.spos = 0
  js.walk = wkRune
  js.quoted = false
  js.after = saValue
  js.nxt = nxDispatch
  js.sep = ""
  js.sepos = 0
  js.blen = 0
  js.bpos = 0
  js.closeSeq = false
  js.stack.setLen(0)

proc serDone*(js: Ser): bool =
  ## Returns whether the rendering is complete and every queued byte drained.
  js.phase == spDone and js.bpos == js.blen and js.sepos == js.sep.len

proc pullSer*(js: var Ser, dst: var openArray[char]): int =
  ## Returns the rendering's next bytes, written into `dst[0 ..< result]`.
  ## Resumable drain, every position advancing only past bytes already handed out, so a window
  ## smaller than the rendering drains across calls through the same `js`:
  ## - the rendering is complete when `serDone` holds
  ## - a call with window room always writes at least one byte unless the rendering is done
  while result < dst.len:
    if js.bpos < js.blen:
      let take = min(dst.len - result, js.blen - js.bpos)
      copyMem(addr dst[result], addr js.buf[js.bpos], take)
      inc js.bpos, take
      inc result, take
    elif js.sepos < js.sep.len:
      let take = min(dst.len - result, js.sep.len - js.sepos)
      copyMem(addr dst[result], unsafeAddr js.sep[js.sepos], take)
      inc js.sepos, take
      inc result, take
    elif js.phase == spDone:
      break
    elif js.phase == spRaw:
      let n = min(dst.len - result, js.s.len - js.spos)
      if n > 0:
        copyMem(addr dst[result], unsafeAddr js.s[js.spos], n)
        inc js.spos, n
        inc result, n
      if js.spos == js.s.len:
        serFinish(js)
    else:
      serStep(js)

proc serString(js: var Ser): string =
  ## Returns the rendering as one fresh string, the caller-side drain-and-grow form.
  ## Buffer growth starts at 256 bytes and doubles until the rendering completes.
  var cap = 256
  result = newString(cap)
  var written = 0
  while true:
    let n = pullSer(js, toOpenArray(result, written, cap - 1))
    inc written, n
    if serDone(js):
      break
    cap = cap * 2
    result.setLen(cap)
  result.setLen(written)

proc pullJSON*(dst: var openArray[char], v: Value, opts = JsonOpts()): int =
  ## Returns the count of `tojson` rendering bytes of `v` written into `dst`.
  ## Window-sized, the count stopping at `dst.len` when the rendering does not fit:
  ## - the unwritten tail is dropped, so a rendering that may exceed the window drains
  ##   through `pullSer` over a held `Ser`
  var js = serValue(v, smJson, opts)
  pullSer(js, dst)

proc pyStr*(v: Value): string =
  ## Returns the value as template output text. Strings pass through unchanged, everything
  ## else takes its Python `str()` form, undefined rendering empty.
  ## Caller-side drain-and-grow over the serializer, one buffer doubling until done.
  if v.kind == vkStr:
    return v.s
  var js = serValue(v, smStr)
  serString(js)

proc toJson*(v: Value, opts = JsonOpts()): string =
  ## Returns the `tojson` filter rendering, non-ASCII as raw UTF-8 unless the template
  ## passes `ensure_ascii`, Jinja's HTML escaping applied as the filter's post-pass.
  ## Caller-side drain-and-grow over the serializer, no presize pass.
  var js = serValue(v, smJson, opts)
  serString(js)
