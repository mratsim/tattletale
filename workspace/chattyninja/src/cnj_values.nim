# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Value model of the chattyninja engine. Carries Jinja-tier values with truthiness and equality,
# plus ordering, stringification, and the Python-repr/JSON renderings a template observes.
# No dependency on the node arena, so the value tier is testable without a compiled template.

import std/[strutils, unicode]
import cnj_errors, cnj_strbuf

type
  ValueKind* = enum
    ## Jinja value tiers the corpus reaches. Float is carried for JSON fidelity only.
    ## No template in the corpus does float arithmetic, so it is not implemented.
    vkUndefined
    vkNone
    vkBool
    vkInt
    vkFloat
    vkStr
    vkSeq
    vkDict
    vkNs
    vkLoop
    vkMacro

  SeqVal* = ref object
    ## Shared sequence of values, the `vkSeq` payload.
    items*: seq[Value]

  DictVal* = ref object
    ## Insertion-ordered mapping. `vkNs` reuses it, and every holder observes changes because
    ## the object is shared by reference.
    keys*: seq[string]
    vals*: seq[Value]

  LoopState* = ref object
    ## Materialized iterable plus the cursor `loop.*` reads. Shared between the driver's frame
    ## and the `loop` value bound in the loop scope, so both readers see one cursor.
    items*: seq[Value]
    idx*: int

  MacroVal* = ref object
    ## A bound macro. `node` is the `nkMacroDef` arena index and the body's terminators land
    ## on it, so a call detects its end by arriving back there. Name and parameters are read
    ## from the definition node's payload slots at call time.
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
    ## `ensureAscii` defaults to false to match the recording environment, which emits
    ## non-ASCII as raw UTF-8 rather than `\uXXXX` escapes.
    ensureAscii*: bool = false
    itemSep*: string = ", "
    kvSep*: string = ": "

func undefinedVal*(): Value =
  ## Returns the absent-binding value, which renders empty, fails truthiness, and equals only itself.
  Value(kind: vkUndefined)

func noneVal*(): Value =
  ## Returns Python's `None`, which renders `None`, fails truthiness, and is distinct from undefined.
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
  ## Returns the value under `key`, or undefined when absent. Absence is a value, never an error:
  ## that is what makes `is defined` and `.get` fall back work.
  ## Comparison reads `key` in place, so a lookup by template span materialises no string.
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
  ## Returns Jinja `==`. Numbers compare across tiers, containers compare element-wise, and undefined
  ## equals only undefined.
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
  ## Returns -1, 0 or 1 for an ordering comparison. Numbers order numerically, text orders
  ## by codepoint, and anything else is a template error, matching Jinja.
  if a.kind in {vkInt, vkFloat} and b.kind in {vkInt, vkFloat}:
    let ai = if a.kind == vkInt: float64 a.i else: a.f
    let bi = if b.kind == vkInt: float64 b.i else: b.f
    return if ai < bi: -1 elif ai > bi: 1 else: 0
  if a.kind == vkStr and b.kind == vkStr:
    return cmp(a.s, b.s)
  raise err("`<` and `>` need two numbers or two strings, got " & $a.kind & " and " & $b.kind)

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
    needle.kind == vkStr and needle.s in haystack.s
  else:
    raise err("`in` needs a sequence, mapping or string on the right, got " & $haystack.kind)

func pyStrip*(s, chars: string, left, right: bool): string =
  ## Returns `s` with leading and/or trailing characters in `chars` removed, the way Python's
  ## `str.strip`, `lstrip` and `rstrip` do. Empty `chars` selects the whitespace set.
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
  s[a ..< b]

proc pyFloat*(f: float64): string =
  ## Returns Python's `str()` for a float:
  ##   integral values keep one decimal place.
  ## A few small allocations, because the measure pass and the render each run `$f`
  ## inside `StrBuf.addFloat`, which float emits pay through scratch as well.
  var sb: StrBuf
  sb.addFloat f
  result = newString(sb.len)
  var dst = over(result)
  dst.addFloat f

proc pyRepr*(v: Value): string

proc pyStrInto*(v: Value, sb: var StrBuf)
proc pyReprInto(v: Value, sb: var StrBuf)

proc pyStrInto*(v: Value, sb: var StrBuf) =
  ## Writes the value as template output text into `sb`:
  ## - strings pass through, scalars format in place
  ## - containers take their Python `repr()` form
  ## - undefined renders empty
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
  ## Returns the value as template output text:
  ##   strings pass through unchanged, everything else
  ## takes its Python `str()` form, undefined renders empty.
  ## Containers materialize one string, presized by a measuring pass.
  result = case v.kind
  of vkUndefined: ""
  of vkNone: "None"
  of vkBool: (if v.b: "True" else: "False")
  of vkInt: $v.i
  of vkFloat: pyFloat(v.f)
  of vkStr: v.s
  of vkSeq, vkDict, vkNs, vkLoop, vkMacro: pyRepr(v)

func reprQuoted(sb: var StrBuf, s: string) =
  ## Writes Python's single-quoted repr of `s`, the form container reprs use for keys
  ## and string items. Non-escaped bytes pass through raw.
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

proc pyReprInto(v: Value, sb: var StrBuf) =
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
  ## One allocation, presized by a measuring pass over the writer.
  var sb: StrBuf
  pyReprInto(v, sb)
  result = newString(sb.len)
  var dst = over(result)
  pyReprInto(v, dst)

func hex4(sb: var StrBuf, c: int) =
  ## Appends `c` as four uppercase hex digits, the payload a `\uXXXX` escape carries.
  const digits = "0123456789ABCDEF"
  sb.add digits[(c shr 12) and 0xF]
  sb.add digits[(c shr 8) and 0xF]
  sb.add digits[(c shr 4) and 0xF]
  sb.add digits[c and 0xF]

proc jsonEscapeInto(sb: var StrBuf, s: string, ensureAscii, html: bool) =
  ## Writes the JSON string body of `s`, mirroring `json.dumps` escaping, ASCII-escaped
  ## when `ensureAscii` is set. `html` additionally escapes `<`, `>`, `&` and `'`.
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

proc toJsonBody(v: Value, sb: var StrBuf, opts: JsonOpts) =
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
  ## Returns the `tojson` filter rendering.
  ## - non-ASCII renders as raw UTF-8 unless the template passes `ensure_ascii`
  ## - Jinja's HTML escaping applies, as the filter's post-pass
  ## A tojson value materializes one string, one allocation bounded by the value size,
  ## whether it is emitted, bound to a name or stored.
  var sb: StrBuf
  toJsonBody(v, sb, opts)
  result = newString(sb.len)
  var dst = over(result)
  toJsonBody(v, dst, opts)
