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
import cnj_errors

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

func pyFloat*(f: float64): string =
  ## Returns Python's `str()` for a float:
  ##   integral values keep one decimal place.
  let r = $f
  if '.' notin r and 'e' notin r and 'E' notin r and 'n' notin r and 'i' notin r:
    return r & ".0"
  r

func pyBool(b: bool): string =
  if b: "True" else: "False"

func jsonBool(b: bool): string =
  if b: "true" else: "false"

func pyRepr*(v: Value): string

func pyStr*(v: Value): string =
  ## Returns the value as template output text:
  ##   strings pass through unchanged, everything else
  ## takes its Python `str()` form, undefined renders empty.
  result = case v.kind
  of vkUndefined: ""
  of vkNone: "None"
  of vkBool: pyBool(v.b)
  of vkInt: $v.i
  of vkFloat: pyFloat(v.f)
  of vkStr: v.s
  of vkSeq, vkDict, vkNs, vkLoop, vkMacro: pyRepr(v)

func quotePy(s: string): string =
  result = "'"
  for c in s:
    case c
    of '\\': result.add "\\\\"
    of '\'': result.add "\\'"
    of '\n': result.add "\\n"
    of '\r': result.add "\\r"
    of '\t': result.add "\\t"
    else: result.add c
  result.add '\''

func pyRepr*(v: Value): string =
  ## Returns Python's `repr()`, the rendering a template sees when it stringifies a container.
  case v.kind
  of vkStr: result = quotePy(v.s)
  of vkSeq:
    result = "["
    for i, x in v.xs.items:
      if i > 0:
        result.add ", "
      result.add pyRepr(x)
    result.add "]"
  of vkDict, vkNs:
    result = "{"
    for i, k in v.d.keys:
      if i > 0:
        result.add ", "
      result.add quotePy(k) & ": " & pyRepr(v.d.vals[i])
    result.add "}"
  of vkLoop:
    result = "<LoopContext>"
  of vkMacro:
    result = "<macro " & $v.mc.name & ">"
  else:
    result = pyStr(v)

func jsonEscape(s: string, ensureAscii: bool): string =
  ## Returns a JSON string body, ASCII-escaped when `ensureAscii` is set and raw UTF-8 otherwise.
  for r in s.runes:
    let c = ord(r)
    if c == ord('"'):
      result.add "\\\""
    elif c == ord('\\'):
      result.add "\\\\"
    elif c == 10:
      result.add "\\n"
    elif c == 13:
      result.add "\\r"
    elif c == 9:
      result.add "\\t"
    elif c == 8:
      result.add "\\b"
    elif c == 12:
      result.add "\\f"
    elif c < 32:
      result.add "\\u" & toHex(c, 4)
    elif c > 127 and ensureAscii:
      if c > 0xFFFF:
        let u = c - 0x10000
        result.add "\\u" & toHex(0xD800 + (u shr 10), 4)
        result.add "\\u" & toHex(0xDC00 + (u and 0x3FF), 4)
      else:
        result.add "\\u" & toHex(c, 4)
    else:
      result.add $r

func jsonEscapeHtml(s: string): string =
  ## Applies Jinja's `tojson` post-pass, escaping the four HTML-significant characters
  ## that `json.dumps` leaves literal.
  for c in s:
    case c
    of '<': result.add "\\u003c"
    of '>': result.add "\\u003e"
    of '&': result.add "\\u0026"
    of '\'': result.add "\\u0027"
    else: result.add c

func toJson*(v: Value, opts = JsonOpts()): string =
  ## Returns the `tojson` filter rendering:
  ##   non-ASCII renders as raw UTF-8 unless the template passes `ensure_ascii`,
  ## then Jinja's HTML escaping.
  case v.kind
  of vkUndefined, vkNone: result = "null"
  of vkBool: result = jsonBool(v.b)
  of vkInt: result = $v.i
  of vkFloat: result = pyFloat(v.f)
  of vkStr: result = jsonEscapeHtml("\"" & jsonEscape(v.s, opts.ensureAscii) & "\"")
  of vkSeq:
    if v.xs.items.len == 0:
      return "[]"
    result = "["
    for i, x in v.xs.items:
      if i > 0:
        result.add opts.itemSep
      result.add toJson(x, opts)
    result.add "]"
  of vkDict, vkNs:
    if v.d.keys.len == 0:
      return "{}"
    result = "{"
    for i, k in v.d.keys:
      if i > 0:
        result.add opts.itemSep
      result.add jsonEscapeHtml("\"" & jsonEscape(k, opts.ensureAscii) & "\"")
      result.add opts.kvSep
      result.add toJson(v.d.vals[i], opts)
    result.add "}"
  of vkLoop, vkMacro:
    result = jsonEscapeHtml("\"" & jsonEscape(pyStr(v), opts.ensureAscii) & "\"")
