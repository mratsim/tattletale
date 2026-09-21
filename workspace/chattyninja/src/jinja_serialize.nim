# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Python `str()`/`repr()` and `tojson` serialization of `JinjaVal`.
## - `Ser` is the defunctional serializer, rendering byte by byte into a caller-owned
##   window with every pause point in its fields, so a drain resumed through the same
##   `Ser` never re-emits a byte
## - the internal renderers `pyStrInto` and `pyReprInto` write a value into a caller-owned `Cursor`
## - `serReset`, `serDone` and `pullSer` drive a held `Ser`, `pyStr`, `toJson` and `pullJSON`
##   are the one-call consumer entry points

import std/unicode
import jinja_data_model

const
  SerChunkCap = 40
    ## Capacity of the literal queue, bounding every queued literal. The longest `tojson`
    ## escape is the astral surrogate pair (12 bytes), the longest atom rendering a float repr
    ## (26 bytes), the macro form `<macro ` plus one int64 plus `>` (30 bytes with quotes).

  SerStartCap = 256
    ## First `serString` drain buffer, doubled by `setLen` until the rendering completes.

type
  SerMode* = enum
    ## Mode of a serializer, `smStr` rendering Python `str()` and `repr()` output text,
    ## `smJson` rendering the `tojson` filter form.
    smStr, smJson

  SerWalk = enum
    ## Character unit a quoted string body advances by, whole runes for the `tojson` escaping
    ## rules and single bytes for the Python repr escaping rules.
    wkRune, wkByte

  SerAfter = enum
    ## Activity following the string body being written, closing the value's quote or closing
    ## a mapping key's quote before the separator and the entry value.
    saValue, saKey

  SerNext = enum
    ## Activity a finished separator hands to, rendering the value in `v` or the mapping key
    ## in `s` as a quoted string body.
    nxDispatch, nxKey

  SerPhase = enum
    ## Pending activity of a serializer. Queued chunk bytes and the pending separator drain
    ## before the phase advances.
    spDispatch, spStr, spRaw, spSep, spClose, spDone

  SerFrame = object
    ## One open container on the serializer's stack.
    val: JinjaVal
      ## the container being rendered
    idx: int
      ## entry the serializer writes next

  Ser* = object
    ## Defunctional serializer for one `JinjaVal`, rendering byte by byte into the caller's
    ## window with every pause point in the fields below, so a drain resumed through the same
    ## `Ser` never re-emits a byte:
    ## - queued literals and separators drain from the chunk buffer and `sep`, unquoted
    ##   string bodies copy straight from `s`
    ## - quoted string bodies, container brackets and entries advance through the phases
    ## - a str-mode `~` tree flattens into `concatTail`, operands dispatching one after
    ##   the other, and a lazy `range` renders its list form arithmetically, one element per index
    mode: SerMode
    opts: JsonOpts
      ## `tojson` knobs, read in `smJson` mode only
    phase: SerPhase
    v: JinjaVal
      ## value the `spDispatch` phase renders
    s: string
      ## string body the `spStr` and `spRaw` phases write
    spos: int
      ## bytes of `s` already written
    send: int
      ## exclusive end of the `s` body, `s.len` for a whole string, a cut's `hi` bound
      ## when the raw phase streams only the cut's sub-span
    walk: SerWalk
    quoted: bool
      ## the `spStr` body sits between quotes the serializer itself writes
    after: SerAfter
    nxt: SerNext
    sep: string
      ## separator the `spSep` phase writes
    sepos: int
      ## bytes of `sep` already written
    buf: array[SerChunkCap, char]
      ## literal queue draining byte by byte
    blen, bpos: int
      ## queued bytes in `buf` and the read position
    stack: seq[SerFrame]
      ## open containers, outermost first
    concatTail: seq[JinjaVal]
      ## operands of a str-mode `~` tree, taken over whole at reset, never sliced
    concatPos: int
      ## index of the next `concatTail` operand after the leaf in `v`
    closeSeq: bool
      ## the `spClose` phase writes a sequence bracket, else a mapping bracket

func pyStrInto(sb: var Cursor, v: JinjaVal)
func pyReprInto(sb: var Cursor, v: JinjaVal, depth: int)

func pyStrInto(sb: var Cursor, v: JinjaVal) =
  ## Writes the value as template output text into `sb`:
  ## - strings pass through, cuts stream their surviving span, scalars format in place,
  ##   undefined renders empty
  ## - containers take their Python `repr()` form
  ## Raises when `sb` cannot hold the rendering, never growing it.
  case v.kind
  of vkUndefined: discard
  of vkNone: sb.add "None"
  of vkBool: sb.add(if v.b: "True" else: "False")
  of vkInt: sb.addInt v.i
  of vkFloat: sb.addFloat v.f
  of vkStr: sb.add v.s
  of vkCut: sb.add v.raw.toOpenArray(v.lo, v.hi - 1)
  of vkSeq, vkDict, vkNs, vkLoop, vkMacro, vkRange: sb.pyReprInto(v, 0)
  of vkCall: raise jinjaErr("a macro call result must be rendered before stringification")
  of vkConcat: raise jinjaErr("a concat must be rendered in emit position before stringification")

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

func pyReprInto(sb: var Cursor, v: JinjaVal, depth: int) =
  ## Writes Python's `repr()` of `v` into `sb`, one level per container, `depth`
  ## counting toward `ValueDepthCap` and raising with `NoOffset` past it, which is
  ## what closes deep and cyclic value graphs on the recursive repr path.
  if depth > ValueDepthCap:
    raise jinjaErr("value nesting deeper than ValueDepthCap = " & $ValueDepthCap &
        " cannot be serialized")
  case v.kind
  of vkStr: reprQuoted(sb, v.s)
  of vkCut: reprQuoted(sb, materializeVal(v).s)
  of vkSeq:
    sb.add '['
    for i in 0 ..< v.xs.items.len:
      if i > 0:
        sb.add ", "
      sb.pyReprInto(v.xs.items[i], depth + 1)
    sb.add ']'
  of vkRange:
    sb.add '['
    for i in 0 ..< v.r.rangeLen:
      if i > 0:
        sb.add ", "
      sb.addInt(rangeAt(v.r, i).i)
    sb.add ']'
  of vkDict, vkNs:
    sb.add '{'
    for i in 0 ..< v.d.keys.len:
      if i > 0:
        sb.add ", "
      reprQuoted(sb, v.d.keys[i])
      sb.add ": "
      sb.pyReprInto(v.d.vals[i], depth + 1)
    sb.add '}'
  of vkLoop: sb.add "<LoopContext>"
  of vkMacro:
    sb.add "<macro "
    sb.addInt v.mc.name.int64
    sb.add '>'
  else:
    sb.pyStrInto(v)

func hex4(sb: var Cursor, c: int) =
  ## Appends `c` as four uppercase hex digits, the payload a `\uXXXX` escape carries.
  const Digits = "0123456789ABCDEF"
  for sh in countdown(12, 0, 4):
    sb.add Digits[(c shr sh) and 0xF]


func serQueue(js: var Ser, s: string) =
  ## Queues literal bytes for draining, `s` at most `SerChunkCap` bytes long.
  ## A longer literal raises `JinjaError` naming the queue's capacity.
  js.blen = s.len
  js.bpos = 0
  if s.len > SerChunkCap:
    raise jinjaErr("serializer literal of " & $s.len & " bytes exceeds the " &
        $SerChunkCap & "-byte queue")
  if s.len > 0:
    copyMem(addr js.buf[0], unsafeAddr s[0], s.len)

func serQueueRune(js: var Ser, r: Rune) =
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

func serQueueByte(js: var Ser, c: char) =
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

func serFinish(js: var Ser) =
  ## Closes the value just rendered. The enclosing container advances to its next entry,
  ## nested containers closing outward, the rendering completing once the stack empties.
  if js.stack.len == 0:
    if js.concatPos < js.concatTail.len:
      js.v = move js.concatTail[js.concatPos]
      inc js.concatPos
      js.phase = spDispatch
      return
    js.phase = spDone
    return
  inc js.stack[^1].idx
  let f = js.stack[^1]
  let n = case f.val.kind
    of vkSeq: f.val.xs.items.len
    of vkRange: f.val.r.rangeLen
    else: f.val.d.keys.len
  if f.idx < n:
    js.sep = if js.mode == smJson: js.opts.itemSep else: ", "
    js.sepos = 0
    case f.val.kind
    of vkSeq:
      js.v = f.val.xs.items[f.idx]
      js.nxt = nxDispatch
    of vkRange:
      js.v = f.val.r.rangeAt(f.idx)
      js.nxt = nxDispatch
    else:
      js.s = f.val.d.keys[f.idx]
      js.nxt = nxKey
    js.phase = spSep
  else:
    js.closeSeq = f.val.kind in {vkSeq, vkRange}
    discard js.stack.pop()
    js.phase = spClose

func serDispatch(js: var Ser) =
  ## Renders the value in `v`, one literal or string body at a time.
  ##
  ## Each container stack entry pushed below counts the value-graph depth toward
  ## `ValueDepthCap`, a breach raising a `JinjaError` with `NoOffset`, `Ser`
  ## carrying no template location:
  ## - deeply nested data raises past the cap
  ## - a cyclic graph, whose nesting is unbounded, raises the same way
  let v = js.v
  template capDepth =
    ## One open container stack entry per nesting level.
    if js.stack.len >= ValueDepthCap:
      raise jinjaErr("value nesting deeper than ValueDepthCap = " & $ValueDepthCap &
          " cannot be serialized")
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
      js.send = v.s.len
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
  of vkCut:
    if js.mode == smStr and js.stack.len == 0:
      # Emit position streams the cut's surviving bytes straight from the shared buffer.
      js.s = v.raw
      js.spos = v.lo.int
      js.send = v.hi.int
      js.blen = 0
      js.bpos = 0
      js.phase = spRaw
    else:
      # Quoted forms (container reprs, tojson) re-compute the cut's string first.
      js.v = materializeVal(v)
      serDispatch(js)
  of vkLoop:
    # tojson escapes the `<` and `>` of the `<LoopContext>` form.
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
  of vkCall:
    raise jinjaErr("a macro call result must be rendered before serialization")
  of vkConcat:
    raise jinjaErr("a concat must be rendered in emit position before serialization")
  of vkSeq:
    if v.xs.items.len == 0:
      serQueue(js, "[]")
      serFinish(js)
    else:
      capDepth
      js.stack.add(SerFrame(val: v, idx: 0))
      serQueue(js, "[")
      js.v = v.xs.items[0]
  of vkRange:
    if v.r.rangeLen == 0:
      serQueue(js, "[]")
      serFinish(js)
    else:
      capDepth
      js.stack.add(SerFrame(val: v, idx: 0))
      serQueue(js, "[")
      js.v = v.r.rangeAt(0)
  of vkDict, vkNs:
    if v.d.keys.len == 0:
      serQueue(js, "{}")
      serFinish(js)
    else:
      capDepth
      js.stack.add(SerFrame(val: v, idx: 0))
      serQueue(js, if js.mode == smJson: "{\"" else: "{'")
      js.s = v.d.keys[0]
      js.spos = 0
      js.quoted = true
      js.walk = if js.mode == smJson: wkRune else: wkByte
      js.after = saKey
      js.phase = spStr

func serStep(js: var Ser) =
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

func serReset*(js: var Ser, v: sink JinjaVal, mode: SerMode, opts = JsonOpts()) =
  ## Repositions `js` before the first byte of `v`'s rendering:
  ## - takes the value over from the caller for the drain
  ## - keeps the container stack's capacity for the next derived value rendered through it
  js.mode = mode
  js.opts = opts
  js.phase = spDispatch
  js.s = ""
  js.spos = 0
  js.send = 0
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
  js.concatTail.setLen(0)
  js.concatPos = 0
  if mode == smStr and v.kind == vkConcat:
    # Operand stack taken over whole, the cursor walks it by index, leaf 0
    # dispatched directly and 1 onward after each completion, render order
    # with no sliced copy.
    js.concatTail = v.xs.items
    js.v = js.concatTail[0]
    js.concatPos = 1
  else:
    js.v = move v

func serValue(v: JinjaVal, mode: SerMode, opts = JsonOpts()): Ser =
  ## Returns a serializer positioned before the first byte of `v`'s rendering.
  result = Ser(mode: mode, opts: opts)
  serReset(result, v, mode, opts)

func serDone*(js: Ser): bool =
  ## Returns whether the rendering is complete and every queued byte drained.
  js.phase == spDone and js.bpos == js.blen and js.sepos == js.sep.len

func pullSer*(js: var Ser, dst: var openArray[char]): int =
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
      let n = min(dst.len - result, js.send - js.spos)
      if n > 0:
        copyMem(addr dst[result], unsafeAddr js.s[js.spos], n)
        inc js.spos, n
        inc result, n
      if js.spos == js.send:
        serFinish(js)
    else:
      serStep(js)

func serString(js: var Ser): string =
  ## Returns the rendering as one fresh string, the caller-side drain-and-grow form.
  ## Buffer growth starts at 256 bytes and doubles until the rendering completes.
  var cap = SerStartCap
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

func pullJSON*(dst: var openArray[char], v: JinjaVal, opts = JsonOpts()): int =
  ## Returns the count of `tojson` rendering bytes of `v` written into `dst`.
  ## Window-sized, the count stopping at `dst.len` when the rendering does not fit:
  ## - the unwritten tail is dropped, so a rendering that may exceed the window drains
  ##   through `pullSer` over a held `Ser`
  var js = serValue(v, smJson, opts)
  pullSer(js, dst)

func pyStr*(v: JinjaVal): string =
  ## Returns the value as template output text. Strings pass through unchanged, everything
  ## else takes its Python `str()` form, undefined rendering empty.
  ## Caller-side drain-and-grow over the serializer, one buffer doubling until done.
  if v.kind == vkStr:
    return v.s
  var js = serValue(v, smStr)
  serString(js)

func toJson*(v: JinjaVal, opts = JsonOpts()): string =
  ## Returns the `tojson` filter rendering, non-ASCII as raw UTF-8 unless the template
  ## passes `ensure_ascii`, Jinja's HTML escaping applied as the filter's post-pass.
  ## Caller-side drain-and-grow over the serializer, no presize pass.
  var js = serValue(v, smJson, opts)
  serString(js)