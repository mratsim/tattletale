# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Scratch-window emit proof for the chattyninja engine.
##
## Caller-owned scratch turns container stringification and `~` concatenation into renders
## with no intermediate strings:
## - every ok corpus row renders byte-exact through pull with scratch attached
## - a scratch-window pending piece drains across pull calls byte-exact
## - an undersized scratch raises `ScratchError` naming the value kind and the capacity
##   in force, and after growth the repull stays byte-exact, a for-filter breach included
##
## Run:
##   $ ./workspace/chattyninja/run_tests.sh t_scratch

import std/[importutils, strutils]
import cnj_errors, cnj_types, cnj_values, cnj_parse, chattyninja
import rows

func bytesOf(buf: openArray[char], n: int): string =
  ## Copies the first `n` bytes of a pull window into a string.
  for i in 0 ..< n:
    result.add buf[i]

func listCtx(): Value =
  ## One context holding `m`, a mixed container whose repr exceeds a tiny scratch.
  var inner = DictVal()
  dictSet(inner, "alpha", strVal("one"))
  dictSet(inner, "beta", strVal("two"))
  dictSet(inner, "gamma", seqVal(@[strVal("x"), strVal("y"), strVal("z")]))
  var cd = DictVal()
  dictSet(cd, "m", dictVal(inner))
  dictVal(cd)

# Every ok corpus row through a pull render with scratch attached stays byte-exact.
# ---------------------------------------------------------------------------
block corpusRowsThroughScratch:
  const parseable = ["deepseekv2lite", "gemma3", "glm47flash", "gptoss20b", "kimi",
      "ling30", "mimo25", "mistral7bv01", "moonlight", "qwen3", "qwen35", "qwen36",
      "qwen38flashnext"]
  var checked = 0
  var gapSkipped = 0
  for suite in parseable:
    let src = templateSource(suite)
    let (nodes, tables) = parseTemplate(src)
    let m = Machine(jinja: src, nodes: nodes)
    for r in rows(suite):
      if r.expectError:
        continue
      var d = newDriver(r.context, r.clock)
      var scr = newSeq[char](4096)
      attachScratch(d, scr)
      var buf = newSeq[char](256)
      var got = ""
      var raised = false
      try:
        while true:
          let n = pull(m, tables, d, buf)
          if n == 0:
            break
          got.add bytesOf(buf, n)
      except CatchableError:
        raised = true
      if raised:
        # A declared engine gap raises before any bytes here. Classify it as a gap only
        # when the string render raises the same way, so a scratch-only failure cannot
        # hide behind the skip.
        var dStr = newDriver(r.context, r.clock)
        try:
          discard pullAll(m, tables, dStr)
        except CatchableError:
          raised = false
        doAssert not raised,
            suite & "/" & r.row & ": the scratch render raised where the string render did not"
        inc gapSkipped
      else:
        doAssert got == r.rendered,
            suite & "/" & r.row & ": the scratch pull render differs from the recorded bytes"
        inc checked
  doAssert checked == 55, "expected 55 rendered ok rows across 13 suites, checked " & $checked
  doAssert gapSkipped == 4, "expected 4 gap rows across 13 suites, skipped " & $gapSkipped

# A container emit rendered into scratch drains across pull calls byte-exact.
# ---------------------------------------------------------------------------
block scratchWindowDrain:
  let ctx = listCtx()
  let src = "{{ m }}"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = renderToString(src, ctx, 0.0)

  var d = newDriver(ctx, 0.0)
  var scr = newSeq[char](4096)
  attachScratch(d, scr)
  var window = newSeq[char](8)
  var acc = ""
  var scratchPieces = 0
  while true:
    if d.pend.kind == pkScratch:
      inc scratchPieces
    let n = pull(m, tables, d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == want, "the scratch-window drain differs from the string render"
  doAssert scratchPieces > 0, "no pull observed a pending scratch piece"

# An undersized scratch raises the typed error naming the value kind and capacity,
# and after growth the repull from the same driver is byte-exact.
# ---------------------------------------------------------------------------
block scratchBreach:
  let ctx = listCtx()
  let src = "{{ m }}"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = renderToString(src, ctx, 0.0)

  var d = newDriver(ctx, 0.0)
  var tiny = newSeq[char](16)
  attachScratch(d, tiny)
  var window = newSeq[char](64)
  var raised = false
  var capacity = 0
  var message = ""
  try:
    while true:
      let n = pull(m, tables, d, window)
      if n == 0:
        break
  except ScratchError as e:
    raised = true
    capacity = e.capacity
    message = e.msg
  doAssert raised, "an undersized scratch did not raise"
  doAssert capacity == 16, "the error named capacity " & $capacity
  doAssert "vkDict" in message, "the error did not name the value kind: " & message

  # Grow scratch on the same driver and repull, no bytes lost, no item re-emitted.
  var grown = newSeq[char](4096)
  attachScratch(d, grown)
  var acc = ""
  while true:
    let n = pull(m, tables, d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == want, "the repull after growth differs from the single-shot render"

# A scratch breach inside a for-filter rolls the loop candidate back, so the repull
# re-evaluates the next item and skips nothing, and every byte returned before the raise
# stays with the caller.
# ---------------------------------------------------------------------------
block filterBreachRepull:
  let longA = repeat("aaaa-", 10)
  var msgs = newSeq[Value]()
  for i in 0 ..< 3:
    var md = DictVal()
    dictSet(md, "a", strVal(longA & $i))
    dictSet(md, "b", strVal("tag" & $i))
    msgs.add dictVal(md)
  var cd = DictVal()
  dictSet(cd, "messages", seqVal(msgs))
  let ctx = dictVal(cd)
  let src = "{% for m in messages if m.a ~ m.b %}[{{ m.a }}]{% endfor %}"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = renderToString(src, ctx, 0.0)
  doAssert want == "[" & longA & "0][" & longA & "1][" & longA & "2]",
      "the string render is not the expected three items"

  # One-byte window. Item 0 renders before item 1's filter concat breaches, so the caller
  # holds item 0's full body at the raise, and a repull from the same driver then yields
  # the single-shot render exactly.
  var d = newDriver(ctx, 0.0)
  var tiny = newSeq[char](8)
  attachScratch(d, tiny)
  var acc = ""
  var raised = false
  var win1 = newSeq[char](1)
  try:
    while true:
      let n = pull(m, tables, d, win1)
      if n == 0:
        break
      acc.add bytesOf(win1, n)
  except ScratchError:
    raised = true
  doAssert raised, "the filter breach did not raise"
  doAssert acc == "[" & longA & "0]",
      "the caller-held bytes at the breach are not exactly item 0's body"

  var grown = newSeq[char](4096)
  attachScratch(d, grown)
  var rest = newSeq[char](64)
  while true:
    let n = pull(m, tables, d, rest)
    if n == 0:
      break
    acc.add bytesOf(rest, n)
  doAssert acc == want,
      "the repull after a filter breach differs from the single-shot render, items skipped"

  # Wide window. Item 0's body fits in the caller window, so the raise lands mid-call,
  # where per the pull contract the bytes written into the window in the failing call
  # are discarded and never reach the caller. A repull resumes after them.
  var dWide = newDriver(ctx, 0.0)
  var tinyWide = newSeq[char](8)
  attachScratch(dWide, tinyWide)
  var wide = newSeq[char](64)
  var wideAcc = ""
  var wideRaised = false
  try:
    while true:
      let n = pull(m, tables, dWide, wide)
      if n == 0:
        break
      wideAcc.add bytesOf(wide, n)
  except ScratchError:
    wideRaised = true
  doAssert wideRaised, "the wide-window filter breach did not raise"
  doAssert wideAcc == "", "the failing call returned bytes: <" & wideAcc & ">"
  var grownWide = newSeq[char](4096)
  attachScratch(dWide, grownWide)
  var wideRest = ""
  while true:
    let n = pull(m, tables, dWide, wide)
    if n == 0:
      break
    wideRest.add bytesOf(wide, n)
  doAssert wideRest == "[" & longA & "1][" & longA & "2]",
      "the wide-window repull did not resume after the discarded item 0 bytes"

  # Growing-concat shape. Here the filter concat length grows per item, so item 0
  # renders with the tiny scratch while item 1's concat exceeds it.
  # Caller-held bytes plus the repull equal the single-shot render, ruling out loss
  # and re-handing of delivered bytes across the raise.
  var msgsGrow = newSeq[Value]()
  for i in 0 ..< 3:
    var md = DictVal()
    dictSet(md, "a", strVal(repeat("p", 2 + 6 * i)))
    dictSet(md, "b", strVal("t" & $i))
    msgsGrow.add dictVal(md)
  var growCd = DictVal()
  dictSet(growCd, "messages", seqVal(msgsGrow))
  let growCtx = dictVal(growCd)
  let growSrc = "[{% for m in messages if m.a ~ m.b %}[{{ m.a }}]{% endfor %}"
  let (growNodes, growTables) = parseTemplate(growSrc)
  let growM = Machine(jinja: growSrc, nodes: growNodes)
  let growWant = renderToString(growSrc, growCtx, 0.0)
  doAssert growWant == "[[pp][pppppppp][pppppppppppppp]",
      "the growing-concat string render is not the expected three items"

  var dGrow = newDriver(growCtx, 0.0)
  var tinyGrow = newSeq[char](8)
  attachScratch(dGrow, tinyGrow)
  var growAcc = ""
  var growRaised = false
  var growWin1 = newSeq[char](1)
  try:
    while true:
      let n = pull(growM, growTables, dGrow, growWin1)
      if n == 0:
        break
      growAcc.add bytesOf(growWin1, n)
  except ScratchError:
    growRaised = true
  doAssert growRaised, "the growing-concat filter breach did not raise"
  doAssert growAcc == "[[pp]",
      "the caller-held bytes at the breach are not exactly the leading span plus item 0"

  var grownGrow = newSeq[char](4096)
  attachScratch(dGrow, grownGrow)
  var growRest = newSeq[char](64)
  while true:
    let n = pull(growM, growTables, dGrow, growRest)
    if n == 0:
      break
    growAcc.add bytesOf(growRest, n)
  doAssert growAcc == growWant,
      "the growing-concat repull differs from the single-shot render, bytes lost or re-handed"

# A container emit inside a macro body drains through the scratch emitter's
# capture-sink branch, and the captured string matches the string render.
# ---------------------------------------------------------------------------
block captureSinkScratch:
  let ctx = listCtx()
  let src = "{%- macro mm(v) -%}[{{ v }}]{%- endmacro -%}{{ mm(m) }}"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = renderToString(src, ctx, 0.0)

  var d = newDriver(ctx, 0.0)
  var scr = newSeq[char](4096)
  attachScratch(d, scr)
  doAssert pullAll(m, tables, d) == want,
      "the capture-sink scratch emit differs from the string render"

# A scratch breach inside a macro body unwinds through the macro runner's `finally`,
# and after growth the repull from the same driver is byte-exact.
# ---------------------------------------------------------------------------
block macroBreachRepull:
  let ctx = listCtx()
  let src = "{%- macro mm(v) -%}[{{ v }}]{%- endmacro -%}{{ mm(m) }}"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = renderToString(src, ctx, 0.0)

  var d = newDriver(ctx, 0.0)
  var tiny = newSeq[char](16)
  attachScratch(d, tiny)
  var raised = false
  var message = ""
  try:
    discard pullAll(m, tables, d)
  except ScratchError as e:
    raised = true
    message = e.msg
  doAssert raised, "the macro-body breach did not raise"
  doAssert "vkDict" in message, "the error did not name the value kind: " & message

  var grown = newSeq[char](4096)
  attachScratch(d, grown)
  doAssert pullAll(m, tables, d) == want,
      "the repull after a macro-body breach differs from the single-shot render"

# `tojson` with `ensure_ascii` exercises every escape shape, control characters included,
# plus the astral-codepoint surrogate pair. The recording environment never
# passes `ensure_ascii`, so this pins the engine rendering with uppercase hex digits.
# ---------------------------------------------------------------------------
block ensureAsciiEscapes:
  let raw = strVal("a\tb\rc\bd\x0Ce\x01f\"g\\h<i>j&k'lém😀n")
  doAssert toJson(raw, JsonOpts(ensureAscii: true)) ==
      "\"a\\tb\\rc\\bd\\fe\\u0001f\\\"g\\\\h\\u003ci\\u003ej\\u0026k\\u0027l\\u00E9m\\uD83D\\uDE00n\"",
      "the ensure_ascii rendering differs from the pinned escapes"
  doAssert toJson(raw) ==
      "\"a\\tb\\rc\\bd\\fe\\u0001f\\\"g\\\\h\\u003ci\\u003ej\\u0026k\\u0027lém😀n\"",
      "the raw-utf8 rendering differs from the pinned escapes"

# Allocation test. Runs last, only under `-d:nimAllocStats`, and a failing doAssert there
# hangs the run with no output instead of failing it.
# ---------------------------------------------------------------------------
when defined(nimAllocStats):
  privateAccess(AllocStats)

  template allocsOf(body: untyped): int =
    ## Counts `alloc` calls made by `body`, with allocator state warmed by the caller.
    let before = getAllocStats()
    body
    (getAllocStats() - before).allocCount

  func toolsVal(): Value =
    ## One function-tool definition, the bench tool schema shape.
    var cityProp = DictVal()
    dictSet(cityProp, "type", strVal("string"))
    var props = DictVal()
    dictSet(props, "city", dictVal(cityProp))
    var params = DictVal()
    dictSet(params, "type", strVal("object"))
    dictSet(params, "properties", dictVal(props))
    var fn = DictVal()
    dictSet(fn, "name", strVal("get_weather"))
    dictSet(fn, "description", strVal("Current weather for one city"))
    dictSet(fn, "parameters", dictVal(params))
    var tool = DictVal()
    dictSet(tool, "type", strVal("function"))
    dictSet(tool, "function", dictVal(fn))
    seqVal(@[dictVal(tool)])

  block allocProbe:
    let iters = 50

    # Direct tojson of the tool schema. The writer renders into one presized string, so
    # the whole serialization costs exactly one allocation per call.
    let tools = toolsVal()
    # warm-up call, excluded from the counted region
    discard toJson(tools)
    let tjAllocs = allocsOf:
      for _ in 0 ..< iters:
        discard toJson(tools)
    doAssert tjAllocs == iters, "toJson of the tool schema cost " & $(tjAllocs div iters) &
        " allocations per call against the measured one"

    # The same schema through the pull render with adequate scratch, driver setup uncounted:
    # the counted region holds only the pull loop, and the render costs two allocations.
    const tJson = "{{ tools|tojson }}"
    var cd = DictVal()
    dictSet(cd, "tools", tools)
    let ctx = dictVal(cd)
    let (nodes, tables) = parseTemplate(tJson)
    let m = Machine(jinja: tJson, nodes: nodes)
    let want = renderToString(tJson, ctx, 0.0)

    var scrWarm = newSeq[char](4096)
    var dWarm = newDriver(ctx, 0.0)
    attachScratch(dWarm, scrWarm)
    var bufWarm = newSeq[char](256)
    var warm = ""
    while true:
      let n = pull(m, tables, dWarm, bufWarm)
      if n == 0:
        break
      warm.add bytesOf(bufWarm, n)
    doAssert warm == want, "the scratch pull render differs from the string render"

    var scr = newSeq[char](4096)
    var d = newDriver(ctx, 0.0)
    attachScratch(d, scr)
    var buf = newSeq[char](256)
    var renderAllocs = 0
    for _ in 0 ..< iters:
      var di = newDriver(ctx, 0.0)
      attachScratch(di, scr)
      let renderCost = allocsOf:
        while true:
          let n = pull(m, tables, di, buf)
          if n == 0:
            break
      renderAllocs += renderCost
    doAssert renderAllocs == 2 * iters, "the tojson pull render cost " &
        $(renderAllocs div iters) & " allocations per render against the measured two"

    # A container emit through scratch costs nothing beyond the loop machinery:
    # the repr writes into scratch and drains as a scratch-window piece.
    var msgs = newSeq[Value]()
    for i in 0 ..< 10:
      var md = DictVal()
      dictSet(md, "n", strVal($i))
      msgs.add dictVal(md)
    var mcd = DictVal()
    dictSet(mcd, "messages", seqVal(msgs))
    let loopCtx = dictVal(mcd)

    template countScratchRenders(src: string, n: int): int =
      ## Warms one scratch pull render uncounted, then totals `n` renders through
      ## `getAllocStats()` deltas with one driver per render, as above.
      let (ns, ts) = parseTemplate(src)
      let mm = Machine(jinja: src, nodes: ns)
      let wantLocal = renderToString(src, loopCtx, 0.0)
      var dWarm2 = newDriver(loopCtx, 0.0)
      var scrWarm2 = newSeq[char](4096)
      attachScratch(dWarm2, scrWarm2)
      var bufWarm2 = newSeq[char](256)
      var accWarm = ""
      while true:
        let got = pull(mm, ts, dWarm2, bufWarm2)
        if got == 0:
          break
        accWarm.add bytesOf(bufWarm2, got)
      doAssert accWarm == wantLocal, "the micro scratch render differs for " & src
      var total = 0
      for _ in 0 ..< n:
        var di = newDriver(loopCtx, 0.0)
        var sci = newSeq[char](4096)
        attachScratch(di, sci)
        var bi = newSeq[char](256)
        let renderCost = allocsOf:
          while true:
            let got = pull(mm, ts, di, bi)
            if got == 0:
              break
        total += renderCost
      total

    let loopOnly = countScratchRenders("{% for m in messages %}x{% endfor %}", iters)
    let dictEmits = countScratchRenders("{% for m in messages %}{{ m }}{% endfor %}", iters)
    doAssert dictEmits == loopOnly, "a container emit via scratch cost " &
        $(dictEmits - loopOnly) & " allocations beyond the loop baseline"

    echo "t_scratch alloc: tojson direct ", tjAllocs div iters, "/call, tojson render ",
        renderAllocs div iters, "/render, container emit via scratch ",
        dictEmits - loopOnly, " allocs beyond the loop baseline over ", iters, " renders"

echo "t_scratch: corpus rows, scratch drains, breaches and repulls all byte-exact"
