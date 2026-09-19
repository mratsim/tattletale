# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Window-contract render proof for the chattyninja engine.
##
## Derived values stream as lazy pieces straight into the caller's window:
## - every ok corpus row renders byte-exact through pull
## - a container emit and a `~` concat drain as lazy pieces across pull calls byte-exact
## - a raise inside a for-filter propagates per the pull contract. The failing call's
##   bytes are discarded and a repull resumes after the failed item
##
## Run:
##   $ nim test_chattyninja

import std/[importutils, strutils]
import cnj_errors, cnj_types, cnj_values, cnj_parse, cnj_engine
import rows

func bytesOf(buf: openArray[char], n: int): string =
  ## Copies the first `n` bytes of a pull window into a string.
  for i in 0 ..< n:
    result.add buf[i]

func listCtx(): Value =
  ## One context holding `m`, a mixed container whose serialization exceeds a tiny window.
  var inner = DictVal()
  dictSet(inner, "alpha", strVal("one"))
  dictSet(inner, "beta", strVal("two"))
  dictSet(inner, "gamma", seqVal(@[strVal("x"), strVal("y"), strVal("z")]))
  var cd = DictVal()
  dictSet(cd, "m", dictVal(inner))
  dictVal(cd)

const listRepr = "{'alpha': 'one', 'beta': 'two', 'gamma': ['x', 'y', 'z']}"
  ## Python `repr()` of the `m` value above, the independent byte truth for the drains.

# Every ok corpus row through a pull render stays byte-exact.
# ---------------------------------------------------------------------------
block corpusRowsThroughPull:
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
        # A declared engine gap raises before any bytes here. Classify it as a gap only
        # when the string render raises the same way, so a window-only failure cannot
        # hide behind the skip.
        var dStr = newDriver(r.context, r.clock)
        try:
          discard pullAll(m, tables, dStr)
        except CatchableError:
          raised = false
        doAssert not raised,
            suite & "/" & r.row & ": the pull render raised where the string render did not"
        inc gapSkipped
        continue
      doAssert got == r.rendered,
          suite & "/" & r.row & ": the pull render differs from the recorded bytes"
      inc checked
  doAssert checked == 55, "expected 55 rendered ok rows across 13 suites, checked " & $checked
  doAssert gapSkipped == 4, "expected 4 gap rows across 13 suites, skipped " & $gapSkipped

# A container emit drains as a lazy piece across pull calls byte-exact.
# ---------------------------------------------------------------------------
block lazyWindowDrain:
  let ctx = listCtx()
  let src = "{{ m }}"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)

  var d = newDriver(ctx, 0.0)
  var window = newSeq[char](8)
  var acc = ""
  var lazyPieces = 0
  while true:
    if d.pend.kind == pkLazy:
      inc lazyPieces
    let n = pull(m, tables, d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == listRepr, "the lazy-piece drain differs from the container repr"
  doAssert lazyPieces > 0, "no pull observed a pending lazy piece"

# A `~` concat emit streams its operands through the lazy-piece machinery, left to right,
# and an 8-byte window forces the drain across several pulls mid-value.
# ---------------------------------------------------------------------------
block concatWindowDrain:
  let ctx = listCtx()
  let src = "{{ m ~ '::' ~ m }}"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = listRepr & "::" & listRepr

  var d = newDriver(ctx, 0.0)
  var window = newSeq[char](8)
  var acc = ""
  var lazyPulls = 0
  while true:
    if d.pend.kind == pkLazy:
      inc lazyPulls
    let n = pull(m, tables, d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == want, "the concat drain differs from the expected operand order"
  doAssert lazyPulls > 2, "the concat drained in fewer than three pulls, the " &
      "mid-value drain is unobserved"

# A raise inside a for-filter propagates per the pull contract. The loop cursor stays
# committed past the failed item and a repull resumes after it. One-byte window first,
# where every byte delivered before the failing call is already with the caller.
# ---------------------------------------------------------------------------
block filterRaiseRepull:
  var msgs = newSeq[Value]()
  msgs.add strVal("aa")
  msgs.add intVal(7)
  msgs.add strVal("ab")
  var cd = DictVal()
  dictSet(cd, "xs", seqVal(msgs))
  let ctx = dictVal(cd)
  # `x[0]` raises on the integer item and passes the strings through the filter comparison.
  let src = "pre{% for x in xs if x[0] == 'a' %}[{{ x }}]{% endfor %}post"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = "pre[aa][ab]post"
  # The one-shot render propagates the same raise, the filtered strings never reaching it.
  try:
    discard renderToString(src, ctx, 0.0)
    doAssert false, "the one-shot render did not propagate the failing filter"
  except TemplateError as e:
    doAssert "not subscriptable" in e.msg, e.msg

  var d = newDriver(ctx, 0.0)
  var win1 = newSeq[char](1)
  var acc = ""
  var raised = false
  var message = ""
  try:
    while true:
      let n = pull(m, tables, d, win1)
      if n == 0:
        break
      acc.add bytesOf(win1, n)
  except CatchableError as e:
    raised = true
    message = e.msg
  doAssert raised, "the failing filter did not raise"
  doAssert "not subscriptable" in message,
      "the error did not name the failed operation: " & message
  doAssert acc == "pre[aa]", "the caller-held bytes at the raise are not exactly the prefix"

  # The repull skips nothing. The integer item stays consumed and the render completes.
  var rest = newSeq[char](64)
  while true:
    let n = pull(m, tables, d, rest)
    if n == 0:
      break
    acc.add bytesOf(rest, n)
  doAssert acc == want, "the repull after the raise differs from the single-shot render"

  # Wide window. The prefix and the failed item's evaluation land in one call, whose
  # window bytes are discarded and never reach the caller. The repull resumes after them,
  # discarded prefix included.
  var dWide = newDriver(ctx, 0.0)
  var wide = newSeq[char](64)
  var wideAcc = ""
  var wideRaised = false
  try:
    while true:
      let n = pull(m, tables, dWide, wide)
      if n == 0:
        break
      wideAcc.add bytesOf(wide, n)
  except CatchableError:
    wideRaised = true
  doAssert wideRaised, "the wide-window filter raise did not raise"
  doAssert wideAcc == "", "the failing call returned bytes: <" & wideAcc & ">"
  var wideRest = ""
  while true:
    let n = pull(m, tables, dWide, wide)
    if n == 0:
      break
    wideRest.add bytesOf(wide, n)
  doAssert wideRest == "[ab]post",
      "the wide-window repull did not resume after the discarded bytes"

# A macro call as a whole emit streams its body's pieces through the caller's window,
# and the streamed bytes match the string render.
# ---------------------------------------------------------------------------
block captureSink:
  let ctx = listCtx()
  let src = "{%- macro mm(v) -%}[{{ v }}]{%- endmacro -%}{{ mm(m) }}"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = renderToString(src, ctx, 0.0)

  var d = newDriver(ctx, 0.0)
  doAssert pullAll(m, tables, d) == want,
      "the capture-sink emit differs from the string render"

# A streamed macro call resolves names against the caller's scopes only before the call
# and against its own scopes only inside the body: the macro scope is popped on close.
# ---------------------------------------------------------------------------
block macroScopePop:
  let leakCaller = "{% macro mm(q) %}[{{ q }}]{% endmacro %}" &
      "{% set q = 'caller' %}{{ mm('inner') }}:{{ q }}"
  doAssert renderToString(leakCaller, listCtx()) == "[inner]:caller",
      "the macro body read a caller binding set after the call"

  let leakBody = "{% macro mm() %}{% set z = 'body' %}{{ z }}{% endmacro %}{{ mm() }}:{{ z }}"
  doAssert renderToString(leakBody, listCtx()) == "body:",
      "a macro body binding leaked into the caller's name resolution"

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

    # Direct tojson of the tool schema. The writer drains into a growable buffer with no
    # presize pass, so a call costs one allocation for the buffer plus one for the stack
    # behind the schema's two nested containers.
    let tools = toolsVal()
    # warm-up call, excluded from the counted region
    discard toJson(tools)
    let tjAllocs = allocsOf:
      for _ in 0 ..< iters:
        discard toJson(tools)
    doAssert tjAllocs == 2 * iters, "toJson of the tool schema cost " & $(tjAllocs div iters) &
        " allocations per call against the measured two"

    # The same schema through the pull render, driver setup uncounted. The counted region
    # holds only the pull loop, and the render costs the filter's argument list plus
    # the serializer's container stack.
    const tJson = "{{ tools|tojson }}"
    var cd = DictVal()
    dictSet(cd, "tools", tools)
    let ctx = dictVal(cd)
    let (nodes, tables) = parseTemplate(tJson)
    let m = Machine(jinja: tJson, nodes: nodes)
    let want = renderToString(tJson, ctx, 0.0)

    var dWarm = newDriver(ctx, 0.0)
    var bufWarm = newSeq[char](256)
    var warm = ""
    while true:
      let n = pull(m, tables, dWarm, bufWarm)
      if n == 0:
        break
      warm.add bytesOf(bufWarm, n)
    doAssert warm == want, "the pull render differs from the string render"

    var buf = newSeq[char](256)
    var renderAllocs = 0
    for _ in 0 ..< iters:
      var di = newDriver(ctx, 0.0)
      let renderCost = allocsOf:
        while true:
          let n = pull(m, tables, di, buf)
          if n == 0:
            break
      renderAllocs += renderCost
    doAssert renderAllocs == 3 * iters, "the tojson pull render cost " &
        $(renderAllocs div iters) & " allocations per render against the measured three"

    # A container emit costs one allocation per emit for the lookup copy plus one per
    # render for the serializer's container stack, over the loop machinery.
    var msgs = newSeq[Value]()
    for i in 0 ..< 10:
      var md = DictVal()
      dictSet(md, "n", strVal($i))
      msgs.add dictVal(md)
    var mcd = DictVal()
    dictSet(mcd, "messages", seqVal(msgs))
    let loopCtx = dictVal(mcd)

    template countRenders(src: string, n: int): int =
      ## Warms one pull render uncounted, then totals `n` renders through
      ## `getAllocStats()` deltas with one driver per render, as above.
      let (ns, ts) = parseTemplate(src)
      let mm = Machine(jinja: src, nodes: ns)
      let wantLocal = renderToString(src, loopCtx, 0.0)
      var dWarm2 = newDriver(loopCtx, 0.0)
      var bufWarm2 = newSeq[char](256)
      var accWarm = ""
      while true:
        let got = pull(mm, ts, dWarm2, bufWarm2)
        if got == 0:
          break
        accWarm.add bytesOf(bufWarm2, got)
      doAssert accWarm == wantLocal, "the micro render differs for " & src
      var total = 0
      for _ in 0 ..< n:
        var di = newDriver(loopCtx, 0.0)
        var bi = newSeq[char](256)
        let renderCost = allocsOf:
          while true:
            let got = pull(mm, ts, di, bi)
            if got == 0:
              break
        total += renderCost
      total

    let loopOnly = countRenders("{% for m in messages %}x{% endfor %}", iters)
    let strEmits = countRenders("{% for m in messages %}{{ m.n }}{% endfor %}", iters)
    let dictEmits = countRenders("{% for m in messages %}{{ m }}{% endfor %}", iters)
    # The runtime-built message values keep the engine's one-lookup-copy residual per emit.
    doAssert strEmits == loopOnly + iters * 10, "the string emit cost " &
        $(strEmits - loopOnly) & " allocations beyond the loop baseline"
    # A container emit through the lazy piece costs one allocation per emit over the string
    # emit and one per render for the serializer's container stack.
    doAssert dictEmits == strEmits + iters * 11, "the container emit cost " &
        $(dictEmits - strEmits) & " allocations beyond the string emit"

    echo "t_scratch alloc: tojson direct ", tjAllocs div iters, "/call, tojson render ",
        renderAllocs div iters, "/render, string emit ", (strEmits - loopOnly) div iters,
        ", container emit ", (dictEmits - loopOnly) div iters,
        " allocs beyond the loop baseline over ", iters, " renders"

echo "t_scratch: corpus rows, lazy drains, the filter raise repull, all byte-exact"
