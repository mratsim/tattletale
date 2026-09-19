# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Presized-buffer pull render proof for the chattyninja engine.
##
## `pull` fills a caller-owned buffer chunk by chunk, resumable across calls through the driver.
## Checked here:
## - a corpus row rendered through a 256-byte buffer equals the recorded bytes and `pullAll`
## - oversized, exact-size, 1-byte and zero-capacity windows, plus a stop-then-resume consumer, deliver byte-exact renders
## - template-text spans and emit values longer than the buffer drain across calls, still equal to the string render
##
## Run:
##   $ ./workspace/chattyninja/run_tests.sh t_pull

import std/[importutils, strutils]
import cnj_types, cnj_values, cnj_parse, cnj_engine
import rows

func bytesOf(buf: openArray[char], n: int): string =
  ## Copies the first `n` bytes of a pull window into a string.
  for i in 0 ..< n:
    result.add buf[i]

proc renderPull(m: Machine, t: Tables, ctx: Value, clock: float64, cap: int): string =
  ## Renders through `pull` with a `cap`-byte caller buffer, accumulating every fill.
  var d = newDriver(ctx, clock)
  var buf = newSeq[char](cap)
  while true:
    let n = pull(m, t, d, buf)
    if n == 0:
      break
    for i in 0 ..< n:
      result.add buf[i]

proc renderAllPull(m: Machine, t: Tables, ctx: Value, clock: float64): string =
  ## Renders through `pullAll` with a fresh driver.
  var d = newDriver(ctx, clock)
  pullAll(m, t, d)

# Tracer bullet: one corpus row, rendered end to end through a 256-byte caller buffer.
# ---------------------------------------------------------------------------
block corpusRowThrough256ByteBuffer:
  let src = templateSource("deepseekv2lite")
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let row = loadRow("deepseekv2lite", "assistant_history")

  let pulled = renderPull(m, tables, row.context, row.clock, 256)
  let whole = renderAllPull(m, tables, row.context, row.clock)
  doAssert whole == row.rendered, "pullAll disagrees with the recorded bytes"
  doAssert pulled == row.rendered, "the 256-byte pull render differs from the recorded bytes"
  doAssert pulled == whole, "the 256-byte pull render differs from pullAll"

# Boundary shapes of the delivery window.
# ---------------------------------------------------------------------------
block boundaryShapes:
  let src = templateSource("deepseekv2lite")
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let row = loadRow("deepseekv2lite", "assistant_history")
  let want = row.rendered

  # A buffer larger than the whole render takes everything in one pull, then reports 0.
  var dBig = newDriver(row.context, row.clock)
  var big = newSeq[char](want.len + 1)
  let n1 = pull(m, tables, dBig, big)
  doAssert n1 == want.len, "an oversized buffer took " & $n1 & " of " & $want.len & " bytes"
  doAssert bytesOf(big, n1) == want, "the one-pull render differs from the recorded bytes"
  doAssert pull(m, tables, dBig, big) == 0, "a completed render kept returning bytes"

  # A buffer exactly the render size also drains in one pull.
  var dExact = newDriver(row.context, row.clock)
  var exact = newSeq[char](want.len)
  let n2 = pull(m, tables, dExact, exact)
  doAssert n2 == want.len, "an exact-size buffer took " & $n2 & " of " & $want.len & " bytes"
  doAssert bytesOf(exact, n2) == want, "the exact-size render differs from the recorded bytes"
  doAssert pull(m, tables, dExact, exact) == 0, "a completed render kept returning bytes"

  # A 1-byte buffer gives every byte its own pull, which forces mid-piece drains.
  var dOne = newDriver(row.context, row.clock)
  var one: array[1, char]
  var acc = ""
  var calls = 0
  while true:
    let n = pull(m, tables, dOne, one)
    if n == 0:
      break
    doAssert n == 1, "a 1-byte buffer pull returned " & $n
    acc.add one[0]
    inc calls
  doAssert acc == want, "the 1-byte render differs from the recorded bytes"
  doAssert calls == want.len, "expected one pull per byte, got " & $calls & " of " & $want.len

  # A consumer that stops mid-drain and resumes never re-receives a byte.
  var dStop = newDriver(row.context, row.clock)
  var window = newSeq[char](16)
  var head = ""
  block stopEarly:
    for _ in 0 ..< 3:
      let n = pull(m, tables, dStop, window)
      if n == 0:
        break
      head.add bytesOf(window, n)
  doAssert head.len > 0 and head.len < want.len, "the early stop covered the whole render"
  doAssert head == want[0 ..< head.len], "the bytes before the stop diverged from the recording"
  doAssert dStop.cur == head.len, "cur is " & $dStop.cur & " but " & $head.len &
      " bytes were received"
  var tail = ""
  while true:
    let n = pull(m, tables, dStop, one)
    if n == 0:
      break
    tail.add one[0]
  doAssert tail == want[head.len ..< want.len], "the resumed bytes overlapped or diverged"
  doAssert head & tail == want, "stop-then-resume is not the whole render"

# A zero-capacity buffer reports 0 without stepping the render.
# ---------------------------------------------------------------------------
block zeroCapacityBuffer:
  let src = templateSource("deepseekv2lite")
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let row = loadRow("deepseekv2lite", "assistant_history")

  var d = newDriver(row.context, row.clock)
  var empty: array[0, char]
  doAssert pull(m, tables, d, empty) == 0, "a zero-capacity buffer did not report 0"
  doAssert d.curNode != noLink, "a zero-capacity pull stepped the render to the end"
  doAssert d.cur == 0, "a zero-capacity pull moved cur"
  doAssert d.pend.kind == pkNone, "a zero-capacity pull started a pending piece"

  # the untouched driver still delivers the whole render byte-exact
  var acc = ""
  var window = newSeq[char](256)
  while true:
    let n = pull(m, tables, d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == row.rendered, "the render after a zero-capacity pull differs from the recorded bytes"

# Resumption across a value boundary:
# an emit value longer than the buffer drains across pull calls through the pending piece.
# ---------------------------------------------------------------------------
block valueBoundary:
  let longA = repeat("alpha-", 50)
  let longB = repeat("beta-", 40)
  let src = "{% for m in messages %}{{ m.content }}{% endfor %}"
  var msgs = newSeq[Value]()
  for c in [longA, longB]:
    var md = DictVal()
    dictSet(md, "content", strVal(c))
    msgs.add dictVal(md)
  var ctxd = DictVal()
  dictSet(ctxd, "messages", seqVal(msgs))
  let ctx = dictVal(ctxd)

  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = renderToString(src, ctx, 0.0)
  doAssert want == longA & longB, "the string render is not the contents concatenation"

  let pulled = renderPull(m, tables, ctx, 0.0, 8)
  doAssert pulled == want, "the 8-byte pull render differs across the value boundary"
  let whole = renderAllPull(m, tables, ctx, 0.0)
  doAssert whole == want, "pullAll differs across the value boundary"

# Span pieces copy out of `Machine.jinja` and drain across calls byte-exact.
# ---------------------------------------------------------------------------
block spanDrain:
  let verbatim = repeat("literal text ", 15)
  let src = verbatim & "{{ m }}"
  var ctxd = DictVal()
  dictSet(ctxd, "m", strVal("emit"))
  let ctx = dictVal(ctxd)
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = renderToString(src, ctx, 0.0)
  doAssert want == verbatim & "emit", "the string render is not the expected bytes"

  # A window far below the verbatim run forces span pieces through several pulls.
  var d = newDriver(ctx, 0.0)
  var window = newSeq[char](8)
  var acc = ""
  var spanPulls = 0
  while true:
    if d.pend.kind == pkSpan:
      inc spanPulls
    let n = pull(m, tables, d, window)
    if n == 0:
      break
    acc.add bytesOf(window, n)
  doAssert acc == want, "the span-drain render differs from the string render"
  doAssert spanPulls > 0, "no pull drained a pending span piece"

# Allocation test. Runs last, only under `-d:nimAllocStats`, and a failing doAssert there
# hangs the run with no output instead of failing it.
# ---------------------------------------------------------------------------
when defined(nimAllocStats):
  privateAccess(AllocStats)

  func pieceRemaining(p: Piece): int =
    ## Bytes of a pending piece not yet delivered, lazy pieces carrying no counted length.
    case p.kind
    of pkNone: 0
    of pkSpan: int(p.hi - p.lo) - p.pos
    of pkStr: p.s.len - p.pos
    of pkScratch: p.shi.int - p.pos
    of pkLazy: 0

  template allocsOf(body: untyped): int =
    ## Counts `alloc` calls made by `body`, with allocator state warmed by the caller.
    let before = getAllocStats()
    body
    (getAllocStats() - before).allocCount

  block allocProbe:
    let src = templateSource("deepseekv2lite")
    let (nodes, tables) = parseTemplate(src)
    let m = Machine(jinja: src, nodes: nodes)
    let row = loadRow("deepseekv2lite", "assistant_history")

    # Warm-up renders, uncounted:
    # first-touch allocator state settles here.
    for _ in 0 ..< 3:
      discard renderPull(m, tables, row.context, row.clock, 1)
      discard renderAllPull(m, tables, row.context, row.clock)
      discard renderToString(src, row.context, row.clock)

    # A pull that enters on a pending piece with bytes left only drains it, no step runs,
    # so it must allocate nothing.
    var d = newDriver(row.context, row.clock)
    var one: array[1, char]
    var acc = ""
    var drainCalls = 0
    while true:
      let pending = d.pend.kind != pkNone and pieceRemaining(d.pend) > 0
      let before = getAllocStats()
      let n = pull(m, tables, d, one)
      let used = (getAllocStats() - before).allocCount
      if n == 0:
        break
      acc.add one[0]
      if pending:
        inc drainCalls
        doAssert used == 0, "a pull that only drained pending bytes allocated " & $used
    doAssert acc == row.rendered, "the counted render disagrees with the recorded bytes"
    doAssert drainCalls > 0, "no pending-piece drain was counted"
    echo "t_pull alloc: ", drainCalls, " pending-piece drain calls, all 0 allocs"

    # Whole-render comparison:
    # the pull path against the string path, whose count also covers parsing the template
    # and therefore bounds the pull total from above.
    var dTotal = newDriver(row.context, row.clock)
    let pullAllocs = allocsOf:
      discard pullAll(m, tables, dTotal)
    let strAllocs = allocsOf:
      discard renderToString(src, row.context, row.clock)
    doAssert pullAllocs <= strAllocs, "the pull render allocated " & $pullAllocs &
        " against the string render's " & $strAllocs
    echo "t_pull alloc: full pull render ", pullAllocs, " allocs, string render ", strAllocs,
        " allocs"

  # Micro attribution over a 10-message for-loop context:
  # a warm-up render per template stays uncounted, then `getAllocStats()` deltas measure
  # the counted renders. DictGet lookup floors are measured in the same run.
  # Every assert below is an exact equality against a value this binary just measured:
  # - an emit-role render costs nothing beyond the loop machinery, its lookups included
  # - an emit-content render costs exactly one lookup copy per emit, the accepted residual
  # - a punctuator evaluation and the pending-piece move of an emit string cost 0
  block allocMicro:
    const msgCount = 10
    let iters = 50

    func msgVal(role, content: string): Value =
      ## Builds one chat message carrying the two keys the templates read.
      var d = DictVal()
      dictSet(d, "role", strVal(role))
      dictSet(d, "content", strVal(content))
      dictVal(d)

    var msgs = newSeq[Value]()
    for i in 0 ..< msgCount:
      msgs.add msgVal(if i mod 2 == 0: "user" else: "assistant",
          "Message " & $i & ": please continue the conversation and stay on topic.")
    var cd = DictVal()
    dictSet(cd, "messages", seqVal(msgs))
    let ctx = dictVal(cd)

    # Lookup floors over one message, each warmed by one uncounted call:
    # `role` strings are literal-backed so a lookup copies nothing, `content` strings are
    # runtime-built so a lookup copies once.
    let msg1 = ctx.d.dictGet("messages").xs.items[1]
    discard msg1.d.dictGet("role")
    discard msg1.d.dictGet("content")
    let dgRole = allocsOf:
      discard msg1.d.dictGet("role")
    let dgContent = allocsOf:
      discard msg1.d.dictGet("content")

    template countRenders(src: string, n: int): int =
      ## Warms one pull render uncounted, then totals `n` pull renders through `getAllocStats()` deltas.
      let (nodes, tables) = parseTemplate(src)
      let m = Machine(jinja: src, nodes: nodes)
      let want = renderToString(src, ctx, 0.0)
      doAssert renderAllPull(m, tables, ctx, 0.0) == want,
          "the micro pull render differs from the string render for " & src
      allocsOf:
        for _ in 0 ..< n:
          discard renderAllPull(m, tables, ctx, 0.0)

    let loopOnly = countRenders("{% for m in messages %}x{% endfor %}", iters)
    let roleRenders = countRenders("{% for m in messages %}{{ m.role }}{% endfor %}", iters)
    let contentRenders = countRenders("{% for m in messages %}{{ m.content }}{% endfor %}", iters)
    let bothSrc = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    let bothRenders = countRenders(bothSrc, iters)

    doAssert roleRenders == loopOnly + iters * msgCount * dgRole,
        "the emit-role render cost " & $(roleRenders - loopOnly) &
        " allocs beyond the loop baseline"
    doAssert contentRenders == loopOnly + iters * msgCount * dgContent,
        "the emit-content render cost " & $(contentRenders - loopOnly) &
        " allocs beyond the loop baseline"
    doAssert bothRenders == loopOnly + iters * msgCount * (dgRole + dgContent),
        "the two-emit render cost " & $(bothRenders - loopOnly) &
        " allocs beyond the loop baseline"
    doAssert (roleRenders - loopOnly) div iters <= msgCount and
        (contentRenders - loopOnly) div iters <= msgCount,
        "an emit cost more than the accepted one-lookup residual"
    echo "t_pull alloc: emit-role ", roleRenders, ", emit-content ", contentRenders,
        ", emit both ", bothRenders, ", loop baseline ", loopOnly,
        " allocs over ", iters, " renders each"

echo "t_pull: corpus row, window shapes, value and span drains, all byte-exact"
