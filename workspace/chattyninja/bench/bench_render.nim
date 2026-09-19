# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Benchmark chattyninja template rendering over real chat templates plus the recorded corpus.
##
## One stimulus, two builds, from `workspace/chattyninja`:
##
##   nim c -d:release --experimental:views --hints:off --warnings:off --path:src --path:tests \
##     --outdir:../../build/chattyninja/bin --nimcache:../../build/chattyninja/nimcache-bench \
##     bench/bench_render.nim && ../../build/chattyninja/bin/bench_render
##
##   nim c -d:release -d:benchAlloc -d:nimAllocStats --experimental:views --hints:off \
##     --warnings:off --path:src --path:tests -o:../../build/chattyninja/bin/bench_render_alloc \
##     bench/bench_render.nim && ../../build/chattyninja/bin/bench_render_alloc
##
## Reports:
##
## - release build, ms/render and renders/s per template x conversation shape, median
##   over 15 timed runs after 500 warm-up renders, spread-flagged when noisy
## - release build, pull-window timing over the corpus anchor rows, median ms/render
##   for 256 B and 4 KiB windows beside the one-shot render, plus pull-call counts
## - benchAlloc build, parse-time and render-time allocations per render, warm-up
##   uncounted through `system.getAllocStats()`, plus spill counts and micro
##   attribution templates isolating loop machinery, emit stringification and JSON serialization
##
## Both builds replay the recorded deepseekv2lite and qwen3 rows as a corpus anchor,
## keeping the numbers here comparable with earlier corpus-row measurements.
##
## Stimulus templates come from the gitignored hf_models links,
## `workspace/transformers/tests/hf_models/<model>/chat_template.jinja`, each link byte-identical
## to the corpus suite it mirrors.
##
## Coverage:
##
## - a template the engine cannot parse, or a construct a shape cannot render, is
##   reported as a coverage gap naming the construct
## - the bench never modifies the engine to fit a stimulus
##
## No `doAssert` anywhere. A failing doAssert hangs under `-d:nimAllocStats`.

import std/[algorithm, importutils, monotimes, os, strformat, strutils, times]
import cnj_errors, cnj_types, cnj_values, cnj_parse, cnj_engine
import workspace/data_structures/src/small_seqs
import ../tests/rows

when defined(benchAlloc) and not defined(nimAllocStats):
  {.error: "build the allocation trace with -d:benchAlloc -d:nimAllocStats".}

# ── Stimulus builders ────────────────────────────────────────────────────────
#
# Contexts are built with the engine's value constructors, not `std/json`, because
# template output observes dict insertion order.

const HfModelsRoot = currentSourcePath().parentDir.parentDir.parentDir /
    "transformers" / "tests" / "hf_models"
  ## the gitignored checkpoint links, `<root>/<model>/chat_template.jinja`

func msgVal(role, content: string): Value =
  ## One chat message with the two keys every corpus template reads.
  var d = DictVal()
  dictSet(d, "role", strVal(role))
  dictSet(d, "content", strVal(content))
  dictVal(d)

func toolCallMsg(name: string, arguments: Value): Value =
  ## One assistant tool-call message, `arguments` a mapping, matching GLM-4.7-Flash's `_args.items()` method.
  var fn = DictVal()
  dictSet(fn, "name", strVal(name))
  dictSet(fn, "arguments", arguments)
  var tc = DictVal()
  dictSet(tc, "function", dictVal(fn))
  var d = DictVal()
  dictSet(d, "role", strVal("assistant"))
  dictSet(d, "content", strVal(""))
  dictSet(d, "tool_calls", seqVal(@[dictVal(tc)]))
  dictVal(d)

func toolRespMsg(id, content: string): Value =
  ## One tool-result message bound to its call id.
  var d = DictVal()
  dictSet(d, "role", strVal("tool"))
  dictSet(d, "tool_call_id", strVal(id))
  dictSet(d, "content", strVal(content))
  dictVal(d)

func weatherTools(): Value =
  ## One function-tool definition, the corpus tools fixture's schema.
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

func contextOf(msgs: seq[Value], tools: Value): Value =
  ## Render context holding `messages`, `tools` and `add_generation_prompt`.
  ## Template kwargs stay absent, the templates guard them with an `is defined` test.
  var d = DictVal()
  dictSet(d, "messages", seqVal(msgs))
  dictSet(d, "tools", tools)
  dictSet(d, "add_generation_prompt", boolVal(true))
  dictVal(d)

type
  Shape = object
    ## One conversation shape, a name plus the context it renders.
    name: string
    ctx: Value

  Compiled = object
    ## One parsed template plus its bench label.
    ##
    ## `Machine` is built from these fields at render time, not at parse time.
    ##
    ## `Machine.jinja` borrows the template text. Borrowing from a local would dangle,
    ## because the local's buffer dies at the end of the parse loop iteration.
    label: string
    src: string
    nodes: seq[Node]
    t: Tables

func shortShape(): Shape =
  ## 2-message short conversation, no tools.
  Shape(name: "short2", ctx: contextOf(@[
      msgVal("user", "Hello!"),
      msgVal("assistant", "Hi there.")], noneVal()))

func typicalShape(): Shape =
  ## 10-message typical conversation, no tools.
  var msgs = @[msgVal("system",
      "You are a helpful assistant. Answer concisely and correctly.")]
  for i in 0 ..< 9:
    msgs.add msgVal(if i mod 2 == 0: "user" else: "assistant",
        "Message " & $i & ": please continue the conversation and stay on topic.")
  Shape(name: "typical10", ctx: contextOf(msgs, noneVal()))

func longShape(withToolRound: bool, name: string): Shape =
  ## 40-message long conversation.
  ##
  ## - a system prompt and a tool definition
  ## - with `withToolRound`, an assistant tool call plus its tool result
  var msgs = @[msgVal("system",
      "You are a helpful assistant with tool access. Call a tool by emitting a JSON " &
      "object with the arguments and wait for the tool result before answering.")]
  let body = if withToolRound: 37 else: 39
  for i in 0 ..< body:
    msgs.add msgVal(if i mod 2 == 0: "user" else: "assistant",
        "Turn " & $i & ": here is more context about the task; keep the details " &
        "accurate and reference the earlier turns when they matter.")
  if withToolRound:
    var args = DictVal()
    dictSet(args, "city", strVal("Tokyo"))
    msgs.add toolCallMsg("get_weather", dictVal(args))
    msgs.add toolRespMsg("call_1", "Sunny, 18 degrees.")
  Shape(name: name, ctx: contextOf(msgs, weatherTools()))

func benchShapes(): array[4, Shape] =
  ## short2 and typical10 are the canonical short and typical shapes.
  ## long40 carries the tool round and long40-nt the same conversation without it, so
  ## the tool round's cost and any construct gap it triggers separate cleanly.
  [shortShape(), typicalShape(), longShape(true, "long40"), longShape(false, "long40-nt")]

const
  HfModels = [("glm47flash", "GLM-4.7-Flash"), ("glm53flash", "GLM-5.3-Flash"),
              ("qwen36", "Qwen3.6-35B-A3B"), ("qwen35", "Qwen3.5-0.8B")]
    ## bench label and hf_models directory per template

# ── Measurement ──────────────────────────────────────────────────────────────

proc renderOnce(m: Machine, t: Tables, ctx: Value, clock = 0.0): string =
  ## Renders once, whole, through the pull interface, exactly as the test suites do.
  var d = newDriver(ctx, clock)
  pullAll(m, t, d)

proc renderN(m: Machine, t: Tables, ctx: Value, clock: float64, n: int): int =
  ## Renders `n` times and returns the accumulated output byte count, so the loop
  ## consumes every render and nothing is optimized away.
  for _ in 0 ..< n:
    result += renderOnce(m, t, ctx, clock).len

template timedRuns(runs, iters: int, body: untyped): seq[float64] =
  ## Runs `body` `iters` times per run, `runs` runs, and returns per-render
  ## milliseconds per run.
  ## No warm-up here. Callers warm up first.
  var samples: seq[float64]
  for _ in 0 ..< runs:
    let start = getMonotime()
    body
    let perRender = (getMonotime() - start).inNanoseconds.float64 / 1e6 / iters.float64
    samples.add perRender
  samples

func median(xs: seq[float64]): float64 =
  ## Median of a non-empty sample.
  let s = sorted(xs)
  s[s.len div 2]

proc timeShape(m: Machine, t: Tables, ctx: Value, iters: int): string =
  ## Times one template x shape.
  ##
  ## 500 warm-up renders, then rounds of 15 timed runs of `iters` renders each.
  ##
  ## - the first round whose spread stays within 20% is reported
  ## - a busy machine gets up to 4 rounds and the most stable one is reported, still
  ##   flagged when its spread exceeds 20%, so noise is never silently averaged away
  let warm = renderN(m, t, ctx, 0, 500)
  var best: seq[float64]
  var bestSpread = 1e9
  for _ in 0 ..< 4:
    let samples = timedRuns(15, iters):
      discard renderN(m, t, ctx, 0, iters)
    let spread = (max(samples) - min(samples)) / median(samples) * 100.0
    when defined(benchDebug):
      echo "    samples: ", samples.mapIt(it.formatFloat(ffDecimal, 4)).join(" ")
    if spread < bestSpread:
      bestSpread = spread
      best = samples
    if bestSpread <= 20.0:
      break
  let mid = median(best)
  let flag = if bestSpread > 20.0: "  VARIANCE" else: ""
  fmt"{mid:9.4f} ms/render {1000.0 / mid:12.1f} renders/s   spread {bestSpread:4.1f}%  output {warm div 500} B{flag}"

# ── Allocation trace ─────────────────────────────────────────────────────────

when defined(benchAlloc):
  privateAccess(AllocStats)

  template allocsOf(body: untyped): int =
    ## Counts `alloc` calls made by `body`.
    ## The delta counts every allocator entry, so a warm allocator pool keeps repeated measurements stable.
    let before = getAllocStats()
    body
    (getAllocStats() - before).allocCount

  func spillCensus(nodes: seq[Node]): tuple[summary: string, maxSlots: int] =
    ## Counts nodes whose payload exceeded the 5-slot inline capacity, per kind.
    ## A payload spills exactly when its public slot length holds more than 5 slots.
    var byKind: array[NodeKind, int]
    for nd in nodes:
      if nd.slots.len > 5:
        inc byKind[nd.kind]
      if nd.slots.len > result.maxSlots:
        result.maxSlots = nd.slots.len
    for kind in NodeKind:
      if byKind[kind] > 0:
        result.summary.add &"{kind}[{byKind[kind]}] "

  proc allocShape(m: Machine, t: Tables, ctx: Value, iters: int): string =
    ## Allocates per render for one template x shape.
    ## One warm-up render goes uncounted, then `iters` renders are counted via `getAllocStats()`.
    discard renderOnce(m, t, ctx)
    let a = allocsOf:
      for _ in 0 ..< iters:
        discard renderOnce(m, t, ctx)
    fmt"{a.float64 / iters.float64:8.2f} allocs/render"

  proc parseAllocs(src: string): string =
    ## Parse-time allocations for one template.
    ## One parse is counted after an uncounted warm-up parse of a tiny template.
    let a = allocsOf:
      discard parseTemplate(src)
    fmt"{a:8d} allocs/parse"

  proc corpusAllocAnchor(suite: string, iters: int): string =
    ## Replays one recorded suite's rows with the earlier corpus measurement's method.
    ##
    ## - one full warm-up pass over every row, uncounted
    ## - then counted passes, reported per row and as the suite mean
    let rs = rows(suite)
    let src = templateSource(suite)
    let (nodes, tables) = parseTemplate(src)
    let m = Machine(jinja: src, nodes: nodes)
    for _ in 0 ..< 3: # warm-up pass over every row, uncounted
      for r in rs:
        discard renderOnce(m, tables, r.context, r.clock)
    var perRow = ""
    var total = 0.0
    for r in rs:
      let a = allocsOf:
        for _ in 0 ..< iters:
          discard renderOnce(m, tables, r.context, r.clock)
      let per = a.float64 / iters.float64
      perRow.add &"{r.row}={per:.1f} "
      total += per
    fmt"suite mean {total / rs.len.float64:.1f} allocs/render over {rs.len} rows: {perRow}"

  proc microAttribution(ctx: Value): void =
    ## Isolates render-time allocation sources with minimal templates over one shared
    ## context through the public render API.
    ## Deltas against the verbatim baseline give per-message and per-emit costs.
    const
      tVerbatim = "hello world"
      tLoop = "{% for m in messages %}x{% endfor %}"
      tEmit = "{% for m in messages %}{{ m.content }}{% endfor %}"
      tEmitConst = "{% for m in messages %}{{ 'c' }}{% endfor %}"
      tEmit2 = "{% for m in messages %}{{ m.content }}{{ m.role }}{% endfor %}"
      tLoopEmpty = "{% for m in messages %}{% endfor %}"
      tIf = "{% for m in messages %}{% if m.role == 'user' %}U{% else %}A{% endif %}{% endfor %}"
      tJson = "{{ tools|tojson }}"
    let iters = 100
    for (name, src) in [("verbatim", tVerbatim), ("for-loop", tLoop),
                        ("for-empty", tLoopEmpty), ("emit", tEmit), ("emit x2", tEmit2),
                        ("emit-const", tEmitConst), ("if/else", tIf), ("tools|tojson", tJson)]:
      let (nodes, tables) = parseTemplate(src)
      let m = Machine(jinja: src, nodes: nodes)
      discard renderOnce(m, tables, ctx) # warm-up render, not counted
      let a = allocsOf:
        for _ in 0 ..< iters:
          discard renderOnce(m, tables, ctx)
      echo &"  micro {name:14} {a.float64 / iters.float64:8.2f} allocs/render"
    # Expression-shape matrix, per-emit cost by token shape, all against the same
    # for-loop baseline, separating punctuator, literal and lookup costs.
    for (name, src) in [("emit-role", "{% for m in messages %}{{ m.role }}{% endfor %}"),
                        ("emit-int", "{% for m in messages %}{{ 1 }}{% endfor %}"),
                        ("emit-bracket", "{% for m in messages %}{{ m['content'] }}{% endfor %}"),
                        ("emit-concat", "{% for m in messages %}{{ m.role ~ 'x' }}{% endfor %}")]:
      let (nodes, tables) = parseTemplate(src)
      let m = Machine(jinja: src, nodes: nodes)
      discard renderOnce(m, tables, ctx) # warm-up render, not counted
      let a = allocsOf:
        for _ in 0 ..< iters:
          discard renderOnce(m, tables, ctx)
      echo &"  micro {name:14} {a.float64 / iters.float64:8.2f} allocs/render"
    # Direct stringifier costs over a representative message content.
    let content = strVal(
        "Turn 12: here is more context about the task; keep the details accurate.")
    let ps = allocsOf:
      for _ in 0 ..< 1000:
        discard pyStr(content)
    echo &"  micro pyStr(msg content) {ps.float64 / 1000.0:5.2f} allocs/call"
    # Direct attribution of the emit path's two owning copies, through public fields.
    # One copy belongs to the context lookup that fills a `Value`.
    # The other belongs to the pending-piece assignment that moves the string into driver storage.
    let msgs = ctx.d.dictGet("messages")
    # Message 1 rather than the system message, whose content is a compile-time
    # constant. A literal-backed string makes both copies below buffer shares
    # measuring 0, so message 1 carries the realistic runtime-built provenance.
    let msg1 = msgs.xs.items[1]
    let dg = allocsOf:
      for _ in 0 ..< 1000:
        discard msg1.d.dictGet("content")
    echo &"  micro dictGet(content)   {dg.float64 / 1000.0:5.2f} allocs/call"
    var drv = newDriver(ctx, 0.0)
    let cv = msg1.d.dictGet("content")
    let pc = allocsOf:
      for _ in 0 ..< 1000:
        drv.pend = Piece(pos: 0, kind: pkStr, s: cv.s)
        drv.pend = Piece(kind: pkNone)
    echo &"  micro pend piece copy    {pc.float64 / 1000.0:5.2f} allocs/call"
    let tj = allocsOf:
      for _ in 0 ..< 100:
        discard toJson(weatherTools())
    echo &"  micro tojson(tools)      {tj.float64 / 100.0:5.2f} allocs/call"

# ── Harness ──────────────────────────────────────────────────────────────────

proc compileHf(): seq[Compiled] =
  ## Parses every hf_models template.
  ## Parse failures are reported as coverage gaps and only the templates that parse
  ## reach the measurement loop.
  when defined(benchAlloc):
    discard parseTemplate("{% if x %}a{% endif %}") # warm-up parse, uncounted
  for (label, dir) in HfModels:
    let path = HfModelsRoot / dir / "chat_template.jinja"
    let src = readFile(path)
    try:
      let (nodes, tables) = parseTemplate(src)
      when defined(benchAlloc):
        let census = spillCensus(nodes)
        echo &"parse {label:11} OK    {nodes.len} nodes, {census.summary}, " &
            &"maxSlots {census.maxSlots}, {parseAllocs(src)}"
      else:
        echo &"parse {label:11} OK    {nodes.len} nodes"
      result.add Compiled(label: label, src: src, nodes: nodes, t: tables)
    except CatchableError as e:
      echo &"parse {label:11} FAIL  coverage gap: {e.msg}"

proc benchHf(): void =
  ## Times and traces every compiled template over every shape.
  ## A shape the template cannot render is reported as a coverage gap.
  var compiled = compileHf()
  echo ""
  let shapes = benchShapes()
  for c in compiled.mitems:
    echo &"render {c.label}"
    let m = Machine(jinja: c.src, nodes: c.nodes)
    for s in shapes:
      try:
        when defined(benchAlloc):
          echo &"  {s.name:10} {allocShape(m, c.t, s.ctx, 50)}"
        else:
          let iters = case s.name
              of "short2": 2000
              of "typical10": 800
              else: 200
          echo &"  {s.name:10} {timeShape(m, c.t, s.ctx, iters)}"
      except CatchableError as e:
        echo &"  {s.name:10} RAISES  coverage gap: {e.msg}"
    echo ""

proc benchCorpus(): void =
  ## Corpus anchor over the recorded deepseekv2lite and qwen3 rows, method identical
  ## to the hf stimulus, so the numbers stay comparable with earlier corpus-row measurements.
  when defined(benchAlloc):
    echo "corpus alloc anchor (warm-up uncounted)"
    echo "  deepseekv2lite " & corpusAllocAnchor("deepseekv2lite", 50)
    echo "  qwen3          " & corpusAllocAnchor("qwen3", 50)
  else:
    echo "corpus timing anchor (median of 15 runs, warm-up uncounted)"
    for (suite, iters) in [("deepseekv2lite", 400), ("qwen3", 150)]:
      let rs = rows(suite)
      let src = templateSource(suite)
      let (nodes, tables) = parseTemplate(src)
      let m = Machine(jinja: src, nodes: nodes)
      discard renderN(m, tables, rs[0].context, rs[0].clock, 500) # warm-up pass, not counted
      var best: seq[float64]
      var bestSpread = 1e9
      for _ in 0 ..< 4:
        let samples = timedRuns(15, iters):
          for _ in 0 ..< iters:
            for r in rs:
              discard renderOnce(m, tables, r.context, r.clock)
        let spread = (max(samples) - min(samples)) / median(samples) * 100.0
        if spread < bestSpread:
          bestSpread = spread
          best = samples
        if bestSpread <= 20.0:
          break
      let mid = median(best)
      let flag = if bestSpread > 20.0: "  VARIANCE" else: ""
      echo &"  {suite:14} {mid:9.4f} ms/render all {rs.len} rows   spread {bestSpread:4.1f}%{flag}"

# ── Pull-window timing ───────────────────────────────────────────────────────

proc renderWindowN(m: Machine, t: Tables, ctx: Value, clock: float64, n,
    windowSize: int): int =
  ## Renders `n` times through a `windowSize`-byte stack window and returns the byte
  ## count accumulated across renders, so the loop consumes every render.
  var buf: array[4096, char]
  for _ in 0 ..< n:
    var d = newDriver(ctx, clock)
    while true:
      let got = pull(m, t, d, buf.toOpenArray(0, windowSize - 1))
      if got == 0:
        break
      result += got

proc pullCallsPerPass(m: Machine, t: Tables, rs: seq[Row], windowSize: int): int =
  ## Pull calls that returned bytes, one untimed pass over every row.
  var buf: array[4096, char]
  for r in rs:
    var d = newDriver(r.context, r.clock)
    while true:
      let got = pull(m, t, d, buf.toOpenArray(0, windowSize - 1))
      if got == 0:
        break
      inc result

proc timedCorpusPasses(m: Machine, t: Tables, rs: seq[Row], iters: int,
    render: proc (m: Machine, t: Tables, ctx: Value, clock: float64): int):
    tuple[mid, spread: float64] =
  ## Rounds of 15 timed runs of `iters` full-corpus passes of `render`, method
  ## identical to the corpus timing anchor:
  ## - the first round whose spread stays within 20% is reported
  ## - a busy machine gets up to 4 rounds
  ## - a spread above 20% flags the sample
  var best: seq[float64]
  var bestSpread = 1e9
  for _ in 0 ..< 4:
    let samples = timedRuns(15, iters):
      for _ in 0 ..< iters:
        for r in rs:
          discard render(m, t, r.context, r.clock)
    let spread = (max(samples) - min(samples)) / median(samples) * 100.0
    if spread < bestSpread:
      bestSpread = spread
      best = samples
    if bestSpread <= 20.0:
      break
  (median(best), bestSpread)

proc benchPullWindows(): void =
  ## Pull-window timing over the corpus anchor rows, harness identical to the corpus
  ## timing anchor:
  ## - 500 warm-up renders, then median of 15 timed runs of `iters` full-corpus passes
  ## - window sizes 256 B and 4 KiB sit beside the one-shot `pullAll` render
  echo "pull-window timing anchor (median of 15 runs, warm-up uncounted)"
  for (suite, iters) in [("deepseekv2lite", 400), ("qwen3", 150)]:
    let rs = rows(suite)
    let src = templateSource(suite)
    let (nodes, tables) = parseTemplate(src)
    let m = Machine(jinja: src, nodes: nodes)
    discard renderN(m, tables, rs[0].context, rs[0].clock, 500) # warm-up pass, not counted
    var line = &"  {suite:14} "
    for windowSize in [256, 4096]:
      let renderRow = proc (mm: Machine, tt: Tables, ctx: Value, clock: float64): int =
        renderWindowN(mm, tt, ctx, clock, 1, windowSize)
      let (mid, spread) = timedCorpusPasses(m, tables, rs, iters, renderRow)
      let flag = if spread > 20.0: "  VARIANCE" else: ""
      line.add &"win {windowSize:4} {mid:9.4f} ms/render  spread {spread:4.1f}%{flag}   "
    let renderWhole = proc (mm: Machine, tt: Tables, ctx: Value, clock: float64): int =
      renderOnce(mm, tt, ctx, clock).len
    let (mid, spread) = timedCorpusPasses(m, tables, rs, iters, renderWhole)
    let flag = if spread > 20.0: "  VARIANCE" else: ""
    line.add &"one-shot {mid:9.4f} ms/render  spread {spread:4.1f}%{flag}"
    echo line
    let calls256 = pullCallsPerPass(m, tables, rs, 256)
    let calls4k = pullCallsPerPass(m, tables, rs, 4096)
    echo &"    pull calls per pass: 256 B {calls256} ({calls256 div rs.len}/render), " &
        &"4 KiB {calls4k} ({calls4k div rs.len}/render)"

proc main(): void =
  benchHf()
  when defined(benchAlloc):
    echo ""
    echo "micro attribution (40-message context, public API only)"
    microAttribution(longShape(false, "long40-nt").ctx)
  echo ""
  benchCorpus()
  when not defined(benchAlloc):
    echo ""
    benchPullWindows()

main()
