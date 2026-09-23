# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Benchmark chattyninja template rendering over real chat templates plus the recorded corpus.
## One stimulus, two builds, from `workspace/chattyninja`:
##
## | Build   | Command                                                                                                                                                                                                                                            |
## | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | release | nim c -d:release --experimental:views --hints:off --warnings:off --path:src --path:tests --outdir:../../build/chattyninja/bin --nimcache:../../build/chattyninja/nimcache-bench bench/bench_render.nim && ../../build/chattyninja/bin/bench_render |
## | alloc   | nim c -d:release -d:benchAlloc -d:nimAllocStats --experimental:views --hints:off --warnings:off --path:src --path:tests -o:../../build/chattyninja/bin/bench_render_alloc bench/bench_render.nim && ../../build/chattyninja/bin/bench_render_alloc |
##
## - release build, ms/render and renders/s per template x conversation shape, median over
##   15 timed runs after 500 warm-up renders, spread-flagged when noisy
## - release build, pull-window timing over the corpus anchor rows, median ms/render beside
##   the one-shot render for 256 B and 4 KiB windows, plus pull-call counts
## - benchAlloc build, parse-time and render-time allocations per render, warm-up uncounted
##   through `system.getAllocStats()`, plus spill counts and micro attribution templates
##   isolating loop machinery, emit stringification and JSON serialization
##
## - both builds replay the recorded deepseekv2lite and qwen3 rows as a corpus anchor
## - the hf stimulus and the corpus anchor share one measurement method, so their numbers compare
##
## - stimulus templates come from the gitignored hf_models links, each byte-identical
##   to the corpus suite it mirrors
## - a template the engine cannot parse, or a construct a shape cannot render, is reported
##   as a coverage gap naming the construct
## - no `doAssert` anywhere, a failing doAssert hangs under `-d:nimAllocStats`

import std/[algorithm, importutils, monotimes, os, strformat, strutils, times]
import cnj_types {.all.}
import jinja_data_model {.all.}
import jinja_serialize {.all.}
import cnj_parse {.all.}
import cnj_engine {.all.}
import workspace/data_structures/src/small_seqs
import corpus/fixture_loader

type
  Shape = object
    ## One conversation shape, a name plus the context it renders.
    name: string
    ctx: JinjaVal

  Compiled = object
    ## One parsed template plus its bench label, the parse pair of one `parseJinjaTemplate` call.
    ##
    ## Borrow contract:
    ## `CompiledTemplate.jinja` borrows the template text, so `src` must outlive
    ## every render built from this record.
    label: string
    src: string
    tmpl: CompiledTemplate
    sym: CompiledSymbols

when defined(benchAlloc) and not defined(nimAllocStats):
  {.error: "build the allocation trace with -d:benchAlloc -d:nimAllocStats".}

# ── Stimulus builders ────────────────────────────────────────────────────────
#
# Render contexts are built with the engine's dict constructors. Template output observes
# dict insertion order, so the construction order the templates rely on is the engine's.

const HfModelsRoot = currentSourcePath().parentDir.parentDir.parentDir /
    "transformers" / "tests" / "hf_models"
  ## the gitignored checkpoint links, `<root>/<model>/chat_template.jinja`

func msgVal(role, content: string): JinjaVal =
  ## One chat message with the two keys every corpus template reads.
  var d = DictVal()
  dictSet(d, "role", strVal(role))
  dictSet(d, "content", strVal(content))
  dictVal(d)

func toolCallMsg(name: string, arguments: JinjaVal): JinjaVal =
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

func toolRespMsg(id, content: string): JinjaVal =
  ## One tool-result message bound to its call id.
  var d = DictVal()
  dictSet(d, "role", strVal("tool"))
  dictSet(d, "tool_call_id", strVal(id))
  dictSet(d, "content", strVal(content))
  dictVal(d)

func weatherTools(): JinjaVal =
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

func contextOf(msgs: seq[JinjaVal], tools: JinjaVal): JinjaVal =
  ## Render context holding `messages`, `tools` and `add_generation_prompt`.
  ## Template kwargs stay absent, the templates guard them with an `is defined` test.
  var d = DictVal()
  dictSet(d, "messages", seqVal(msgs))
  dictSet(d, "tools", tools)
  dictSet(d, "add_generation_prompt", boolVal(true))
  dictVal(d)


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
  ## long40 carries the tool round, long40-nt the same conversation without it,
  ## the pair separating the tool round's cost and any construct gap it triggers.
  [shortShape(), typicalShape(), longShape(true, "long40"), longShape(false, "long40-nt")]

const
  HfModels = [("glm47flash", "GLM-4.7-Flash"), ("glm53flash", "GLM-5.3-Flash"),
              ("qwen36", "Qwen3.6-35B-A3B"), ("qwen35", "Qwen3.5-0.8B")]
    ## bench label and hf_models directory per template

# ── Corpus anchor rows ───────────────────────────────────────────────────────
#
# Anchor rows come from the corpus fixture loader, one reading pass per suite
# so the timed and counted passes never touch the recorded rows.

func anchorContext(req: ChatRenderRequest): JinjaVal =
  ## Render context of one recorded row, the standard keys in recording order,
  ## then the row's kwargs.
  var ps: seq[tuple[k: string, v: JinjaVal]] =
    @[("messages", req.messages), ("tools", req.tools),
      ("documents", req.documents),
      ("add_generation_prompt", boolVal(req.addGenerationPrompt))]
  for pair in req.kwargs:
    ps.add pair
  dictVal(ps)

type
  AnchorRow = object
    ## One corpus anchor row, the render inputs materialized once.
    row: string
    ctx: JinjaVal
    clock: float64

proc anchorRows(suite: string): seq[AnchorRow] =
  ## Every recorded row of one suite, sorted row order, read once.
  for name in suiteRowNames(suite):
    let r = loadRow(suite, name)
    result.add AnchorRow(row: r.row, ctx: anchorContext(r.request),
        clock: r.request.clockEpoch)

# ── Measurement ──────────────────────────────────────────────────────────────

proc renderOnce(tmpl: CompiledTemplate, sym: CompiledSymbols, ctx: JinjaVal, clock = 0.0): string =
  ## Renders once, whole, through the buffered `pullInto` drain loop, exactly as the test suites do.
  var c = startJinjaRender(tmpl, sym, ctx, clock)
  var buf: array[4096, char]
  while true:
    let n = pullInto(c, buf)
    if n == 0:
      break
    for i in 0 ..< n:
      result.add buf[i]

proc renderN(tmpl: CompiledTemplate, sym: CompiledSymbols, ctx: JinjaVal, clock: float64, n: int): int =
  ## Renders `n` times and returns the accumulated output byte count, so the loop
  ## consumes every render and nothing is optimized away.
  for _ in 0 ..< n:
    result += renderOnce(tmpl, sym, ctx, clock).len

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

proc timeShape(tmpl: CompiledTemplate, sym: CompiledSymbols, ctx: JinjaVal, iters: int): string =
  ## Times one template x shape.
  ##
  ## 500 warm-up renders, then rounds of 15 timed runs of `iters` renders each.
  ##
  ## - the first round whose spread stays within 20% is reported
  ## - a busy machine gets up to 4 rounds and the most stable one is reported, still
  ##   flagged when its spread exceeds 20%, so noise is never silently averaged away
  let warm = renderN(tmpl, sym, ctx, 0, 500)
  var best: seq[float64]
  var bestSpread = 1e9
  for _ in 0 ..< 4:
    let samples = timedRuns(15, iters):
      discard renderN(tmpl, sym, ctx, 0, iters)
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

  proc allocShape(tmpl: CompiledTemplate, sym: CompiledSymbols, ctx: JinjaVal, iters: int): string =
    ## Allocates per render for one template x shape.
    ## One warm-up render goes uncounted, then `iters` renders are counted via `getAllocStats()`.
    discard renderOnce(tmpl, sym, ctx)
    let a = allocsOf:
      for _ in 0 ..< iters:
        discard renderOnce(tmpl, sym, ctx)
    fmt"{a.float64 / iters.float64:8.2f} allocs/render"

  proc parseAllocs(src: string): string =
    ## Parse-time allocations for one template.
    ## One parse is counted after an uncounted warm-up parse of a tiny template.
    let a = allocsOf:
      discard parseJinjaTemplate(src)
    fmt"{a:8d} allocs/parse"

  proc corpusAllocAnchor(suite: string, iters: int): string =
    ## Replays one recorded suite's rows, measured like the hf stimulus so the numbers compare.
    ##
    ## - one full warm-up pass over every row, uncounted
    ## - then counted passes, reported per row and as the suite mean
    let (m, sym) = parseJinjaTemplate(suiteTemplateSource(suite))
    let rs = anchorRows(suite)
    for _ in 0 ..< 3: # warm-up pass over every row, uncounted
      for r in rs:
        discard renderOnce(m, sym, r.ctx, r.clock)
    var perRow = ""
    var total = 0.0
    for r in rs:
      let a = allocsOf:
        for _ in 0 ..< iters:
          discard renderOnce(m, sym, r.ctx, r.clock)
      let per = a.float64 / iters.float64
      perRow.add &"{r.row}={per:.1f} "
      total += per
    fmt"suite mean {total / rs.len.float64:.1f} allocs/render over {rs.len} rows: {perRow}"

  proc microAttribution(ctx: JinjaVal): void =
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
      let (m, sym) = parseJinjaTemplate(src)
      discard renderOnce(m, sym, ctx) # warm-up render, not counted
      let a = allocsOf:
        for _ in 0 ..< iters:
          discard renderOnce(m, sym, ctx)
      echo &"  micro {name:14} {a.float64 / iters.float64:8.2f} allocs/render"
    # Expression-shape matrix, per-emit cost by token shape, all against the same
    # for-loop baseline, separating punctuator, literal and lookup costs.
    for (name, src) in [("emit-role", "{% for m in messages %}{{ m.role }}{% endfor %}"),
                        ("emit-int", "{% for m in messages %}{{ 1 }}{% endfor %}"),
                        ("emit-bracket", "{% for m in messages %}{{ m['content'] }}{% endfor %}"),
                        ("emit-concat", "{% for m in messages %}{{ m.role ~ 'x' }}{% endfor %}")]:
      let (m, sym) = parseJinjaTemplate(src)
      discard renderOnce(m, sym, ctx) # warm-up render, not counted
      let a = allocsOf:
        for _ in 0 ..< iters:
          discard renderOnce(m, sym, ctx)
      echo &"  micro {name:14} {a.float64 / iters.float64:8.2f} allocs/render"
    # Direct stringifier costs over a representative message content.
    let content = strVal(
        "Turn 12: here is more context about the task; keep the details accurate.")
    let ps = allocsOf:
      for _ in 0 ..< 1000:
        discard pyStr(content)
    echo &"  micro pyStr(msg content) {ps.float64 / 1000.0:5.2f} allocs/call"
    # Direct attribution of the emit path's two owning copies, through public fields.
    # One copy belongs to the context lookup that fills a `JinjaVal`.
    # The other belongs to the pending-piece assignment that moves the string into render-state storage.
    let msgs = ctx.d.dictGet("messages")
    # Message 1's content is runtime-built, matching the templates' real message
    # strings. The system message's content is a compile-time constant backed
    # by a literal, measuring 0 buffer shares in both copies below.
    let msg1 = msgs.xs.items[1]
    let dg = allocsOf:
      for _ in 0 ..< 1000:
        discard msg1.d.dictGet("content")
    echo &"  micro dictGet(content)   {dg.float64 / 1000.0:5.2f} allocs/call"
    let (pm, psyms) = parseJinjaTemplate("{{ m.content }}")
    var pcx = startJinjaRender(pm, psyms, ctx, 0.0)
    let cv = msg1.d.dictGet("content")
    let pc = allocsOf:
      for _ in 0 ..< 1000:
        pcx.state.pend = Piece(pos: 0, kind: pkStr, s: cv.s)
        pcx.state.pend = Piece(kind: pkNone)
    echo &"  micro pend piece copy    {pc.float64 / 1000.0:5.2f} allocs/call"
    let tj = allocsOf:
      for _ in 0 ..< 100:
        discard toJson(weatherTools())
    echo &"  micro tojson(tools)      {tj.float64 / 100.0:5.2f} allocs/call"

# ── Harness ──────────────────────────────────────────────────────────────────

const parseGapLabels = ["glm53flash"]
  ## hf_models templates the engine cannot parse yet, the declared parse gaps.
  ## `glm53flash` trips the unimplemented `nkBreak` node, at 7 sites inside
  ## the macro `has_dup_tool_result_id`.

proc compileHf(): seq[Compiled] =
  ## Parses every hf_models template into a `Compiled` record.
  ## Templates in `parseGapLabels` are skipped with a note, any other parse
  ## failure raises and fails the bench.
  when defined(benchAlloc):
    discard parseJinjaTemplate("{% if x %}a{% endif %}") # warm-up parse, uncounted
  for (label, dir) in HfModels:
    if label in parseGapLabels:
      echo &"parse {label:11} SKIP  declared gap, skipped"
      continue
    let path = HfModelsRoot / dir / "chat_template.jinja"
    let src = readFile(path)
    let (tmpl, sym) = parseJinjaTemplate(src)
    when defined(benchAlloc):
      let census = spillCensus(tmpl.nodes)
      echo &"parse {label:11} OK    {tmpl.nodes.len} nodes, {census.summary}, " &
          &"maxSlots {census.maxSlots}, {parseAllocs(src)}"
    else:
      echo &"parse {label:11} OK    {tmpl.nodes.len} nodes"
    result.add Compiled(label: label, src: src, tmpl: tmpl, sym: sym)

const renderGapShapes = [("qwen36", "long40"), ("qwen35", "long40")]
  ## Template and shape pairs the engine cannot render yet, the declared render gaps.
  ## Both raise on the `long40` tool round through the unimplemented `|items` filter.

proc benchHf(): void =
  ## Times and traces every compiled template over every shape.
  ## Pairs in `renderGapShapes` are skipped with a note, any other raise
  ## propagates and fails the bench.
  var compiled = compileHf()
  echo ""
  let shapes = benchShapes()
  for c in compiled.mitems:
    echo &"render {c.label}"
    for s in shapes:
      if (c.label, s.name) in renderGapShapes:
        echo &"  {s.name:10} SKIP   declared gap, skipped"
        continue
      when defined(benchAlloc):
        echo &"  {s.name:10} {allocShape(c.tmpl, c.sym, s.ctx, 50)}"
      else:
        let iters = case s.name
            of "short2": 2000
            of "typical10": 800
            else: 200
        echo &"  {s.name:10} {timeShape(c.tmpl, c.sym, s.ctx, iters)}"
    echo ""

proc benchCorpus(): void =
  ## Corpus anchor over the recorded deepseekv2lite and qwen3 rows, one measurement
  ## method with the hf stimulus, so the numbers compare.
  when defined(benchAlloc):
    echo "corpus alloc anchor (warm-up uncounted)"
    echo "  deepseekv2lite " & corpusAllocAnchor("deepseekv2lite", 50)
    echo "  qwen3          " & corpusAllocAnchor("qwen3", 50)
  else:
    echo "corpus timing anchor (median of 15 runs, warm-up uncounted)"
    for (suite, iters) in [("deepseekv2lite", 400), ("qwen3", 150)]:
      let (m, sym) = parseJinjaTemplate(suiteTemplateSource(suite))
      let rs = anchorRows(suite)
      discard renderN(m, sym, rs[0].ctx, rs[0].clock, 500) # warm-up pass, not counted
      var best: seq[float64]
      var bestSpread = 1e9
      for _ in 0 ..< 4:
        let samples = timedRuns(15, iters):
          for _ in 0 ..< iters:
            for r in rs:
              discard renderOnce(m, sym, r.ctx, r.clock)
        let spread = (max(samples) - min(samples)) / median(samples) * 100.0
        if spread < bestSpread:
          bestSpread = spread
          best = samples
        if bestSpread <= 20.0:
          break
      let mid = median(best) / rs.len.float64 # a sample is ms/pass, reported per render
      let flag = if bestSpread > 20.0: "  VARIANCE" else: ""
      echo &"  {suite:14} {mid:9.4f} ms/render all {rs.len} rows   spread {bestSpread:4.1f}%{flag}"

# ── Pull-window timing ───────────────────────────────────────────────────────

proc renderWindowN(tmpl: CompiledTemplate, sym: CompiledSymbols, ctx: JinjaVal, clock: float64, n, windowSize: int): int =
  ## Renders `n` times through a `windowSize`-byte stack window and returns the byte
  ## count accumulated across renders, so the loop consumes every render.
  var buf: array[4096, char]
  for _ in 0 ..< n:
    var c = startJinjaRender(tmpl, sym, ctx, clock)
    while true:
      let got = pullInto(c, buf.toOpenArray(0, windowSize - 1))
      if got == 0:
        break
      result += got

proc pullCallsPerPass(tmpl: CompiledTemplate, sym: CompiledSymbols, rs: seq[AnchorRow], windowSize: int): int =
  ## Pull calls that returned bytes, one untimed pass over every row.
  var buf: array[4096, char]
  for r in rs:
    var c = startJinjaRender(tmpl, sym, r.ctx, r.clock)
    while true:
      let got = pullInto(c, buf.toOpenArray(0, windowSize - 1))
      if got == 0:
        break
      inc result

proc timedCorpusPasses(tmpl: CompiledTemplate, sym: CompiledSymbols,
    rs: seq[AnchorRow], iters: int,
    render: proc (tmpl: CompiledTemplate, sym: CompiledSymbols,
        ctx: JinjaVal, clock: float64): int):
    tuple[mid, spread: float64] =
  ## Rounds of 15 timed runs of `iters` full-corpus passes of `render`, method
  ## identical to the corpus timing anchor:
  ## - the first round whose spread stays within 20% is reported
  ## - a busy machine gets up to 4 rounds
  ## - a spread above 20% flags the sample
  ##
  ## Returns the median in milliseconds per render, next to the per-render hf
  ## stimulus numbers:
  ##   - a timed run measures `iters` passes over all `rs` rows
  ##   - the median is divided by `rs.len`
  var best: seq[float64]
  var bestSpread = 1e9
  for _ in 0 ..< 4:
    let samples = timedRuns(15, iters):
      for _ in 0 ..< iters:
        for r in rs:
          discard render(tmpl, sym, r.ctx, r.clock)
    let spread = (max(samples) - min(samples)) / median(samples) * 100.0
    if spread < bestSpread:
      bestSpread = spread
      best = samples
    if bestSpread <= 20.0:
      break
  # each timed run iterates `iters` full-corpus passes, a sample is
  # milliseconds per pass, divided by the row count to report per render
  (median(best) / rs.len.float64, bestSpread)

proc benchPullWindows(): void =
  ## Pull-window timing over the corpus anchor rows, harness identical to the corpus
  ## timing anchor:
  ## - 500 warm-up renders, then median of 15 timed runs of `iters` full-corpus passes
  ## - window sizes 256 B and 4 KiB sit beside the one-shot `renderToString` render
  echo "pull-window timing anchor (median of 15 runs, warm-up uncounted)"
  for (suite, iters) in [("deepseekv2lite", 400), ("qwen3", 150)]:
    let (m, sym) = parseJinjaTemplate(suiteTemplateSource(suite))
    let rs = anchorRows(suite)
    discard renderN(m, sym, rs[0].ctx, rs[0].clock, 500) # warm-up pass, not counted
    var line = &"  {suite:14} "
    for windowSize in [256, 4096]:
      let renderRow = proc (mm: CompiledTemplate, sym: CompiledSymbols,
          ctx: JinjaVal, clock: float64): int =
        renderWindowN(mm, sym, ctx, clock, 1, windowSize)
      let (mid, spread) = timedCorpusPasses(m, sym, rs, iters, renderRow)
      let flag = if spread > 20.0: "  VARIANCE" else: ""
      line.add &"win {windowSize:4} {mid:9.4f} ms/render  spread {spread:4.1f}%{flag}   "
    let renderWhole = proc (mm: CompiledTemplate, sym: CompiledSymbols,
        ctx: JinjaVal, clock: float64): int =
      renderOnce(mm, sym, ctx, clock).len
    let (mid, spread) = timedCorpusPasses(m, sym, rs, iters, renderWhole)
    let flag = if spread > 20.0: "  VARIANCE" else: ""
    line.add &"one-shot {mid:9.4f} ms/render  spread {spread:4.1f}%{flag}"
    echo line
    let calls256 = pullCallsPerPass(m, sym, rs, 256)
    let calls4k = pullCallsPerPass(m, sym, rs, 4096)
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
