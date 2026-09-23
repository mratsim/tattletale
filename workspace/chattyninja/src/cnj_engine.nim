# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Compiled chattyninja templates and their render driver.
# | Step     | Behavior                                                                                                                                                      |
# | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | parse    | `parseTemplate` appends nodes in source order, resolving the whitespace policy and interning names into `CompiledSymbols`                                     |
# | load     | `parseTemplate` returns the artifact borrowing the template text, so it holds no mutable state and cannot outlive the text it points into                     |
# | render   | `startRender` opens a `JinjaRenderContext` over the artifact and `pullInto` walks the node list through `steps`, all control state in the context's `RenderState` |
# | dispatch | `steps` is total over `NodeKind`, so a node's meaning is a pure function of its kind and no node carries a proc field or program counter                      |
# | force    | `startRender` binds the macro forcer into the context, expressions reading the render state's scopes, root and clock directly                                 |
# Resumption state for a re-entered step lives in the render state's row stack, never in a node.
# `nkFor`, `nkSetBlock` and `nkGeneration` are re-entered by their bodies, `nkIf`
# single-entry, parse time backpatching its branch bodies past the whole chain.
#
# Run:
#   from the repo root, `nim test_chattyninja` builds and runs every suite with its variants.

# Public API:
#   startRender, pullInto, pullAll and renderToString. Everything else is engine plumbing.

import cnj_types {.all.}
import jinja_data_model {.all.}
import jinja_serialize {.all.}
import cnj_parse {.all.}
import jinja_interpolation {.all.}

type
  Step = proc (c: JinjaRenderContext, n: int32) {.nimcall.}
    ## One construct's step, writing only through `c.state` and leaving `curNode` on the node control enters next.

proc forceMacro(c: JinjaRenderContext, mc: MacroVal, args: var Args): JinjaVal

proc startMacro(c: JinjaRenderContext, lo, hi: int, call: DeferredMacroCall, retNode: int32)

proc forceCondCall(c: JinjaRenderContext, v: JinjaVal, lo, hi: int32): JinjaVal =
  ## Renders a macro call read in boolean position to its output value, `lo` and `hi` bounding
  ## the raise reported when no forcer was supplied.
  if c.force.isNil:
    raise jinjaErr("a macro call result was consumed where no macro forcer was supplied", lo, hi - lo)
  c.force(c, v.pc.mc, v.pc.args)

func lookupNameById(c: JinjaRenderContext, id: int32): JinjaVal =
  ## Returns the binding of an interned name, undefined when absent. The scope key is
  ## the id, no string rebuilt per lookup.
  if id == NoLink:
    return undefinedVal()
  var got: JinjaVal
  if c.state.scopeHas(id, got):
    return got
  if c.state.root.kind == vkDict and id < c.symbols.names.len.int32:
    return c.state.root.d.dictGet(c.symbols.names[id])
  undefinedVal()

# Output:

func emitSpan(st: var RenderState, lo, hi: int32) =
  ## Makes a template-text span the pending piece.
  if hi <= lo:
    return
  # One piece is pending at a time, drained before the next dispatch, so a second
  # piece here would silently drop the first one's bytes.
  doAssert st.pend.kind == pkNone, "a step queued a piece while one was still pending"
  st.pend = Piece(pos: 0, kind: pkSpan, lo: lo, hi: hi)

func emitStr(st: var RenderState, s: sink string) =
  ## Makes a materialized string the pending piece, a runtime-built emit string never copied,
  ## an empty string queuing nothing.
  if s.len == 0:
    return
  doAssert st.pend.kind == pkNone, "a step queued a piece while one was still pending"
  st.pend = Piece(pos: 0, kind: pkStr, s: s)

func emitCut(st: var RenderState, v: sink JinjaVal) =
  ## Makes a cut value the pending piece, moving the string out of the caller's value, an empty cut queueing nothing.
  if v.lo == v.hi:
    return
  doAssert st.pend.kind == pkNone, "a step queued a piece while one was still pending"
  st.pend = Piece(pos: 0, kind: pkCut, raw: move v.raw, clo: v.lo, chi: v.hi)

func emitValue(st: var RenderState, v: sink JinjaVal) =
  ## Makes a derived value the pending piece, taking over the caller's value, the serializer
  ## in `st.lazy` draining across pullInto calls byte-exact with `pyStr`.
  doAssert st.pend.kind == pkNone, "a step queued a piece while one was still pending"
  serReset(st.lazy, v, smStr)
  st.pend = Piece(kind: pkLazy)

template pieceLen(p: Piece): int =
  ## Length in bytes of a pending piece, lazy pieces drained through the serializer instead.
  case p.kind
  of pkNone: 0
  of pkSpan: (p.hi - p.lo).int
  of pkStr: p.s.len
  of pkCut: (p.chi - p.clo).int
  of pkLazy: 0

# Binding:

func bindName(st: var RenderState, name: int32, val: JinjaVal) =
  ## Binds a name in the innermost scope, replacing an existing binding there.
  var sc = st.scopes.len - 1
  for bi in 0 ..< st.scopes[sc].len:
    if st.scopes[sc][bi].name == name:
      st.scopes[sc][bi].val = val
      return
  st.scopes[sc].add Binding(name: name, val: val)

# Steps:
#
# Every step reads its node's payload through the `nd` slot-accessor templates, which expand
# textually onto the node entry. Render code never binds a `Node` value, a binding running
# SmallSeq's `=copy` and heap-allocating a spilled payload's block (7% of corpus nodes spill).

func stepVerbatim(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## Streams the final text run, whose span already reflects every whitespace rule.
  template nd: Node = c.tmpl.nodes[n]
  c.state.emitSpan(nd.lo, nd.hi)
  c.state.curNode = nd.succ

proc stepEmit(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## Evaluates the expression span, a whole-expression macro call entering its body instead,
  ## the body's pieces draining until the row closes.
  template nd: Node = c.tmpl.nodes[n]
  var v = evalSpan(c, nd.lo, nd.hi)
  if v.kind == vkCall:
    startMacro(c, nd.lo.int, nd.hi.int, v.pc, nd.succ)
    return
  if v.kind == vkCut:
    c.state.emitCut(move v)
  elif v.kind == vkStr:
    c.state.emitStr(move v.s)
  else:
    c.state.emitValue(v)
  c.state.curNode = nd.succ

proc stepIf(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## Chooses a branch once, branch bodies terminating past the chain, so no row exists for it.
  ## A pending macro call in the condition renders to its output bytes before the truth test.
  template nd: Node = c.tmpl.nodes[n]
  var v = evalSpan(c, nd.lo, nd.hi)
  if v.kind == vkCall:
    v = forceCondCall(c, v, nd.lo, nd.hi)
  if isTruthy(v, nd.lo):
    c.state.curNode = if nd.child == NoLink: nd.succ else: nd.child
  elif nd.alt != NoLink:
    c.state.curNode = nd.alt
  else:
    c.state.curNode = nd.succ

func iterSeq(v: JinjaVal): LoopState =
  ## Iterable leg for a sequence value, borrowing the shared payload without copying.
  ## Sequence payloads never mutate in place at render time.
  LoopState(xs: v.xs)

func iterMapping(v: JinjaVal): LoopState =
  ## Iterable leg for a mapping, materializing once the key strings the loop binds.
  var acc = newSeq[JinjaVal](v.d.keys.len)
  for i, k in v.d.keys:
    acc[i] = strVal(k)
  LoopState(xs: SeqVal(items: acc))

func iterChars(v: JinjaVal): LoopState =
  ## Iterable leg for a string, materializing one single-codepoint value per codepoint.
  let v = if v.kind == vkCut: materializeVal(v) else: v
  LoopState(xs: SeqVal(items: codepointVals(v.s)))

func iterRange(v: JinjaVal): LoopState =
  ## Iterable leg for a lazy range, the cursor carrying the bounds and elements computing
  ## per index through `loopItem`.
  LoopState(r: v.r, isRange: true)

func notIterable(v: JinjaVal, lo, hi: int): void {.noreturn.} =
  ## Shared raise leg of the iterable dispatch, one report for every kind no loop walks.
  ## `lo` and `hi` bound the iterable expression, the loop header the raise reports.
  if v.kind == vkUndefined:
    raise jinjaErr("cannot iterate an undefined value", lo, hi - lo)
  raise jinjaErr("cannot iterate a " & $v.kind, lo, hi - lo)

func loopStateOf(v: JinjaVal, lo, hi: int): LoopState =
  ## Dispatch at the loop's chain entry, one leg per iterable kind the corpus supports
  ## and the shared raise leg for everything else. `stepFor` coerces a pending call first.
  case v.kind
  of vkSeq: iterSeq(v)
  of vkDict, vkNs: iterMapping(v)
  of vkStr, vkCut: iterChars(v)
  of vkRange: iterRange(v)
  else: notIterable(v, lo, hi)

func bindTargets(c: JinjaRenderContext, n: int32, item: JinjaVal) =
  ## Binds the `nkFor` loop targets at `n`, more than one target unpacking a sequence, which
  ## is what `x.items()` feeds through `{% for k, v in x.items() %}`.
  template nd: Node = c.tmpl.nodes[n]
  let ntargets = int(nd.targetCount)
  if ntargets == 1:
    c.state.bindName(nd.targetAt(0), item)
  else:
    if item.kind != vkSeq or item.xs.items.len != ntargets:
      raise jinjaErr("`for` unpacks " & $ntargets & " targets from a value that is not a " &
          $ntargets & "-element sequence", nd.lo.int, nd.hi.int - nd.lo.int)
    for i in 0 ..< ntargets:
      c.state.bindName(nd.targetAt(i), item.xs.items[i])

proc filterKeep(c: JinjaRenderContext, lo, hi: int32): bool =
  ## Evaluates one filter clause in boolean position, a macro call rendering
  ## to its output value before the truth test. Returns the keep decision.
  var evaluated = evalSpan(c, lo, hi)
  if evaluated.kind == vkCall:
    evaluated = forceCondCall(c, evaluated, lo, hi)
  isTruthy(evaluated, lo)

proc forStep(c: JinjaRenderContext, n: int32, lp: LoopState): bool =
  ## Per-item loop step moving the shared cursor one item, binding the loop targets, running
  ## the filter clause, the keep decision returned. Exhaustion reads off the cursor.
  template nd: Node = c.tmpl.nodes[n]
  inc lp.idx
  let idx = lp.idx
  if idx >= lp.loopLen:
    return false
  result = nd.filterLo == NoLink
  bindTargets(c, n, lp.loopItem(idx))
  if nd.filterLo != NoLink:
    result = filterKeep(c, nd.filterLo, nd.filterHi)

func closeRow(st: var RenderState, at: int, next: int32) =
  ## Leaves the row's construct, truncating scopes to the row's `scopeAt` mark,
  ## rows to `at`, landing `curNode` on `next`. A raise abandons the render, no partial close.
  st.scopes.setLen(st.rows[at].scopeAt)
  st.curNode = next
  st.rows.setLen(at)

proc advanceFor(c: JinjaRenderContext, n: int32) =
  ## Re-entry path moving the cursor to the next filter-passing item, re-entering the body, or closing the row and continuing past the loop.
  template nd: Node = c.tmpl.nodes[n]
  while true:
    let fi = c.state.rows.len - 1
    if forStep(c, n, c.state.rows[fi].loop):
      break
    closeRow(c.state, fi, nd.succ)
    return
  c.state.curNode = nd.child

proc stepFor(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## `{% for %}` advances on a matching row, otherwise sets up the iteration, an empty body completing inline and a filter clause per item.
  template nd: Node = c.tmpl.nodes[n]
  if c.state.rows.len > 0 and c.state.rows[^1].kind == frFor and c.state.rows[^1].node == n:
    advanceFor(c, n)
    return
  var iterable = evalSpan(c, nd.lo, nd.hi)
  if iterable.kind == vkCall:
    # A macro call in the iterable position renders at its call site, its output
    # what the loop walks, matching upstream.
    iterable = forceCondCall(c, iterable, nd.lo, nd.hi)
  let lp = loopStateOf(iterable, nd.lo.int, nd.hi.int)
  if lp.loopLen == 0:
    c.state.curNode = nd.succ
    return
  let scopeBase = c.state.scopes.len
  c.state.scopes.add @[]
  c.state.rows.add Row(node: n, kind: frFor, loop: lp,
      scopeAt: scopeBase, filterLo: nd.filterLo, filterHi: nd.filterHi)
  lp.idx = 0
  bindTargets(c, n, lp.loopItem(0))
  c.state.bindName(nd.loopName, loopVal(lp))
  if nd.child == NoLink:
    if nd.filterLo != NoLink:
      # Every item's clause runs, item 0's included, the keep decision
      # discarded there, the clause's raises and side effects the observable behavior.
      # Loop control reads the shared cursor, which every step commits, so a rejected
      # item does not end the walk.
      discard filterKeep(c, nd.filterLo, nd.filterHi)
      while lp.idx < lp.loopLen:
        discard forStep(c, n, lp)
    closeRow(c.state, c.state.rows.len - 1, nd.succ)
    return
  # Item 0's clause runs before body entry. A rejected item 0 takes the advance walk,
  # the body entering on the first kept item or the row closing past the loop's end.
  if nd.filterLo != NoLink and not filterKeep(c, nd.filterLo, nd.filterHi):
    while true:
      if forStep(c, n, lp):
        break
      closeRow(c.state, c.state.rows.len - 1, nd.succ)
      return
  c.state.curNode = nd.child

proc stepSet(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## Single-target `{% set %}`, the target carried as an interned name id in the child slot,
  ## emitting nothing, the pending piece untouched.
  template nd: Node = c.tmpl.nodes[n]
  var v = evalSpan(c, nd.lo, nd.hi)
  if v.kind == vkCall:
    # A macro call evaluates at its call site, its output bound, matching upstream:
    # the body's side effects land once, never re-run by a later use.
    v = forceCondCall(c, v, nd.lo, nd.hi)
  c.state.bindName(nd.child, v)
  c.state.curNode = nd.succ

func gap(kindName, corpusSite: string, lo, hi: int): void {.noreturn.} =
  ## Reports a declared construct that is not implemented, naming `kindName`
  ## and the corpus site that demands it, `lo` and `hi` bounding the construct's node.
  raise jinjaErr(kindName & " is not implemented; " & corpusSite, lo, hi - lo,
      cause = ceUnimplemented)

func outsideEveryFor(tmpl: CompiledTemplate, lo, hi: int32): void {.noreturn.} =
  ## Raises the no-enclosing-`{% for %}` report for a break or continue,
  ## `lo` and `hi` bounding the keyword.
  raise jinjaErr("`{% " & spanString(tmpl.jinja.toOpenArray(int(lo), int(hi) - 1)) &
      " %}` ran outside every `{% for %}`", int(lo), int(hi - lo))

func closeMacroRow(st: var RenderState, at: int, next: int32) =
  ## Closes the macro row at index `at` through `closeRow`, the boundary stop a break takes,
  ## control continuing at the row's return node.
  closeRow(st, at, next)
  dec st.macroDepth

func stepBreak(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## Unwinds to the nearest for-row. A break stops at the next macro-call boundary, a continue
  ## leaves the for-row and its scope in place, generation rows above abandon their partial spans.
  template nd: Node = c.tmpl.nodes[n]
  # Continue discriminates from break over the keyword span, trailing whitespace trimmed.
  var kwHi = int(nd.hi)
  while kwHi > int(nd.lo) and c.tmpl.jinja[kwHi - 1] in Whitespace:
    dec kwHi
  let cont = c.tmpl.jinja.toOpenArray(int(nd.lo), kwHi - 1) == "continue"
  var k = c.state.rows.len - 1
  while k >= 0:
    let r = c.state.rows[k]
    if r.kind == frMacro:
      if cont:
        outsideEveryFor(c.tmpl, nd.lo, nd.hi)
      closeMacroRow(c.state, k, c.state.rows[k].retNode)
      return
    if r.kind == frFor:
      if cont:
        c.state.rows.setLen(k + 1)
        c.state.curNode = r.node
      else:
        closeRow(c.state, k, c.tmpl.nodes[r.node].succ)
      return
    if r.kind == frGeneration:
      # An abandoned generation body still ran its bytes, the span closing at the position reached.
      c.state.spans.add (r.spanStart, c.state.cur)
    # Generation rows push no scope, nothing to pop.
    dec k
  outsideEveryFor(c.tmpl, nd.lo, nd.hi)

proc stepSetNs(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## `ns.field = expr`, mutating the shared namespace mapping in place, visible to every
  ## holder of the `DictVal` ref, and emitting nothing.
  template nd: Node = c.tmpl.nodes[n]
  let ns = lookupNameById(c, nd.target)
  if ns.kind != vkNs:
    raise jinjaErr("`" & c.symbols.names[nd.target] & "` is not a namespace, so it has no `" &
        c.symbols.names[nd.field] & "` to set", nd.lo.int, nd.hi.int - nd.lo.int)
  var v = evalSpan(c, nd.lo, nd.hi)
  if v.kind == vkCall:
    # A macro call evaluates at its call site, its output bound, matching upstream:
    # the body's side effects land once, never re-run by a later use.
    v = forceCondCall(c, v, nd.lo, nd.hi)
  dictSet(ns.d, c.symbols.names[nd.field], v)
  c.state.curNode = nd.succ

func stepSetBlock(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## Opens a capture sink for the body and, on re-entry, binds the capture to the target name.
  template nd: Node = c.tmpl.nodes[n]
  gap("nkSetBlock", "corpus demand is 2 sites: gemma4.jinja:322 and northminicode10.jinja:2",
      nd.lo.int, nd.hi.int)

func stepGeneration(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## `{% generation %}` renders its body byte for byte, the model's turn span recorded around
  ## it in `RenderState.spans`, an empty body an empty span and the body adding no scope.
  template nd: Node = c.tmpl.nodes[n]
  if c.state.rows.len > 0 and c.state.rows[^1].kind == frGeneration and c.state.rows[^1].node == n:
    c.state.spans.add (c.state.rows[^1].spanStart, c.state.cur)
    closeRow(c.state, c.state.rows.len - 1, nd.succ)
    return
  if nd.child == NoLink:
    # An empty body never re-enters, the span closing at once, empty.
    c.state.spans.add (c.state.cur, c.state.cur)
    c.state.curNode = nd.succ
    return
  # no scope of its own, the entry mark left untouched so the close pops nothing
  var r = Row(node: n, kind: frGeneration, scopeAt: c.state.scopes.len)
  r.spanStart = c.state.cur
  c.state.rows.add r
  c.state.curNode = nd.child

func stepMacroDef(c: JinjaRenderContext, n: int32) {.nimcall.} =
  ## Binds a macro value and emits nothing, the body never running here. A row arriving
  ## back on the definition node closes and drains its pieces.
  template nd: Node = c.tmpl.nodes[n]
  if c.state.rows.len > 0 and c.state.rows[^1].kind == frMacro and c.state.rows[^1].node == n:
    closeMacroRow(c.state, c.state.rows.len - 1, c.state.rows[^1].retNode)
    return
  c.state.bindName(nd.macroName, macroVal(
      MacroVal(name: nd.macroName, body: nd.child, node: n)))
  c.state.curNode = nd.succ

const
  CaptureDrainCap = 256
    ## Stack buffer `capturePend` hands to `pullSer` per drain call, sized to hold a whole
    ## scalar rendering so the capture copies in one grow.

  Steps: array[NodeKind, Step] = [
    stepVerbatim, stepEmit, stepIf, stepFor, stepBreak, stepSet, stepSetNs, stepSetBlock,
    stepGeneration, stepMacroDef
  ]
    ## Dispatch table, total over `NodeKind`, a new kind without a step a compile error.

proc bindMacroArgs(c: JinjaRenderContext, n: int32, args: var Args, lo, hi: int) =
  ## Binds one macro call's parameters in a fresh scope, positional arguments by count,
  ## keyword arguments by name, defaults evaluated in order inside the macro scope. Mismatches raise located.
  template nd: Node = c.tmpl.nodes[n]
  template argErr(what: string) {.dirty.} =
    if lo == NoOffset:
      raise jinjaErr(what)
    raise jinjaErr(what, lo, hi - lo)
  let nparams = int(nd.paramCount)
  let macroName = c.symbols.names[nd.macroName]
  template keywordName(a: Arg): untyped =
    c.tmpl.jinja.toOpenArray(a.nameLo.int, a.nameHi.int - 1)
  var used: array[ArgsCap, bool]
  var posCount = 0
  var firstKeyword = -1
  for i in 0 ..< args.len:
    if args[i].nameLo == NoLink:
      inc posCount
    elif firstKeyword < 0:
      firstKeyword = i
  if posCount > nparams:
    argErr("macro `" & macroName & "` takes at most " & $nparams & " positional argument(s)")
  if firstKeyword >= 0:
    for i in firstKeyword + 1 ..< args.len:
      if args[i].nameLo == NoLink:
        argErr("a positional argument follows a keyword argument in the call to `" &
            macroName & "`")
  for k in 0 ..< nparams:
    var val = undefinedVal()
    var bound = false
    if k < posCount:
      var seen = 0
      for i in 0 ..< args.len:
        if args[i].nameLo == NoLink:
          if seen == k:
            val = args[i].val
            used[i] = true
            bound = true
            break
          inc seen
    else:
      for i in 0 ..< args.len:
        if not used[i] and args[i].nameLo != NoLink and
            keywordName(args[i]) == c.symbols.names[nd.paramNameAt(k)]:
          val = args[i].val
          used[i] = true
          bound = true
          break
    if not bound:
      if nd.paramDefLoAt(k) == NoLink:
        val = undefinedVal()
      else:
        val = evalSpan(c, nd.paramDefLoAt(k), nd.paramDefHiAt(k))
    c.state.bindName(nd.paramNameAt(k), val)
  for i in 0 ..< args.len:
    if used[i] or args[i].nameLo == NoLink:
      continue
    var known = false
    for k in 0 ..< nparams:
      if keywordName(args[i]) == c.symbols.names[nd.paramNameAt(k)]:
        known = true
        break
    if known:
      argErr("macro `" & macroName & "` got multiple values for argument `" &
          spanString(keywordName(args[i])) & "`")
    argErr("macro `" & macroName & "` takes no keyword argument `" &
        spanString(keywordName(args[i])) & "`")

func capturePend(c: JinjaRenderContext, outp: var string) =
  ## Appends the pending piece's bytes to `outp` and retires the piece, the capture form
  ## of a forced macro body whose output never reaches the caller's window.
  case c.state.pend.kind
  of pkNone:
    discard
  of pkSpan:
    let at = outp.len
    let n = int(c.state.pend.hi - c.state.pend.lo) - c.state.pend.pos
    outp.setLen(at + n)
    copyMem(addr outp[at], unsafeAddr c.tmpl.jinja[int c.state.pend.lo + c.state.pend.pos], n)
    c.state.pend = Piece(kind: pkNone)
  of pkStr:
    addView(outp, c.state.pend.s.toOpenArray(c.state.pend.pos, c.state.pend.s.len - 1))
    c.state.pend = Piece(kind: pkNone)
  of pkCut:
    addView(outp, c.state.pend.raw.toOpenArray(c.state.pend.clo + c.state.pend.pos, c.state.pend.chi - 1))
    c.state.pend = Piece(kind: pkNone)
  of pkLazy:
    var buf: array[CaptureDrainCap, char]
    while true:
      let n = pullSer(c.state.lazy, buf)
      if n == 0:
        break
      let at = outp.len
      outp.setLen(at + n)
      copyMem(addr outp[at], addr buf[0], n)
    c.state.pend = Piece(kind: pkNone)

proc startMacro(c: JinjaRenderContext, lo, hi: int, call: DeferredMacroCall, retNode: int32) =
  ## Opens a macro row and enters the body, the body's pieces draining until the row closes
  ## on the definition node, depth capped with a located raise on breach.
  if c.state.macroDepth >= TTT_CNJ_MacroDepthCap:
    raise jinjaErr("macro nesting reached TTT_CNJ_MacroDepthCap = " & $TTT_CNJ_MacroDepthCap & " on `" &
        c.symbols.names[call.mc.name] & "`", lo, hi - lo)
  inc c.state.macroDepth
  let scopeBase = c.state.scopes.len
  c.state.scopes.add @[]
  bindMacroArgs(c, call.mc.node, call.args, lo, hi)
  c.state.rows.add Row(node: call.mc.node, kind: frMacro, pc: call.mc.body,
      retNode: retNode, scopeAt: scopeBase)
  if call.mc.body == NoLink:
    # An empty body emits nothing, the row closing at once so the render tail
    # continues at the call's return node
    closeMacroRow(c.state, c.state.rows.len - 1, retNode)
  else:
    c.state.curNode = call.mc.body

proc forceMacro(c: JinjaRenderContext, mc: MacroVal, args: var Args): JinjaVal =
  ## Statement tier side of the macro forcer, the body running on a second render context
  ## over the caller's artifact refs, the caller's render state untouched by construction.

  ## Capture is transient, the nested render context discarded once its pieces drain into
  ## the result value's string, expressions resolving against a snapshot of the caller's
  ## scope chain, depth capped against the inherited depth.
  doAssert c.state.pend.kind == pkNone,
      "a macro body was forced while the driver still held a pending piece"
  if c.state.macroDepth >= TTT_CNJ_MacroDepthCap:
    raise jinjaErr("macro nesting reached TTT_CNJ_MacroDepthCap = " & $TTT_CNJ_MacroDepthCap & " on `" &
        c.symbols.names[mc.name] & "` (forced call)")
  # Nested-context shape, reproducing the field-copy semantics the corpus verifies:
  # - the same artifact refs and the same stateless force callable
  # - a snapshot of the caller's render state taken at the force point, the caller's
  #   scopes staying visible to the body through the innermost-first scan
  # - the body's bindings and rows accumulating on the snapshot, the whole nested context
  #   abandoned once the capture drains
  var c2 = JinjaRenderContext(tmpl: c.tmpl, symbols: c.symbols,
      state: c.state, force: c.force)
  inc c2.state.macroDepth
  let scopeBase = c2.state.scopes.len
  c2.state.scopes.add @[]
  c2.state.rows.add Row(node: mc.node, kind: frMacro, pc: mc.body,
      retNode: mc.node, scopeAt: scopeBase)
  bindMacroArgs(c2, mc.node, args, NoOffset, 0)
  c2.state.curNode = mc.body
  result = strVal("")
  # A nested streamed call flows back onto the same definition node this force
  # entered through, so node == mc.node alone cannot mean the body is done.
  #   - the outer row closed and dropped a row below the entry count
  #   - the flow reached mc.node with no row beyond the entry set open
  # TTT_CNJ_MacroDepthCap bounds nesting, not iterations, so the force walk keeps
  # its own step counter with the budget `pullInto` enforces, a breach raising
  # located at the node the walk reached.
  let baseRows = c2.state.rows.len
  var node = mc.body
  var steps = 0
  while node != NoLink:
    inc steps
    if steps > TTT_CNJ_StepBudget:
      raise jinjaErr("one macro force stepped past TTT_CNJ_StepBudget = " & $TTT_CNJ_StepBudget &
          ", the body walk is not terminating", c2.tmpl.nodes[node].lo.int,
          c2.tmpl.nodes[node].hi.int - c2.tmpl.nodes[node].lo.int)
    Steps[c2.tmpl.nodes[node].kind](c2, node)
    node = c2.state.curNode
    while c2.state.pend.kind != pkNone:
      capturePend(c2, result.s)
    if c2.state.rows.len < baseRows or (node == mc.node and c2.state.rows.len == baseRows):
      break


# Render driver:

func startRender*(tmpl: CompiledTemplate, sym: CompiledSymbols, root: JinjaVal, clock = 0.0): JinjaRenderContext =
  ## Returns a render context over the shared artifact, ready to render `root`, the render
  ## context dict with `messages`, `tools`, `add_generation_prompt` and template kwargs.
  ##
  ## Contract:
  ## - `clock` is the epoch `strftime_now` reads, never artifact state, so one artifact
  ##   renders reproducibly under different clocks
  ## - `sym` is the parse-built interned-name table by ref, every render over the artifact
  ##   holding the same heap object, no borrow contract
  # A zero-node artifact (empty or comment-only text) dispatches nothing, its render
  # completing on the first pullInto, so the program counter starts past the node list.
  JinjaRenderContext(tmpl: tmpl, symbols: sym, force: forceMacro,
      state: RenderState(curNode: (if tmpl.nodes.len == 0: NoLink else: 0), cur: 0,
          pend: Piece(kind: pkNone),
          scopes: @[(default(Scope))], root: root, clock: clock))

proc pullInto*(c: JinjaRenderContext, buf: var openArray[char]): int =
  ## Returns the render's next bytes, written into `buf[0 ..< result]`.
  ##
  ## Ownership sits with the caller, whose buffer capacity is the delivery window.
  ## Resumption state is `c.state`, so consumers over one artifact each hold a context
  ## from `startRender` and own their delivery position.
  ##
  ## Delivery contract:
  ## - `c.state.pend.pos` and `c.state.cur` advance before the call returns, so a consumer
  ##   that stops mid-drain and resumes never re-receives a byte
  ## - a piece longer than the window drains across calls, a lazy piece resuming
  ##   through the serializer in `c.state.lazy`
  ##
  ## Termination and budget:
  ## - 0 means the render is complete, nothing pending and `c.state.curNode == NoLink`
  ## - one call dispatches at most `TTT_CNJ_StepBudget` steps, a breach raising located at the reached node
  ##
  ## A raise discards the bytes already written into `buf` in the failing call, the caller
  ## never receiving them and the render state having advanced past their render, so a repull
  ## resumes after them:
  ## - a consumer that must hold every byte across a raise keeps the window at one byte,
  ##   which makes each delivered byte a returned byte
  ## - span pieces copy out of `CompiledTemplate.jinja`, string pieces and cut pieces copy
  ##   out of render-state storage, lazy pieces out of the serializer state in `c.state.lazy`
  ## - a zero-capacity buffer returns 0 without stepping the render
  template tmpl: CompiledTemplate = c.tmpl
  template st: RenderState = c.state
  if buf.len == 0:
    return 0
  var steps = 0
  while true:
    # Retire a piece whose bytes are all delivered. A lazy piece completes when its serializer
    # is done, which a window-sized drain reports by leaving the piece queued.
    if st.pend.kind == pkLazy:
      if serDone(st.lazy):
        st.pend = Piece(kind: pkNone)
    elif st.pend.kind != pkNone and st.pend.pos >= pieceLen(st.pend):
      st.pend = Piece(kind: pkNone)
    if st.pend.kind == pkLazy:
      let n = pullSer(st.lazy, toOpenArray(buf, result, buf.len - 1))
      st.cur += n
      result += n
      if result == buf.len:
        return
      continue
    if st.pend.kind != pkNone:
      let take = min(buf.len - result, pieceLen(st.pend) - st.pend.pos)
      let base = st.pend.pos
      # Commit the delivery position before returning, not after. The caller may stop after any call,
      # so a post-return update would strand `pos` and `cur` at their pre-call values, re-handing bytes.
      st.pend.pos += take
      st.cur += take
      case st.pend.kind
      of pkSpan:
        copyMem(addr buf[result], unsafeAddr tmpl.jinja[int st.pend.lo + base], take)
      of pkStr:
        copyMem(addr buf[result], unsafeAddr st.pend.s[base], take)
      of pkCut:
        copyMem(addr buf[result], unsafeAddr st.pend.raw[st.pend.clo + base], take)
      of pkNone, pkLazy:
        discard
      result += take
      if result == buf.len:
        return
      continue
    if st.curNode == NoLink:
      if st.rows.len > 0:
        let r = st.rows[^1]
        raise jinjaErr("the render walk drained with an open `" & $r.kind & "` row",
            tmpl.nodes[r.node].lo.int, tmpl.nodes[r.node].hi.int - tmpl.nodes[r.node].lo.int)
      return
    let n = st.curNode
    inc steps
    if steps > TTT_CNJ_StepBudget:
      raise jinjaErr("one pullInto call stepped past TTT_CNJ_StepBudget = " & $TTT_CNJ_StepBudget &
          ", the render walk is not terminating", tmpl.nodes[n].lo.int,
          tmpl.nodes[n].hi.int - tmpl.nodes[n].lo.int)
    Steps[c.tmpl.nodes[n].kind](c, n)

proc pullAll*(c: JinjaRenderContext): string =
  ## Returns the whole render in one call, one stack buffer drained through `pullInto`
  ## until the render reports 0.
  ## - `pullAll` owns its window, the only delivery path with an engine-chosen size
  ## - the caller's window composes with `cur`, so a consumer that counts bytes first
  ##   can redeliver from a fresh context without a counting pass
  var buf: array[4096, char]
  while true:
    let n = pullInto(c, buf)
    if n == 0:
      break
    addView(result, buf.toOpenArray(0, n - 1))

proc renderToString*(src: string, root: JinjaVal, clock = 0.0): string =
  ## Compiles and renders in one call, compiling at the scope that owns `src`, the artifact
  ## borrowing the template text and never outliving it.
  let (tmpl, sym) = parseTemplate(src)
  var c = startRender(tmpl, sym, root, clock)
  pullAll(c)
