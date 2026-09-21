# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Compiled chattyninja templates and their render driver.
#
# | Step     | Behavior                                                                                                                                                      |
# | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | parse    | `parseTemplate` appends nodes in source order, resolving the whitespace policy and interning names into `CompiledSymbols`                                     |
# | load     | `parseTemplate` returns the artifact borrowing the template text, so it holds no mutable state and cannot outlive the text it points into                     |
# | render   | `startRender` opens a `Context` over the artifact and `pull` walks the arena through `steps`, all control state in the context's `RenderState`                |
# | dispatch | `steps` is total over `NodeKind`, so a node's meaning is a pure function of its kind and no node carries a proc field or program counter                      |
# | ports    | every dispatch builds the `PortEnv` adapter on the stack and hands the steps `Ports`, so the expression tier reads the render only through the injected ports |
#
# Resumption state for a re-entered step lives in the render state's frame stack, never in a node.
# `nkFor`, `nkSetBlock` and `nkGeneration` are re-entered by their bodies, `nkIf`
# single-entry, parse time backpatching its branch bodies past the whole chain.
#
# Run:
#   from the repo root, `nim test_chattyninja` builds and runs every suite with its variants.

import std/unicode
import cnj_types, jinja_data_model, jinja_serialize, cnj_parse, jinja_interpolation

type
  Step = proc (tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState,
      ports: Ports, n: int32) {.nimcall, noSideEffect.}
    ## One construct's step. Writes only through `st`, always leaving `st.curNode` holding
    ## the node control enters next. Expressions evaluate through `ports`, the injected
    ## render services, so no expression proc ever sees the render state.

  PortEnv = object
    ## Adapter state one dispatch's ports read. Holds the artifact, the shared symbol arena,
    ## the render state the port procs serve. Built on the stack per dispatch,
    ## never stored in the render state, so a copied `Context` never carries a dangling adapter.
    tmpl: CompiledTemplate
    sym: ptr CompiledSymbols
    st: ptr RenderState

func forceMacro(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, mc: MacroVal, args: Args): JinjaVal
  ## Runs one macro body to completion and returns the captured text, the macro forcer
  ## the expression tier receives through the ports. A parameter here keeps the compiled
  ## artifact read-only and keeps both tiers clear of an import cycle.

func startMacro(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState,
    ports: Ports, lo, hi: int, call: PendingCallVal, retNode: int32)
  ## Opens a macro frame and enters the body, the body's output pieces draining through
  ## the caller's window until the frame closes on the definition node.

func scopeHas(st: RenderState, id: int32, val: var JinjaVal): bool =
  ## Scope scan innermost first, returning true with `val` set when `id` is bound.
  ## A binding to an undefined value is still a binding, so the root lookup never sees it.
  for si in countdown(st.scopes.len - 1, 0):
    for b in st.scopes[si]:
      if b.name == id:
        val = b.val
        return true
  false

func portLookup(env: pointer, name: openArray[char]): JinjaVal {.nimcall.} =
  ## Lookup port forwarder. Returns the binding of `name` in the scopes, else in the render
  ## context root, else undefined. Absence is a value, never an error.
  ## `is defined` tests for exactly that shape.
  let e = cast[ptr PortEnv](env)
  let id = e.sym[].findName(name)
  var got: JinjaVal
  if id != NoLink and e.st[].scopeHas(id, got):
    return got
  if e.st[].root.kind == vkDict:
    return e.st[].root.d.dictGet(name)
  undefinedVal()

func portClock(env: pointer): float64 {.nimcall.} =
  ## Clock port forwarder returning the render's injected epoch.
  cast[ptr PortEnv](env)[].st[].clock

func portForce(env: pointer, mc: MacroVal, args: Args): JinjaVal {.nimcall.} =
  ## Macro-forcer port forwarder, running the body to completion on the adapter's render
  ## state and returning the captured text value.
  let e = cast[ptr PortEnv](env)
  forceMacro(e.tmpl, e.sym, e.st[], mc, args)

func lookupNameById(sym: CompiledSymbols, st: var RenderState, id: int32): JinjaVal =
  ## Returns the binding of an interned name, undefined when absent. The scope key is
  ## the id, so no string is rebuilt per lookup.
  if id == NoLink:
    return undefinedVal()
  var got: JinjaVal
  if st.scopeHas(id, got):
    return got
  if st.root.kind == vkDict and id < sym.names.len.int32:
    return st.root.d.dictGet(sym.names[id])
  undefinedVal()

# Output:

func emitSpan(st: var RenderState, tmpl: CompiledTemplate, lo, hi: int32) =
  ## Makes a template-text span the pending piece.
  if hi <= lo:
    return
  # One piece is pending at a time, drained before the next dispatch, so a second
  # piece here would silently drop the first one's bytes.
  doAssert st.pend.kind == pkNone, "a step queued a piece while one was still pending"
  st.pend = Piece(pos: 0, kind: pkSpan, lo: lo, hi: hi)

func emitStr(st: var RenderState, s: sink string) =
  ## Makes a materialized string the pending piece, moving it out of the caller's value so
  ## a runtime-built emit string is never copied, an empty string queuing nothing.
  if s.len == 0:
    return
  doAssert st.pend.kind == pkNone, "a step queued a piece while one was still pending"
  st.pend = Piece(pos: 0, kind: pkStr, s: s)

func emitCut(st: var RenderState, v: sink JinjaVal) =
  ## Makes a cut value the pending piece, moving the value's string out of the caller's
  ## value so the cut drains from render-state storage with no copy, an empty cut
  ## queueing nothing, the same empty check `emitStr` applies.
  if v.lo == v.hi:
    return
  doAssert st.pend.kind == pkNone, "a step queued a piece while one was still pending"
  st.pend = Piece(pos: 0, kind: pkCut, raw: move v.raw, clo: v.lo, chi: v.hi)

func emitValue(st: var RenderState, v: sink JinjaVal) =
  ## Makes a derived value the pending piece, taking over the caller's value so the engine
  ## never copies an emit value, the serializer in `st.lazy` draining into the caller's
  ## window across pull calls, byte-exact with `pyStr`.
  ##
  ## The caller routes `vkStr` to `emitStr` and `vkCut` to `emitCut` first, so the value
  ## here never carries either kind.
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
# textually onto the arena entry. Render code never binds a `Node` value, a binding running
# SmallSeq's `=copy` and heap-allocating a spilled payload's block (7% of corpus nodes spill).

func stepVerbatim(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## Streams the final text run, whose span already reflects every whitespace rule.
  template nd: Node = tmpl.nodes[n]
  st.emitSpan(tmpl, nd.lo, nd.hi)
  st.curNode = nd.succ

func stepEmit(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## Evaluates the expression span and hands the result on as the pending piece, or enters
  ## a whole-expression macro call's body instead, the body's output pieces draining
  ## through the caller's window until the frame closes.
  template nd: Node = tmpl.nodes[n]
  var v = evalSpan(tmpl, ports, nd.lo, nd.hi)
  if v.kind == vkCall:
    startMacro(tmpl, sym, st, ports, nd.lo.int, nd.hi.int, v.pc, nd.succ)
    return
  if v.kind == vkCut:
    st.emitCut(move v)
  elif v.kind == vkStr:
    st.emitStr(move v.s)
  else:
    st.emitValue(v)
  st.curNode = nd.succ

func stepIf(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## Chooses a branch once, branch bodies terminating past the chain, so no frame exists for it.
  template nd: Node = tmpl.nodes[n]
  let v = evalSpan(tmpl, ports, nd.lo, nd.hi)
  if isTruthy(v):
    st.curNode = if nd.child == NoLink: nd.succ else: nd.child
  elif nd.alt != NoLink:
    st.curNode = nd.alt
  else:
    st.curNode = nd.succ

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
  LoopState(r: v.r)

func notIterable(v: JinjaVal, lo, hi: int): void {.noreturn.} =
  ## Shared raise leg of the iterable dispatch, one report for every kind no loop walks.
  ## `lo` and `hi` bound the iterable expression, the loop header the raise reports.
  if v.kind == vkUndefined:
    raise jinjaErr("cannot iterate an undefined value", lo, hi - lo)
  raise jinjaErr("cannot iterate a " & $v.kind, lo, hi - lo)

func loopStateOf(v: JinjaVal, lo, hi: int): LoopState =
  ## Dispatch at the loop's chain entry, one leg per iterable kind the corpus supports
  ## and the shared raise leg for everything else. The cursor answers the random access
  ## that `loop.previtem` and `loop.nextitem` need, per index.
  case v.kind
  of vkSeq: iterSeq(v)
  of vkDict, vkNs: iterMapping(v)
  of vkStr, vkCut: iterChars(v)
  of vkRange: iterRange(v)
  else: notIterable(v, lo, hi)

func bindTargets(tmpl: CompiledTemplate, st: var RenderState, n: int32, item: JinjaVal) =
  ## Binds the `nkFor` loop targets at `n`, more than one target unpacking a sequence, which
  ## is what `x.items()` feeds through `{% for k, v in x.items() %}`.
  template nd: Node = tmpl.nodes[n]
  let ntargets = int(nd.targetCount)
  if ntargets == 1:
    st.bindName(nd.targetAt(0), item)
  else:
    if item.kind != vkSeq or item.xs.items.len != ntargets:
      raise jinjaErr("`for` unpacks " & $ntargets & " targets from a value that is not a " &
          $ntargets & "-element sequence", nd.lo.int, nd.hi.int - nd.lo.int)
    for i in 0 ..< ntargets:
      st.bindName(nd.targetAt(i), item.xs.items[i])

func advanceFor(tmpl: CompiledTemplate, st: var RenderState, ports: Ports, n: int32) =
  ## Re-entry path. Moves the shared cursor to the next item passing the filter clause, re-enters
  ## the body, or closes the frame and continues past the loop.
  ##
  ## Contract:
  ## - the cursor increment stays committed while `bindTargets` and the filter clause run.
  ##   Corpus filters read `loop.index0` and friends through the shared cursor
  ## - a raise in either propagates to the caller per the pull contract. The bytes written
  ##   in the failing call are discarded and a repull resumes after the failed item
  template nd: Node = tmpl.nodes[n]
  while true:
    let fi = st.frames.len - 1
    let lp = st.frames[fi].loop
    inc lp.idx
    let idx = lp.idx
    if idx >= lp.loopLen:
      st.scopes.setLen(st.frames[fi].scopeAt - 1)
      st.frames.setLen(st.frames.len - 1)
      st.curNode = nd.succ
      return
    var keep = nd.filterLo == NoLink
    bindTargets(tmpl, st, n, lp.loopItem(idx))
    if nd.filterLo != NoLink:
      let evaluated = evalSpan(tmpl, ports, nd.filterLo, nd.filterHi)
      keep = isTruthy(evaluated)
    if keep:
      break
  st.curNode = nd.child

func stepFor(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## `{% for %}`:
  ##   a matching frame on top of the stack means advance, anything else means set up the iteration.
  template nd: Node = tmpl.nodes[n]
  if st.frames.len > 0 and st.frames[^1].kind == frFor and st.frames[^1].node == n:
    advanceFor(tmpl, st, ports, n)
    return
  let lp = loopStateOf(evalSpan(tmpl, ports, nd.lo, nd.hi), nd.lo.int, nd.hi.int)
  if lp.loopLen == 0:
    st.curNode = nd.succ
    return
  st.scopes.add @[]
  st.frames.add Frame(node: n, kind: frFor, loop: lp,
      scopeAt: st.scopes.len, filterLo: nd.filterLo, filterHi: nd.filterHi)
  lp.idx = 0
  bindTargets(tmpl, st, n, lp.loopItem(0))
  st.bindName(nd.loopName, loopVal(lp))
  st.curNode = if nd.child == NoLink: nd.succ else: nd.child

func stepSet(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## Single-target `{% set %}`, the target carried as an interned name id in the child slot,
  ## emitting nothing, the pending piece untouched.
  template nd: Node = tmpl.nodes[n]
  st.bindName(nd.child, evalSpan(tmpl, ports, nd.lo, nd.hi))
  st.curNode = nd.succ

func gap(kindName, corpusSite: string, lo, hi: int): void {.noreturn.} =
  ## Reports a declared construct that is not implemented, naming `kindName`
  ## and the corpus site that demands it, `lo` and `hi` bounding the construct's node.
  raise jinjaErr(kindName & " is not implemented; " & corpusSite, lo, hi - lo,
      cause = ceUnimplemented)

func stepBreak(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## Unwinds to the nearest for-frame and continues at its successor, stopping at a macro-call
  ## boundary so a break cannot cross out of its macro.
  template nd: Node = tmpl.nodes[n]
  gap("nkBreak", "corpus demand is 8 sites: 7 in glm53flash.jinja inside the macro " &
      "has_dup_tool_result_id, 1 in northminicode10.jinja", nd.lo.int, nd.hi.int)

func stepSetNs(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## `ns.field = expr`, mutating the shared namespace mapping in place, visible to every
  ## holder of the `DictVal` ref, and emitting nothing.
  template nd: Node = tmpl.nodes[n]
  let ns = sym[].lookupNameById(st, nd.target)
  if ns.kind != vkNs:
    raise jinjaErr("`" & sym[].names[nd.target] & "` is not a namespace, so it has no `" &
        sym[].names[nd.field] & "` to set", nd.lo.int, nd.hi.int - nd.lo.int)
  dictSet(ns.d, sym[].names[nd.field], evalSpan(tmpl, ports, nd.lo, nd.hi))
  st.curNode = nd.succ

func stepSetBlock(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## Opens a capture sink for the body and, on re-entry, binds the capture to the target name.
  template nd: Node = tmpl.nodes[n]
  gap("nkSetBlock", "corpus demand is 2 sites: gemma4.jinja:322 and northminicode10.jinja:2",
      nd.lo.int, nd.hi.int)

func stepGeneration(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## Records the root-output span of the model's turn:
  ##   the frame holds the opening position, and the re-entry closes it.
  template nd: Node = tmpl.nodes[n]
  gap("nkGeneration", "corpus demand is 2 sites: lagunaxs21.jinja:44 and lfm25.jinja:77, with " &
      "8 recorded rows carrying codepoint spans", nd.lo.int, nd.hi.int)

func stepMacroDef(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32) {.nimcall.} =
  ## Binds a macro value and emits nothing, the body never running here. A macro frame
  ## arriving back on the definition node closes instead, the body's output pieces drained
  ## through the caller's window, control continuing at the frame's return node.
  template nd: Node = tmpl.nodes[n]
  if st.frames.len > 0 and st.frames[^1].kind == frMacro and st.frames[^1].node == n:
    st.scopes.setLen(st.frames[^1].scopeAt - 1)
    st.curNode = st.frames[^1].retNode
    st.frames.setLen(st.frames.len - 1)
    dec st.macroDepth
    return
  st.bindName(nd.macroName, macroVal(
      MacroVal(name: nd.macroName, body: nd.child, node: n)))
  st.curNode = nd.succ

const
  CaptureDrainCap = 256
    ## Stack buffer `capturePend` hands to `pullSer` per drain call, sized to hold a whole
    ## scalar rendering in the common case so the capture copies in one grow.

  Steps*: array[NodeKind, Step] = [
    stepVerbatim, stepEmit, stepIf, stepFor, stepBreak, stepSet, stepSetNs, stepSetBlock,
    stepGeneration, stepMacroDef
  ]
    ## Dispatch table, total over `NodeKind`, a new kind without a step a compile error.

func bindMacroArgs(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, ports: Ports, n: int32, args: Args) =
  ## Binds one macro call's parameters in a fresh scope, read from the `nkMacroDef` node at `n`,
  ## each parameter carrying its interned name id and default expression span in the node tail.
  ##
  ## Positionals bind first, then keywords, then defaults, each default evaluated after those
  ## before it are bound, inside the macro scope that a default sees in Jinja.
  template nd: Node = tmpl.nodes[n]
  var pos = 0
  let nparams = int(nd.paramCount)
  for k in 0 ..< nparams:
    var val = undefinedVal()
    var bound = false
    while pos < args.n and args.vals[pos].nameLo == NoLink:
      if pos == k:
        val = args.vals[pos].val
        bound = true
      inc pos
      break
    if not bound:
      for a in args.argItems:
        if a.nameLo != NoLink and
            tmpl.jinja.toOpenArray(a.nameLo.int, a.nameHi.int - 1) == sym[].names[nd.paramNameAt(k)]:
          val = a.val
          bound = true
          break
    if not bound:
      if nd.paramDefLoAt(k) == NoLink:
        val = undefinedVal()
      else:
        val = evalSpan(tmpl, ports, nd.paramDefLoAt(k), nd.paramDefHiAt(k))
    st.bindName(nd.paramNameAt(k), val)

func capturePend(tmpl: CompiledTemplate, st: var RenderState, outp: var string) =
  ## Appends the pending piece's bytes to `outp` and retires the piece, the capture form
  ## of a forced macro body whose output never reaches the caller's window.
  case st.pend.kind
  of pkNone:
    discard
  of pkSpan:
    let at = outp.len
    let n = int(st.pend.hi - st.pend.lo) - st.pend.pos
    outp.setLen(at + n)
    copyMem(addr outp[at], unsafeAddr tmpl.jinja[int st.pend.lo + st.pend.pos], n)
    st.pend = Piece(kind: pkNone)
  of pkStr:
    outp.add st.pend.s[st.pend.pos ..< st.pend.s.len]
    st.pend = Piece(kind: pkNone)
  of pkCut:
    outp.add st.pend.raw[st.pend.clo + st.pend.pos ..< st.pend.chi]
    st.pend = Piece(kind: pkNone)
  of pkLazy:
    var buf: array[CaptureDrainCap, char]
    while true:
      let n = pullSer(st.lazy, buf)
      if n == 0:
        break
      let at = outp.len
      outp.setLen(at + n)
      copyMem(addr outp[at], addr buf[0], n)
    st.pend = Piece(kind: pkNone)

func startMacro(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState,
    ports: Ports, lo, hi: int, call: PendingCallVal, retNode: int32) =
  ## Opens a macro frame and enters the body.
  ## Contract:
  ## - the body's output pieces drain through the caller's window until the frame closes on the definition node
  ## - depth is capped, and a breach raises, `lo` and `hi` bounding the call's site
  if st.macroDepth >= MacroDepthCap:
    raise jinjaErr("macro nesting reached MacroDepthCap = " & $MacroDepthCap & " on `" &
        sym[].names[call.mc.name] & "`", lo, hi - lo)
  inc st.macroDepth
  st.scopes.add @[]
  bindMacroArgs(tmpl, sym, st, ports, call.mc.node, call.args)
  st.frames.add Frame(node: call.mc.node, kind: frMacro, pc: call.mc.body,
      retNode: retNode, scopeAt: st.scopes.len)
  st.curNode = call.mc.body

func forceMacro(tmpl: CompiledTemplate, sym: ptr CompiledSymbols, st: var RenderState, mc: MacroVal, args: Args): JinjaVal =
  ## Statement tier side of the macro forcer.
  ## Contract:
  ## - the body runs on a copy of the driver, so the caller's scopes, frames, program counter,
  ##   depth and pending piece are untouched by construction, and a raise inside the body
  ##   abandons the copy wholesale
  ## - the capture is transient, the copy discarded once its pieces drain into the result
  ##   value's string, shared dict writes staying visible
  ##
  ## Ports and depth:
  ## - the body's expressions evaluate through ports over the copy, so a body binding
  ##   or a nested streamed call resolves against the body's own scopes, never the caller's scopes
  ## - depth is capped against the inherited depth, so the cap chains across nested forces,
  ##   and a breach raises
  doAssert st.pend.kind == pkNone,
      "a macro body was forced while the driver still held a pending piece"
  if st.macroDepth >= MacroDepthCap:
    raise jinjaErr("macro nesting reached MacroDepthCap = " & $MacroDepthCap & " on `" &
        sym[].names[mc.name] & "`")
  var st2 = st
  inc st2.macroDepth
  st2.scopes.add @[]
  var env2 = PortEnv(tmpl: tmpl, sym: sym, st: addr st2)
  let ports2 = Ports(lookup: portLookup, clock: portClock, force: portForce, env: addr env2)
  bindMacroArgs(tmpl, sym, st2, ports2, mc.node, args)
  st2.curNode = mc.body
  result = strVal("")
  var node = mc.body
  while node != mc.node and node != NoLink:
    Steps[tmpl.nodes[node].kind](tmpl, sym, st2, ports2, node)
    node = st2.curNode
    while st2.pend.kind != pkNone:
      capturePend(tmpl, st2, result.s)


# Render driver:

func startRender*(tmpl: CompiledTemplate, sym: var CompiledSymbols, root: JinjaVal, clock = 0.0): Context =
  ## Returns a render context over the shared artifact, ready to render `root`, the render
  ## context dict with `messages`, `tools`, `add_generation_prompt` and template kwargs.
  ##
  ## Contract:
  ## - `clock` is the epoch `strftime_now` reads, never artifact state, so one artifact
  ##   renders reproducibly under different clocks
  ## - `sym` is borrowed, the context's symbol pointer must not outlive the binding it was
  ##   taken from, the same class of contract as the artifact's borrow of the template text
  Context(tmpl: tmpl, symbols: addr sym,
      state: RenderState(curNode: 0, cur: 0, pend: Piece(kind: pkNone),
          scopes: @[(default(Scope))], root: root, clock: clock))

func pull*(c: var Context, buf: var openArray[char]): int =
  ## Returns the render's next bytes, written into `buf[0 ..< result]`.
  ##
  ## Ownership sits with the caller, whose buffer capacity is the delivery window.
  ## Resumption state is `c.state`, so consumers holding separate `Context` copies over one
  ## artifact each own their delivery position.
  ##
  ## Delivery contract:
  ## - `c.state.pend.pos` and `c.state.cur` advance before the call returns, so a consumer
  ##   that stops mid-drain and resumes never re-receives a byte
  ## - a piece longer than the window drains across calls, a lazy piece resuming
  ##   through the serializer in `c.state.lazy`
  ## - 0 means the render is complete, nothing pending and `c.state.curNode == NoLink`
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
  template sym: ptr CompiledSymbols = c.symbols
  template st: RenderState = c.state
  if buf.len == 0:
    return 0
  # Adapter lifetime is one dispatch. Every step this call runs evaluates expressions
  # through ports over this env, and nothing escapes the call.
  var env = PortEnv(tmpl: tmpl, sym: sym, st: addr st)
  let ports = Ports(lookup: portLookup, clock: portClock, force: portForce, env: addr env)
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
      return
    let n = st.curNode
    Steps[tmpl.nodes[n].kind](tmpl, sym, st, ports, n)

iterator items*(c: var Context): openArray[char] =
  ## Pulls the render in chunks of at most `ChunkSize` bytes, one `pull` call per chunk.
  ## - a chunk borrows the iterator's local window, so a consumer must finish with it before
  ##   advancing the loop
  ## - `cur` counts bytes handed out, so a stop mid-render resumes consistently
  var buf: array[ChunkSize, char]
  while true:
    let n = pull(c, buf)
    if n == 0:
      break
    yield buf.toOpenArray(0, n - 1)

func pullAll*(c: var Context): string =
  ## Returns the whole render in one call. Chunking composes with `cur`, so a consumer
  ## that counts bytes first can redeliver from a fresh `Context` without a counting pass.
  var buf: array[ChunkSize, char]
  while true:
    let n = pull(c, buf)
    if n == 0:
      break
    let at = result.len
    result.setLen(at + n)
    if n > 0:
      copyMem(addr result[at], unsafeAddr buf[0], n)

proc renderToString*(src: string, root: JinjaVal, clock = 0.0): string =
  ## Compiles and renders in one call, compiling at the scope that owns `src`, the artifact
  ## borrowing the template text and never outliving it.
  var (tmpl, sym) = parseTemplate(src)
  var c = startRender(tmpl, sym, root, clock)
  pullAll(c)
