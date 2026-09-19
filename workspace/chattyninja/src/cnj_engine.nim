# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Compiled chattyninja templates and their render driver.
#
# | Step     | Behavior                                                                                                                                                      |
# | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | parse    | `parseTemplate` appends nodes in source order, resolving the whitespace policy and interning names into `Tables`                                              |
# | load     | the caller builds `Machine` from the arena plus the borrowed template text, so the artifact holds no mutable state and cannot outlive the text it points into |
# | render   | `pull` walks the arena through `steps`, all control state in a caller-supplied `Driver`, so two drivers over one `Machine` are independent                    |
# | dispatch | `steps` is total over `NodeKind`, so a node's meaning is a pure function of its kind and no node carries a proc field or program counter                      |
#
# Resumption state for a re-entered step lives in the driver's frame stack, never in a node.
# `nkFor`, `nkSetBlock` and `nkGeneration` are re-entered by their bodies, `nkIf`
# single-entry because parse time backpatches its branch bodies past the whole chain.
#
# Run:
#   `nim c --experimental:views -r tests/t_render.nim`, or `./run_tests.sh` for every suite.

import std/unicode
import cnj_errors, cnj_types, cnj_values, cnj_expr, cnj_parse

type
  Step* = proc (m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.}
    ## One construct's step. Writes only through `d`, always leaving `d.curNode` on the node control enters next.

proc forceMacro(m: Machine, t: Tables, d: var Driver, mc: MacroVal, args: seq[CallArg]): string
  ## Runs one macro body to completion and returns the captured text, the macro forcer
  ## the expression tier receives. Carried as a parameter so the compiled artifact
  ## stays read-only and `cnj_engine` and `cnj_expr` stay free of an import cycle.

proc startMacro(m: Machine, t: Tables, d: var Driver, call: PendingCallVal, retNode: int32)
  ## Opens a macro frame and enters the body, the body's output pieces draining through
  ## the caller's window until the frame closes on the definition node.

# Output
# ---------------------------------------------------------------------------

proc emitSpan(m: Machine, d: var Driver, lo, hi: int32) =
  ## Makes a template-text span the pending piece.
  if hi <= lo:
    return
  # One piece is pending at a time, drained before the next dispatch, so a second
  # piece here would silently drop the first one's bytes.
  doAssert d.pend.kind == pkNone, "a step queued a piece while one was still pending"
  d.pend = Piece(pos: 0, kind: pkSpan, lo: lo, hi: hi)

proc emitStr(d: var Driver, s: sink string) =
  ## Makes a materialized string the pending piece, moving it out of the caller's value so
  ## a runtime-built emit string is never copied, an empty string queuing nothing.
  if s.len == 0:
    return
  doAssert d.pend.kind == pkNone, "a step queued a piece while one was still pending"
  d.pend = Piece(pos: 0, kind: pkStr, s: s)

proc emitValue(d: var Driver, v: Value) =
  ## Queues a derived value's rendering as the lazy piece, the serializer in `d.lazy`
  ## draining into the caller's window across pull calls, byte-exact with `pyStr`.
  doAssert d.pend.kind == pkNone, "a step queued a piece while one was still pending"
  serReset(d.lazy, v, smStr)
  d.pend = Piece(kind: pkLazy)

template pieceLen(p: Piece): int =
  ## Length in bytes of a pending piece, lazy pieces drained through the serializer instead.
  case p.kind
  of pkNone: 0
  of pkSpan: (p.hi - p.lo).int
  of pkStr: p.s.len
  of pkScratch: p.shi.int
  of pkLazy: 0

# Binding
# ---------------------------------------------------------------------------

func bindName(d: var Driver, name: int32, val: Value) =
  ## Binds a name in the innermost scope, replacing an existing binding there.
  var sc = d.scopes.len - 1
  for bi in 0 ..< d.scopes[sc].len:
    if d.scopes[sc][bi].name == name:
      d.scopes[sc][bi].val = val
      return
  d.scopes[sc].add Binding(name: name, val: val)

# Steps
# ---------------------------------------------------------------------------
#
# Every step reads its node's payload through the `nd` slot-accessor templates, which expand
# textually onto the arena entry. Render code never binds a `Node` value, a binding running
# SmallSeq's `=copy` and heap-allocating a spilled payload's block (7% of corpus nodes spill).

proc stepVerbatim(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Streams the final text run, whose span already reflects every whitespace rule.
  template nd: Node = m.nodes[n]
  emitSpan(m, d, nd.lo, nd.hi)
  d.curNode = nd.succ

proc stepEmit(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Evaluates the expression span and hands the result on as the pending piece, or enters
  ## a whole-expression macro call's body instead, the body's output pieces draining
  ## through the caller's window until the frame closes.
  template nd: Node = m.nodes[n]
  var v = evalSpan(m, t, d, nd.lo, nd.hi, forceMacro)
  if v.kind == vkCall:
    startMacro(m, t, d, v.pc, nd.succ)
    return
  if v.kind == vkStr:
    emitStr(d, move v.s)
  else:
    emitValue(d, v)
  d.curNode = nd.succ

proc stepIf(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Chooses a branch once, branch bodies terminating past the chain, so no frame exists for it.
  template nd: Node = m.nodes[n]
  let v = evalSpan(m, t, d, nd.lo, nd.hi, forceMacro)
  if isTruthy(v):
    d.curNode = if nd.child == noLink: nd.succ else: nd.child
  elif nd.alt != noLink:
    d.curNode = nd.alt
  else:
    d.curNode = nd.succ

func materialize(v: Value): seq[Value] =
  ## Returns the iterable as a materialized sequence. `loop.previtem` and `loop.nextitem`
  ## need random access, so a lazy cursor would need a peek buffer anyway.
  case v.kind
  of vkSeq: v.xs.items
  of vkDict, vkNs:
    var acc = newSeq[Value](v.d.keys.len)
    for i, k in v.d.keys:
      acc[i] = strVal(k)
    acc
  of vkStr: codepointVals(v.s)
  of vkUndefined:
    raise err("cannot iterate an undefined value")
  else:
    raise err("cannot iterate a " & $v.kind)

proc bindTargets(m: Machine, t: Tables, d: var Driver, n: int32, item: Value) =
  ## Binds the `nkFor` loop targets at `n`, more than one target unpacking a sequence, which
  ## is what `x.items()` feeds through `{% for k, v in x.items() %}`.
  template nd: Node = m.nodes[n]
  let ntargets = int(nd.targetCount)
  if ntargets == 1:
    d.bindName(nd.targetAt(0), item)
  else:
    if item.kind != vkSeq or item.xs.items.len != ntargets:
      raise err("`for` unpacks " & $ntargets & " targets from a value that is not a " &
          $ntargets & "-element sequence")
    for i in 0 ..< ntargets:
      d.bindName(nd.targetAt(i), item.xs.items[i])

proc advanceFor(m: Machine, t: Tables, d: var Driver, n: int32) =
  ## Re-entry path. Move the shared cursor to the next item passing the filter clause, re-enter
  ## the body, or close the frame and continue past the loop.
  ##
  ## A raise in `bindTargets` or the filter clause rolls the candidate back, so a caller growing
  ## scratch and repulling re-evaluates the same item and skips nothing.
  template nd: Node = m.nodes[n]
  while true:
    let fi = d.frames.len - 1
    let items = d.frames[fi].loop.items
    inc d.frames[fi].loop.idx
    let idx = d.frames[fi].loop.idx
    if idx >= items.len:
      d.scopes.setLen(d.frames[fi].scopeAt - 1)
      d.frames.setLen(d.frames.len - 1)
      d.curNode = nd.succ
      return
    # The increment stays committed while the filter runs, because corpus filters read `loop.index0` and friends through the shared cursor.
    # Rolling the candidate back when a raise escapes keeps a repull from skipping that item.
    var keep = nd.filterLo == noLink
    try:
      bindTargets(m, t, d, n, items[idx])
      if nd.filterLo != noLink:
        let evaluated = evalSpan(m, t, d, nd.filterLo, nd.filterHi, forceMacro)
        keep = isTruthy(evaluated)
    except CatchableError:
      dec d.frames[fi].loop.idx
      raise
    if keep:
      break
  d.curNode = nd.child

proc stepFor(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## `{% for %}`:
  ##   a matching frame on top of the stack means advance, anything else means set up the iteration.
  template nd: Node = m.nodes[n]
  if d.frames.len > 0 and d.frames[^1].kind == frFor and d.frames[^1].node == n:
    advanceFor(m, t, d, n)
    return
  let items = materialize(evalSpan(m, t, d, nd.lo, nd.hi, forceMacro))
  if items.len == 0:
    d.curNode = nd.succ
    return
  d.scopes.add @[]
  d.frames.add Frame(node: n, kind: frFor, loop: LoopState(items: items, idx: -1),
      scopeAt: d.scopes.len, filterLo: nd.filterLo, filterHi: nd.filterHi)
  let fi = d.frames.len - 1
  d.frames[fi].loop.idx = 0
  bindTargets(m, t, d, n, items[0])
  d.bindName(nd.loopName, loopVal(d.frames[fi].loop))
  d.curNode = if nd.child == noLink: nd.succ else: nd.child

proc stepSet(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Single-target `{% set %}`, the target carried as an interned name id in the child slot,
  ## emitting nothing, the pending piece untouched.
  template nd: Node = m.nodes[n]
  d.bindName(nd.child, evalSpan(m, t, d, nd.lo, nd.hi, forceMacro))
  d.curNode = nd.succ

proc gap(kindName, corpusSite: string): void {.noreturn.} =
  ## Reports a declared construct that is not implemented, naming `kindName`
  ## and the corpus site that demands it.
  raise newImplementError(kindName & " is not implemented; " & corpusSite)

proc stepBreak(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Unwinds to the nearest for-frame and continues at its successor, stopping at a macro-call
  ## boundary so a break cannot cross out of its macro.
  gap("nkBreak", "corpus demand is 8 sites: 7 in glm53flash.jinja inside the macro " &
      "has_dup_tool_result_id, 1 in northminicode10.jinja")

proc stepSetNs(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## `ns.field = expr`, mutating the shared namespace mapping in place, visible to every
  ## holder of the `DictVal` ref, and emitting nothing.
  template nd: Node = m.nodes[n]
  let ns = lookupNameById(t, d, nd.target)
  if ns.kind != vkNs:
    raise err("`" & t.names[nd.target] & "` is not a namespace, so it has no `" &
        t.names[nd.field] & "` to set")
  dictSet(ns.d, t.names[nd.field], evalSpan(m, t, d, nd.lo, nd.hi, forceMacro))
  d.curNode = nd.succ

proc stepSetBlock(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Opens a capture sink for the body and, on re-entry, binds the capture to the target name.
  gap("nkSetBlock", "corpus demand is 2 sites: gemma4.jinja:322 and northminicode10.jinja:2")

proc stepGeneration(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Records the root-output span of the model's turn:
  ##   the frame holds the opening position, and the re-entry closes it.
  gap("nkGeneration", "corpus demand is 2 sites: lagunaxs21.jinja:44 and lfm25.jinja:77, with " &
      "8 recorded rows carrying codepoint spans")

proc stepMacroDef(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Binds a macro value and emits nothing, the body never running here. A macro frame
  ## arriving back on the definition node closes instead, the body's output pieces drained
  ## through the caller's window, control continuing at the frame's return node.
  template nd: Node = m.nodes[n]
  if d.frames.len > 0 and d.frames[^1].kind == frMacro and d.frames[^1].node == n:
    d.scopes.setLen(d.frames[^1].scopeAt)
    d.curNode = d.frames[^1].retNode
    d.frames.setLen(d.frames.len - 1)
    dec d.macroDepth
    return
  d.bindName(nd.macroName, macroVal(
      MacroVal(name: nd.macroName, body: nd.child, node: n)))
  d.curNode = nd.succ

const
  steps*: array[NodeKind, Step] = [
    stepVerbatim, stepEmit, stepIf, stepFor, stepBreak, stepSet, stepSetNs, stepSetBlock,
    stepGeneration, stepMacroDef
  ]
    ## Dispatch table, total over `NodeKind`, a new kind without a step a compile error.

proc bindMacroArgs(m: Machine, t: Tables, d: var Driver, n: int32, args: seq[CallArg]) =
  ## Binds one macro call's parameters in a fresh scope, read from the `nkMacroDef` node at `n`,
  ## each parameter carrying its interned name id and default expression span in the node tail.
  ##
  ## Positionals bind first, then keywords, then defaults, each default evaluated after those
  ## before it are bound, inside the macro scope that a default sees in Jinja.
  template nd: Node = m.nodes[n]
  var pos = 0
  let nparams = int(nd.paramCount)
  for k in 0 ..< nparams:
    var val = undefinedVal()
    var bound = false
    while pos < args.len and args[pos].nameLo == noLink:
      if pos == k:
        val = args[pos].val
        bound = true
      inc pos
      break
    if not bound:
      for a in args:
        if a.nameLo != noLink and
            m.jinja.toOpenArray(a.nameLo.int, a.nameHi.int - 1) == t.names[nd.paramNameAt(k)]:
          val = a.val
          bound = true
          break
    if not bound:
      if nd.paramDefLoAt(k) == noLink:
        val = undefinedVal()
      else:
        val = evalSpan(m, t, d, nd.paramDefLoAt(k), nd.paramDefHiAt(k), forceMacro)
    d.bindName(nd.paramNameAt(k), val)

proc capturePend(m: Machine, d: var Driver, outp: var string) =
  ## Appends the pending piece's bytes to `outp` and retires the piece, the capture form
  ## of a forced macro body whose output never reaches the caller's window.
  case d.pend.kind
  of pkNone:
    discard
  of pkSpan:
    let at = outp.len
    let n = int(d.pend.hi - d.pend.lo) - d.pend.pos
    outp.setLen(at + n)
    copyMem(addr outp[at], unsafeAddr m.jinja[int d.pend.lo + d.pend.pos], n)
    d.pend = Piece(kind: pkNone)
  of pkStr:
    outp.add d.pend.s[d.pend.pos ..< d.pend.s.len]
    d.pend = Piece(kind: pkNone)
  of pkScratch:
    let at = outp.len
    let n = d.pend.shi.int - d.pend.pos
    outp.setLen(at + n)
    copyMem(addr outp[at], unsafeAddr d.scratch[d.pend.pos], n)
    d.pend = Piece(kind: pkNone)
  of pkLazy:
    var buf: array[256, char]
    while true:
      let n = pullSer(d.lazy, buf)
      if n == 0:
        break
      let at = outp.len
      outp.setLen(at + n)
      copyMem(addr outp[at], addr buf[0], n)
    d.pend = Piece(kind: pkNone)

proc startMacro(m: Machine, t: Tables, d: var Driver, call: PendingCallVal, retNode: int32) =
  ## Opens a macro frame and enters the body.
  ## Contract:
  ## - the body's output pieces drain through the caller's window until the frame closes on the definition node
  ## - depth is capped, and a breach raises
  if d.macroDepth >= MacroDepthCap:
    raise err("macro nesting reached MacroDepthCap = " & $MacroDepthCap & " on `" &
        t.names[call.mc.name] & "`")
  inc d.macroDepth
  d.scopes.add @[]
  bindMacroArgs(m, t, d, call.mc.node, call.args)
  d.frames.add Frame(node: call.mc.node, kind: frMacro, pc: call.mc.body,
      retNode: retNode, scopeAt: d.scopes.len)
  d.curNode = call.mc.body

proc forceMacro(m: Machine, t: Tables, d: var Driver, mc: MacroVal, args: seq[CallArg]): string =
  ## Statement tier side of the macro forcer.
  ## Contract:
  ## - the body runs on a copy of the driver, so the caller's scopes, frames, program counter,
  ##   depth and pending piece are untouched by construction, and a raise inside the body
  ##   abandons the copy wholesale
  ## - the capture is transient, the copy discarded once its pieces drain into the result, shared dict writes staying visible
  ## - depth is capped against the inherited depth, so the cap chains across nested forces, and a breach raises
  doAssert d.pend.kind == pkNone,
      "a macro body was forced while the driver still held a pending piece"
  if d.macroDepth >= MacroDepthCap:
    raise err("macro nesting reached MacroDepthCap = " & $MacroDepthCap & " on `" &
        t.names[mc.name] & "`")
  var d2 = d
  inc d2.macroDepth
  d2.scopes.add @[]
  bindMacroArgs(m, t, d2, mc.node, args)
  d2.curNode = mc.body
  var node = mc.body
  while node != mc.node and node != noLink:
    steps[m.nodes[node].kind](m, t, d2, node)
    node = d2.curNode
    while d2.pend.kind != pkNone:
      capturePend(m, d2, result)


# Driver
# ---------------------------------------------------------------------------

proc attachScratch*(d: var Driver, buf: var openArray[char]) =
  ## Attaches caller-owned scratch to the driver, which must outlive the render exactly like the template text behind `Machine.jinja`.
  ## An empty buffer detaches, and the emit path then materializes one string per derived value.
  if buf.len == 0:
    d.scratch = nil
    d.scratchCap = 0
  else:
    d.scratch = cast[ptr UncheckedArray[char]](addr buf[0])
    d.scratchCap = buf.len

func newDriver*(ctx: Value, clock = 0.0): Driver =
  ## Returns a driver ready to render `ctx`, the render context dict with `messages`, `tools`, `add_generation_prompt` and template kwargs.
  ## `clock` is the epoch `strftime_now` reads, never `Machine` state, so one artifact renders reproducibly under different clocks.
  Driver(curNode: 0, cur: 0, pend: Piece(kind: pkNone), scopes: @[(default(Scope))], root: ctx,
      clock: clock)

proc pull*(m: Machine, t: Tables, d: var Driver, buf: var openArray[char]): int =
  ## Returns the render's next bytes, written into `buf[0 ..< result]`.
  ##
  ## Ownership sits with the caller, whose buffer capacity is the delivery window.
  ## Resumption state is the driver, so consumers over one `Machine` with separate drivers each
  ## own their delivery position.
  ##
  ## Delivery contract:
  ## - `d.pend.pos` and `d.cur` advance before the call returns, so a consumer that stops
  ##   mid-drain and resumes never re-receives a byte
  ## - a piece longer than the window drains across calls, a lazy piece resuming
  ##   through the serializer in `d.lazy`
  ## - 0 means the render is complete, nothing pending and `d.curNode == noLink`
  ##
  ## A raise discards the bytes already written into `buf` in the failing call, the caller
  ## never receiving them and the driver having advanced past their render, so a repull
  ## resumes after them:
  ## - a consumer that must hold every byte across a raise keeps the window at one byte,
  ##   which makes each delivered byte a returned byte
  ## - span pieces copy out of `Machine.jinja`, string and scratch pieces out of driver storage,
  ##   lazy pieces out of the serializer state in `d.lazy`
  ## - a zero-capacity buffer returns 0 without stepping the render
  if buf.len == 0:
    return 0
  while true:
    # Retire a piece whose bytes are all delivered. A lazy piece completes when its serializer
    # is done, which a window-sized drain reports by leaving the piece queued.
    if d.pend.kind == pkLazy:
      if serDone(d.lazy):
        d.pend = Piece(kind: pkNone)
    elif d.pend.kind != pkNone and d.pend.pos >= pieceLen(d.pend):
      d.pend = Piece(kind: pkNone)
    if d.pend.kind == pkLazy:
      let n = pullSer(d.lazy, toOpenArray(buf, result, buf.len - 1))
      d.cur += n
      result += n
      if result == buf.len:
        return
      continue
    if d.pend.kind != pkNone:
      let take = min(buf.len - result, pieceLen(d.pend) - d.pend.pos)
      let base = d.pend.pos
      # Commit the delivery position before returning, not after. The caller may stop after any call,
      # so a post-return update would strand `pos` and `cur` at their pre-call values, re-handing bytes.
      d.pend.pos += take
      d.cur += take
      case d.pend.kind
      of pkSpan:
        copyMem(addr buf[result], unsafeAddr m.jinja[int d.pend.lo + base], take)
      of pkStr:
        copyMem(addr buf[result], unsafeAddr d.pend.s[base], take)
      of pkScratch:
        copyMem(addr buf[result], unsafeAddr d.scratch[base], take)
      of pkNone, pkLazy:
        discard
      result += take
      if result == buf.len:
        return
      continue
    if d.curNode == noLink:
      return
    let n = d.curNode
    steps[m.nodes[n].kind](m, t, d, n)

iterator items*(m: Machine, t: Tables, d: var Driver): openArray[char] =
  ## Pulls the render in chunks of at most `ChunkSize` bytes, one `pull` call per chunk.
  ## - a chunk borrows the iterator's local window, so a consumer must finish with it before
  ##   advancing the loop
  ## - `cur` counts bytes handed out, so a stop mid-render resumes consistently
  var buf: array[ChunkSize, char]
  while true:
    let n = pull(m, t, d, buf)
    if n == 0:
      break
    yield buf.toOpenArray(0, n - 1)

proc pullAll*(m: Machine, t: Tables, d: var Driver): string =
  ## Returns every render byte, chunking composing with `cur`, so the two-pass counting contract needs no separate counting pass.
  var buf: array[ChunkSize, char]
  while true:
    let n = pull(m, t, d, buf)
    if n == 0:
      break
    let at = result.len
    result.setLen(at + n)
    if n > 0:
      copyMem(addr result[at], unsafeAddr buf[0], n)

proc renderToString*(src: string, ctx: Value, clock = 0.0): string =
  ## Compiles and renders in one call, building `Machine` at the scope that owns `src`,
  ## the artifact borrowing the template text and never outliving it.
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  var d = newDriver(ctx, clock)
  pullAll(m, tables, d)
