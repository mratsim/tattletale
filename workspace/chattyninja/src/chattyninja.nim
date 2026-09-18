# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Compiled chattyninja templates and their render driver.
#
# Lifecycle:
#
# | Step     | Behavior                                                                                                                                                              |
# | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | parse    | `parseTemplate` reads template text and appends nodes in source order, resolving the whitespace policy and interning names into `Tables`                              |
# | load     | the caller builds `Machine` from the arena plus the borrowed template text, so the artifact holds no mutable state and cannot outlive the text it points into         |
# | render   | `items` walks the arena through `steps`, owning every piece of control state in a `Driver` passed in by the caller, so two drivers over one `Machine` are independent |
# | dispatch | `steps` is total over `NodeKind`, so the executable meaning of a node is a pure function of its kind and no node carries a proc field or program counter              |
#
# The re-entry contract. A step may be re-entered, and resumption state comes from the top
# of the driver's frame stack (the frame plus its resume state), never from a node. The re-entering kinds
# are `nkFor`, `nkSetBlock` and `nkGeneration`, whose body returns control to them. `nkIf`
# is single-entry and single-activation because parse time backpatches its branch bodies past the whole chain.
#
# Run:
#   `nim c --experimental:views -r tests/t_render.nim`, or `./run_tests.sh` for every suite.

import std/unicode
import cjn_errors, cjn_types, cjn_values, cjn_expr, cjn_parse

export cjn_types, cjn_values, cjn_parse, cjn_errors

type
  Step* = proc (m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.}
    ## One construct's step. Writes only through `d`, and always leaves `d.curNode` on the node
    ## control enters next.

proc runMacroBody(m: Machine, t: Tables, d: var Driver, mc: MacroVal,
    args: seq[Arg]): string
  ## Runs one macro body to completion and returns the captured text. Declared ahead of the steps
  ## because each one hands it to the expression tier as the macro runner, and its own body drives `steps`.
  ## Why a parameter and not a field:
  ## - `chattyninja` cannot import `cjn_expr` and be imported back
  ## - a proc in `Machine` or `Tables` would put a proc field in the read-only artifact

# Output
# ---------------------------------------------------------------------------

proc emitSpan(m: Machine, d: var Driver, lo, hi: int32) =
  ## Makes a template-text span the pending piece, or appends it to the capture sink when one is open.
  if hi <= lo:
    return
  # One piece is pending at a time, and the `items` loop drains it before dispatching again, so
  # a second piece here would silently drop the first one's bytes.
  doAssert d.pend.kind == pkNone, "a step queued a piece while one was still pending"
  if d.sinks.len > 0:
    for i in lo ..< hi:
      d.sinks[^1].add m.jinja[i]
    return
  d.pend = Piece(pos: 0, kind: pkSpan, lo: lo, hi: hi)

proc emitStr(d: var Driver, s: string) =
  ## Makes a materialized string the pending piece, or appends it to the capture sink.
  if s.len == 0:
    return
  doAssert d.pend.kind == pkNone, "a step queued a piece while one was still pending"
  if d.sinks.len > 0:
    d.sinks[^1].add s
    return
  d.pend = Piece(pos: 0, kind: pkStr, s: s)

template pieceLen(p: Piece): int =
  ## Length in bytes of a pending piece.
  case p.kind
  of pkNone: 0
  of pkSpan: (p.hi - p.lo).int
  of pkStr: p.s.len

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

proc stepVerbatim(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Streams the final text run. The span already reflects every whitespace rule, so this step has
  ## nothing to decide.
  let nd = m.nodes[n]
  emitSpan(m, d, nd.lo, nd.hi)
  d.curNode = nd.succ

proc stepEmit(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Evaluates the expression span, stringifies it, and hands the result on as the pending piece.
  let nd = m.nodes[n]
  let v = evalSpan(m, t, d, nd.lo, nd.hi, runMacroBody)
  emitStr(d, if v.kind == vkStr: v.s else: pyStr(v))
  d.curNode = nd.succ

proc stepIf(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Chooses a branch once. Branch bodies terminate past the chain, so this node is never entered
  ## a second time and no frame exists for it.
  let nd = m.nodes[n]
  let v = evalSpan(m, t, d, nd.lo, nd.hi, runMacroBody)
  if isTruthy(v):
    d.curNode = if nd.child == noLink: nd.succ else: nd.child
  elif nd.alt != noLink:
    d.curNode = nd.alt
  else:
    d.curNode = nd.succ

func materialize(v: Value): seq[Value] =
  ## Returns the iterable as a materialized sequence. `loop.previtem` and `loop.nextitem`
  ## are exercised by the corpus, so a lazy cursor would need a peek buffer anyway.
  case v.kind
  of vkSeq: v.xs.items
  of vkDict, vkNs:
    var acc = newSeq[Value](v.d.keys.len)
    for i, k in v.d.keys:
      acc[i] = strVal(k)
    acc
  of vkStr:
    var acc = newSeq[Value]()
    for r in v.s.runes:
      acc.add strVal($r)
    acc
  of vkUndefined:
    raise err("cannot iterate an undefined value")
  else:
    raise err("cannot iterate a " & $v.kind)

proc bindTargets(m: Machine, t: Tables, d: var Driver, aux: Aux, item: Value) =
  ## Binds the loop targets. More than one target unpacks a sequence, which is what `x.items()`
  ## feeds through `{% for k, v in x.items() %}`.
  if aux.targets.len == 1:
    d.bindName(aux.targets[0], item)
  else:
    if item.kind != vkSeq or item.xs.items.len != aux.targets.len:
      raise err("`for` unpacks " & $aux.targets.len & " targets from a value that is not a " &
          $aux.targets.len & "-element sequence")
    for i, name in aux.targets:
      d.bindName(name, item.xs.items[i])

proc advanceFor(m: Machine, t: Tables, d: var Driver, n: int32, aux: Aux) =
  ## Re-entry path. Move the shared cursor to the next item passing the filter clause and re-enter
  ## the body, or close the frame and continue past the loop. The frame index is re-read after every
  ## evaluation because a nested construct can grow the frame stack and move it.
  while true:
    let fi = d.frames.len - 1
    let items = d.frames[fi].loop.items
    inc d.frames[fi].loop.idx
    let idx = d.frames[fi].loop.idx
    if idx >= items.len:
      d.scopes.setLen(d.frames[fi].scopeAt - 1)
      d.frames.setLen(d.frames.len - 1)
      d.curNode = m.nodes[n].succ
      return
    bindTargets(m, t, d, aux, items[idx])
    if aux.filterLo == noLink:
      break
    let keep = evalSpan(m, t, d, aux.filterLo, aux.filterHi, runMacroBody)
    if isTruthy(keep):
      break
  d.curNode = m.nodes[n].child

proc stepFor(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## `{% for %}`:
  ##   one function, two modes. A matching frame on top of the stack means advance.
  ##   Anything else means set up the iteration.
  let nd = m.nodes[n]
  let aux = t.aux[nd.alt]
  if d.frames.len > 0 and d.frames[^1].kind == frFor and d.frames[^1].node == n:
    advanceFor(m, t, d, n, aux)
    return
  let items = materialize(evalSpan(m, t, d, nd.lo, nd.hi, runMacroBody))
  if items.len == 0:
    d.curNode = nd.succ
    return
  d.scopes.add @[]
  d.frames.add Frame(node: n, kind: frFor, loop: LoopState(items: items, idx: -1),
      scopeAt: d.scopes.len, filterLo: aux.filterLo, filterHi: aux.filterHi)
  let fi = d.frames.len - 1
  d.frames[fi].loop.idx = 0
  bindTargets(m, t, d, aux, items[0])
  d.bindName(aux.loopName, loopVal(d.frames[fi].loop))
  d.curNode = if nd.child == noLink: nd.succ else: nd.child

proc stepSet(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Single-target `{% set %}`. It emits nothing, so the pending piece is untouched.
  let nd = m.nodes[n]
  d.bindName(nd.child, evalSpan(m, t, d, nd.lo, nd.hi, runMacroBody))
  d.curNode = nd.succ

proc gap(kindName, corpusSite: string): void {.noreturn.} =
  ## Reports a declared construct that is not implemented, naming what the corpus demands of it
  ## rather than leaving the gap silent.
  raise newImplementError(kindName & " is not implemented; " & corpusSite)

proc stepBreak(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Unwinds to the nearest for-frame and continues at its successor, stopping at a macro-call
  ## boundary so a break cannot cross out of the macro that contains it.
  let _ = (m, t, d, n)
  gap("nkBreak", "corpus demand is 8 sites: 7 in glm53flash.jinja inside the macro " &
      "has_dup_tool_result_id, 1 in northminicode10.jinja")

proc stepSetNs(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## `ns.field = expr`, mutating the shared namespace mapping in place. The value is a `DictVal` ref
  ## shared by every holder, so the write is visible wherever the namespace was bound,
  ## and nothing is emitted.
  let nd = m.nodes[n]
  let aux = t.aux[nd.child]
  let ns = lookupNameById(t, d, aux.target)
  if ns.kind != vkNs:
    raise err("`" & t.names[aux.target] & "` is not a namespace, so it has no `" &
        t.names[aux.field] & "` to set")
  dictSet(ns.d, t.names[aux.field], evalSpan(m, t, d, nd.lo, nd.hi, runMacroBody))
  d.curNode = nd.succ

proc stepSetBlock(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Opens a capture sink for the body and, on re-entry, binds the capture to the target name.
  let _ = (m, t, d, n)
  gap("nkSetBlock", "corpus demand is 2 sites: gemma4.jinja:322 and northminicode10.jinja:2")

proc stepGeneration(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Records the root-output span of the model's turn:
  ##   the frame holds the opening position, and the re-entry closes it.
  let _ = (m, t, d, n)
  gap("nkGeneration", "corpus demand is 2 sites: lagunaxs21.jinja:44 and lfm25.jinja:77, with " &
      "8 recorded rows carrying codepoint spans")

proc stepMacroDef(m: Machine, t: Tables, d: var Driver, n: int32) {.nimcall.} =
  ## Binds a macro value and emits nothing. The body never runs here:
  ##   a macro's output is a string value, so only a call can run it, and a call runs it to completion.
  let nd = m.nodes[n]
  let aux = t.aux[nd.alt]
  d.bindName(aux.name, macroVal(
      MacroVal(name: aux.name, body: nd.child, aux: nd.alt, node: n)))
  d.curNode = nd.succ

const
  steps*: array[NodeKind, Step] = [
    stepVerbatim, stepEmit, stepIf, stepFor, stepBreak, stepSet, stepSetNs, stepSetBlock,
    stepGeneration, stepMacroDef
  ]
    ## Dispatch table, total over `NodeKind`:
    ##   a new kind without a step is a compile error.

proc bindMacroArgs(m: Machine, t: Tables, d: var Driver, aux: Aux, args: seq[Arg]) =
  ## Binds one macro call's parameters in a fresh scope. The order is positionals, then keywords,
  ## then the defaults. A default is evaluated after the parameters before it are bound, and inside
  ## the macro scope, which is the frame a default sees in Jinja.
  var pos = 0
  for k, param in aux.params:
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
        if a.nameLo != noLink and argName(m, a) == t.names[param.name]:
          val = a.val
          bound = true
          break
    if not bound:
      if param.defLo == noLink:
        val = undefinedVal()
      else:
        val = evalSpan(m, t, d, param.defLo, param.defHi, runMacroBody)
    d.bindName(param.name, val)

proc runMacroBody(m: Machine, t: Tables, d: var Driver, mc: MacroVal, args: seq[Arg]): string =
  ## Statement tier side of the macro runner. A macro body cannot stream, because its output is
  ## a string value, so the call runs the body to completion into a capture sink and never yields
  ## a piece. That keeps this the engine's only render-time recursion and makes it unsuspendable.
  ##
  ## Control state is driver-owned throughout:
  ##   a scope for the parameters, one capture sink, and the program counter, which is restored
  ## on the way out. Depth is capped, and a breach raises.
  if d.macroDepth >= MacroDepthCap:
    raise err("macro nesting reached MacroDepthCap = " & $MacroDepthCap & " on `" &
        t.names[mc.name] & "`")
  inc d.macroDepth
  let scopeAt = d.scopes.len
  let sinkAt = d.sinks.len
  let savedNode = d.curNode
  d.scopes.add @[]
  d.sinks.add ""
  bindMacroArgs(m, t, d, t.aux[mc.aux], args)
  var node = mc.body
  while node != mc.node and node != noLink:
    steps[m.nodes[node].kind](m, t, d, node)
    node = d.curNode
  result = d.sinks[sinkAt]
  d.sinks.setLen(sinkAt)
  d.scopes.setLen(scopeAt)
  d.curNode = savedNode
  dec d.macroDepth




# Driver
# ---------------------------------------------------------------------------

func newDriver*(ctx: Value, clock = 0.0): Driver =
  ## Returns a driver ready to render `ctx`, the render context dict with `messages`, `tools`,
  ## `add_generation_prompt` and template kwargs. `clock` is the epoch `strftime_now` reads, per
  ## render and never `Machine` state, so one artifact renders reproducibly under different clocks.
  Driver(curNode: 0, cur: 0, pend: Piece(kind: pkNone), scopes: @[(default(Scope))], root: ctx,
      clock: clock)

func append(a: var string, c: openArray[char]) =
  ## Appends a chunk's bytes to a string in one copy.
  let at = a.len
  a.setLen(at + c.len)
  if c.len > 0:
    copyMem(addr a[at], unsafeAddr c[0], c.len)

iterator items*(m: Machine, t: Tables, d: var Driver): openArray[char] =
  ## Pulls the render in chunks of at most `ChunkSize` bytes. A chunk borrows template text for a span
  ## piece and driver storage for a materialized one, so no output byte is copied on the way out.
  ## Delivery contract:
  ## - `cur` counts bytes handed out, so a consumer that stops mid-render and resumes leaves
  ##   the driver's position consistent with what it received
  while true:
    if d.pend.kind != pkNone and d.pend.pos >= pieceLen(d.pend):
      d.pend = Piece(kind: pkNone)
    if d.pend.kind != pkNone:
      let total = pieceLen(d.pend)
      let take = min(ChunkSize, total - d.pend.pos)
      let base = d.pend.pos
      # Commit the delivery position before `yield`, not after:
      #   a consumer that leaves the loop through `break` never resumes the iterator, so a post-yield
      # update strands `pos` and `cur` at their pre-yield values and re-hands the same chunk
      # forever. A yielded slice aliases `pend.s`, so a fully delivered piece is reclaimed at the loop
      # top above, never while its slice is live.
      d.pend.pos += take
      d.cur += take
      case d.pend.kind
      of pkSpan:
        yield m.jinja.toOpenArray(int d.pend.lo + base, int d.pend.lo + base + take - 1)
      of pkStr:
        yield d.pend.s.toOpenArray(base, base + take - 1)
      of pkNone:
        discard
      continue
    if d.curNode == noLink:
      break
    let n = d.curNode
    steps[m.nodes[n].kind](m, t, d, n)

proc pullAll*(m: Machine, t: Tables, d: var Driver): string =
  ## Renders to completion and returns every byte. Chunking composes with `cur`, so the two-pass
  ## counting contract needs no separate counting pass.
  for c in items(m, t, d):
    result.append c

proc renderToString*(src: string, ctx: Value, clock = 0.0): string =
  ## Compiles and renders in one call. `Machine` is built here, at the scope that owns `src`, because
  ## the artifact borrows the template text.
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  var d = newDriver(ctx, clock)
  pullAll(m, tables, d)
