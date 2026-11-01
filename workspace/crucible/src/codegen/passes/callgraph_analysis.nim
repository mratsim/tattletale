# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Call-graph analysis: the `pkAnalysis` pass filling `CirModule.callGraph`
## with the read-write classification of the module's buffer parameters,
## and the shared device-function recursion refusal pass.
##
## What the call-graph pass computes:
## - the argument-flow edges: each call argument rooted at a parameter symbol flows into the callee's matching parameter
## - the read-write classification: a buffer parameter is read-write when its proc stores through it or a transitive callee does
##
## What the recursion pass refuses:
## - a recursive device function would not terminate
##
## Contract:
## - the target registers each pass after the last transform, before the check-only validation stage
## - analysis results are valid only if no transform runs after them: the registry runs in registration order and never invalidates
## - the printer reads the stored record in `CirModule.callGraph`, it never rescans

import std/[sets, strutils, tables]
import ../ir/cirv2_lang
import ./pass_helpers
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Argument-flow edges
# ═══════════════════════════════════════════════════════════════════════

proc directCalls(ss: seq[CirStmt], outp: var seq[CirExpr]) =
  ## Call expressions of one statement list, in walk order.
  ##
  ## A call value may appear in any operand position, every position is visited:
  ##
  ## - statement calls
  ## - assignment, declaration, and return values
  ## - operands nested inside other calls
  ## - loop conditions and ternary branches
  var calls: seq[CirExpr]
  let onStmt = proc(s: CirStmt) =
    if s.kind == sCall:
      calls.add CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
  let onExpr = proc(e: CirExpr) =
    if e != nil and e.kind == eCall:
      calls.add e
  ss.walkStmts(onStmt, onExpr)
  outp = calls

proc scanStoredStmts(graph: var CallGraph, fn: CirProc, ss: seq[CirStmt]) =
  ## Records the parameters `fn` stores through, anywhere in the statement tree:
  ##
  ## - an assignment whose target roots at a parameter symbol
  ## - a statement call whose callee or argument roots at one
  ##   (a var parameter that the callee stores through, or a buffer passed deeper)
  ## - the verbatim text of an emit, which can name a buffer
  var g = graph
  proc markStored(root: CirExpr) =
    if root != nil:
      for p in fn.params:
        if p.name == root.name:
          g.stored.incl(fn.name & "/" & p.name)
  let onStmt = proc(s: CirStmt) =
    case s.kind
    of sAsgn:
      markStored(exprRootSym(s.lhs))
    of sCall:
      markStored(exprRootSym(s.callCallee))
      for a in s.callArgs:
        markStored(exprRootSym(a))
    of sEmit:
      # a store can hide in an emit: the verbatim text names the buffer,
      # every parameter the text mentions counts as stored
      var emitText = ""
      for part in s.emitParts:
        emitText.add part
      for op in s.emitOperands:
        emitText.add op.name
      for p in fn.params:
        if p.typ.kind in {ctPtr, ctAptr} and emitText.contains(p.name):
          g.stored.incl(fn.name & "/" & p.name)
    else:
      discard
  ss.walkStmts(onStmt, nil)
  graph = g

proc scanCallStmts(graph: var CallGraph, fn: CirProc, ss: seq[CirStmt]) =
  ## Records one edge per call argument rooted at a parameter symbol:
  ## `caller/arg -> callee/paramI`.
  ##
  ## Call sites match `directCalls`, every operand position included,
  ## each nested call contributes one edge.
  var g = graph
  proc record(callee: CirExpr, args: seq[CirExpr]) =
    let calleeName = if callee != nil and callee.kind == eSym: callee.name else: ""
    if calleeName.len == 0 or not g.procs.hasKey(calleeName):
      return
    let calleeFn = g.procs[calleeName]
    for i, a in args:
      let root = exprRootSym(a)
      if root == nil or i >= calleeFn.params.len:
        continue
      for p in fn.params:
        if p.name == root.name:
          g.edges.add (fn.name, p.name, calleeName, calleeFn.params[i].name)
  let onStmt = proc(s: CirStmt) =
    if s.kind == sCall:
      record(s.callCallee, s.callArgs)
  let onExpr = proc(e: CirExpr) =
    if e != nil and e.kind == eCall:
      record(e.callee, e.args)
  ss.walkStmts(onStmt, onExpr)
  graph = g

# ═══════════════════════════════════════════════════════════════════════
# Read-write fixpoint and recursion refusal
# ═══════════════════════════════════════════════════════════════════════

proc classifyRwBuffers(graph: var CallGraph) =
  ## Marks every buffer parameter stored through, directly or transitively
  ## over the argument-flow edges: a read-write fixpoint over the call graph.
  ## Each edge propagates read-write both ways:
  ## - a store through the callee's parameter writes the caller's buffer
  ## - a stored caller buffer flows into the callee's parameter
  graph.reached = graph.stored
  var grew = true
  while grew:
    grew = false
    for e in graph.edges:
      let src = e.caller & "/" & e.arg
      let dst = e.callee & "/" & e.param
      if src in graph.reached and dst notin graph.reached:
        graph.reached.incl(dst)
        grew = true
      if dst in graph.reached and src notin graph.reached:
        graph.reached.incl(src)
        grew = true

proc checkDeviceFnRecursion(graph: CallGraph): string =
  ## Returns the recursion refusal over the device-function call graph,
  ## "" when the graph is acyclic.
  var state = initTable[string, int]() # fn name: 1 = on the visit stack, 2 = fully visited

  proc visit(state: var Table[string, int], calleesOf: Table[string, seq[string]], fn: string): string =
    state[fn] = 1
    for callee in calleesOf.getOrDefault(fn, @[]):
      case state.getOrDefault(callee, 0)
      of 1:
        return "recursive device function `" & callee &
          "` would not terminate, recursion is rejected"
      of 0:
        result = visit(state, calleesOf, callee)
        if result.len > 0:
          return
      else:
        discard
    state[fn] = 2

  var calleesOf: Table[string, seq[string]]
  for fn in graph.procs.values:
    if fn.isKernel:
      continue
    var calls: seq[CirExpr]
    directCalls(fn.body, calls)
    for c in calls:
      if c.callee.kind == eSym and graph.procs.hasKey(c.callee.name) and
          not graph.procs[c.callee.name].isKernel:
        calleesOf.mgetOrPut(fn.name, @[]).add(c.callee.name)
  for fn in graph.procs.values:
    if not fn.isKernel and state.getOrDefault(fn.name, 0) == 0:
      result = visit(state, calleesOf, fn.name)
      if result.len > 0:
        return

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

proc hlslRun(m: CirModule): string {.nimcall.} =
  var graph = CallGraph()
  for fn in m.procs:
    graph.procs[fn.name] = fn
  for fn in m.procs:
    graph.scanStoredStmts(fn, fn.body)
    graph.scanCallStmts(fn, fn.body)
  graph.classifyRwBuffers()
  m.callGraph = graph

proc deviceRecursionRefusal(m: CirModule): string {.nimcall.} =
  ## Pass run over `m`: the device-function recursion refusal.
  var graph = CallGraph()
  for fn in m.procs:
    graph.procs[fn.name] = fn
  result = checkDeviceFnRecursion(graph)

proc registerDeviceRecursionPass*(reg: var PassRegistry) =
  ## Registers the device-function recursion refusal over CIRv2 modules.
  ##
  ## Contract:
  ## - refuses a recursive device function, which would not terminate
  ## - registers after the last transform, before the check-only validation stage
  reg.register("validateDeviceFnRecursion", pkValidation, phaseMain,
    "Refuses recursive device functions: a recursion would not terminate",
    run = deviceRecursionRefusal)

proc registerHlslCallGraphPass*(reg: var PassRegistry) =
  ## Registers the HLSL call-graph analysis pass over CIRv2 modules.
  ##
  ## Contract:
  ## - fills `CirModule.callGraph` with the argument-flow edges and the read-write classification of every buffer parameter
  ## - registers after the last transform, before the check-only validation stage
  ## - analysis results are valid only if no transform runs after them
  reg.register("hlslCallGraphAnalysis", pkAnalysis, phaseMain,
    "Builds the device-function call graph once: argument-flow edges, the" &
    " read-write classification of buffer parameters",
    run = hlslRun)
