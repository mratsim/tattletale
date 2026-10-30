# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Call-graph analysis: one `pkAnalysis` pass filling `CirModule.callGraph`
## with the read-write classification of the module's buffer parameters,
## plus the device-function recursion refusal.
##
## What the pass computes:
## - the argument-flow edges: each call argument rooted at a parameter symbol flows into the callee's matching parameter
## - the read-write classification: a buffer parameter is read-write when its proc stores through it or a transitive callee does
## - the recursion refusal: a recursive device function would not terminate, the HLSL frontend accepts recursive source while miscompiling it
##
## Contract:
## - the target registers the pass after the last transform, before the check-only validation stage
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
  ## Call expressions of one statement list, a call binds to its destination:
  ## the sites are statement calls plus the right-hand sides of assignments,
  ## declarations and returns.
  for s in ss:
    case s.kind
    of sCall:
      outp.add CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
    of sAsgn:
      if s.rhs != nil and s.rhs.kind == eCall:
        outp.add s.rhs
    of sVar, sConst:
      if s.init != nil and s.init.kind == eCall:
        outp.add s.init
    of sRet:
      if s.retVal != nil and s.retVal.kind == eCall:
        outp.add s.retVal
    of sIte:
      directCalls(s.thenBody, outp)
      directCalls(s.elseBody, outp)
    of sLoop:
      directCalls(s.loopBody, outp)
    of sScope:
      directCalls(s.scopeBody, outp)
    of sJmp, sLab, sAssume, sDiscard, sCase, sAsm, sEmit:
      discard

proc scanStoredStmts(graph: var CallGraph, fn: CirProc, ss: seq[CirStmt]) =
  ## Records the parameters `fn` stores through: an assignment whose target
  ## roots at a parameter symbol, anywhere in the statement tree.
  for s in ss:
    case s.kind
    of sAsgn:
      let root = exprRootSym(s.lhs)
      if root != nil:
        for p in fn.params:
          if p.name == root.name:
            graph.stored.incl(fn.name & "/" & p.name)
    of sIte:
      graph.scanStoredStmts(fn, s.thenBody)
      graph.scanStoredStmts(fn, s.elseBody)
    of sLoop:
      graph.scanStoredStmts(fn, s.loopBody)
    of sScope:
      graph.scanStoredStmts(fn, s.scopeBody)
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
          graph.stored.incl(fn.name & "/" & p.name)
    of sCall:
      # a store can hide in a call-bound destination expression
      let root = exprRootSym(s.callCallee)
      if root != nil:
        for p in fn.params:
          if p.name == root.name:
            graph.stored.incl(fn.name & "/" & p.name)
      for a in s.callArgs:
        let ar = exprRootSym(a)
        if ar != nil:
          for p in fn.params:
            if p.name == ar.name:
              graph.stored.incl(fn.name & "/" & p.name)
    of sVar, sConst, sJmp, sLab, sAssume, sRet, sDiscard, sCase, sAsm:
      discard

proc scanCallStmts(graph: var CallGraph, fn: CirProc, ss: seq[CirStmt]) =
  ## Records one edge per call argument rooted at a parameter symbol:
  ## `caller/arg -> callee/paramI`. Call sites match `directCalls`:
  ## statement calls plus assignment, declaration and return right-hand sides.
  for s in ss:
    case s.kind
    of sCall:
      let calleeName = if s.callCallee.kind == eSym: s.callCallee.name else: ""
      if calleeName.len > 0 and graph.procs.hasKey(calleeName):
        let callee = graph.procs[calleeName]
        for i, a in s.callArgs:
          let root = exprRootSym(a)
          if root == nil:
            continue
          for p in fn.params:
            if p.name == root.name:
              if i < callee.params.len:
                graph.edges.add (fn.name, p.name, calleeName, callee.params[i].name)
    of sAsgn:
      if s.rhs != nil and s.rhs.kind == eCall:
        graph.scanCallStmts(fn, @[CirStmt(kind: sCall, callCallee: s.rhs.callee,
          callArgs: s.rhs.args)])
    of sVar, sConst:
      if s.init != nil and s.init.kind == eCall:
        graph.scanCallStmts(fn, @[CirStmt(kind: sCall, callCallee: s.init.callee,
          callArgs: s.init.args)])
    of sRet:
      if s.retVal != nil and s.retVal.kind == eCall:
        graph.scanCallStmts(fn, @[CirStmt(kind: sCall, callCallee: s.retVal.callee,
          callArgs: s.retVal.args)])
    of sIte:
      graph.scanCallStmts(fn, s.thenBody)
      graph.scanCallStmts(fn, s.elseBody)
    of sLoop:
      graph.scanCallStmts(fn, s.loopBody)
    of sScope:
      graph.scanCallStmts(fn, s.scopeBody)
    of sJmp, sLab, sAssume, sDiscard, sCase, sAsm, sEmit:
      discard

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
  result = checkDeviceFnRecursion(graph)

proc registerHlslCallGraphPass*(reg: var PassRegistry) =
  ## Registers the HLSL call-graph analysis pass over CIRv2 modules.
  ##
  ## Contract:
  ## - fills `CirModule.callGraph` with the argument-flow edges and the read-write classification of every buffer parameter
  ## - refuses recursive device functions
  ## - registers after the last transform, before the check-only validation stage
  ## - analysis results are valid only if no transform runs after them
  reg.register("hlslCallGraphAnalysis", pkAnalysis, phaseMain,
    "Builds the device-function call graph once: argument-flow edges, the" &
    " read-write classification of buffer parameters, the recursion refusal",
    run = hlslRun)
