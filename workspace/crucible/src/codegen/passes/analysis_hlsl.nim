# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[sets, strutils, tables]
import ../ir/cirv2_lang
import ./pass_helpers
import ./pass_datatypes

# ############################################################
#
#            HLSL analysis
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Argument-flow edges
# ═══════════════════════════════════════════════════════════════════════

proc collectDirectCalls(stmts: seq[CirStmt]): seq[CirExpr] =
  let onStmt = proc(s: CirStmt) =
    if s.kind == sCall:
      result.add CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
  let onExpr = proc(e: CirExpr) =
    if e != nil and e.kind == eCall:
      result.add e
  stmts.walkStmts(onStmt, onExpr)

proc scanStoredStmts(stored: var HashSet[string], fn: CirProc, ss: seq[CirStmt]) =
  ## Records the parameters `fn` stores through, anywhere in the statement tree:
  ##
  ## - an assignment whose target roots at a parameter symbol
  ## - a statement call whose callee or argument roots at one
  ##   (a var parameter that the callee stores through, or a buffer passed deeper)
  ## - the verbatim text of an emit, which can name a buffer
  var g = stored
  proc markStored(root: CirExpr) =
    if root != nil:
      for p in fn.params:
        if p.name == root.name:
          g.incl(fn.name & "/" & p.name)
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
          g.incl(fn.name & "/" & p.name)
    else:
      discard
  ss.walkStmts(onStmt, nil)
  stored = g

proc scanCallStmts(edges: var seq[tuple[caller, arg, callee, param: string]],
    procs: Table[string, CirProc], fn: CirProc, ss: seq[CirStmt]) =
  ## Records one edge per call argument rooted at a parameter symbol:
  ## `caller/arg -> callee/paramI`.
  ##
  ## Call sites match `directCalls`, every operand position included,
  ## each nested call contributes one edge.
  var es = edges
  proc record(callee: CirExpr, args: seq[CirExpr]) =
    let calleeName = if callee != nil and callee.kind == eSym: callee.name else: ""
    if calleeName.len == 0 or not procs.hasKey(calleeName):
      return
    let calleeFn = procs[calleeName]
    for i, a in args:
      let root = exprRootSym(a)
      if root == nil or i >= calleeFn.params.len:
        continue
      for p in fn.params:
        if p.name == root.name:
          es.add (fn.name, p.name, calleeName, calleeFn.params[i].name)
  let onStmt = proc(s: CirStmt) =
    if s.kind == sCall:
      record(s.callCallee, s.callArgs)
  let onExpr = proc(e: CirExpr) =
    if e != nil and e.kind == eCall:
      record(e.callee, e.args)
  ss.walkStmts(onStmt, onExpr)
  edges = es

# ═══════════════════════════════════════════════════════════════════════
#  Read-write fixpoint
# ═══════════════════════════════════════════════════════════════════════

proc classifyRwBuffers(stored: HashSet[string], edges: seq[tuple[caller, arg, callee, param: string]]): HashSet[string] =
  ## Marks every buffer parameter stored through, directly or transitively
  ## over the argument-flow edges: a read-write fixpoint over the call graph.
  ## Each edge propagates read-write both ways:
  ## - a store through the callee's parameter writes the caller's buffer
  ## - a stored caller buffer flows into the callee's parameter
  result = stored
  var grew = true
  while grew:
    grew = false
    for e in edges:
      let src = e.caller & "/" & e.arg
      let dst = e.callee & "/" & e.param
      if src in result and dst notin result:
        result.incl(dst)
        grew = true
      if dst in result and src notin result:
        result.incl(src)
        grew = true

proc checkDeviceFnRecursion(procs: Table[string, CirProc]): string =
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
  for fn in procs.values:
    if fn.isKernel:
      continue
    for c in fn.body.collectDirectCalls:
      if c.callee.kind == eSym and procs.hasKey(c.callee.name) and
          not procs[c.callee.name].isKernel:
        calleesOf.mgetOrPut(fn.name, @[]).add(c.callee.name)
  for fn in procs.values:
    if not fn.isKernel and state.getOrDefault(fn.name, 0) == 0:
      result = visit(state, calleesOf, fn.name)
      if result.len > 0:
        return

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

proc hlslRun(m: CirModule): string {.nimcall.} =
  var procs: Table[string, CirProc]
  for fn in m.procs:
    procs[fn.name] = fn
  var stored: HashSet[string]
  var edges: seq[tuple[caller, arg, callee, param: string]]
  for fn in m.procs:
    stored.scanStoredStmts(fn, fn.body)
    scanCallStmts(edges, procs, fn, fn.body)
  m.callGraph = CallGraph(stored: stored,
    reached: classifyRwBuffers(stored, edges))
  result = checkDeviceFnRecursion(procs)

proc registerHlslCallGraphPass*(reg: var PassRegistry) =
  reg.register("hlslCallGraphAnalysis", pkAnalysis,
    "Builds the device-function call graph once: argument-flow edges, the" &
    " read-write classification of buffer parameters, the recursion refusal",
    run = hlslRun)
