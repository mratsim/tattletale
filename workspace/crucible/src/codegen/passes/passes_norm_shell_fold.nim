# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Normalization: constant-shell folding over CIRv2.
##
## A shell is what Nim's semcheck leaves of a generic proc instantiated with only static arguments:
## a body computing a constant with no parameter read.
## Layout algebra (`logical_divide`, `slice`, static shapes) crosses the kernel boundary through such shells.
##
##   shell call site ──▶ its resolved constant, substituted before verification and printing
##   dead shell definition ──▶ deleted
##   dead constant local ──▶ deleted by the optimization stage's dead-local pass
##
## Detection is syntactic:
## - single-assignment local bindings
## - no control flow and no calls in the candidate body
## - a resolved return value of literals, wrappers over literals, and constant constructors only

import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers
import std/[sets, tables, sequtils]

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerShellFolding*(reg: var PassRegistry) =
  ## Registers the constant-shell folding pass, the normalization stage.
  reg.register("foldConstantShells", pkTransform, phaseEarly,
    "Dissolves shell calls into their constants, deletes the dead shell definitions",
    run = foldConstantShells)

type
  FoldCtx = object
    fnName: string
    err: string

proc failing(ctx: FoldCtx): bool =
  ctx.err.len > 0

proc fail(ctx: var FoldCtx, detail: string) =
  if ctx.err.len == 0:
    ctx.err = "cirv2 fold: at " & ctx.fnName & ": " & detail

# ═══════════════════════════════════════════════════════════════════════
# Constant resolution
# ═══════════════════════════════════════════════════════════════════════

proc resolveConstant(e: CirExpr, binding: Table[string, CirExpr], seen: var HashSet[string]): CirExpr =
  ## Returns the constant `e` denotes, nil when it is not constant. A constant is a literal,
  ## a wrapper over constants, a constant constructor, or a local resolved through
  ## its single-assignment binding.
  if e.isNil:
    return nil
  case e.kind
  of eIntLit, eFloatLit, eNil, eTrue, eFalse:
    return e
  of eSym:
    if e.name in seen or not binding.hasKey(e.name):
      return nil
    seen.incl e.name
    return resolveConstant(binding[e.name], binding, seen)
  of eOconstr:
    var fields: seq[CirKv]
    for f in e.kvFields:
      let v = resolveConstant(f.value, binding, seen)
      if v.isNil:
        return nil
      fields.add kv(f.name, v)
    return oconstrE(e.ocType, fields)
  of eAconstr:
    var elems: seq[CirExpr]
    for el in e.elems:
      let v = resolveConstant(el, binding, seen)
      if v.isNil:
        return nil
      elems.add v
    return aconstrE(e.acType, elems)
  of eDot:
    let parent = resolveConstant(e.dotBase, binding, seen)
    if parent.isNil or parent.kind != eOconstr:
      return nil
    for f in parent.kvFields:
      if f.name == e.fieldName:
        return f.value
    return nil
  of eCast, eConv:
    let v = resolveConstant(e.castOperand, binding, seen)
    if v.isNil:
      return nil
    result = CirExpr(kind: eCast, castType: e.castType, castOperand: v)
    if e.kind == eConv:
      result = CirExpr(kind: eConv, castType: e.castType, castOperand: v)
  else:
    return nil

proc scanConstBindings(stmts: seq[CirStmt], bodyLocals: var HashSet[string],
    binding: var Table[string, CirExpr], returns: var seq[CirExpr],
    ctx: var FoldCtx): bool =
  ## Collects the single-assignment local bindings of a candidate body.
  ##
  ## Disqualifying statements:
  ## - a call or control flow
  ## - a local bound twice
  ## - an assignment to a non-local target
  for s in stmts:
    case s.kind
    of sDiscard:
      discard
    of sVar:
      if s.space != caThread:
        # smem and const_mem locals never fold
        return false
      if s.name in binding:
        return false
      bodyLocals.incl s.name
      if s.init != nil:
        binding[s.name] = s.init
    of sConst:
      if s.name in binding:
        return false
      bodyLocals.incl s.name
      if s.init != nil:
        binding[s.name] = s.init
    of sAsgn:
      if s.lhs.kind != eSym or s.lhs.name notin bodyLocals:
        return false
      if s.lhs.name in binding:
        return false
      binding[s.lhs.name] = s.rhs
    of sRet:
      if s.retVal != nil:
        returns.add s.retVal
    of sScope:
      if not scanConstBindings(s.scopeBody, bodyLocals, binding, returns, ctx):
        return false
    else:
      return false
  result = true

proc shellConstant(fn: CirProc): CirExpr =
  ## Returns the constant a proc computes when its body is a single constant expression,
  ## nil otherwise. A shell reads no parameter and runs no control flow.
  if fn.isKernel or fn.params.len > 0:
    return nil
  if fn.ret.isNil or fn.ret.kind == ctVoid:
    return nil
  var binding = initTable[string, CirExpr]()
  var returns: seq[CirExpr] = @[]
  var bodyLocals = initHashSet[string]()
  var ctx = FoldCtx(fnName: fn.name)
  if not scanConstBindings(fn.body, bodyLocals, binding, returns, ctx):
    return nil
  if returns.len != 1:
    return nil
  var seen = initHashSet[string]()
  return resolveConstant(returns[0], binding, seen)

# ═══════════════════════════════════════════════════════════════════════
# Substitution and elimination
# ═══════════════════════════════════════════════════════════════════════

proc substituteShells(stmts: var seq[CirStmt], folds: Table[string, CirExpr]) =
  ## Replaces assignments of shell calls with the shell's constant.
  ## The call args stay untouched, a folded shell reads no parameter.
  for i in 0 ..< stmts.len:
    let s = addr stmts[i]
    case s[].kind
    of sAsgn:
      if s[].rhs != nil and s[].rhs.kind == eCall and
          s[].rhs.callee.kind == eSym and folds.hasKey(s[].rhs.callee.name):
        s[].rhs = folds[s[].rhs.callee.name]
    of sIte:
      substituteShells(s[].thenBody, folds)
      substituteShells(s[].elseBody, folds)
    of sLoop:
      substituteShells(s[].loopBody, folds)
    of sScope:
      substituteShells(s[].scopeBody, folds)
    else:
      discard

proc foldConstantShells(m: CirModule): string =
  ## Dissolves shell calls into their constants, deletes the dead definitions
  ## and the dead constant locals, iterating to a fixpoint.
  ## Returns "" when clean, else the first refusal.
  var ctx = FoldCtx(fnName: "<module>")
  # fold runs until no shell call survives, each round may turn callers
  # of shells into shells themselves
  const maxRounds = 64
  for round in 0 ..< maxRounds:
    var folds = initTable[string, CirExpr]()
    for fn in m.procs:
      let constant = fn.shellConstant()
      if not constant.isNil:
        folds[fn.name] = constant
    if folds.len == 0:
      break
    for fn in m.procs:
      fn.body.substituteShells(folds)
    var surviving = initHashSet[string]()
    for fn in m.procs:
      fn.body.collectCallTargets(surviving)
    m.procs = m.procs.filterIt(not (folds.hasKey(it.name) and
        it.name notin surviving))
  return ctx.err
