# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/sets
import cirv2_lang
import cirv2_pretty

# ############################################################
#
#            CIRv2 pass-boundary verifier
#
# ############################################################

## CIRv2 pass-boundary verifier. Run by `runPasses` once at the final pass boundary.
## Dev-only: `CirModule.verifyIR`, defaulted from `-d:CIR_verify`.
## Read-only: the first violation is reported, rendered with the offending proc where one exists.
##
## Per module:
## - emitted proc names unique (a kernel emits its `exportName` when set, else its name)
## - global names unique, disjoint from emitted proc names
## - globals are `sVar`s whose initializers resolve against prior declarations
## - a nil global refuses
##
## Per proc:
## - param names unique and non-empty
## - one declaration per name per statement list
## - a body-top-level declaration never shadows a parameter
## - a nil statement refuses
##
## Per scope:
## - every symbol resolves to a param, an earlier `sVar`, a global, or a module proc
## - a `sVar` joins scope after its initializer checks
## - branch and loop declarations scope to their body
## - `{.builtin.}` symbols and `IntrinsicCallees` below are exempt, each pass-backed

const IntrinsicCallees = ["__builtin_memcpy", "max", "min"]
  ## Each entry is emitted by a named pass and provided by the target languages:
  ##
  ## - `__builtin_memcpy`: the decomposeArrayAssign pass, C targets provide the intrinsic
  ## - `max`, `min`: the foldMaxMinToBuiltins pass emits them as callees.
  ##   Consumers: WGSL/HLSL printers, SPIR-V encoder, and the CUDA math mapping.

proc getEmittedName(fn: CirProc): string =
  result = if fn.isKernel and fn.exportName.len > 0: fn.exportName else: fn.name

proc refuseProc(fn: CirProc, violation: string) =
  raiseAssert(violation)

proc checkScopeExpr(e: CirExpr, visible, moduleNames: HashSet[string], where = "") =
  ## A `{.builtin.}`-marked symbol is a target intrinsic, never a declaration.
  ## `where` prefixes the violation, the global checks name the offending global with it.
  if e.isNil:
    return
  if e.kind == eStmts:
    raiseAssert(where & "statement-expression survived elimination," &
      " the rewriteStmtsExprs pass must run before the printers")
  if e.kind == eSym:
    if e.builtin or e.name in IntrinsicCallees or e.name in visible or e.name in moduleNames:
      return
    raiseAssert(where & "symbol '" & e.name & "' resolves to no declaration in scope")
  for ch in e.getExprChildren():
    checkScopeExpr(ch, visible, moduleNames, where)

proc checkScopeStmts(ss: seq[CirStmt], visible: var HashSet[string], moduleNames: HashSet[string], outermost = false, where = "") =
  ## `where` prefixes every violation raised below the call, the proc check names
  ## the offending proc with it.
  ##
  ## A `sVar` joins `visible` after its own initializer checks, so a declaration never resolves its own initializer.
  ##
  ## Two declarations of one name in the same statement list refuse: the printers emit
  ## one list into one brace scope. A body-top-level declaration that shadows a parameter
  ## refuses too: parameters and the body share one scope in the emitted source.
  ##
  ## Branch and loop bodies take a copy of `visible`, a declaration inside a body is scoped to that body,
  ## matching the braces the C-family printers emit.
  var declared: HashSet[string]
  for s in ss:
    if s.isNil:
      raiseAssert(where & "nil statement in body")
    case s.kind
    of sVar:
      checkScopeExpr(s.init, visible, moduleNames, where)
      if s.name in declared:
        raiseAssert(where & "duplicate local name '" & s.name & "'")
      if outermost and s.name in visible:
        raiseAssert(where & "local '" & s.name & "' shadows a parameter")
      declared.incl s.name
      visible.incl s.name
    of sAsgn:
      checkScopeExpr(s.lhs, visible, moduleNames, where)
      checkScopeExpr(s.rhs, visible, moduleNames, where)
    of sIte:
      checkScopeExpr(s.cond, visible, moduleNames, where)
      var thenVisible = visible
      checkScopeStmts(s.thenBody, thenVisible, moduleNames, where = where)
      var elseVisible = visible
      checkScopeStmts(s.elseBody, elseVisible, moduleNames, where = where)
    of sWhile:
      checkScopeExpr(s.whileCond, visible, moduleNames, where)
      var loopVisible = visible
      checkScopeStmts(s.whileBody, loopVisible, moduleNames, where = where)
    of sRet:
      checkScopeExpr(s.retVal, visible, moduleNames, where)
    of sDiscard:
      checkScopeExpr(s.discardVal, visible, moduleNames, where)
    of sCall:
      checkScopeExpr(s.callCallee, visible, moduleNames, where)
      for a in s.callArgs:
        checkScopeExpr(a, visible, moduleNames, where)
    of sAsm:
      for op in s.asmOperands:
        checkScopeExpr(op, visible, moduleNames, where)
    of sEmit:
      for op in s.emitOperands:
        checkScopeExpr(op, visible, moduleNames, where)

proc checkUniqueParamNames(fn: CirProc, names: seq[string]) =
  var seen: HashSet[string]
  for n in names:
    if n in seen:
      refuseProc(fn, "at " & fn.name & ": duplicate parameter name '" & n & "'")
    seen.incl n

proc verifyModule*(m: CirModule) =
  ## Raises on the first violation of the pass-boundary contracts (catalogue in the module header),
  ## rendered with the offending proc for context.
  var moduleNames: HashSet[string]
  var emittedNames: HashSet[string]
  for fn in m.procs:
    moduleNames.incl fn.name
    emittedNames.incl fn.getEmittedName()
  var seenGlobals: HashSet[string]
  for g in m.globals:
    # Globals come from the lowering passes, which hoist `var` statements only.
    if g.isNil:
      raiseAssert("nil statement in module globals")
    if g.kind != sVar:
      raiseAssert("unsupported global kind '" & $g.kind & "', only sVar globals are supported")
    if g.name in seenGlobals:
      raiseAssert("duplicate global name '" & g.name & "'")
    if g.name in moduleNames or g.name in emittedNames:
      raiseAssert("global name '" & g.name & "' collides with a proc name")
    seenGlobals.incl g.name
    # Printers emit globals in declaration order at file scope: an initializer resolves only
    # against the names collected before it, never the global's own name.
    checkScopeExpr(g.init, moduleNames, moduleNames, where = "at global '" & g.name & "': ")
    moduleNames.incl g.name
  var seenProcs: HashSet[string]
  for fn in m.procs:
    let emitted = fn.getEmittedName()
    if emitted in seenProcs:
      refuseProc(fn, "duplicate emitted name '" & emitted & "'")
    seenProcs.incl emitted
    var paramNames: seq[string]
    var visible: HashSet[string]
    for p in fn.params:
      if p.name.len == 0:
        # an empty parameter name would pass checkUniqueParamNames, it refuses here
        refuseProc(fn, "at " & fn.name & ": empty parameter name")
      paramNames.add p.name
      visible.incl p.name
    checkUniqueParamNames(fn, paramNames)
    checkScopeStmts(fn.body, visible, moduleNames, outermost = true,
      where = "at " & fn.name & ": ")
