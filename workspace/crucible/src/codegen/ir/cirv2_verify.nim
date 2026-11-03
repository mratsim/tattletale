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

## Verifier for the two contracts every pass boundary relies on, run by `runPasses`
## once after the last transform pass.
##
## Dev-only: `CirModule.verifyIR`, defaulted from `-d:CIR_verify`.
## Read-only: returns the first violation, rendered with the offending proc, never rewrites.
##
## Checks, per module:
## - proc names unique: printers emit names verbatim, the runtime caches binaries on kernel names
##
## Checks, per proc (kernels and device functions alike, lowering rewrites bodies in place):
## - param names unique: duplicates emit ill-formed target source
## - every symbol resolves to a param, a `sVar` declared earlier in the body,
##   a module global, or another module proc
##   - a `sVar` joins scope after its initializer checks
##   - declarations inside a branch or loop body scope to that body, matching the braces every printer emits
##
## Exempt from the scope walk: `{.builtin.}`-marked symbols and `IntrinsicCallees` below, each backed by the pass that emits it.
##
## Not checked, deliberately:
## - feature-flag consistency: the pass registry refuses transforms after analyses
## - forward/backward pair ABI: the autodiff refusals own it
## - enum dispatch coverage: the compiler guarantees it
## - any canonical form: none is imposed

const IntrinsicCallees = ["__builtin_memcpy", "max", "min"]
  ## Each entry is emitted by a named pass and provided by the target languages:
  ##
  ## - `__builtin_memcpy`: the decomposeArrayAssign pass, C targets provide the intrinsic
  ## - `max`, `min`: the foldMaxMinToBuiltins pass, WGSL and HLSL provide the native builtins

proc checkScopeExpr(e: CirExpr, visible, moduleNames: HashSet[string]): string =
  ## Returns "" while every symbol in `e` resolves, else the violation.
  ## A `{.builtin.}`-marked symbol is a target intrinsic, never a declaration.
  if e.isNil:
    return ""
  if e.kind == eSym:
    if e.builtin or e.name in IntrinsicCallees or e.name in visible or e.name in moduleNames:
      return ""
    return "symbol '" & e.name & "' resolves to no declaration in scope"
  for ch in e.getExprChildren():
    result = checkScopeExpr(ch, visible, moduleNames)
    if result.len > 0:
      return

proc checkScopeStmts(ss: seq[CirStmt], visible: var HashSet[string], moduleNames: HashSet[string]): string =
  ## Returns "" while every symbol in `ss` resolves in statement order, else the first violation.
  ##
  ## A `sVar` joins `visible` after its own initializer checks, so a declaration never resolves its own initializer.
  ##
  ## Branch and loop bodies take a copy of `visible`, a declaration inside a body is scoped to that body,
  ## matching the braces every printer emits, and a use after the body resolves to nothing.
  for s in ss:
    case s.kind
    of sVar:
      result = checkScopeExpr(s.init, visible, moduleNames)
      if result.len > 0:
        return
      visible.incl s.name
    of sAsgn:
      result = checkScopeExpr(s.lhs, visible, moduleNames)
      if result.len > 0:
        return
      result = checkScopeExpr(s.rhs, visible, moduleNames)
      if result.len > 0:
        return
    of sIte:
      result = checkScopeExpr(s.cond, visible, moduleNames)
      if result.len > 0:
        return
      var thenVisible = visible
      result = checkScopeStmts(s.thenBody, thenVisible, moduleNames)
      if result.len > 0:
        return
      var elseVisible = visible
      result = checkScopeStmts(s.elseBody, elseVisible, moduleNames)
      if result.len > 0:
        return
    of sWhile:
      result = checkScopeExpr(s.whileCond, visible, moduleNames)
      if result.len > 0:
        return
      var loopVisible = visible
      result = checkScopeStmts(s.whileBody, loopVisible, moduleNames)
      if result.len > 0:
        return
    of sRet:
      result = checkScopeExpr(s.retVal, visible, moduleNames)
      if result.len > 0:
        return
    of sDiscard:
      result = checkScopeExpr(s.discardVal, visible, moduleNames)
      if result.len > 0:
        return
    of sCall:
      result = checkScopeExpr(s.callCallee, visible, moduleNames)
      if result.len > 0:
        return
      for a in s.callArgs:
        result = checkScopeExpr(a, visible, moduleNames)
        if result.len > 0:
          return
    of sAsm:
      for op in s.asmOperands:
        result = checkScopeExpr(op, visible, moduleNames)
        if result.len > 0:
          return
    of sEmit:
      for op in s.emitOperands:
        result = checkScopeExpr(op, visible, moduleNames)
        if result.len > 0:
          return

proc checkUniqueNames(names: seq[string]): string =
  ## Returns "" while every name in `names` is unique, else the first repeated name.
  var seen: HashSet[string]
  for n in names:
    if n in seen:
      return n
    seen.incl n

proc renderProcRefusal(fn: CirProc, violation: string): string =
  violation & "\n" & renderProc(fn)

proc verifyModule*(m: CirModule): string =
  ## Returns "" while the module upholds the pass-boundary contracts (catalogue in the module header),
  ## else the first violation rendered as a refusal line plus the offending proc for context.
  var moduleNames: HashSet[string]
  for g in m.globals:
    if g.kind == sVar:
      moduleNames.incl g.name
  for fn in m.procs:
    moduleNames.incl fn.name
  var seenProcs: HashSet[string]
  for fn in m.procs:
    if fn.name in seenProcs:
      # Duplicate proc names refuse on the second occurrence and the walk renders that proc.
      return renderProcRefusal(fn, "duplicate proc name '" & fn.name & "'")
    seenProcs.incl fn.name
    var paramNames: seq[string]
    var visible: HashSet[string]
    for p in fn.params:
      paramNames.add p.name
      visible.incl p.name
    result = checkUniqueNames(paramNames)
    if result.len > 0:
      return renderProcRefusal(fn, "at " & fn.name & ": duplicate parameter name '" & result & "'")
    result = checkScopeStmts(fn.body, visible, moduleNames)
    if result.len > 0:
      return renderProcRefusal(fn, "at " & fn.name & ": " & result)
