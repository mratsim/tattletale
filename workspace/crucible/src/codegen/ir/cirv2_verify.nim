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
## once at the final pass boundary, after the last transform pass or after the last
## pass when the registry has none.
##
## Dev-only: `CirModule.verifyIR`, defaulted from `-d:CIR_verify`.
## Read-only: returns the first violation, rendered with the offending proc where one exists, never rewrites.
## Checks, per module:
## - emitted proc names unique, a kernel emits its `exportName` when set, else its name:
##   the printers emit names verbatim and the runtime caches binaries on kernel names
## - global names unique and disjoint from emitted proc names: hoisted declarations emit at file scope
## - every global is a `sVar` whose initializer resolves against the declarations before it,
##   a nil global refuses
##
## Checks, per proc (kernels and device functions alike, lowering rewrites bodies in place):
## - param names unique and non-empty: duplicates emit ill-formed target source, an empty name passes the uniqueness scan
## - declarations of one name in one statement list refuse: the printers emit one list into one brace scope
## - a body-top-level declaration shadowing a parameter refuses: parameters share the body's scope
## - a nil statement refuses, malformed input fails fast before the walk dereferences it
##
## Scope contract:
## - every symbol resolves to a param, a `sVar` declared earlier in the body, a module global, or another module proc
## - a `sVar` joins scope after its initializer checks, so a declaration never resolves its own initializer
## - declarations inside a branch or loop body scope to that body, matching the braces C-family printers emit
##
## Exempt from the scope walk: `{.builtin.}`-marked symbols and `IntrinsicCallees` below, each backed by the pass that emits it.
##
## Not checked, deliberately:
## - feature-flag consistency: the pass registry refuses transforms after analyses
## - forward/backward pair ABI: the autodiff refusals own it
## - enum dispatch coverage: the compiler guarantees it
## - any canonical form: none is imposed
##
## Out of scope by construction:
## - pass-precise blame: the verifier sees only the final pass boundary, bisect the registry to name the introducing pass
## - analysis-record checks (features, call graph, ref blocks): verifyModule sees IR shape only,
##   such checks belong after the analyses

const IntrinsicCallees = ["__builtin_memcpy", "max", "min"]
  ## Each entry is emitted by a named pass and provided by the target languages:
  ##
  ## - `__builtin_memcpy`: the decomposeArrayAssign pass, C targets provide the intrinsic
  ## - `max`, `min`: the foldMaxMinToBuiltins pass emits them as callees.
  ##   Consumers: WGSL/HLSL printers, SPIR-V encoder, and the CUDA math mapping.

proc getEmittedName(fn: CirProc): string =
  ## Returns the file-scope name the C-family printers emit for `fn`,
  ## its `exportName` when the kernel sets one and its name otherwise.
  result = if fn.isKernel and fn.exportName.len > 0: fn.exportName else: fn.name

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

proc checkScopeStmts(ss: seq[CirStmt], visible: var HashSet[string], moduleNames: HashSet[string], outermost = false): string =
  ## Returns "" while every symbol in `ss` resolves in statement order, else the first violation.
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
      return "nil statement in body"
    case s.kind
    of sVar:
      result = checkScopeExpr(s.init, visible, moduleNames)
      if result.len > 0:
        return
      if s.name in declared:
        return "duplicate local name '" & s.name & "'"
      if outermost and s.name in visible:
        return "local '" & s.name & "' shadows a parameter"
      declared.incl s.name
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
  var emittedNames: HashSet[string]
  for fn in m.procs:
    moduleNames.incl fn.name
    emittedNames.incl fn.getEmittedName()
  var seenGlobals: HashSet[string]
  for g in m.globals:
    # Globals come from the lowering passes, which hoist `var` statements only.
    if g.isNil:
      return "nil statement in module globals"
    if g.kind != sVar:
      return "unsupported global kind '" & $g.kind & "', only sVar globals are supported"
    if g.name in seenGlobals:
      return "duplicate global name '" & g.name & "'"
    if g.name in moduleNames or g.name in emittedNames:
      return "global name '" & g.name & "' collides with a proc name"
    seenGlobals.incl g.name
    # Printers emit globals in declaration order at file scope: an initializer resolves only
    # against the names collected before it, never the global's own name.
    result = checkScopeExpr(g.init, moduleNames, moduleNames)
    if result.len > 0:
      return "at global '" & g.name & "': " & result
    moduleNames.incl g.name
  var seenProcs: HashSet[string]
  for fn in m.procs:
    let emitted = fn.getEmittedName()
    if emitted in seenProcs:
      return renderProcRefusal(fn, "duplicate emitted name '" & emitted & "'")
    seenProcs.incl emitted
    var paramNames: seq[string]
    var visible: HashSet[string]
    for p in fn.params:
      if p.name.len == 0:
        # checkUniqueNames reports "" for a clean scan, an empty name would pass it
        return renderProcRefusal(fn, "at " & fn.name & ": empty parameter name")
      paramNames.add p.name
      visible.incl p.name
    result = checkUniqueNames(paramNames)
    if result.len > 0:
      return renderProcRefusal(fn, "at " & fn.name & ": duplicate parameter name '" & result & "'")
    result = checkScopeStmts(fn.body, visible, moduleNames, outermost = true)
    if result.len > 0:
      return renderProcRefusal(fn, "at " & fn.name & ": " & result)
