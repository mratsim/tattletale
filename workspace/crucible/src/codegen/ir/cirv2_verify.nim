# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/sets
import cirv2_lang

# ############################################################
#
#            CIRv2 pass-boundary verifier
#
# ############################################################

const IntrinsicCallees* = ["__builtin_memcpy", "max", "min"]
  ## Callees that no declaration ever backs, exempt from the scope walk.
  ## The printers emit each name verbatim and the target languages provide it:
  ##
  ## - `__builtin_memcpy`: the decomposeArrayAssign pass emits it, C targets provide the intrinsic
  ## - `max`, `min`: the foldMaxMinToBuiltins pass emits ternary max/min idioms
  ##   as calls to these names, WGSL and HLSL provide the native builtins

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

proc verifyModule*(m: CirModule): string =
  ## Returns "" while every symbol in every proc resolves, else the first violation naming the proc and the symbol.
  ## Read-only: the walk never rewrites or repairs the IR.
  ## Runs at the final pass boundary: runPasses calls it once, after the last transform pass.
  ##
  ## Contract, per proc:
  ## - a symbol resolves to the proc's params, a `sVar` declared earlier in the body, a module global, or another module proc
  ## - `{.builtin.}`-marked symbols are target intrinsics, exempt
  ## - declaration precedes use inside the proc body, globals and procs carry no statement order
  ##
  ## Module procs and kernels are walked the same way: the lowering passes rewrite device function bodies in place,
  ## so a transform can dangle a symbol in either.
  var moduleNames: HashSet[string]
  for g in m.globals:
    if g.kind == sVar:
      moduleNames.incl g.name
  for fn in m.procs:
    moduleNames.incl fn.name
  for fn in m.procs:
    var visible: HashSet[string]
    for p in fn.params:
      visible.incl p.name
    result = checkScopeStmts(fn.body, visible, moduleNames)
    if result.len > 0:
      return "at " & fn.name & ": " & result
