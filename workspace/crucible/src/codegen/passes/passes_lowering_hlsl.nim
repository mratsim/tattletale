# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/[sets, strutils, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ../backends/lang_hlsl
import ./pass_datatypes
import ./pass_helpers
import ./passes_low_pointers

# ############################################################
#
#            HLSL lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerHlslMaterializeNumWorkgroups*(reg: var PassRegistry) =
  reg.register("hlslMaterializeNumWorkgroups", pkTransform,
    "Rewrites threadgroups_per_grid reads to the materialized trailing" &
    " kernel parameter the engine stages",
    run = materializeNumWorkgroups
  )

func registerHlslExplodePointerLocals*(reg: var PassRegistry) =
  registerExplodePointerLocals(reg, "hlslExplodePointerLocals",
    "Rewrites pointer locals, pointer-holding struct locals and var" &
    " parameters to the HLSL storage model",
    "HLSL")

func registerHlslHoistProgramScopeConstants*(reg: var PassRegistry) =
  reg.register("hlslHoistProgramScopeConstants", pkTransform,
    "Relocates const_mem locals into the module's program scope",
    run = hoistConstMemLocals
  )

# ═══════════════════════════════════════════════════════════════════════
#  The materialized dispatch-grid parameter
# ═══════════════════════════════════════════════════════════════════════

const NumWorkgroupsParam* = "crucible_numworkgroups"
  ## Trailing kernel parameter the dispatch grid size materializes into:
  ## the engine stages the launch's dispatch counts into the uniform block
  ## member of the same name.

proc findNumWorkgroupsExpr(e: CirExpr, coordTy: var CirType): bool =
  if e.isNil:
    return false
  if e.kind == eSym and e.builtin and e.name == "threadgroups_per_grid":
    coordTy = e.typ
    return true
  for ch in getExprChildren(e):
    if findNumWorkgroupsExpr(ch, coordTy):
      return true
  return false

proc findNumWorkgroupsStmts(ss: seq[CirStmt], coordTy: var CirType): bool =
  for s in ss:
    case s.kind
    of sVar:
      if s.init != nil and findNumWorkgroupsExpr(s.init, coordTy):
        return true
    of sAsgn:
      if findNumWorkgroupsExpr(s.lhs, coordTy) or
          findNumWorkgroupsExpr(s.rhs, coordTy):
        return true
    of sIte:
      if findNumWorkgroupsExpr(s.cond, coordTy) or
          findNumWorkgroupsStmts(s.thenBody, coordTy) or
          findNumWorkgroupsStmts(s.elseBody, coordTy):
        return true
    of sWhile:
      if findNumWorkgroupsExpr(s.whileCond, coordTy) or
          findNumWorkgroupsStmts(s.whileBody, coordTy):
        return true
    of sRet:
      if s.retVal != nil and findNumWorkgroupsExpr(s.retVal, coordTy):
        return true
    of sDiscard:
      if findNumWorkgroupsExpr(s.discardVal, coordTy):
        return true
    of sCall:
      if findNumWorkgroupsExpr(s.callCallee, coordTy):
        return true
      for a in s.callArgs:
        if findNumWorkgroupsExpr(a, coordTy):
          return true
    of sAsm, sEmit:
      discard
  return false

proc rewriteNumWorkgroupsExpr(e: CirExpr, coordTy: CirType) =
  if e.isNil:
    return
  if e.kind == eSym and e.builtin and e.name == "threadgroups_per_grid":
    e.name = NumWorkgroupsParam
    e.typ = coordTy
    e.builtin = false
    return
  for ch in getExprChildren(e):
    rewriteNumWorkgroupsExpr(ch, coordTy)

proc rewriteNumWorkgroupsStmts(ss: seq[CirStmt], coordTy: CirType) =
  ss.walkStmts(nil, proc(e: CirExpr) = rewriteNumWorkgroupsExpr(e, coordTy))

proc materializeNumWorkgroups(m: CirModule): string =
  ## Kernels without such reads stay untouched, device functions never see one,
  ## the printer refuses a coordinate builtin in a device function.
  for fn in m.procs:
    if not fn.isKernel:
      continue
    var coordTy: CirType = nil
    if not findNumWorkgroupsStmts(fn.body, coordTy):
      continue
    if coordTy.isNil:
      return "kernel " & fn.name & " reads threadgroups_per_grid through an" &
        " untyped expression, the materialized parameter has no type"
    var taken: seq[string]
    for p in fn.params:
      taken.add p.name
    var locals: seq[tuple[name: string, typ: CirType]]
    fn.body.collectDeclaredLocals(locals)
    for l in locals:
      taken.add l.name
    if NumWorkgroupsParam in taken:
      return "kernel " & fn.name & ": the materialized parameter name '" &
        NumWorkgroupsParam & "' is taken"
    fn.params.add CirParam(name: NumWorkgroupsParam, typ: coordTy)
    rewriteNumWorkgroupsStmts(fn.body, coordTy)
  return ""

