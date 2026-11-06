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

proc findNumWorkgroupsExpr(coordTy: var CirType, e: CirExpr): bool =
  if e.isNil:
    return false
  if e.kind == eSym and e.builtin and e.name == "threadgroups_per_grid":
    coordTy = e.typ
    return true
  for ch in getExprChildren(e):
    if findNumWorkgroupsExpr(coordTy, ch):
      return true
  return false

proc findNumWorkgroupsStmts(coordTy: var CirType, ss: seq[CirStmt]): bool =
  for s in ss:
    case s.kind
    of sVar:
      if s.init != nil and findNumWorkgroupsExpr(coordTy, s.init):
        return true
    of sAsgn:
      if findNumWorkgroupsExpr(coordTy, s.lhs) or
          findNumWorkgroupsExpr(coordTy, s.rhs):
        return true
    of sIte:
      if findNumWorkgroupsExpr(coordTy, s.cond) or
          findNumWorkgroupsStmts(coordTy, s.thenBody) or
          findNumWorkgroupsStmts(coordTy, s.elseBody):
        return true
    of sWhile:
      if findNumWorkgroupsExpr(coordTy, s.whileCond) or
          findNumWorkgroupsStmts(coordTy, s.whileBody):
        return true
    of sRet:
      if s.retVal != nil and findNumWorkgroupsExpr(coordTy, s.retVal):
        return true
    of sDiscard:
      if findNumWorkgroupsExpr(coordTy, s.discardVal):
        return true
    of sCall:
      if findNumWorkgroupsExpr(coordTy, s.callCallee):
        return true
      for a in s.callArgs:
        if findNumWorkgroupsExpr(coordTy, a):
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

proc materializeNumWorkgroups(m: CirModule) =
  ## Kernels without such reads stay untouched, device functions never see one,
  ## the printer refuses a coordinate builtin in a device function.
  for fn in m.procs:
    if not fn.isKernel:
      continue
    var coordTy: CirType = nil
    if not findNumWorkgroupsStmts(coordTy, fn.body):
      continue
    if coordTy.isNil:
      raiseAssert("kernel " & fn.name & " reads threadgroups_per_grid through an" &
        " untyped expression, the materialized parameter has no type")
    var taken: seq[string]
    for p in fn.params:
      taken.add p.name
    var locals: seq[tuple[name: string, typ: CirType]]
    collectDeclaredLocals(locals, fn.body)
    for l in locals:
      taken.add l.name
    if NumWorkgroupsParam in taken:
      raiseAssert("kernel " & fn.name & ": the materialized parameter name '" &
        NumWorkgroupsParam & "' is taken")
    fn.params.add CirParam(name: NumWorkgroupsParam, typ: coordTy)
    rewriteNumWorkgroupsStmts(fn.body, coordTy)

