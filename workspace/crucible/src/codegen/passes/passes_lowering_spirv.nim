{.experimental: "codeReordering".}

## SPIR-V per-target lowering: bool storage retype.
##
## The 4-byte push-constant convention has no bool member.
## Module-level bool storage and kernel-boundary bool symbols retype
## to i32 before printing:
##
## - reads of the retyped symbols wrap in conv(bool)
## - assignment right-hand sides into them wrap in conv(i32)
## - device functions are untouched, their own bool parameters and locals
##   stay real bools

import std/sets
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSpirvLoweringPasses*(reg: var PassRegistry) =
  ## Registers the SPIR-V per-target lowering passes:
  ##
  ## - `spirvPatchBoolToI32`, before printing: bool kernel parameters and bool
  ##   globals retype to i32 storage, reads and stores wrap in conversions
  reg.register("spirvPatchBoolToI32", pkTransform, phaseMain,
    "Retypes bool kernel parameters and bool globals to i32 storage",
    run = spirvPatchBoolToI32
  )

proc patchBoolReadsExpr(e: CirExpr, boolVals, boolPtrs: HashSet[string]) =
  ## Wraps every value-position read of an i32-stored bool back into bool.
  ## Expressions are refs, the rewrite applies to every sharer.
  if e.isNil:
    return
  if e.kind == eSym and e.name in boolVals:
    let b = e.typ
    let symName = e.name
    e.kind = eConv
    e.castType = b
    e.castOperand = CirExpr(kind: eSym, name: symName, typ: tInt(32))
    return
  if e.kind == eDeref and e.derefOperand.kind == eSym and
      e.derefOperand.name in boolPtrs:
    let b = e.typ
    e.kind = eConv
    e.castType = b
    e.castOperand = CirExpr(kind: eDeref, derefOperand: e.derefOperand)
    return
  for ch in getExprChildren(e):
    patchBoolReadsExpr(ch, boolVals, boolPtrs)

proc isBoolStoreTarget(s: CirStmt, boolVals, boolPtrs: HashSet[string]): bool =
  ## True when the statement stores into an i32-retyped bool symbol.
  if s.kind != sAsgn:
    return false
  if s.lhs.kind == eSym and s.lhs.name in boolVals:
    return true
  if s.lhs.kind == eDeref and s.lhs.derefOperand.kind == eSym and
      s.lhs.derefOperand.name in boolPtrs:
    return true
  return false

proc patchBoolReadsStmts(ss: var seq[CirStmt], boolVals, boolPtrs: HashSet[string]) =
  ## Applies the read and store rewrites over a statement list.
  for s in mitems(ss):
    case s.kind
    of sVar, sConst:
      patchBoolReadsExpr(s.init, boolVals, boolPtrs)
    of sAsgn:
      if isBoolStoreTarget(s, boolVals, boolPtrs):
        # one conv wraps the whole right-hand side, storage is i32
        # reads inside it already rewrapped to bool keep their values
        s.rhs = convE(tInt(32), s.rhs)
      else:
        patchBoolReadsExpr(s.rhs, boolVals, boolPtrs)
    of sIte:
      patchBoolReadsExpr(s.cond, boolVals, boolPtrs)
      patchBoolReadsStmts(s.thenBody, boolVals, boolPtrs)
      patchBoolReadsStmts(s.elseBody, boolVals, boolPtrs)
    of sLoop:
      patchBoolReadsStmts(s.loopBody, boolVals, boolPtrs)
    of sScope:
      patchBoolReadsStmts(s.scopeBody, boolVals, boolPtrs)
    of sAssume:
      patchBoolReadsExpr(s.assumeCond, boolVals, boolPtrs)
    of sRet:
      patchBoolReadsExpr(s.retVal, boolVals, boolPtrs)
    of sDiscard:
      patchBoolReadsExpr(s.discardVal, boolVals, boolPtrs)
    of sCall:
      for a in mitems(s.callArgs):
        patchBoolReadsExpr(a, boolVals, boolPtrs)
    of sJmp, sLab, sCase, sAsm, sEmit:
      discard

proc spirvPatchBoolToI32(m: CirModule): string =
  ## Retypes bool kernel parameters and bool globals to i32 storage,
  ## rewriting the reads and stores of every proc body to match.
  ## Returns "" when clean.
  var globalVals, globalPtrs: HashSet[string]
  for g in m.globals:
    if g.declType.kind == ctBool:
      g.declType = tInt(32)
      globalVals.incl g.name
    elif g.declType.kind in {ctPtr, ctAptr} and g.declType.pElem != nil and
        g.declType.pElem.kind == ctBool:
      g.declType.pElem = tInt(32)
      globalPtrs.incl g.name
  for fn in m.procs:
    # a kernel's bool parameter names apply to that kernel only:
    # a device function with a same-named bool local keeps real bools
    var boolVals = globalVals
    var boolPtrs = globalPtrs
    if fn.isKernel:
      for p in mitems(fn.params):
        if p.typ.kind == ctBool:
          p.typ = tInt(32)
          boolVals.incl p.name
        elif p.typ.kind in {ctPtr, ctAptr} and p.typ.pElem != nil and
            p.typ.pElem.kind == ctBool:
          p.typ.pElem = tInt(32)
          boolPtrs.incl p.name
    patchBoolReadsStmts(fn.body, boolVals, boolPtrs)
  return ""
