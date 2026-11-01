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
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSpirvLoweringPasses*(reg: var PassRegistry) =
  ## Registers the SPIR-V per-target lowering passes:
  ##
  ## - `spirvPatchBoolToI32`, before printing: bool kernel parameters and bool
  ##   globals retype to i32 storage, reads and stores wrap in conversions
  reg.register("spirvPatchBoolToI32", pkTransform,
    "Retypes bool kernel parameters and bool globals to i32 storage",
    run = spirvPatchBoolToI32
  )

proc patchBoolReadsExpr(e: CirExpr, boolVals, boolPtrs: HashSet[string]) =
  ## Wraps every value-position read of an i32-stored bool back into bool.
  ##
  ## Expressions are refs, so the rewrite applies to every sharer and checks
  ## the symbol's own type: a proc-typed callee position never fires
  ## and a bool-typed symbol is never a callee. The conv's operand is
  ## a fresh node. The rewrite returns before the walk re-enters it.
  if e.isNil:
    return
  if e.kind == eSym and e.typ != nil and e.typ.kind == ctBool and
      e.name in boolVals:
    let b = e.typ
    let symName = e.name
    e.kind = eConv
    e.castType = b
    e.castOperand = CirExpr(kind: eSym, name: symName, typ: tInt(32))
    return
  if e.kind == eDeref and e.typ != nil and e.typ.kind == ctBool and
      e.derefOperand.kind == eSym and
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

proc wrapBoolStores(s: CirStmt, boolVals, boolPtrs: HashSet[string]): bool =
  ## Wraps the whole right-hand side of an assignment into i32-stored bool
  ## storage in `conv(i32)`. Returns `true` when the statement is a store,
  ## the shared walk skips its operands: storage is i32 and the reads inside keep their 0/1 values.
  if s.kind == sAsgn and isBoolStoreTarget(s, boolVals, boolPtrs):
    s.rhs = convE(tInt(32), s.rhs)
    true
  else:
    false

proc collectBoolGlobals(m: CirModule, vals, ptrs: var HashSet[string]) =
  ## Retypes the module's bool storage to i32 and collects the names:
  ## bool globals as values, bool-pointee globals as pointers.
  for g in m.globals:
    if g.declType.kind == ctBool:
      g.declType = tInt(32)
      vals.incl g.name
    elif g.declType.kind in {ctPtr, ctAptr} and g.declType.pElem != nil and
        g.declType.pElem.kind == ctBool:
      g.declType.pElem = tInt(32)
      ptrs.incl g.name

proc collectKernelBoolParams(fn: CirProc, vals, ptrs: var HashSet[string]) =
  ## Retypes the kernel's bool parameters to i32 and collects the names,
  ## bool-pointee parameters as pointers. A kernel's bool parameter names
  ## apply to that kernel only: a device function with a same-named bool
  ## local keeps real bools.
  if not fn.isKernel:
    return
  for p in mitems(fn.params):
    if p.typ.kind == ctBool:
      p.typ = tInt(32)
      vals.incl p.name
    elif p.typ.kind in {ctPtr, ctAptr} and p.typ.pElem != nil and
        p.typ.pElem.kind == ctBool:
      p.typ.pElem = tInt(32)
      ptrs.incl p.name

proc spirvPatchBoolToI32(m: CirModule): string =
  ## Retypes bool kernel parameters and bool globals to i32 storage,
  ## rewriting the reads and stores of every proc body to match.
  ## Returns "" when clean.
  ##
  ## The shared walk covers every expression position, callee positions included.
  ## A callee names a proc, never a retyped bool symbol.
  m.patchBoolStorage(BoolPatchPolicy(
    collectGlobals: collectBoolGlobals,
    collectParams: collectKernelBoolParams,
    walks: proc(fn: CirProc): bool = true,
    rewriteRead: patchBoolReadsExpr,
    rewriteStore: wrapBoolStores))
