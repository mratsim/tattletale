# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Validation stage over CIRv2: semantic checks plus the normal-form verifier,
## all registry passes run after the transforms, right before printing.

import std/sets
import ../ir/cirv2_lang
import ../ir/cirv2_verify
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerValidationPostPasses*(reg: var PassRegistry) =
  ## Registers the validation stage every backend bundle runs last:
  ## two semantic checks, then the normal-form verifier.
  ##
  ## Placement contract: the stage sits after every transform, before printing.
  ##
  ## - the constant-storage check reads hoisted declarations.
  ##   The const_mem hoist relocates program-scope constants into `m.globals`
  ## - the per-address-space callee instantiation settles which pointer
  ##   spaces reach the printer, so the cast check reads final spaces
  ## - the verifier assumes sound casts and read-only constant storage,
  ##   so it runs after the semantic checks
  reg.register("validateAddressSpaceCasts", pkValidation, phaseMain,
    "Rejects pointer casts whose target space differs from the operand's storage provenance",
    run = validateAddressSpaceCastsPass)
  reg.register("validateConstantWrites", pkValidation, phaseMain,
    "Rejects assignments into constant storage, direct or through a pointer",
    run = validateConstantWritesPass)
  reg.register("verify", pkValidation, phaseMain,
    "Verifies the module, refuses on the first invariant violation",
    run = verifyPass)

proc verifyPass(m: CirModule): string =
  verifyMessage(m)

# ═══════════════════════════════════════════════════════════════════════
# Address-space cast validation
# ═══════════════════════════════════════════════════════════════════════

type
  Provenance = tuple[space: CirAddrSpace, named: bool]
    ## Storage an expression's pointer provenance names:
    ##
    ## - `named` is false for a value with no pointer anywhere in its chain,
    ##   a plain integer or an array value
    ## - `caDevice` reads as the unprovenanced default then, matching the frontend's birth-space derivation

proc ptrProvenance(e: CirExpr): Provenance =
  ## Storage the pointer provenance of one expression names:
  ##
  ## - provenance survives element reads, field reads, value casts, and integer arithmetic over cast pointers
  ## - the frontend's birth-space derivation walks the same chain when it stamps a cast's target space
  if e.isNil:
    return (caDevice, false)
  case e.kind
  of eSym:
    let t = e.typ
    if t != nil and t.kind in {ctPtr, ctAptr}:
      return (t.space, true)
    result = (caDevice, false)
  of eDeref:
    result = ptrProvenance(e.derefOperand)
  of eAt:
    result = ptrProvenance(e.atBase)
  of ePat:
    result = ptrProvenance(e.patBase)
  of eDot:
    let t = getExprType(e)
    if t != nil and t.kind in {ctPtr, ctAptr}:
      return (t.space, true)
    result = ptrProvenance(e.dotBase)
  of eCast, eConv:
    let t = e.castType
    if t != nil and t.kind in {ctPtr, ctAptr}:
      return (t.space, true)
    result = ptrProvenance(e.castOperand)
  of eAddr:
    let t = e.addrType
    if t != nil and t.kind in {ctPtr, ctAptr}:
      return (t.space, true)
    result = (caDevice, false)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    result = ptrProvenance(e.left)
    if not result.named:
      result = ptrProvenance(e.right)
  else:
    result = (caDevice, false)

proc spaceCastRejection(e: CirExpr, fnName: string): string =
  ## Rejection message of one cast or conversion whose pointer target names
  ## storage the operand's provenance does not, "" when the cast passes:
  ##
  ## - a provenanced operand keeps its storage: the frontend derives the cast
  ##   target from the operand at birth, so a mismatch means the target space
  ##   was written independently of the operand
  ## - an unprovenanced integer names no storage, only the default device
  ##   space is claimable (the address route)
  if e.kind notin {eCast, eConv}:
    return ""
  let target = e.castType
  if target == nil or target.kind notin {ctPtr, ctAptr}:
    return ""
  let (space, named) = ptrProvenance(e.castOperand)
  if (named and space == target.space) or
      (not named and target.space == caDevice):
    return ""
  let construct = if e.kind == eCast: "cast" else: "conversion"
  let detail = if named:
    "the operand accesses " & $space & " storage, the target space is " &
    $target.space & ". Cross-space pointer casts are never implicit: " &
    "derive the pointer from storage of the target space"
  else:
    "the operand has no pointer provenance, the target space is " &
    $target.space & ". An integer names no storage: derive the pointer " &
    "from storage of the target space"
  result = "at " & fnName & ": cross-space " & construct & ": " & detail

proc checkSpaceCastsStmts(ss: seq[CirStmt], fnName: string): string =
  ## First cross-space cast or conversion in one statement list, walk order,
  ## nested bodies included.
  var err = ""
  let check = proc(e: CirExpr) =
    if err.len > 0:
      return
    err = spaceCastRejection(e, fnName)
  ss.walkStmts(nil, check)
  return err

proc validateAddressSpaceCastsPass(m: CirModule): string =
  ## Rejects cross-space pointer casts and conversions over every module proc.
  ## Returns "" when clean, else the first refusal.
  for fn in m.procs:
    let err = fn.body.checkSpaceCastsStmts(fn.name)
    if err.len > 0:
      return err
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Constant-storage write validation
# ═══════════════════════════════════════════════════════════════════════

proc collectConstLocals(ss: seq[CirStmt], seeded: HashSet[string]): HashSet[string] =
  ## Every `{.const_mem.}` local name of one statement list over `seeded`,
  ## scopes and case branches included.
  ##
  ## The hoisting passes relocate these locals on most targets.
  ## The CUDA bundle keeps them in-body, and a write to one refuses here.
  var names = seeded
  let collect = proc(s: CirStmt) =
    if s.kind == sVar and s.space == caConstant:
      names.incl s.name
  ss.walkStmts(collect, nil)
  return names

proc constStorageTarget(e: CirExpr, constNames: HashSet[string]): string =
  ## Name of the constant-storage target an lvalue designates.
  ## Returns "" when the lvalue lives in another memory class or names no storage.
  ##
  ## Chain resolution:
  ##
  ## - an indexed or field write uses the base's storage
  ## - a write through a pointer uses the pointer's address space
  if e.isNil:
    return ""
  case e.kind
  of eSym:
    if e.name in constNames:
      return e.name
  of eDeref:
    let t = getExprType(e.derefOperand)
    if t != nil and t.kind == ctPtr and t.space == caConstant:
      let root = exprRootSym(e.derefOperand)
      return if root != nil: root.name else: $e.derefOperand.kind
  of ePat:
    let t = getExprType(e.patBase)
    if t != nil and t.kind == ctAptr and t.space == caConstant:
      let root = exprRootSym(e.patBase)
      return if root != nil: root.name else: $e.patBase.kind
  of eAt:
    return constStorageTarget(e.atBase, constNames)
  of eDot:
    return constStorageTarget(e.dotBase, constNames)
  else:
    discard
  return ""

proc constWriteRejection(lhs: CirExpr, fnName: string, constNames: HashSet[string]): string =
  ## Rejection message of one assignment into constant storage.
  ## Returns "" when the target lives in another memory class.
  let target = constStorageTarget(lhs, constNames)
  if target.len == 0:
    return ""
  result = "at " & fnName & ": assignment into constant storage: the target " &
    target & " lives in read-only program-scope memory. Globals are " &
    "read-only; pass mutable state as a pointer parameter"

proc checkConstWritesStmts(ss: seq[CirStmt], fnName: string, constNames: HashSet[string]): string =
  ## First assignment into constant storage in one statement list, walk order,
  ## nested bodies included. Only assignment targets are checked, reads of constant storage stay legal.
  var err = ""
  let check = proc(s: CirStmt) =
    if err.len > 0:
      return
    if s.kind == sAsgn:
      err = constWriteRejection(s.lhs, fnName, constNames)
  ss.walkStmts(check, nil)
  return err

proc validateConstantWritesPass(m: CirModule): string =
  ## Rejects writes into constant storage across every module proc and every
  ## program-scope statement. Returns "" when clean, else the first refusal.
  var constGlobals: HashSet[string]
  for g in m.globals:
    if g.kind == sVar and g.space == caConstant:
      constGlobals.incl g.name
  for fn in m.procs:
    let constNames = fn.body.collectConstLocals(constGlobals)
    let err = fn.body.checkConstWritesStmts(fn.name, constNames)
    if err.len > 0:
      return err
  return m.globals.checkConstWritesStmts("global scope", constGlobals)
