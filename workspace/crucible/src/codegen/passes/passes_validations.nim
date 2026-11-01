# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Validation stage over CIRv2: semantic checks over the transformed module, right before printing.

import std/sets
import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerValidationPostPasses*(reg: var PassRegistry) =
  ## Registers the validation stage that every backend bundle runs last:
  ## the two semantic checks.
  ##
  ## Placement contract: the stage sits after every transform, before printing.
  ##
  ## - the constant-storage check reads hoisted declarations.
  ##   The const_mem hoist relocates program-scope constants into `m.globals`
  ## - the per-address-space callee instantiation settles which pointer
  ##   spaces reach the printer, so the cast check reads final spaces
  reg.register("validateAddressSpaceCasts", pkValidation,
    "Rejects pointer casts whose target space differs from the operand's storage provenance",
    run = validateAddressSpaceCastsPass)
  reg.register("validateConstantWrites", pkValidation,
    "Rejects assignments into constant storage, direct or through a pointer",
    run = validateConstantWritesPass)
# ═══════════════════════════════════════════════════════════════════════
# Address-space cast validation
# ═══════════════════════════════════════════════════════════════════════

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
  let space = resolveAddressSpace(e.castOperand)
  if space == target.space:
    return ""
  let construct = if e.kind == eCast: "cast" else: "conversion"
  let detail = if space == caDevice:
    "the operand has no pointer provenance, the target space is " &
    $target.space & ". An integer names no storage: derive the pointer " &
    "from storage of the target space"
  else:
    "the operand accesses " & $space & " storage, the target space is " &
    $target.space & ". Cross-space pointer casts are never implicit: " &
    "derive the pointer from storage of the target space"
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

type
  ConstStorage = object
    ## Name classes one constant-write check resolves a bare symbol against:
    ##
    ## - `constLocals`: the const-space locals of the checked statement list
    ## - `otherLocals`: every parameter and every other local declaration, they shadow the program-scope globals of the same name
    ## - `constGlobals`: the program-scope constants of the module
    constLocals*: HashSet[string]
    otherLocals*: HashSet[string]
    constGlobals*: HashSet[string]

proc collectConstStorage(fn: CirProc, constGlobals: HashSet[string]): ConstStorage =
  ## Classifies the declarations one proc's constant-write check resolves a bare symbol against:
  ## the parameters first, then the local declarations of the body, nested scopes and case branches included.
  ##
  ## The hoisting passes relocate const-space locals on most targets.
  ## The CUDA bundle keeps them in-body, and a write to one refuses here.
  var found = ConstStorage(constGlobals: constGlobals)
  for p in fn.params:
    found.otherLocals.incl p.name
  let collect = proc(s: CirStmt) =
    if s.kind == sConst:
      found.constLocals.incl s.name
    elif s.kind == sVar:
      if s.space == caConstant:
        found.constLocals.incl s.name
      else:
        found.otherLocals.incl s.name
  fn.body.walkStmts(collect, nil)
  return found

proc constStorageTarget(e: CirExpr, storage: ConstStorage): string =
  ## Name of the constant-storage target an lvalue designates.
  ## Returns "" when the lvalue lives in another memory class or names no storage.
  ##
  ## Resolution:
  ##
  ## Bare symbols:
  ##
  ## - a `const` declaration is constant storage whatever its space
  ## - a const-space local is constant storage
  ## - any other local declaration or parameter shadows the program-scope globals of the same name
  ## - an undeclared name resolves against the program-scope globals
  ##
  ## Chains:
  ##
  ## - a write through a pointer or array pointer reads the pointer's space, the pointee may be an array or a scalar alike
  ## - an indexed or field write uses the base's storage
  if e.isNil:
    return ""
  case e.kind
  of eSym:
    if e.name in storage.constLocals:
      return e.name
    if e.name in storage.otherLocals:
      return ""
    if e.name in storage.constGlobals:
      return e.name
  of eDeref, ePat:
    let base = if e.kind == eDeref: e.derefOperand else: e.patBase
    let t = getExprType(base)
    if t != nil and t.kind in {ctPtr, ctAptr} and t.space == caConstant:
      let root = exprRootSym(base)
      return if root != nil: root.name else: $base.kind
  of eAt:
    return constStorageTarget(e.atBase, storage)
  of eDot:
    return constStorageTarget(e.dotBase, storage)
  else:
    discard
  return ""

proc constWriteRejection(lhs: CirExpr, fnName: string, storage: ConstStorage): string =
  ## Rejection message of one assignment into constant storage.
  ## Returns "" when the target lives in another memory class.
  let target = constStorageTarget(lhs, storage)
  if target.len == 0:
    return ""
  result = "at " & fnName & ": assignment into constant storage: the target " &
    target & " lives in read-only program-scope memory. Globals are " &
    "read-only; pass mutable state as a pointer parameter"

proc checkConstWritesStmts(ss: seq[CirStmt], fnName: string, storage: ConstStorage): string =
  ## First assignment into constant storage in one statement list, walk order,
  ## nested bodies included. Only assignment targets are checked, reads of constant storage stay legal.
  var err = ""
  let check = proc(s: CirStmt) =
    if err.len > 0:
      return
    if s.kind == sAsgn:
      err = constWriteRejection(s.lhs, fnName, storage)
  ss.walkStmts(check, nil)
  return err

proc validateConstantWritesPass(m: CirModule): string =
  ## Rejects writes into constant storage across every module proc and every
  ## program-scope statement. Returns "" when clean, else the first refusal.
  ##
  ## Reachability: the frontend emits an sConst declaration for every Nim `const` section.
  ## Nim sem refuses any assignment to a `const`, so the frontend never emits a write to an sConst target.
  ## The refusal has teeth over hand-built IR and any future producer.
  var constGlobals: HashSet[string]
  for g in m.globals:
    if g.kind == sConst or (g.kind == sVar and g.space == caConstant):
      constGlobals.incl g.name
  for fn in m.procs:
    let err = fn.body.checkConstWritesStmts(fn.name,
      fn.collectConstStorage(constGlobals))
    if err.len > 0:
      return err
  return m.globals.checkConstWritesStmts("global scope",
    ConstStorage(constGlobals: constGlobals))
