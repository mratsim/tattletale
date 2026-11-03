# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}


import std/[sequtils, sets, strutils]
import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers

# ############################################################
#
#            Validation stage
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerValidationPostPasses*(reg: var PassRegistry) =
  ## One semantic check.
  ##
  ## Placement contract: the stage sits after every transform, before printing.
  ## The constant-storage check reads hoisted declarations: the const_mem hoist
  ## relocates program-scope constants into `m.globals`
  reg.register("validateConstantWrites", pkValidation,
    "Rejects assignments into constant storage, direct or through a pointer",
    run = validateConstantWritesPass)

# ═══════════════════════════════════════════════════════════════════════
#  Constant-storage write validation
# ═══════════════════════════════════════════════════════════════════════

type
  ConstStorage = object
    ## Name classes one constant-write check resolves an unqualified symbol against:
    ##
    ## - `constLocals`: the const-space locals of the checked statement list
    ## - `otherLocals`: every parameter and every other local declaration, they shadow the program-scope globals of the same name
    ## - `constGlobals`: the program-scope constants of the module
    constLocals*: HashSet[string]
    otherLocals*: HashSet[string]
    constGlobals*: HashSet[string]

proc collectConstStorage(fn: CirProc, constGlobals: HashSet[string]): ConstStorage =
  ## Classifies the declarations one proc's constant-write check resolves an unqualified symbol against:
  ## the parameters first, then the local declarations of the body.
  ##
  ## The hoisting passes relocate const-space locals on most targets.
  ## The CUDA bundle keeps them in-body, and the pass refuses a write to one here.
  var found = ConstStorage(constGlobals: constGlobals)
  for p in fn.params:
    found.otherLocals.incl p.name
  let collect = proc(s: CirStmt) =
    if s.kind == sVar:
      if s.addressSpace == caConstant:
        found.constLocals.incl s.name
      else:
        found.otherLocals.incl s.name
  fn.body.walkStmts(collect, nil)
  return found

proc resolveConstStorageTarget(e: CirExpr, storage: ConstStorage): string =
  ## Name of the constant-storage target an lvalue designates.
  ## Returns "" when the lvalue is in another memory class or has no storage.
  ##
  ## Resolution:
  ##
  ## Unqualified symbols:
  ##
  ## - a `const` declaration is constant storage whatever its space
  ## - a const-space local is constant storage
  ## - any other local declaration or parameter shadows the program-scope globals of the same name
  ## - an undeclared name resolves against the program-scope globals
  ##
  ## Chains:
  ##
  ## - a write through a pointer or array pointer reads the pointer's space, array or scalar pointee alike
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
    if t != nil and t.kind in {ctPtr, ctAptr} and t.addressSpace == caConstant:
      let root = exprRootSym(base)
      return if root != nil: root.name else: $base.kind
  of eAt:
    return resolveConstStorageTarget(e.atBase, storage)
  of eDot:
    return resolveConstStorageTarget(e.dotBase, storage)
  else:
    discard
  return ""

proc constWriteRejection(lhs: CirExpr, fnName: string, storage: ConstStorage): string =
  ## Rejection message of one assignment into constant storage.
  ## Returns "" when the target is in another memory class.
  let target = resolveConstStorageTarget(lhs, storage)
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
  ## Reachability: Nim sem refuses any assignment to a `const`, so no producer
  ## emits a write to a `const`-space target.
  ## The refusal has teeth over hand-built IR and any future producer.
  var constGlobals: HashSet[string]
  for g in m.globals:
    if g.kind == sVar and g.addressSpace == caConstant:
      constGlobals.incl g.name
  for fn in m.procs:
    let err = fn.body.checkConstWritesStmts(fn.name,
      fn.collectConstStorage(constGlobals))
    if err.len > 0:
      return err
  return m.globals.checkConstWritesStmts("global scope",
    ConstStorage(constGlobals: constGlobals))
