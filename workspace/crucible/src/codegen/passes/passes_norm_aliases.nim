# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Normalization: builtin kind assignment over CIRv2.
##
## Flow: canonical MSL name --> typed kind --> printer dispatch.
##
## The IR already names the builtins by their canonical MSL names.
##
## Canonical names come from the public catalog (`src/builtins/builtins_catalog.nim`),
## whose per-backend alias templates expand during Nim semantic analysis,
## no alias name reaches the IR.
##
## After this pass:
##
## - every builtin symbol is kind-assigned from the `cirv2_lang` name tables,
##   one of coordinate, synchronization, reduction, atomic
## - the printers dispatch on the typed kinds alone, no name recognition survives downstream
##
## A builtin outside the four categories (a math intrinsic like `rsqrt`) keeps
## `none` in all four, its native name resolves in the printers' own tables.
##
## Ordering: the pass runs in the legalization stage, before the inline splice.
## The kernel-only-reads check there sees the typed kinds. A later ordering
## would let a helper proc survive as a device function reading a builtin name
## that no backend defines.

import std/macros

import ../../builtins/builtins_catalog
import ../../builtins/builtins_warp_sync_reductions
import ../../builtins/builtins_atomics
import ../ir/cirv2_lang
import ./pass_datatypes

macro builtinSymNames(syms: varargs[typed]): untyped =
  ## Args:
  ## - `syms`: the `{.builtin.}` catalog lets and procs, generic procs included
  ##   (an overload choice unwraps to its first candidate)
  ##
  ## Returns their declared names as a bracket of string literals, one per symbol.
  result = newNimNode(nnkBracket)
  for s in syms:
    var n = s
    if n.kind in {nnkOpenSymChoice, nnkClosedSymChoice}:
      n = n[0]
    result.add newLit(n.strVal)

static:
  ## Cross-check: every canonical catalog builtin resolves in the IR name tables,
  ## a catalog rename without the table update fails at compile time.
  for n in builtinSymNames(thread_position_in_grid, threadgroup_position_in_grid,
      thread_position_in_threadgroup, threads_per_threadgroup,
      threadgroups_per_grid, thread_index_in_threadgroup):
    doAssert coordBuiltinKind(n) != cbkNone
  for n in builtinSymNames(threadgroup_barrier, threadgroup_barrier_device):
    doAssert synchroBuiltinKind(n) != sbkNone
  for n in builtinSymNames(reduceAdd, reduceMax):
    doAssert reductionBuiltinKind(n) != rbkNone
  for n in builtinSymNames(atomicFetchAdd):
    doAssert atomicsBuiltinKind(n) != abkNone

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerBuiltinKindAssignment*(reg: var PassRegistry) =
  ## Registers the builtin kind assignment pass.
  ##
  ## Ordering:
  ## - runs before the inline splice, the registration order encodes the dependency
  reg.register("builtinKindAssignment", pkTransform, phaseEarly,
    "Kind-assigns every builtin symbol from the canonical name tables",
    run = builtinKindAssignment)

# ═══════════════════════════════════════════════════════════════════════
# The pass
# ═══════════════════════════════════════════════════════════════════════

proc assignBuiltinKinds(e: CirExpr) =
  e.coordBuiltin = coordBuiltinKind(e.name)
  e.synchroBuiltin = synchroBuiltinKind(e.name)
  e.reductionBuiltin = reductionBuiltinKind(e.name)
  e.atomicsBuiltin = atomicsBuiltinKind(e.name)

proc assignBuiltinKindsExpr(e: CirExpr) =
  ## Kind-assigns one expression's builtin-marked symbols in place.
  ##
  ## Contract:
  ## - only builtin-marked symbols kind-assign
  ## - a local or parameter sharing a builtin's name is an ordinary symbol,
  ##   untouched
  ## - expressions are refs, the assignment applies to the shared node
  if e.isNil:
    return
  if e.kind == eSym:
    if e.builtin:
      e.assignBuiltinKinds()
    return
  for ch in getExprChildren(e):
    assignBuiltinKindsExpr(ch)

proc assignBuiltinKindsStmts(ss: var seq[CirStmt]) =
  ## Kind-assigns the builtin-marked symbols in one statement list,
  ## nested bodies included.
  for s in mitems(ss):
    case s.kind
    of sVar, sConst:
      assignBuiltinKindsExpr(s.init)
    of sAsgn:
      assignBuiltinKindsExpr(s.lhs)
      assignBuiltinKindsExpr(s.rhs)
    of sIte:
      assignBuiltinKindsExpr(s.cond)
      assignBuiltinKindsStmts(s.thenBody)
      assignBuiltinKindsStmts(s.elseBody)
    of sLoop:
      assignBuiltinKindsStmts(s.loopBody)
    of sScope:
      assignBuiltinKindsStmts(s.scopeBody)
    of sAssume:
      assignBuiltinKindsExpr(s.assumeCond)
    of sRet:
      assignBuiltinKindsExpr(s.retVal)
    of sDiscard:
      assignBuiltinKindsExpr(s.discardVal)
    of sCall:
      assignBuiltinKindsExpr(s.callCallee)
      for a in mitems(s.callArgs):
        assignBuiltinKindsExpr(a)
    else:
      discard

proc builtinKindAssignment(m: CirModule): string =
  for fn in m.procs:
    fn.body.assignBuiltinKindsStmts()
