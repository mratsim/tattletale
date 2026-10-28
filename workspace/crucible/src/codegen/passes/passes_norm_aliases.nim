# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Normalization: builtin alias canonicalization over CIRv2.
##
## A kernel author may write any column of the public contract table:
##
## - CUDA (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`)
## - GLSL (`gl_GlobalInvocationID`, `gl_LocalInvocationIndex`, ...)
## - WGSL (`global_id`, `workgroup_id`, `local_invocation_id`, ...)
##
## Flow: alias name --> canonical MSL name --> typed kind --> printer dispatch.
##
## After this pass:
##
## - the IR names the coordinate builtins by their canonical MSL names
## - every builtin symbol is kind-assigned from the `cirv2_lang` name tables: coordinate, synchronization, reduction, atomic
## - the printers dispatch on the typed kinds alone, no name recognition survives downstream
##
## The alias relation comes from the `{.builtin.}` alias declarations:
##
## - `builtins_aliases.nim` is read through `aliasCanonicalTable`
## - no parallel name table exists
##
## Ordering: the pass runs in the legalization stage, before the inline splice.
## The kernel-only-reads check there sees canonical names. Without that ordering
## a helper proc survives as a device function reading a builtin no backend defines.

import std/tables
import ../../builtins/builtins_aliases
import ../../builtins/builtins_catalog
import ../../builtins/builtins_warp_sync_reductions
import ../../builtins/builtins_atomics
import ../ir/cirv2_lang
import ./pass_datatypes

static:
  ## Cross-check: every canonical catalog builtin resolves in the IR name tables,
  ## a catalog rename without the table update fails at compile time.
  for n in CanonicalCoordBuiltins:
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

func registerAliasCanonicalization*(reg: var PassRegistry) =
  ## Registers the alias canonicalization pass.
  ##
  ## Ordering:
  ## - runs before the inline splice, the registration order encodes the dependency
  reg.register("builtinAliasCanonicalization", pkTransform, phaseEarly,
    "Renames every alias-named builtin symbol to its canonical MSL name",
    run = builtinAliasCanonicalization)

# ═══════════════════════════════════════════════════════════════════════
# The pass
# ═══════════════════════════════════════════════════════════════════════

proc assignBuiltinKinds(e: CirExpr) =
  ## Assigns the four typed builtin kinds of one builtin-marked symbol.
  ## Reads the canonical name tables, value reads and call callees alike.
  ##
  ## A builtin outside the four categories (a math intrinsic like `rsqrt`)
  ## keeps `none` in all four.
  ##
  ## The printers resolve a `none` builtin's native name in their own tables.
  e.coordBuiltin = coordBuiltinKind(e.name)
  e.synchroBuiltin = synchroBuiltinKind(e.name)
  e.reductionBuiltin = reductionBuiltinKind(e.name)
  e.atomicsBuiltin = atomicsBuiltinKind(e.name)

proc renameAliasExpr(e: CirExpr, aliases: Table[string, string]) =
  ## Renames one expression's alias-named builtin symbols in place.
  ## Kind-assigns every builtin-marked symbol through the canonical tables.
  ##
  ## Contract:
  ## - only builtin-marked symbols rename and kind-assign
  ## - a local or parameter sharing an alias name is an ordinary symbol,
  ##   untouched
  ## - expressions are refs, the rename applies to the shared node
  if e.isNil:
    return
  if e.kind == eSym:
    if e.builtin:
      if e.name in aliases:
        e.name = aliases[e.name]
      e.assignBuiltinKinds()
    return
  for ch in getExprChildren(e):
    renameAliasExpr(ch, aliases)

proc renameAliasStmts(ss: var seq[CirStmt], aliases: Table[string, string]) =
  ## Renames the alias-named builtin symbols in one statement list,
  ## nested bodies included.
  for s in mitems(ss):
    case s.kind
    of sVar, sConst:
      renameAliasExpr(s.init, aliases)
    of sAsgn:
      renameAliasExpr(s.lhs, aliases)
      renameAliasExpr(s.rhs, aliases)
    of sIte:
      renameAliasExpr(s.cond, aliases)
      renameAliasStmts(s.thenBody, aliases)
      renameAliasStmts(s.elseBody, aliases)
    of sLoop:
      renameAliasStmts(s.loopBody, aliases)
    of sScope:
      renameAliasStmts(s.scopeBody, aliases)
    of sAssume:
      renameAliasExpr(s.assumeCond, aliases)
    of sRet:
      renameAliasExpr(s.retVal, aliases)
    of sDiscard:
      renameAliasExpr(s.discardVal, aliases)
    of sCall:
      renameAliasExpr(s.callCallee, aliases)
      for a in mitems(s.callArgs):
        renameAliasExpr(a, aliases)
    else:
      discard

proc builtinAliasCanonicalization(m: CirModule): string =
  ## Renames every alias-named builtin symbol to its canonical MSL name,
  ## one pass over every proc body.
  ##
  ## Returns "" when clean, the first refusal otherwise.
  ##
  ## Registered by `registerAliasCanonicalization`.
  let aliases = aliasCanonicalTable()
  for fn in m.procs:
    fn.body.renameAliasStmts(aliases)
  result = ""
