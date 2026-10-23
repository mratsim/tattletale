# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, tables]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./pass_datatypes

{.experimental: "codeReordering".}

# ############################################################
#
# Validations
#
# ############################################################
#
# Invariant checks over the IR, a violated invariant is a compile-time error.

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerValidationPrePasses*(reg: var PassRegistry) =
  ## Register passes that check IR invariants.

  reg.register("ensureNoCustomResult", pkValidation, phaseMain,
    "Rejects var result at any nesting depth",
    proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        fn.pBody.walk(proc(n: var GpuAst): void =
          if n.kind == gpuVar and n.vName.ident() == "result":
            error fn.pName.ident() & " has a custom `result` variable which shadows the implicit `result` (not allowed in GPU code)"
        )
    )

  reg.register("ensureResultAssignedBeforeRead", pkValidation, phaseMain,
    "Warns if result is read before being written",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        warnUnassigned(fn.pBody, fn.pName.ident())
    )
    
  reg.register("validateScopeResolution", pkValidation, phaseMain,
    "Verifies every gpuIdent has a non-nil Symbol",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        let fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          fn.pBody.walk(proc(n: var GpuAst): void =
            if n.kind == gpuIdent and n.symbol.isNil:
              error("gpuIdent without Symbol in " & fn.pName.ident() & ": " & $n))
      for fnKey in ctx.genericInsts.keys:
        let fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          fn.pBody.walk(proc(n: var GpuAst): void =
            if n.kind == gpuIdent and n.symbol.isNil:
              error("gpuIdent without Symbol in " & fn.pName.ident() & ": " & $n))
    )

  reg.register("validateFnTable", pkValidation, phaseMain,
    "Verifies fnTable has valid entries (no nil idents, consistent kinds)",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for key, entry in ctx.fnTable:
        if entry.ident.isNil:
          error "fnTable entry '" & key & "' has nil ident"
        if entry.ident.kind != gpuIdent:
          error "fnTable entry '" & key & "' ident is not a gpuIdent, got " & $entry.ident.kind
        if card(entry.kind) == 0:
          error "fnTable entry '" & key & "' has empty kind set"
        if fkDefined in entry.kind or fkGenericInst in entry.kind:
          if entry.body.isNil:
            error "fnTable entry '" & key & "' marked as defined/generic but has nil body"
        if fkBuiltin in entry.kind:
          if entry.namePolicy != npUnassigned:
            error "fnTable entry '" & key & "' builtin should not have namePolicy assigned"
    )

func registerValidateAddressSpaceCasts*(reg: var PassRegistry) =
  ## Registers the cross-space pointer-cast validation, shared by every
  ## backend bundle through registerValidationPostPasses
  reg.register("validateAddressSpaceCasts", pkValidation, phaseMain,
    "Rejects pointer casts whose target space differs from the operand's accessed space",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.validateAddressSpaceCasts()
  )

func registerValidateConstantWrites*(reg: var PassRegistry) =
  reg.register("validateConstantWrites", pkValidation, phaseMain,
    "Rejects assignments into constant storage (read-only program-scope globals)",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.validateConstantWrites()
  )

func registerValidationPostPasses*(reg: var PassRegistry) =
  ## Register passes that check IR invariants AFTER all transforms.
  ## Runs just before codegen.
  # Cast validation first: a rejected cross-space cast is the earliest
  # defect to surface, the emitted-source checks below assume sound casts
  reg.registerValidateAddressSpaceCasts()
  reg.registerValidateConstantWrites()
  reg.register("ensureNoExprBlocks", pkValidation, phaseMain,
    "Rejects gpuBlock(isExpr: true) anywhere in the IR",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        let fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          fn.pBody.walk(proc(n: var GpuAst): void =
            if n.kind == gpuBlock and n.isExpr:
              error "Block expression survived to codegen in " & fn.pName.ident() & ": " & $n)
      for fnKey in ctx.genericInsts.keys:
        let fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          fn.pBody.walk(proc(n: var GpuAst): void =
            if n.kind == gpuBlock and n.isExpr:
              error "Block expression survived to codegen in " & fn.pName.ident() & ": " & $n)
    )

proc warnUnassigned(n: GpuAst, fnName: string): void =
  ## Emit a warning if `result` is read before being assigned in any branch.
  case n.kind
  of gpuBlock:
    var assigned = false
    for ch in n:
      if ch.kind == gpuAssign and ch.aLeft.kind == gpuIdent and ch.aLeft.ident() == "result":
        assigned = true
      elif ch.kind == gpuIf:
        ch.ifThen.warnUnassigned(fnName)
        for el in ch.ifElifs:
          el.body.warnUnassigned(fnName)
        if ch.ifElse.kind != gpuDiscard:
          ch.ifElse.warnUnassigned(fnName)
      elif ch.kind in {gpuFor, gpuWhile}:
        discard
      else:
        if not assigned:
          var readsResult = false
          var tmp = ch.clone()
          tmp.walk(proc(m: var GpuAst): void =
            if m.kind == gpuIdent and m.ident() == "result":
              readsResult = true
          )
          if readsResult:
            warning "result may be used before being initialized in " & fnName & ". " &
              "Assign to `result = ...` before reading it."
  of gpuAssign:
    discard  # nothing to warn about
  else:
    discard

# ═══════════════════════════════════════════════════════════════════════
# Address-space cast validation
# ═══════════════════════════════════════════════════════════════════════

proc validateAddressSpaceCasts(ctx: var GpuContext) =
  ## Rejects implicit cross-space pointer casts.
  ##
  ## Contract for every `gpuCast`/`gpuConv` whose target is a pointer, span, or array:
  ## - the target address space must equal the address space the operand accesses
  ## - the frontend derives pointer targets from the operand at birth,
  ##   a derived pointer stays in its source storage, so a mismatch means
  ##   a target space written independently of the operand: the C-style
  ##   `cast[ptr T](rawInt)` shape
  ## - pointer provenance is the explicit route across kinds:
  ##   `cast[uint64](p) + off` keeps `p`'s space and the resulting
  ##   `cast[ptr T](...)` re-derives it, so provenance-carrying casts pass
  ## - a cast from an unprovenanced plain integer reads `asRMEM` on both sides, it passes only when the target does too
  ##
  ## Nim has no construct that re-annotates a pointer to a different space, every cross-space cast in the IR is a defect.
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      ctx.validateSpaceCastsIn(fn.pBody, fn.pName.ident())
  for fnKey in ctx.genericInsts.keys:
    let fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      ctx.validateSpaceCastsIn(fn.pBody, fn.pName.ident())
  # top-level type and var statements are emitted verbatim, they get the same rejection
  for blk in ctx.globalBlocks:
    ctx.validateSpaceCastsIn(blk, "global scope")

proc validateSpaceCastsIn(ctx: var GpuContext, n: GpuAst, fnName: string) =
  ## Validates every cast/conv in depth-first pre-order, other kinds pass
  if n.isNil:
    return
  ctx.validateSpaceCast(n, fnName)
  for ch in n:
    ctx.validateSpaceCastsIn(ch, fnName)

proc spaceCastRejection(ctx: GpuContext, n: GpuAst, fnName: string): string =
  ## Rejection message of a cross-space cast/conv, empty when accepted
  ## `validateSpaceCast` errors on a non-empty message
  if n.kind notin {gpuCast, gpuConv}:
    return ""
  let target = if n.kind == gpuCast: n.cTo else: n.convTo
  if target.kind notin {gtPtr, gtSpan, gtArray}:
    return ""
  let operand = if n.kind == gpuCast: n.cExpr else: n.convExpr
  let operandSpace = ctx.resolveAddressSpace(operand)
  let targetSpace = getTypeAccessedSpace(target)
  if operandSpace == targetSpace:
    return ""
  let construct = if n.kind == gpuCast: "cast" else: "conversion"
  result = fnName & ": cross-space " & construct & ": the operand accesses " &
    $operandSpace & " storage, the target space is " & $targetSpace &
    " (" & $target.kind & "). Cross-space pointer casts are never implicit: " &
    "derive the pointer from storage of the target space"

proc validateSpaceCast(ctx: var GpuContext, n: GpuAst, fnName: string) =
  ## One cast/conv node against the operand's accessed space, other kinds pass
  let rejection = ctx.spaceCastRejection(n, fnName)
  if rejection.len > 0:
    error rejection

# ═══════════════════════════════════════════════════════════════════════
# Constant-storage write validation
# ═══════════════════════════════════════════════════════════════════════

proc constantWriteRejection(ctx: GpuContext, n: GpuAst, fnName: string): string =
  ## Rejection message of an assignment into constant storage, empty when the target
  ## lives in another space
  if n.kind != gpuAssign:
    return ""
  if ctx.resolveAddressSpace(n.aLeft) != asConstant:
    return ""
  let target = if n.aLeft.kind == gpuIdent: n.aLeft.ident() else: $n.aLeft.kind
  result = fnName & ": assignment into constant storage: the target " &
    target & " lives in read-only program-scope memory. Globals " &
    "are read-only; pass mutable state as a pointer parameter"

proc validateConstantWritesIn(ctx: var GpuContext, n: GpuAst, fnName: string) =
  ## Validates every assignment in depth-first pre-order, other kinds pass
  if n.isNil:
    return
  let rejection = ctx.constantWriteRejection(n, fnName)
  if rejection.len > 0:
    error rejection
  for ch in n:
    ctx.validateConstantWritesIn(ch, fnName)

proc validateConstantWrites(ctx: var GpuContext) =
  ## Rejects writes into constant storage, direct or through a pointer: the target's space decides
  ## in function bodies and in top-level statements alike
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      ctx.validateConstantWritesIn(fn.pBody, fn.pName.ident())
  for fnKey in ctx.genericInsts.keys:
    let fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      ctx.validateConstantWritesIn(fn.pBody, fn.pName.ident())
  for blk in ctx.globalBlocks.mitems:
    ctx.validateConstantWritesIn(blk, "global scope")
