## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, tables]
import ../ir/gpu_types
import ./pass_datatypes

proc warnUnassigned(n: GpuAst; fnName: string): void =
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

proc checkReservedKeywords*(ctx: var GpuContext; reserved: openArray[string]; backendName: string) =
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.pName.ident() in reserved:
      error "'" & fn.pName.ident() & "' is a reserved keyword in " & backendName &
            ". Rename the function."
  
proc registerValidationPrePasses*(reg: var PassRegistry) =
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
              error "gpuIdent without Symbol in " & fn.pName.ident() & ": " & $n)
      for fnKey in ctx.genericInsts.keys:
        let fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          fn.pBody.walk(proc(n: var GpuAst): void =
            if n.kind == gpuIdent and n.symbol.isNil:
              error "gpuIdent without Symbol in " & fn.pName.ident() & ": " & $n)
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

proc registerValidationPostPasses*(reg: var PassRegistry) =
  ## Register passes that check IR invariants AFTER all transforms.
  ## Runs just before codegen.
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
