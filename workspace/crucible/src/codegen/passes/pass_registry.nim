# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, sets, tables]
import ../ir/gpu_types

export gpu_types

# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

type
  PassKind* = enum
    pkValidation     ## Check-only: error/warn if invariant violated
    pkTransform      ## Mutates IR
    pkAnalysis       ## Computes metadata, no mutation

  PassPhase* = enum
    phaseEarly       ## Right after IR construction (normalization)
    phaseMain        ## Before lowering (optimizations, analysis, validation)

  GpuPass* = ref object of RootObj
    name*: string
    phase*: PassPhase
    kind*: PassKind
    description*: string
    dependsOn*: seq[string]
    run*: proc(ctx: var GpuContext): void {.nimcall.}

  PassRegistry* = ref object
    passes*: seq[GpuPass]
    donePasses*: HashSet[string]

# ─── Walk ──

proc walk*(body: var GpuAst; pre: proc(n: var GpuAst): void): void =
  ## Depth-first pre-order traversal. Calls `pre` at every node,
  ## then recurses into children via `mitems`.
  pre(body)
  for child in body.mitems:
    child.walk(pre)

# ─── Registration ──

proc register*(reg: var PassRegistry; name: string; kind: PassKind;
               phase: PassPhase; description: string;
               run: proc(ctx: var GpuContext): void {.nimcall.};
               dependsOn: seq[string] = @[]) =
  reg.passes.add GpuPass(name: name, kind: kind, phase: phase,
                         description: description,
                         dependsOn: dependsOn, run: run)

# ─── Execution ──

proc runPasses*(ctx: var GpuContext; reg: var PassRegistry) =
  ## Run all registered passes in order, verifying dependencies.
  ## TODO: pass the AST explicitly (coupled to ctx.allFnTab)
  for p in reg.passes:
    for dep in p.dependsOn:
      if dep notin reg.donePasses:
        error "\"" & p.name & "\" requires \"" & dep & "\" to run first"
    when defined(debugPasses):
      echo "[pass] ", p.name, " (", p.phase, ")"
    p.run(ctx)
    reg.donePasses.incl p.name

# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

proc assigned*(n: GpuAst; fnName: string): bool =
  case n.kind
  of gpuBlock:
    var assigned = false
    for ch in n:
      if ch.kind == gpuAssign and ch.aLeft.kind == gpuIdent and ch.aLeft.ident() == "result":
        assigned = true
      elif ch.kind == gpuIf:
        let thenAssigned = ch.ifThen.assigned(fnName)
        let elseAssigned = ch.ifElse.kind != gpuVoid and ch.ifElse.assigned(fnName)
        assigned = thenAssigned and elseAssigned
      elif ch.kind in {gpuFor, gpuWhile}:
        discard
      else:
        if not assigned:
          # Walk a copy to check if this statement actually reads `result`
          var readsResult = false
          var tmp = ch.clone()
          tmp.walk(proc(m: var GpuAst): void =
            if m.kind == gpuIdent and m.ident() == "result":
              readsResult = true
          )
          if readsResult:
            warning "result may be used before being initialized in " & fnName & ". " &
              "Assign to `result = ...` before reading it."
    result = assigned
  of gpuAssign:
    result = n.aLeft.kind == gpuIdent and n.aLeft.ident() == "result"
  else:
    result = false

proc insertResult*(ctx: var GpuContext; fn: GpuAst) =
  if fn.pRetType.kind == gtVoid: return

  proc lastIsReturn(n: GpuAst): bool =
    doAssert n.kind == gpuBlock
    if n.statements[^1].kind == gpuReturn: return true

  if not lastIsReturn(fn.pBody):
    let resId = GpuAst(kind: gpuIdent, iName: "result",
                       iSym: "result",
                       iTyp: fn.pRetType,
                       symbolKind: gsLocal)
    let res = GpuAst(kind: gpuVar, vName: resId,
                     vType: fn.pRetType,
                     vInit: GpuAst(kind: gpuVoid),
                     vRequiresMemcpy: false,
                     vMutable: true)
    fn.pBody.statements.insert(res, 0)

    if not lastIsReturn(fn.pBody):
      fn.pBody.statements.add GpuAst(kind: gpuReturn, rValue: resId)

    for i in countdown(fn.pBody.statements.high, 0):
      let stmt = fn.pBody.statements[i]
      if stmt.kind notin {gpuVar, gpuComment, gpuVoid, gpuReturn, gpuIf, gpuFor, gpuWhile}:
        if stmt.kind == gpuBlock and stmt.isExpr:
          if stmt.statements.len == 1:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt.statements[0])
          else:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        elif stmt.kind != gpuAssign:
          fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        break

# ═══════════════════════════════════════════════════════════════════
# Default pipeline
# ═══════════════════════════════════════════════════════════════════

proc newDefaultRegistry*(): PassRegistry =
  result = PassRegistry(passes: @[], donePasses: initHashSet[string]())

  result.register("ensureBlock", pkTransform, phaseEarly,
    "Wraps non-block bodies in gpuBlock",
    proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.pBody.kind != gpuBlock:
          fn.pBody = GpuAst(kind: gpuBlock, statements: @[fn.pBody])
        fn.pBody.walk(proc(n: var GpuAst): void =
          case n.kind
          of gpuIf:
            if n.ifThen.kind != gpuBlock:
              n.ifThen = GpuAst(kind: gpuBlock, statements: @[n.ifThen])
            if n.ifElse.kind != gpuVoid and n.ifElse.kind != gpuBlock:
              n.ifElse = GpuAst(kind: gpuBlock, statements: @[n.ifElse])
          of gpuFor:
            if n.fBody.kind != gpuBlock:
              n.fBody = GpuAst(kind: gpuBlock, statements: @[n.fBody])
          of gpuWhile:
            if n.wBody.kind != gpuBlock:
              n.wBody = GpuAst(kind: gpuBlock, statements: @[n.wBody])
          else: discard
        )
    )

  result.register("ensureNoCustomResult", pkValidation, phaseMain,
    "Rejects var result at any nesting depth",
    proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        fn.pBody.walk(proc(n: var GpuAst): void =
          if n.kind == gpuVar and n.vName.ident() == "result":
            error fn.pName.ident() & " has a custom `result` variable which shadows the implicit `result` (not allowed in GPU code)"
        )
    )

  result.register("ensureResultAssignedBeforeRead", pkValidation, phaseMain,
    "Warns if result is read before being written",
    proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        discard assigned(fn.pBody, fn.pName.ident())
    )

  result.register("maybeInsertResult", pkTransform, phaseMain,
    "Inserts var result and return result",
    dependsOn = @["ensureNoCustomResult", "ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        insertResult(ctx, fn)
    )

