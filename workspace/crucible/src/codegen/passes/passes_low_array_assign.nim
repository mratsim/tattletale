# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
# Array assignment lowering (C-family targets)
#
# ############################################################
#
# Whole-array assignment decomposes into memcpy-call composition,
# registered by C-family bundles only, WGSL surfaces the node unlowered.

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

import std / [hashes, sequtils, sets, strutils, tables]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ../ir/ast_type_constructors
import ./pass_datatypes
import ./pass_registry
import ./pass_helpers

func registerDecomposeArrayAssign*(reg: var PassRegistry) =
  reg.register("decomposeArrayAssign", pkTransform, phaseMain,
    "Splits gpuVar/gpuAssign with RequiresMemcpy into decl + memcpy call",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          decomposeArrayAssignImpl(ctx, fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          decomposeArrayAssignImpl(ctx, fn.pBody)
  )


# ═══════════════════════════════════════════════════════════════════════
# Memcpy decomposition
# ═══════════════════════════════════════════════════════════════════════

proc renameSymsInTree(n: var GpuAst, oldName, newName: string) =
  if n == nil:
    return
  case n.kind
  of gpuIdent:
    if n.symbol.name == oldName:
      n.symbol.name = newName
      n.symbol.iSym = newName
  of gpuVar:
    # mitems does not yield vName
    if n.vName.symbol.name == oldName:
      n.vName.symbol.name = newName
      n.vName.symbol.iSym = newName
    renameSymsInTree(n.vInit, oldName, newName)
  else:
    for child in n.mitems:
      renameSymsInTree(child, oldName, newName)

proc dedupVarNames(stmts: var seq[GpuAst]) =
  ## Renames duplicate gpuVar declarations across sibling blocks,
  ## template expansions collide without C++ scope isolation
  var seenNames: Table[string, int]
  for i in 0 ..< stmts.len:
    if stmts[i].kind == gpuBlock:
      var localRenames: seq[(string, string)]
      for s in stmts[i].statements:
        if s.kind == gpuVar:
          let name = s.vName.symbol.name
          if name in seenNames:
            seenNames[name] += 1
            let newName = name & "_" & $seenNames[name]
            localRenames.add (name, newName)
          else:
            seenNames[name] = 0
      for (oldName, newName) in localRenames:
        renameSymsInTree(stmts[i], oldName, newName)
    elif stmts[i].kind == gpuVar:
      let name = stmts[i].vName.symbol.name
      if name in seenNames:
        seenNames[name] += 1
        let newName = name & "_" & $seenNames[name]
        renameSymsInTree(stmts[i], name, newName)
      else:
        seenNames[name] = 0

proc genMemcpyCall(lhs, rhs: GpuAst, sizeExpr: GpuAst): GpuAst =
  let memcpySym = newSymbol("__builtin_memcpy", iSym = "__builtin_memcpy", symKind = gsProc)
  let memcpyIdent = GpuAst(kind: gpuIdent, symbol: memcpySym)
  var call = GpuAst(kind: gpuCall)
  call.cName = memcpyIdent
  call.cArgs = @[lhs, rhs, sizeExpr]
  result = call

func genSizeofCall(ctx: var GpuContext, expr: GpuAst): GpuAst =
  let sizeSym = newSymbol("__builtin_sizeof", iSym = "__builtin_sizeof", symKind = gsProc)
  let sizeIdent = GpuAst(kind: gpuIdent, symbol: sizeSym)
  var call = GpuAst(kind: gpuCall)
  call.cName = sizeIdent
  call.cArgs = @[expr]
  result = call

proc decomposeArrayAssignImpl(ctx: var GpuContext, n: var GpuAst) =
  ## After this pass, backends only emit `=` for simple assignments
  case n.kind
  of gpuVar:
    if n.vRequiresMemcpy and n.vInit.kind != gpuDiscard:
      let addrOfVar = GpuAst(kind: gpuAddr, aOf: n.vName)
      let addrOfInit = GpuAst(kind: gpuAddr, aOf: n.vInit)
      let sizeCall = genSizeofCall(ctx, n.vName)
      let memcpyNode = genMemcpyCall(addrOfVar, addrOfInit, sizeCall)

      n.vRequiresMemcpy = false
      n.vInit = GpuAst(kind: gpuDiscard)
      n = GpuAst(kind: gpuBlock, blockLabel: "_memcpy", statements: @[n, memcpyNode])
    else:
      for ch in mitems(n):
        decomposeArrayAssignImpl(ctx, ch)

  of gpuAssign:
    if n.aRequiresMemcpy:
      let addrOfLeft = GpuAst(kind: gpuAddr, aOf: n.aLeft)
      let addrOfRight = GpuAst(kind: gpuAddr, aOf: n.aRight)
      let sizeCall = genSizeofCall(ctx, n.aLeft)
      n = genMemcpyCall(addrOfLeft, addrOfRight, sizeCall)
    else:
      for ch in mitems(n):
        decomposeArrayAssignImpl(ctx, ch)

  of gpuIf:
    decomposeArrayAssignImpl(ctx, n.ifCond)
    decomposeArrayAssignImpl(ctx, n.ifThen)
    for el in n.ifElifs.mitems:
      decomposeArrayAssignImpl(ctx, el.cond)
      decomposeArrayAssignImpl(ctx, el.body)
    if n.ifElse.kind != gpuDiscard:
      decomposeArrayAssignImpl(ctx, n.ifElse)

  of gpuFor:
    decomposeArrayAssignImpl(ctx, n.fBody)

  of gpuWhile:
    decomposeArrayAssignImpl(ctx, n.wBody)

  of gpuBlock:
    var newStmts: seq[GpuAst]
    for i in 0 ..< n.statements.len:
      var stmt = n.statements[i]
      decomposeArrayAssignImpl(ctx, stmt)
      if stmt.kind == gpuBlock and stmt.blockLabel == "_memcpy":
        # only anonymous _memcpy blocks inline, labeled blocks keep C++ scope isolation
        for s in stmt.statements:
          newStmts.add s
      else:
        newStmts.add stmt
    n.statements = newStmts
    # template expansions duplicate var names across blocks flattened by codegen
    dedupVarNames(n.statements)

  of gpuProc:
    decomposeArrayAssignImpl(ctx, n.pBody)

  else:
    for ch in mitems(n):
      decomposeArrayAssignImpl(ctx, ch)
