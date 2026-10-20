## Symbol ref identity test
##
## Test that changes propagate through shared Symbol refs
## and that clone() preserves Symbol identity
##
## - clone() must preserve Symbol refs, a wrong clone corrupts rewrites
## - changes via a cloned ident reach the original through the shared ref
## - a proc clone preserves param and body Symbol sharing
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/symbol_resolution/t_ir_symbols.nim

import workspace/crucible/src/codegen/ir/gpu_ast_datatypes

block:
  # mutating symbol.name updates all idents
  let sym = newSymbol("oldName", symKind = gsLocal)
  let ident1 = GpuAst(kind: gpuIdent, symbol: sym)
  let ident2 = GpuAst(kind: gpuIdent, symbol: sym)

  sym.name = "newName"
  doAssert ident1.ident() == "newName", "ident1 should reflect name change"
  doAssert ident2.ident() == "newName", "ident2 should reflect name change"

block:
  # clone() preserves Symbol ref identity
  let sym = newSymbol("sharedVar", symKind = gsLocal)
  let ident1 = GpuAst(kind: gpuIdent, symbol: sym)
  let ident2 = GpuAst(kind: gpuIdent, symbol: sym)
  let blockNode = GpuAst(kind: gpuBlock, statements: @[ident1, ident2])

  let cloned = blockNode.clone()
  doAssert cloned.statements.len == 2, "clone should have 2 statements"

  let clonedIdent1 = cloned.statements[0]
  let clonedIdent2 = cloned.statements[1]
  doAssert clonedIdent1.symbol == clonedIdent2.symbol,
    "Cloned idents should share the same Symbol ref"
  doAssert clonedIdent1.symbol == ident1.symbol,
    "Cloned ident should share Symbol ref with original"

  # a change through clone's symbol reaches the original
  clonedIdent1.symbol.name = "mutatedViaClone"
  doAssert ident1.ident() == "mutatedViaClone",
    "Mutation via clone should propagate to original"

block:
  # a proc clone preserves all Symbol refs, the body ident shares the param Symbol
  let fnSym = newSymbol("myFunc", iSym = "func_hash", symKind = gsProc)
  let fnName = GpuAst(kind: gpuIdent, symbol: fnSym)
  let paramSym = newSymbol("a", iSym = "a_hash", typ = GpuType(kind: gtInt32), symKind = gsDeviceKernelParam)
  let paramIdent = GpuAst(kind: gpuIdent, symbol: paramSym)
  let bodyIdent = GpuAst(kind: gpuIdent, symbol: paramSym)  # same Symbol as param

  let procNode = GpuAst(kind: gpuProc,
    pName: fnName,
    pRetType: GpuType(kind: gtInt32),
    pParams: @[GpuParam(ident: paramIdent, typ: GpuType(kind: gtInt32))],
    pBody: GpuAst(kind: gpuBlock, statements: @[bodyIdent]))

  let clonedProc = procNode.clone()

  doAssert clonedProc.pName.symbol == fnSym, "Proc name Symbol preserved"
  doAssert clonedProc.pParams[0].ident.symbol == paramSym, "Param Symbol preserved"
  doAssert clonedProc.pBody.kind == gpuBlock
  doAssert clonedProc.pBody.statements[0].symbol == paramSym,
    "Body ident should share Symbol with param"

echo "t_ir_symbols: ref identity and clone identity verified"
