## Phase 1a: Symbol ref identity test
##
## Verifies:
## - Two references to the same variable share the same Symbol ref
## - Changing symbol.name updates both idents' ident() return value
## - clone() preserves Symbol identity (cloned ident has same Symbol ref)
## - ident() returns symbol.name
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_symbols.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types

# ═══════════════════════════════════════════════════════════════════════
# 1. Symbol creation and field access
# ═══════════════════════════════════════════════════════════════════════
block:
  let s = newSymbol("x", iSym = "x_abc123", typ = GpuType(kind: gtInt32), symKind = gsLocal)
  doAssert s.name == "x", "Symbol.name should be 'x'"
  doAssert s.iSym == "x_abc123", "Symbol.iSym should be 'x_abc123'"
  doAssert s.typ.kind == gtInt32, "Symbol.typ.kind should be gtInt32"
  doAssert s.symKind == gsLocal, "Symbol.symKind should be gsLocal"
  doAssert s.module == "", "Symbol.module should be empty string"
  echo "  OK — Symbol creation and field access"

# ═══════════════════════════════════════════════════════════════════════
# 2. Two idents sharing the same Symbol ref
# ═══════════════════════════════════════════════════════════════════════
block:
  let sym = newSymbol("myVar", iSym = "myVar_hash", symKind = gsLocal)
  let ident1 = GpuAst(kind: gpuIdent, symbol: sym)
  let ident2 = GpuAst(kind: gpuIdent, symbol: sym)
  
  # Both idents should have the same Symbol ref
  doAssert ident1.symbol == ident2.symbol, "Two idents should share the same Symbol ref"
  doAssert ident1.ident() == "myVar", "ident() should return symbol.name"
  doAssert ident2.ident() == "myVar", "ident() should return symbol.name"
  echo "  OK — Two idents sharing same Symbol ref"

# ═══════════════════════════════════════════════════════════════════════
# 3. Mutating symbol.name updates all idents
# ═══════════════════════════════════════════════════════════════════════
block:
  let sym = newSymbol("oldName", symKind = gsLocal)
  let ident1 = GpuAst(kind: gpuIdent, symbol: sym)
  let ident2 = GpuAst(kind: gpuIdent, symbol: sym)
  
  sym.name = "newName"
  doAssert ident1.ident() == "newName", "ident1 should reflect name change"
  doAssert ident2.ident() == "newName", "ident2 should reflect name change"
  echo "  OK — Mutating symbol.name updates all idents"

# ═══════════════════════════════════════════════════════════════════════
# 4. clone() preserves Symbol ref identity
# ═══════════════════════════════════════════════════════════════════════
block:
  let sym = newSymbol("sharedVar", symKind = gsLocal)
  let ident1 = GpuAst(kind: gpuIdent, symbol: sym)
  let ident2 = GpuAst(kind: gpuIdent, symbol: sym)
  let blockNode = GpuAst(kind: gpuBlock, statements: @[ident1, ident2])
  
  let cloned = blockNode.clone()
  doAssert cloned.statements.len == 2, "clone should have 2 statements"
  
  # Both cloned idents should share the SAME Symbol ref as each other
  let clonedIdent1 = cloned.statements[0]
  let clonedIdent2 = cloned.statements[1]
  doAssert clonedIdent1.symbol == clonedIdent2.symbol,
    "Cloned idents should share the same Symbol ref"
  
  # The cloned idents should also share the SAME Symbol ref as the originals
  doAssert clonedIdent1.symbol == ident1.symbol,
    "Cloned ident should share Symbol ref with original"
  
  # Mutation through clone's symbol should affect original too
  clonedIdent1.symbol.name = "mutatedViaClone"
  doAssert ident1.ident() == "mutatedViaClone",
    "Mutation via clone should propagate to original"
  echo "  OK — clone() preserves Symbol ref identity"

# ═══════════════════════════════════════════════════════════════════════
# 5. Clone of gpuBlock with multiple children preserves all refs
# ═══════════════════════════════════════════════════════════════════════
block:
  # Create a more complex AST: proc with ident references
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
  
  # The body ident should share Symbol with the param ident (both reference same parameter)
  doAssert clonedProc.pBody.kind == gpuBlock
  doAssert clonedProc.pBody.statements[0].symbol == paramSym,
    "Body ident should share Symbol with param"
  echo "  OK — Complex clone preserves all Symbol refs"

# ═══════════════════════════════════════════════════════════════════════
# 6. newGpuIdent creates Symbol correctly
# ═══════════════════════════════════════════════════════════════════════
block:
  let ident = newGpuIdent("testVar", gsLocal)
  doAssert ident.kind == gpuIdent, "newGpuIdent should create gpuIdent"
  doAssert ident.symbol != nil, "newGpuIdent should create non-nil Symbol"
  doAssert ident.symbol.name == "testVar", "newGpuIdent should set name"
  doAssert ident.symbol.iSym == "testVar", "newGpuIdent iSym should equal name"
  doAssert ident.symbol.symKind == gsLocal, "newGpuIdent should set symKind"
  doAssert ident.ident() == "testVar", "ident() should return name"
  echo "  OK — newGpuIdent creates Symbol correctly"

# ═══════════════════════════════════════════════════════════════════════
# 7. shortHash is accessible from gpu_types
# ═══════════════════════════════════════════════════════════════════════
block:
  let h = shortHash(42)
  doAssert h.len == 7, "shortHash should produce 7 characters"
  doAssert h == shortHash(42), "shortHash should be deterministic"
  echo "  OK — shortHash from gpu_types"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All symbol ref identity tests passed."
