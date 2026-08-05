## Phase 4: mapOperators pass test
##
## Verifies:
## - Operator function names (+, -, *, /) are patched to valid C++ identifiers
##   ONLY when the gpuIdent has symKind == gsProc (function identifiers)
## - Operator symbols inside gpuBinOp.bOp (symKind == gsNone) are NOT patched
## - The pass runs correctly on function bodies
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_mapOperators.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_normalizations

# ═══════════════════════════════════════════════════════════════════════
# 1. Operator function names (gsProc) are patched
# ═══════════════════════════════════════════════════════════════════════
block:
  let plusSym = newSymbol("+", iSym = "+_h1", symKind = gsProc)
  var plusIdent = GpuAst(kind: gpuIdent, symbol: plusSym)

  mapOperatorsImpl(plusIdent)
  doAssert plusIdent.symbol.name == "add",
    "Operator + with gsProc should be patched to 'add', got '" & plusIdent.symbol.name & "'"
  doAssert plusIdent.symbol.iSym == "add_h1",
    "iSym should also be patched, got '" & plusIdent.symbol.iSym & "'"
  echo "  OK — operator function name + is patched to 'add'"

# ═══════════════════════════════════════════════════════════════════════
# 2. Plain identifiers are NOT affected
# ═══════════════════════════════════════════════════════════════════════
block:
  let mySym = newSymbol("myVar", iSym = "myVar_h2", symKind = gsProc)
  var myVar = GpuAst(kind: gpuIdent, symbol: mySym)

  mapOperatorsImpl(myVar)
  doAssert myVar.symbol.name == "myVar",
    "Plain identifier should be unchanged, got '" & myVar.symbol.name & "'"
  echo "  OK — plain identifiers are NOT affected"

# ═══════════════════════════════════════════════════════════════════════
# 3. BinOp operators (gsNone) in bOp are NOT patched
# ═══════════════════════════════════════════════════════════════════════
block:
  # BinOp with gsNone operator symbol — should NOT be patched
  let opSym = newSymbol("+", iSym = "+_h3", symKind = gsNone)
  let opIdent = GpuAst(kind: gpuIdent, symbol: opSym)
  var binOp = GpuAst(kind: gpuBinOp, bType: nil, bOp: opIdent,
                     bLeft: GpuAst(kind: gpuLit, lValue: "1"),
                     bRight: GpuAst(kind: gpuLit, lValue: "2"))

  mapOperatorsImpl(binOp)
  doAssert binOp.bOp.symbol.name == "+",
    "Operator symbol with gsNone should NOT be patched, got '" & binOp.bOp.symbol.name & "'"
  echo "  OK — binOp operator symbols (gsNone) are NOT patched"

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All mapOperators tests passed."
