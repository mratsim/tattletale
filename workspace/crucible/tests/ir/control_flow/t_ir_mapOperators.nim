## mapOperators operator-name patching on the raw IR
##
## Test that mapOperators rewrites operator function idents to valid C++
## names and leaves binop operator symbols untouched
##
## - `+`, `-`, `*`, `/` idents with symKind gsProc become add, sub, mul, div
## - plain non-operator idents with symKind gsProc keep their name and iSym
## - gsNone operator symbols inside gpuBinOp.bOp stay unpatched
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_mapOperators.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_normalizations {.all.}
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

# ═══════════════════════════════════════════════════════════════════════
# 1. Operator function names (gsProc) are patched
# ═══════════════════════════════════════════════════════════════════════
static:
  let plusSym = newSymbol("+", iSym = "+_h1", symKind = gsProc)
  let plusIdent = GpuAst(kind: gpuIdent, symbol: plusSym)

  let patched = registerMapOperators.runPassOnNode(plusIdent)[0]
  doAssert patched.symbol.name == "add",
    "Operator + with gsProc should be patched to 'add', got '" & patched.symbol.name & "'"
  doAssert patched.symbol.iSym == "add_h1",
    "iSym should also be patched, got '" & patched.symbol.iSym & "'"
  echo "  OK — operator function name + is patched to 'add'"

# ═══════════════════════════════════════════════════════════════════════
# 2. Plain identifiers are NOT affected
# ═══════════════════════════════════════════════════════════════════════
static:
  let mySym = newSymbol("myVar", iSym = "myVar_h2", symKind = gsProc)
  let myVar = GpuAst(kind: gpuIdent, symbol: mySym)

  let untouched = registerMapOperators.runPassOnNode(myVar)[0]
  doAssert untouched.symbol.name == "myVar",
    "Plain identifier should be unchanged, got '" & untouched.symbol.name & "'"
  echo "  OK — plain identifiers are NOT affected"

# ═══════════════════════════════════════════════════════════════════════
# 3. BinOp operators (gsNone) in bOp are NOT patched
# ═══════════════════════════════════════════════════════════════════════
static:
  let opSym = newSymbol("+", iSym = "+_h3", symKind = gsNone)
  let opIdent = GpuAst(kind: gpuIdent, symbol: opSym)
  let binOp = GpuAst(kind: gpuBinOp, bType: nil, bOp: opIdent,
                     bLeft: GpuAst(kind: gpuLit, lValue: "1"),
                     bRight: GpuAst(kind: gpuLit, lValue: "2"))

  let untouched = registerMapOperators.runPassOnNode(binOp)[0]
  doAssert untouched.bOp.symbol.name == "+",
    "Operator symbol with gsNone should NOT be patched, got '" & untouched.bOp.symbol.name & "'"
  echo "  OK — binOp operator symbols (gsNone) are NOT patched"
