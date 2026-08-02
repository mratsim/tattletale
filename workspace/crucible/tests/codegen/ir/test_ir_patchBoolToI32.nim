## Phase 6: patchBoolToI32 pass tests
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_patchBoolToI32.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. Bool global ident gets gpuConv(expr, gtBool)
# ═══════════════════════════════════════════════════════════════════════════
block:
  let boolTyp = GpuType(kind: gtBool)
  let sym = newSymbol("flag", iSym = "flag_h1", typ = boolTyp)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  var ctx = GpuContext()
  ctx.globals["flag_h1"] = GpuParam(ident: ident, typ: boolTyp, addressSpace: asStorage)
  var n = ident
  ctx.patchBoolToI32Impl(n)
  doAssert n.kind == gpuConv, "Bool global should get gpuConv, got: " & $n.kind
  doAssert n.convTo.kind == gtBool, "Conv target should be gtBool"
  doAssert n.convExpr.kind == gpuIdent, "Conv expr should be the ident"
  echo "  OK — bool global ident gets gpuConv"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Non-global bool ident is left unchanged
# ═══════════════════════════════════════════════════════════════════════════
block:
  let boolTyp = GpuType(kind: gtBool)
  let sym = newSymbol("localFlag", iSym = "local_flag_h2", typ = boolTyp)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  var ctx = GpuContext()
  # Don't add to globals
  var n = ident
  ctx.patchBoolToI32Impl(n)
  doAssert n.kind == gpuIdent, "Non-global bool should stay as gpuIdent, got: " & $n.kind
  echo "  OK — non-global bool ident left unchanged"

# ═══════════════════════════════════════════════════════════════════════════
# 3. Int32 global ident is left unchanged
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let sym = newSymbol("val", iSym = "val_h3", typ = int32)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  var ctx = GpuContext()
  ctx.globals["val_h3"] = GpuParam(ident: ident, typ: int32, addressSpace: asStorage)
  var n = ident
  ctx.patchBoolToI32Impl(n)
  doAssert n.kind == gpuIdent, "Int32 global should stay as gpuIdent, got: " & $n.kind
  echo "  OK — int32 global ident left unchanged"

echo ""
echo "  All patchBoolToI32 tests passed."
