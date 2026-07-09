## Phase 5: rewriteIndexDeref pass test
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_rewriteIndexDeref.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. Rewrites gpuIndex(gpuDeref(ptrIdent)) -> gpuIndex(ptrIdent)
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let ptrTyp = GpuType(kind: gtPtr, to: int32)
  let ptrSym = newSymbol("p", iSym = "p_h1", typ = ptrTyp)
  let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
  let deref = GpuAst(kind: gpuDeref, dOf: ptrIdent)
  let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: int32)
  var idx = GpuAst(kind: gpuIndex, iArr: deref, iIndex: lit0)
  var ctx = GpuContext()

  rewriteIndexDerefImpl(ctx, idx)
  doAssert idx.kind == gpuIndex, "Should remain gpuIndex"
  doAssert idx.iArr.kind != gpuDeref, "gpuDeref should be removed for ptr (not ptr-to-array)"
  doAssert idx.iArr.kind == gpuIdent, "iArr should now be the raw ident"
  echo "  OK — ptr without array: gpuDeref removed"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Preserves gpuIndex(gpuDeref(ptrToArrayIdent)) — ptr to static array
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let arrTyp = GpuType(kind: gtArray, aTyp: int32, aLen: 4)
  let ptrToArr = GpuType(kind: gtPtr, to: arrTyp)
  let arrSym = newSymbol("arr", iSym = "arr_h2", typ = ptrToArr)
  let arrIdent = GpuAst(kind: gpuIdent, symbol: arrSym)
  let deref = GpuAst(kind: gpuDeref, dOf: arrIdent)
  let lit1 = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
  var idx = GpuAst(kind: gpuIndex, iArr: deref, iIndex: lit1)
  var ctx = GpuContext()

  rewriteIndexDerefImpl(ctx, idx)
  doAssert idx.kind == gpuIndex, "Should remain gpuIndex"
  doAssert idx.iArr.kind == gpuDeref, "gpuDeref should be preserved for ptr-to-array"
  echo "  OK — ptr to static array: gpuDeref preserved"

# ═══════════════════════════════════════════════════════════════════════════
# 3. Does not affect simple gpuIndex (no gpuDeref)
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let arrType = GpuType(kind: gtArray, aTyp: int32, aLen: 10)
  let arrSym = newSymbol("a", iSym = "a_h3", typ = arrType)
  let arrIdent = GpuAst(kind: gpuIdent, symbol: arrSym)
  let lit2 = GpuAst(kind: gpuLit, lValue: "2", lType: int32)
  var idx = GpuAst(kind: gpuIndex, iArr: arrIdent, iIndex: lit2)
  var ctx = GpuContext()

  rewriteIndexDerefImpl(ctx, idx)
  doAssert idx.kind == gpuIndex, "Should remain gpuIndex"
  doAssert idx.iArr.kind == gpuIdent, "Simple ident index should be preserved"
  echo "  OK — simple gpuIndex without deref is unchanged"

echo ""
echo "  All rewriteIndexDeref tests passed."
