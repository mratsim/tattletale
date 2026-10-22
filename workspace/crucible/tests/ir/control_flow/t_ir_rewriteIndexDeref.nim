## rewriteIndexDeref index-base deref fold on the raw IR
##
## Test that rewriteIndexDeref strips a gpuDeref from an index base only
## when the pointer does not point at a static array
##
## - gpuIndex(gpuDeref(p), i) with p: ptr int32 folds to gpuIndex(p, i)
## - p: ptr array[4, int32] keeps the deref so p[i] reads the array
## - a gpuIndex with no gpuDeref base passes through unchanged
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_rewriteIndexDeref.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_legalizations {.all.}
import workspace/crucible/tests/ir/ir_testutils

# ═══════════════════════════════════════════════════════════════════════
# 1. Rewrites gpuIndex(gpuDeref(ptrIdent)) -> gpuIndex(ptrIdent)
# ═══════════════════════════════════════════════════════════════════════
static:
  let int32 = GpuType(kind: gtInt32)
  let ptrTyp = GpuType(kind: gtPtr, to: int32)
  let ptrSym = newSymbol("p", iSym = "p_h1", typ = ptrTyp)
  let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
  let deref = GpuAst(kind: gpuDeref, dOf: ptrIdent)
  let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: int32)
  var idx = GpuAst(kind: gpuIndex, iArr: deref, iIndex: lit0)
  idx = registerRewriteIndexDeref.runPassOnNode(idx)[0]
  doAssert idx.kind == gpuIndex, "Should remain gpuIndex"
  doAssert idx.iArr.kind != gpuDeref, "gpuDeref should be removed for ptr (not ptr-to-array)"
  doAssert idx.iArr.kind == gpuIdent, "iArr should now be the raw ident"
  echo "  OK — ptr without array: gpuDeref removed"

# ═══════════════════════════════════════════════════════════════════════
# 2. Preserves gpuIndex(gpuDeref(ptrToArrayIdent)) — ptr to static array
# ═══════════════════════════════════════════════════════════════════════
static:
  let int32 = GpuType(kind: gtInt32)
  let arrTyp = GpuType(kind: gtArray, aTyp: int32, aLen: 4)
  let ptrToArr = GpuType(kind: gtPtr, to: arrTyp)
  let arrSym = newSymbol("arr", iSym = "arr_h2", typ = ptrToArr)
  let arrIdent = GpuAst(kind: gpuIdent, symbol: arrSym)
  let deref = GpuAst(kind: gpuDeref, dOf: arrIdent)
  let lit1 = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
  var idx = GpuAst(kind: gpuIndex, iArr: deref, iIndex: lit1)
  idx = registerRewriteIndexDeref.runPassOnNode(idx)[0]
  doAssert idx.kind == gpuIndex, "Should remain gpuIndex"
  doAssert idx.iArr.kind == gpuDeref, "gpuDeref should be preserved for ptr-to-array"
  echo "  OK — ptr to static array: gpuDeref preserved"

# ═══════════════════════════════════════════════════════════════════════
# 3. Does not affect simple gpuIndex (no gpuDeref)
# ═══════════════════════════════════════════════════════════════════════
static:
  let int32 = GpuType(kind: gtInt32)
  let arrType = GpuType(kind: gtArray, aTyp: int32, aLen: 10)
  let arrSym = newSymbol("a", iSym = "a_h3", typ = arrType)
  let arrIdent = GpuAst(kind: gpuIdent, symbol: arrSym)
  let lit2 = GpuAst(kind: gpuLit, lValue: "2", lType: int32)
  var idx = GpuAst(kind: gpuIndex, iArr: arrIdent, iIndex: lit2)
  idx = registerRewriteIndexDeref.runPassOnNode(idx)[0]
  doAssert idx.kind == gpuIndex, "Should remain gpuIndex"
  doAssert idx.iArr.kind == gpuIdent, "Simple ident index should be preserved"
  echo "  OK — simple gpuIndex without deref is unchanged"

# ═══════════════════════════════════════════════════════════════════════
# 4. Rewrites nested gpuIndex(gpuDeref(...)) under a rewritten base
# ═══════════════════════════════════════════════════════════════════════
static:
  # `(*a)[(*b)[i]]`: after the outer rewrite the exposed base itself
  # indexes a deref, so the pass must keep scanning the rebuilt node
  let int32 = GpuType(kind: gtInt32)
  let ptrTyp = GpuType(kind: gtPtr, to: int32)
  let symA = newSymbol("a", iSym = "a_h4", typ = ptrTyp)
  let symB = newSymbol("b", iSym = "b_h5", typ = ptrTyp)
  let litI = GpuAst(kind: gpuLit, lValue: "3", lType: int32)
  let inner = GpuAst(kind: gpuIndex,
                     iArr: GpuAst(kind: gpuDeref, dOf: GpuAst(kind: gpuIdent, symbol: symB)),
                     iIndex: litI)
  let outer = GpuAst(kind: gpuIndex,
                     iArr: GpuAst(kind: gpuDeref, dOf: GpuAst(kind: gpuIdent, symbol: symA)),
                     iIndex: inner)
  let res = registerRewriteIndexDeref.runPassOnNode(outer)[0]
  doAssert res.iArr.kind == gpuIdent, "outer base deref stripped"
  doAssert res.iIndex.kind == gpuIndex, "nested index preserved"
  doAssert res.iIndex.iArr.kind == gpuIdent, "nested base deref stripped"
  echo "  OK — nested deref under rewritten base stripped"

# ═══════════════════════════════════════════════════════════════════════
# 5. Rewrites nested gpuIndex(gpuDeref(...)) under a non-deref base
# ═══════════════════════════════════════════════════════════════════════
static:
  # `(*a)[x][y]`: the outer base is a gpuIndex, the pass still scans
  # gpuIndex children and strips the inner base deref
  let int32 = GpuType(kind: gtInt32)
  let ptrTyp = GpuType(kind: gtPtr, to: int32)
  let symA = newSymbol("a", iSym = "a_h6", typ = ptrTyp)
  let litX = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
  let litY = GpuAst(kind: gpuLit, lValue: "2", lType: int32)
  let inner = GpuAst(kind: gpuIndex,
                     iArr: GpuAst(kind: gpuDeref, dOf: GpuAst(kind: gpuIdent, symbol: symA)),
                     iIndex: litX)
  let outer = GpuAst(kind: gpuIndex, iArr: inner, iIndex: litY)
  let res = registerRewriteIndexDeref.runPassOnNode(outer)[0]
  doAssert res.iArr.kind == gpuIndex, "outer base stays an index"
  doAssert res.iArr.iArr.kind == gpuIdent, "inner base deref stripped"
  echo "  OK — nested deref under non-deref index base stripped"

echo ""
echo "  All rewriteIndexDeref tests passed."
