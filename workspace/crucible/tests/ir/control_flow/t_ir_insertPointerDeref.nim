## insertPointerDeref pass test: explicit derefs at value positions for pointer-indexing backends
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_insertPointerDeref.nim

import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_lowering {.all.}

# Pass impls run at compile time, getExprType raises through the `error` macro.

static:
  # ── 1. Wraps gpuIndex over a pointer to unsized array ─────────────────────
  block:
    let int32 = GpuType(kind: gtInt32)
    let uaTyp = GpuType(kind: gtUA, uaTo: int32)
    let ptrTyp = GpuType(kind: gtPtr, to: uaTyp)
    let ptrSym = newSymbol("p", iSym = "p_h1", typ = ptrTyp)
    let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
    let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: int32)
    var idx = GpuAst(kind: gpuIndex, iArr: ptrIdent, iIndex: lit0)
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, idx)
    doAssert idx.kind == gpuIndex, "Should remain gpuIndex"
    doAssert idx.iArr.kind == gpuDeref, "iArr should be wrapped in gpuDeref"
    doAssert idx.iArr.dOf.kind == gpuIdent, "deref operand should be the raw ident"
    echo "  OK - ptr to unsized array: gpuDeref inserted"

  # ── 2. Wraps gpuIndex over a pointer to static array ──────────────────────
  block:
    let int32 = GpuType(kind: gtInt32)
    let arrTyp = GpuType(kind: gtArray, aTyp: int32, aLen: 4)
    let ptrTyp = GpuType(kind: gtPtr, to: arrTyp)
    let ptrSym = newSymbol("p", iSym = "p_h2", typ = ptrTyp)
    let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
    let lit1 = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
    var idx = GpuAst(kind: gpuIndex, iArr: ptrIdent, iIndex: lit1)
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, idx)
    doAssert idx.iArr.kind == gpuDeref, "ptr-to-array indexing should be wrapped"
    echo "  OK - ptr to static array: gpuDeref inserted"

  # ── 3. Leaves gpuIndex over a pointer to scalar alone ─────────────────────
  block:
    let int32 = GpuType(kind: gtInt32)
    let ptrTyp = GpuType(kind: gtPtr, to: int32)
    let ptrSym = newSymbol("p", iSym = "p_h3", typ = ptrTyp)
    let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
    let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: int32)
    var idx = GpuAst(kind: gpuIndex, iArr: ptrIdent, iIndex: lit0)
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, idx)
    doAssert idx.iArr.kind == gpuIdent, "scalar-pointee index must not be wrapped"
    echo "  OK - ptr to scalar: no gpuDeref"

  # ── 4. Wraps gpuDot over a pointer parent ─────────────────────────────────
  block:
    let int32 = GpuType(kind: gtInt32)
    let structTyp = GpuType(kind: gtObject)
    let ptrTyp = GpuType(kind: gtPtr, to: structTyp)
    let ptrSym = newSymbol("p", iSym = "p_h4", typ = ptrTyp)
    let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
    let fieldSym = newSymbol("f", iSym = "f_h4", typ = int32)
    let fieldIdent = GpuAst(kind: gpuIdent, symbol: fieldSym)
    var dot = GpuAst(kind: gpuDot, dParent: ptrIdent, dField: fieldIdent)
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, dot)
    doAssert dot.kind == gpuDot, "Should remain gpuDot"
    doAssert dot.dParent.kind == gpuDeref, "ptr parent should be wrapped in gpuDeref"
    echo "  OK - ptr parent: gpuDeref inserted"

  # ── 5. Never derefs under addr-of ─────────────────────────────────────────
  block:
    let int32 = GpuType(kind: gtInt32)
    let uaTyp = GpuType(kind: gtUA, uaTo: int32)
    let ptrTyp = GpuType(kind: gtPtr, to: uaTyp)
    let ptrSym = newSymbol("p", iSym = "p_h5", typ = ptrTyp)
    let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
    let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: int32)
    let idx = GpuAst(kind: gpuIndex, iArr: ptrIdent, iIndex: lit0)
    var addrOf = GpuAst(kind: gpuAddr, aOf: idx)
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, addrOf)
    doAssert addrOf.aOf.iArr.kind == gpuIdent, "no deref under addr-of"
    echo "  OK - addr-of subtree untouched"

  # ── 6. Double deref: gpuDeref over a pointer operand ──────────────────────
  block:
    let int32 = GpuType(kind: gtInt32)
    let ptrPtr = GpuType(kind: gtPtr, to: GpuType(kind: gtPtr, to: int32))
    let ptrSym = newSymbol("p", iSym = "p_h6", typ = ptrPtr)
    let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
    var deref = GpuAst(kind: gpuDeref, dOf: ptrIdent)
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, deref)
    doAssert deref.dOf.kind == gpuDeref, "ptr operand of a deref should be wrapped"
    echo "  OK - double deref inserted"

  # ── 7. Leaves plain array indexing alone ──────────────────────────────────
  block:
    let int32 = GpuType(kind: gtInt32)
    let arrType = GpuType(kind: gtArray, aTyp: int32, aLen: 10)
    let arrSym = newSymbol("a", iSym = "a_h7", typ = arrType)
    let arrIdent = GpuAst(kind: gpuIdent, symbol: arrSym)
    let lit2 = GpuAst(kind: gpuLit, lValue: "2", lType: int32)
    var idx = GpuAst(kind: gpuIndex, iArr: arrIdent, iIndex: lit2)
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, idx)
    doAssert idx.iArr.kind == gpuIdent, "plain array index must not be wrapped"
    echo "  OK - plain array: untouched"

echo "insertPointerDeref: all cases pass"
