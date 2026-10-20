## insertPointerDeref pass
##
## Test that the pass wraps pointer-indexed reads in gpuDeref and leaves
## address-of subtrees, explicit derefs, and plain array indexing alone.
##
## - gpuIndex over a pointer to unsized array, static array, or scalar wraps its base
## - gpuDot over a pointer parent wraps the parent, including inside a gpuIndex iArr
## - `m[i][j]` over an array of pointers wraps only the outer index base
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_insertPointerDeref.nim

import workspace/crucible/src/crucible_ir
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

  # ── 3. Wraps gpuIndex over a pointer to scalar ───────────────────────────
  block:
    let int32 = GpuType(kind: gtInt32)
    let ptrTyp = GpuType(kind: gtPtr, to: int32)
    let ptrSym = newSymbol("p", iSym = "p_h3", typ = ptrTyp)
    let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
    let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: int32)
    var idx = GpuAst(kind: gpuIndex, iArr: ptrIdent, iIndex: lit0)
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, idx)
    doAssert idx.iArr.kind == gpuDeref, "scalar-pointee index must be wrapped: bare pointer indexing is unrenderable on Vulkan"
    echo "  OK - ptr to scalar: gpuDeref inserted"

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

  # ── 6. Single deref of a pointer-to-pointer stays one deref ───────────────
  block:
    let int32 = GpuType(kind: gtInt32)
    let ptrPtr = GpuType(kind: gtPtr, to: GpuType(kind: gtPtr, to: int32))
    let ptrSym = newSymbol("p", iSym = "p_h6", typ = ptrPtr)
    let ptrIdent = GpuAst(kind: gpuIdent, symbol: ptrSym)
    var deref = GpuAst(kind: gpuDeref, dOf: ptrIdent)
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, deref)
    doAssert deref.dOf.kind == gpuIdent,
      "the deref node itself is the wrap: re-wrapping its pointer operand changes the read depth"
    echo "  OK - explicit deref left alone"

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

  # ── 8. Recurses into gpuIndex over a dot: p.f[i] wraps the dot's pointer parent ─
  block:
    let int32 = GpuType(kind: gtInt32)
    let arrType = GpuType(kind: gtArray, aTyp: int32, aLen: 4)
    let structTyp = GpuType(kind: gtObject)
    let ptrTyp = GpuType(kind: gtPtr, to: structTyp)
    let fieldSym = newSymbol("f", iSym = "f_h8", typ = arrType)
    let ptrSym = newSymbol("p", iSym = "p_h8", typ = ptrTyp)
    var dot = GpuAst(kind: gpuDot,
                     dParent: GpuAst(kind: gpuIdent, symbol: ptrSym),
                     dField: GpuAst(kind: gpuIdent, symbol: fieldSym))
    var idx = GpuAst(kind: gpuIndex, iArr: dot, iIndex: GpuAst(kind: gpuLit, lValue: "0", lType: int32))
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, idx)
    doAssert idx.iArr.kind == gpuDot, "the dot itself stays the index base"
    doAssert idx.iArr.dParent.kind == gpuDeref, "the pointer parent inside iArr must be wrapped"
    echo "  OK - p.f[i]: deref reaches inside iArr"

  # ── 9. Wraps gpuIndex whose base is a nested gpuIndex over an array of pointers ─
  block:
    let int32 = GpuType(kind: gtInt32)
    let ptrTyp = GpuType(kind: gtPtr, to: int32)
    let arrOfPtr = GpuType(kind: gtArray, aTyp: ptrTyp, aLen: 4)
    let arrSym = newSymbol("m", iSym = "m_h9", typ = arrOfPtr)
    let inner = GpuAst(kind: gpuIndex,
                       iArr: GpuAst(kind: gpuIdent, symbol: arrSym),
                       iIndex: GpuAst(kind: gpuLit, lValue: "0", lType: int32))
    var idx = GpuAst(kind: gpuIndex, iArr: inner, iIndex: GpuAst(kind: gpuLit, lValue: "1", lType: int32))
    var ctx = GpuContext()
    insertPointerDerefImpl(ctx, idx)
    doAssert idx.iArr.kind == gpuDeref, "index over an indexed pointer element must be wrapped"
    echo "  OK - m[i][j] over array of pointers: outer index wrapped"

echo "insertPointerDeref: all cases pass"
