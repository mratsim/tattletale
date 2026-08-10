## getExprType gpuDeref unwrap test (COV-B-002):
## `getExprType(gpuDeref(p))` where `p` is a gtPtr-typed ident must return the
## POINTEE (by-ref params: the ident is typed as ptr, the deref is the
## pointee), and a non-ptr ident must pass through unchanged. This pins the
## unwrap added for by-ref params — a revert would fail here.
##
## getExprType uses the compile-time `error` macro, so it can only be invoked
## in compile-time context — the cases run in `static:` blocks.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_getExprType.nim

import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_legalizations

static:
  block: # 1. gpuDeref over a gtPtr ident unwraps to the pointee
    let f32 = GpuType(kind: gtFloat32)
    let ptrF32 = GpuType(kind: gtPtr, to: f32)
    let sym = newSymbol("p", iSym = "p_deref", typ = ptrF32)
    let ident = GpuAst(kind: gpuIdent, symbol: sym)
    let deref = GpuAst(kind: gpuDeref, dOf: ident)
    var ctx = GpuContext()

    let t = ctx.getExprType(deref)

    doAssert t != nil and t.kind == gtFloat32,
      "gpuDeref over gtPtr must return the pointee, got " & $t.kind
    echo "  OK — gpuDeref(gtPtr ident) unwraps to pointee (gtFloat32)"

  block: # 2. control: gpuDeref over a non-ptr ident passes through
    let int32 = GpuType(kind: gtInt32)
    let sym = newSymbol("v", iSym = "v_deref", typ = int32)
    let ident = GpuAst(kind: gpuIdent, symbol: sym)
    let deref = GpuAst(kind: gpuDeref, dOf: ident)
    var ctx = GpuContext()

    let t = ctx.getExprType(deref)

    doAssert t != nil and t.kind == gtInt32,
      "non-ptr ident must pass through unchanged, got " & $t.kind
    echo "  OK — gpuDeref(non-ptr ident) passes through (gtInt32)"

  block: # 3. control: gpuIdent alone returns its own type
    let f64 = GpuType(kind: gtFloat64)
    let sym = newSymbol("x", iSym = "x_ident", typ = f64)
    let ident = GpuAst(kind: gpuIdent, symbol: sym)
    var ctx = GpuContext()

    let t = ctx.getExprType(ident)

    doAssert t != nil and t.kind == gtFloat64,
      "plain ident must return its own type, got " & $t.kind
    echo "  OK — gpuIdent returns its own type (gtFloat64)"

  block: # 4. consistency: gpuIndex self-unwraps the same way — gpuIndex(gpuDeref(p)) is gtFloat32
    let f32 = GpuType(kind: gtFloat32)
    let ptrF32 = GpuType(kind: gtPtr, to: f32)
    let sym = newSymbol("p", iSym = "p_idx", typ = ptrF32)
    let ident = GpuAst(kind: gpuIdent, symbol: sym)
    let deref = GpuAst(kind: gpuDeref, dOf: ident)
    let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: GpuType(kind: gtInt32))
    let idx = GpuAst(kind: gpuIndex, iArr: deref, iIndex: lit0)
    var ctx = GpuContext()

    let t = ctx.getExprType(idx)

    doAssert t != nil and t.kind == gtFloat32,
      "gpuIndex(gpuDeref(p)) must agree with the gpuDeref unwrap, got " & $t.kind
    echo "  OK — gpuIndex(gpuDeref(p)) consistent with the unwrap (gtFloat32)"

  block: # 5. F3: gpuIndex(gpuDeref(p)) with p: ptr array[N, T] returns T, not the array
    let int32 = GpuType(kind: gtInt32)
    let arr4 = GpuType(kind: gtArray, aTyp: int32, aLen: 4)
    let ptrArr = GpuType(kind: gtPtr, to: arr4)
    let sym = newSymbol("p", iSym = "p_arr", typ = ptrArr)
    let ident = GpuAst(kind: gpuIdent, symbol: sym)
    let deref = GpuAst(kind: gpuDeref, dOf: ident)
    let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: GpuType(kind: gtInt32))
    let idx = GpuAst(kind: gpuIndex, iArr: deref, iIndex: lit0)
    var ctx = GpuContext()

    let t = ctx.getExprType(idx)

    doAssert t != nil and t.kind == gtInt32,
      "gpuIndex(gpuDeref(p: ptr array)) must return the ELEMENT type, got " & $t.kind
    echo "  OK — gpuIndex(gpuDeref(ptr array)) returns element type (gtInt32)"


echo "  ALL PASS — getExprType gpuDeref unwrap pinned (compile-time)"
