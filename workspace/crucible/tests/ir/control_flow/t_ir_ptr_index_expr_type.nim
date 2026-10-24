## getExprType of a pointer-indexed read
##
## Test that `p[i]` types as the element for a pointer to an array,
## fixed-size or unchecked, matching Nim pointer indexing and the text
## backends' type printers, while a pointer to a scalar passes through.
##
## - gtPtr(gtUncheckedArray(T)) indexes to T, the blit-temp typing bug fixed here
## - gtPtr(gtArray(T)) indexes to T
## - gtPtr(scalar) passes through, `p[i]` is pointer arithmetic
## - the explicit-deref form `(*p)[i]` keeps the element type it already had
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_ptr_index_expr_type.nim

import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/ir/ast_type_resolvers

static:
  let f32 = GpuType(kind: gtFloat32)
  let i32 = GpuType(kind: gtInt32)
  let lit0 = GpuAst(kind: gpuLit, lValue: "0", lType: i32)
  let iSym = newSymbol("i", iSym = "i_h", typ = i32)
  let iIdent = GpuAst(kind: gpuIdent, symbol: iSym)

  # ── 1. gtPtr(gtUncheckedArray(T)) indexes to the element ────────────────
  block:
    let uaTyp = GpuType(kind: gtUncheckedArray, uaTo: f32)
    let ptrTyp = GpuType(kind: gtPtr, to: uaTyp)
    let ptrIdent = GpuAst(kind: gpuIdent,
                          symbol: newSymbol("p", iSym = "p_h1", typ = ptrTyp))
    let idx = GpuAst(kind: gpuIndex, iArr: ptrIdent, iIndex: iIdent)
    let ctx = GpuContext()
    doAssert ctx.getExprType(idx).kind == gtFloat32,
      "p[i] over ptr-to-UncheckedArray is the element, got " &
      ctx.getExprType(idx).pretty()
    echo "  OK - ptr to unchecked array indexes to the element"

  # ── 2. gtPtr(gtArray(T)) indexes to the element ──────────────────────────
  block:
    let arrTyp = GpuType(kind: gtArray, aTyp: f32, aLen: 4)
    let ptrTyp = GpuType(kind: gtPtr, to: arrTyp)
    let ptrIdent = GpuAst(kind: gpuIdent,
                          symbol: newSymbol("p", iSym = "p_h2", typ = ptrTyp))
    let idx = GpuAst(kind: gpuIndex, iArr: ptrIdent, iIndex: iIdent)
    let ctx = GpuContext()
    doAssert ctx.getExprType(idx).kind == gtFloat32,
      "p[i] over ptr-to-array is the element, got " &
      ctx.getExprType(idx).pretty()
    echo "  OK - ptr to static array indexes to the element"

  # ── 3. gtPtr(scalar) passes through, pointer arithmetic ──────────────────
  block:
    let ptrTyp = GpuType(kind: gtPtr, to: f32)
    let ptrIdent = GpuAst(kind: gpuIdent,
                          symbol: newSymbol("p", iSym = "p_h3", typ = ptrTyp))
    let idx = GpuAst(kind: gpuIndex, iArr: ptrIdent, iIndex: iIdent)
    let ctx = GpuContext()
    doAssert ctx.getExprType(idx).kind == gtFloat32,
      "p[i] over a scalar pointer is the pointee"
    echo "  OK - ptr to scalar passes through"

  # ── 4. the explicit-deref form keeps the element type ────────────────────
  block:
    let uaTyp = GpuType(kind: gtUncheckedArray, uaTo: f32)
    let ptrTyp = GpuType(kind: gtPtr, to: uaTyp)
    let ptrIdent = GpuAst(kind: gpuIdent,
                          symbol: newSymbol("p", iSym = "p_h4", typ = ptrTyp))
    let idx = GpuAst(kind: gpuIndex,
                     iArr: GpuAst(kind: gpuDeref, dOf: ptrIdent),
                     iIndex: iIdent)
    let ctx = GpuContext()
    doAssert ctx.getExprType(idx).kind == gtFloat32,
      "(*p)[i] over ptr-to-UncheckedArray is the element"
    echo "  OK - explicit deref keeps the element type"

echo "t_ir_ptr_index_expr_type: all static assertions passed"
