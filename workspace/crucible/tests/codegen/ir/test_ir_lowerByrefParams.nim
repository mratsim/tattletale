## Phase 6: lowerByrefParams pass tests
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_lowerByrefParams.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. lowerByrefParamsImpl renames passByRef params to _p_ prefix
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let sym = newSymbol("data", iSym = "data_byref", typ = int32)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let param = GpuParam(ident: ident, typ: int32, addressSpace: asRMEM, passByRef: true)
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuDiscard)
  var procNode = GpuAst(kind: gpuProc, pName: ident, pParams: @[param], pBody: body)
  var ctx = GpuContext()

  ctx.lowerByrefParamsImpl(procNode)

  doAssert procNode.pParams[0].ident.ident() == "_p_data",
    "Byref param should be renamed to '_p_data', got: '" & procNode.pParams[0].ident.ident() & "'"
  echo "  OK — passByRef param renamed to _p_data"

# ═══════════════════════════════════════════════════════════════════════════
# 2. lowerByrefParamsImpl deref-wraps body idents (no deref-init prepend)
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let sym = newSymbol("val", iSym = "val_byref2", typ = int32)
  # two DISTINCT ident nodes sharing the byref param's symbol — the body is
  # `val = val` (both sides reference the byref param)
  let identL = GpuAst(kind: gpuIdent, symbol: sym)
  let identR = GpuAst(kind: gpuIdent, symbol: sym)
  let param = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: int32,
                       addressSpace: asRMEM, passByRef: true)
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuAssign, aLeft: identL, aRight: identR)
  var procNode = GpuAst(kind: gpuProc, pName: GpuAst(kind: gpuIdent, symbol: sym),
                        pParams: @[param], pBody: body)
  var ctx = GpuContext()

  ctx.lowerByrefParamsImpl(procNode)

  # the body keeps its original statements — NO deref-init is prepended
  doAssert procNode.pBody.statements.len == 1,
    "Body must keep its original statements (no deref-init prepend), got: " &
    $procNode.pBody.statements.len
  let stmt = procNode.pBody.statements[0]
  doAssert stmt.kind == gpuAssign,
    "Body stmt must remain the original gpuAssign, got: " & $stmt.kind
  # idents referencing the byref param are wrapped in gpuDeref: t → (*_p_t)
  doAssert stmt.aLeft.kind == gpuDeref and stmt.aLeft.dOf.kind == gpuIdent,
    "LHS ident must be deref-wrapped (t → (*_p_t)), got: " & $stmt.aLeft.kind
  doAssert stmt.aRight.kind == gpuDeref and stmt.aRight.dOf.kind == gpuIdent,
    "RHS ident must be deref-wrapped (t → (*_p_t)), got: " & $stmt.aRight.kind
  echo "  OK — body idents deref-wrapped (no deref-init prepend)"

# ═══════════════════════════════════════════════════════════════════════════
# 3. Kernel functions are not modified by lowerByrefParamsImpl
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let sym = newSymbol("kernel_data", iSym = "kd_h3", typ = int32)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let param = GpuParam(ident: ident, typ: int32, addressSpace: asRMEM, passByRef: true)
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuDiscard)
  var procNode = GpuAst(kind: gpuProc, pName: ident, pParams: @[param], pBody: body)
  procNode.pAttributes = {attGlobal}
  var ctx = GpuContext()

  ctx.lowerByrefParamsImpl(procNode)

  doAssert procNode.pParams[0].ident.ident() == "kernel_data",
    "Kernel byref param should NOT be renamed, got: '" & procNode.pParams[0].ident.ident() & "'"
  echo "  OK — kernel function params not modified"

echo ""
echo "  All lowerByrefParams tests passed."
