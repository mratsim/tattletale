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
  let param = GpuParam(ident: ident, typ: int32, addressSpace: asFunction, passByRef: true)
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuDiscard)
  var procNode = GpuAst(kind: gpuProc, pName: ident, pParams: @[param], pBody: body)
  var ctx = GpuContext()

  ctx.lowerByrefParamsImpl(procNode)

  doAssert procNode.pParams[0].ident.ident() == "_p_data",
    "Byref param should be renamed to '_p_data', got: '" & procNode.pParams[0].ident.ident() & "'"
  echo "  OK — passByRef param renamed to _p_data"

# ═══════════════════════════════════════════════════════════════════════════
# 2. lowerByrefParamsImpl inserts deref-init statement in function body
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let sym = newSymbol("val", iSym = "val_byref2", typ = int32)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let param = GpuParam(ident: ident, typ: int32, addressSpace: asFunction, passByRef: true)
  var body = GpuAst(kind: gpuBlock, statements: @[])
  var procNode = GpuAst(kind: gpuProc, pName: ident, pParams: @[param], pBody: body)
  var ctx = GpuContext()

  ctx.lowerByrefParamsImpl(procNode)

  doAssert procNode.pBody.statements.len >= 1, "Body should have at least 1 statement (deref-init), got: " & $procNode.pBody.statements.len
  let firstStmt = procNode.pBody.statements[0]
  doAssert firstStmt.kind == gpuAssign, "First body stmt should be gpuAssign (deref-init), got: " & $firstStmt.kind
  doAssert firstStmt.aLeft.kind == gpuIdent, "LHS of deref-init should be ident"
  doAssert firstStmt.aRight.kind == gpuDeref, "RHS of deref-init should be deref"
  echo "  OK — deref-init statement inserted in function body"

# ═══════════════════════════════════════════════════════════════════════════
# 3. Kernel functions are not modified by lowerByrefParamsImpl
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let sym = newSymbol("kernel_data", iSym = "kd_h3", typ = int32)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let param = GpuParam(ident: ident, typ: int32, addressSpace: asFunction, passByRef: true)
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
