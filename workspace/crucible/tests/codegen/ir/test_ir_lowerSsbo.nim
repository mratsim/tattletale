## Phase 6: lowerSsboParams pass tests
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_lowerSsbo.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. lowerSsboParamsImpl builds canonical SSBO list from kernel params
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let ptrInt32 = GpuType(kind: gtPtr, to: int32)
  let sym = newSymbol("buf", iSym = "buf_ssbo1", typ = ptrInt32)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  let param = GpuParam(ident: ident, typ: ptrInt32, addressSpace: asDevice)
  var kernelBody = GpuAst(kind: gpuBlock, statements: @[GpuAst(kind: gpuDiscard)])
  var kernel = GpuAst(kind: gpuProc, pName: ident, pParams: @[param], pBody: kernelBody)
  kernel.pAttributes = {attGlobal}
  var ctx = GpuContext()
  ctx.fnTab[ident] = kernel

  lowerSsboParamsImpl(ctx)

  doAssert ctx.ssboCanonicalInfo.len == 1, "Should have 1 SSBO slot, got: " & $ctx.ssboCanonicalInfo.len
  doAssert ctx.ssboCanonicalInfo[0].name == "buf", "SSBO name should be 'buf', got: '" & ctx.ssboCanonicalInfo[0].name & "'"
  echo "  OK — SSBO canonical list built from kernel param"

# ═══════════════════════════════════════════════════════════════════════════
# 2. lowerSsboParamsImpl normalizes names across kernels
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let ptrInt32 = GpuType(kind: gtPtr, to: int32)
  let sym1 = newSymbol("bufferA", iSym = "buf_a_sym", typ = ptrInt32)
  let ident1 = GpuAst(kind: gpuIdent, symbol: sym1)
  let param1 = GpuParam(ident: ident1, typ: ptrInt32, addressSpace: asDevice)
  var body1 = GpuAst(kind: gpuBlock, statements: @[GpuAst(kind: gpuDiscard)])
  var kernel1 = GpuAst(kind: gpuProc, pName: ident1, pParams: @[param1], pBody: body1)
  kernel1.pAttributes = {attGlobal}

  let sym2 = newSymbol("bufferB", iSym = "buf_b_sym", typ = ptrInt32)
  let ident2 = GpuAst(kind: gpuIdent, symbol: sym2)
  let param2 = GpuParam(ident: ident2, typ: ptrInt32, addressSpace: asDevice)
  var body2 = GpuAst(kind: gpuBlock, statements: @[GpuAst(kind: gpuDiscard)])
  var kernel2 = GpuAst(kind: gpuProc, pName: ident2, pParams: @[param2], pBody: body2)
  kernel2.pAttributes = {attGlobal}

  var ctx = GpuContext()
  ctx.fnTab[ident1] = kernel1
  ctx.fnTab[ident2] = kernel2

  lowerSsboParamsImpl(ctx)

  doAssert ctx.ssboCanonicalInfo.len == 1, "Should have 1 SSBO slot for both kernels"
  let canonName = ctx.ssboCanonicalInfo[0].name
  # Second kernel's param should be renamed to match first
  doAssert kernel1.pParams[0].ident.ident() == canonName, "First kernel param should match canonical name"
  doAssert kernel2.pParams[0].ident.ident() == canonName, "Second kernel param should be renamed to match canonical name"
  echo "  OK — SSBO param names normalized across kernels"

echo ""
echo "  All lowerSsbo tests passed."
