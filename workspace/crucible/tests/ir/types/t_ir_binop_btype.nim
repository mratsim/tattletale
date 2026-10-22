## gpuBinOp self-carried result type (bType) on the raw IR
##
## Test that a primitive fold-tail gpuBinOp carries a non-nil gtInt32 bType
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/types/t_ir_binop_btype.nim

import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible

# ── Fold do-block (ceramic foldZipWith style, primitive int32) ──
template foldDo(body: untyped): untyped =
  block:
    let acc {.inject.} = int32(1)
    let it_a {.inject.} = int32(8)
    let it_b {.inject.} = int32(2)
    body

proc findBinOp(n: GpuAst): GpuAst =
  ## First gpuBinOp in the tree, nil when absent.
  if n == nil:
    return nil
  if n.kind == gpuBinOp:
    return n
  for ch in n.items:
    let found = findBinOp(ch)
    if not found.isNil:
      return found

static:
  let ir = toCrucibleAst:
    proc probeKernel(x: int32) {.device.} =
      let y = foldDo:
        acc * it_a
  doAssert ir.kind == gpuBlock, "Expected gpuBlock at top level, got " & $ir.kind
  let fn = ir.statements[0]
  doAssert fn.kind == gpuProc, "Expected gpuProc first, got " & $fn.kind
  let tail = findBinOp(fn.pBody)
  doAssert not tail.isNil, "Expected a gpuBinOp in the fold body"
  doAssert not tail.bIsOverloaded, "Primitive fold must stay a gpuBinOp (not gpuCall)"
  doAssert not tail.bType.isNil, "Fold-tail gpuBinOp must carry non-nil bType"
  doAssert tail.bType == initGpuType(gtInt32),
    "acc * it_a must carry gtInt32 bType, got " & $tail.bType
  echo "  OK — fold-tail gpuBinOp carries gtInt32 bType"
