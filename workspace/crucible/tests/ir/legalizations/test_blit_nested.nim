## Verify nested expression blocks structure before and after blitting.
##
##   let a = block:
##     let b = block:
##       int32(1)
##     b
##
## Before blitting: 3 gpuBlock(isExpr: true)
## After blitting:  0 gpuBlock(isExpr: true), nested _blit_scope blocks

import std/[sequtils]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types

proc countPred(n: GpuAst; pred: proc(n: GpuAst): bool): int =
  if n == nil: return 0
  result = if pred(n): 1 else: 0
  for child in n.items: result += countPred(child, pred)

proc testInitialIr() =
  let ir = toGpuAst:
    proc kernel(output: ptr UncheckedArray[int32]) {.global.} =
      let a = block:
        let b = block:
          int32(1)
        b
      output[0] = a

  echo "=== Initial IR ==="
  echo ir.pretty()

  let exprBlocks = countPred(ir,
    proc(n: GpuAst): bool = n.kind == gpuBlock and n.isExpr)
  doAssert exprBlocks == 3,
    "Expected 3 gpuBlock(isExpr: true), found " & $exprBlocks
  echo "  OK — 3 expression blocks"

proc testPostBlitInvariants() =
  ## TODO: Requires IR pass-runner fixture that can:
  ##   1. Construct IR via toIr
  ##   2. Clone or re-construct with GpuContext populated from the same AST
  ##   3. Run ensureBlock + blitBlockExprs passes
  ##   4. Verify: countPred(ir, n.kind == gpuBlock and n.isExpr) == 0
  ##   5. Verify: countPred(ir, n.blockLabel == "_blit_scope") == 2
  echo "  SKIP — IR pass-runner fixture not yet available"

when isMainModule:
  echo "--- testInitialIr ---"
  testInitialIr()
  echo ""
  echo "--- testPostBlitInvariants ---"
  testPostBlitInvariants()
  echo ""
  echo "  All tests done"
