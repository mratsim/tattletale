# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## gpuBinOp self-carried result type (bType) on the raw IR
##
## Test that a primitive fold-tail gpuBinOp carries a non-nil gtInt32 bType
##
## Run:
##   nim c -r --hints:off --warnings:off \
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

proc runTest() =
  block:
    let ir = toGpuAst:
      proc probeKernel(x: int32) {.device.} =
        let y = foldDo:
          acc * it_a
    doAssert ir.kind == gpuBlock, "Expected gpuBlock at top level, got " & $ir.kind
    let fn = ir.statements[0]
    doAssert fn.kind == gpuProc, "Expected gpuProc first, got " & $fn.kind
    var tail: GpuAst = nil
    proc findBinOp(n: GpuAst) =
      if n == nil or not tail.isNil: return
      if n.kind == gpuBinOp:
        tail = n
        return
      for ch in n.items:
        findBinOp(ch)
    findBinOp(fn.pBody)
    doAssert not tail.isNil, "Expected a gpuBinOp in the fold body"
    doAssert not tail.bIsOverloaded, "Primitive fold must stay a gpuBinOp (not gpuCall)"
    doAssert not tail.bType.isNil, "Fold-tail gpuBinOp must carry non-nil bType"
    doAssert tail.bType == initGpuType(gtInt32),
      "acc * it_a must carry gtInt32 bType, got " & $tail.bType

runTest()
echo "t_ir_binop_btype: all assertions passed"
