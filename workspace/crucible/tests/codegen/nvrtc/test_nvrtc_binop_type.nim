## gpuBinOp self-carried result type (bType) — end-to-end NVRTC oracle
##
## A fold do-block (`acc * it`, `acc + it_a * it_b`) is pulled into device
## procs from OUTSIDE the cuda block. The fold body lowers to a block
## expression whose tail is a primitive gpuBinOp. Pre-fix, blitting that
## tail crashed the compiler: "getExprType: unhandled node kind gpuBinOp".
##
## This test is MANDATORY: it runs the crash path through the full pipeline
## via execute() and additionally asserts the fold-tail IR node is a
## primitive gpuBinOp carrying its own gtInt32 result type. The fold element
## is the proc parameter, so the kernel result genuinely depends on its input.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests/nvrtc --nimcache:nimcache/tests/nvrtc \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_binop_type.nim

import std/[unittest]
import workspace/crucible
import workspace/crucible/src/codegen/ir/gpu_type_constructors

# ── Fold do-block (ceramic foldZipWith style, primitive int32) ──
template foldDo(body: untyped): untyped =
  block:
    let acc {.inject.} = int32(1)
    let it_a {.inject.} = int32(8)
    let it_b {.inject.} = int32(2)
    body

# Fold bodies pulled into procs from OUTSIDE the cuda block —
# this is the crash path: the proc body lowers to
# `result = gpuBlock(isExpr: true, @[..., gpuBinOp])`.
# The fold element is the parameter, so the kernel result depends on its input.
proc foldedMul(x: int32): int32 =
  foldDo:
    acc * x

proc foldedFma(x: int32): int32 =
  foldDo:
    acc + x * it_b

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[int32]) {.global.} =
    C[0] = foldedMul(C[0] + 8)
    C[1] = foldedFma(C[1] + 2)

# ── Fold-tail-kind + bType oracle on the raw IR ──
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

suite "gpuBinOp self-carried bType":
  test "fold do-block pulled into device procs compiles and executes":
    var output: array[2, int32]
    var engine = bkCuda.init()
    engine.ingest(kernel)
    engine.run<<(1, 1)>>("kernel", output, ())
    check output[0] == 8  # foldedMul(C[0]+8) = 1 * (0 + 8)
    check output[1] == 5  # foldedFma(C[1]+2) = 1 + (0 + 2) * 2
