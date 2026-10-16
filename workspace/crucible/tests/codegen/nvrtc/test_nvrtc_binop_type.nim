## gpuBinOp self-carried result type (bType): end-to-end NVRTC test
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_binop_type.nim
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
proc runTest() =   # private — tests run in a proc so engines are destroyed at return
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
      let oDev = engine.allocDevice(sizeof(output))
      engine.copyToDeviceFromHost(oDev, addr output[0], sizeof(output))
      engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr output[0], oDev, sizeof(output))
      engine.freeDevice(oDev)
      check output[0] == 8  # foldedMul(C[0]+8) = 1 * (0 + 8)
      check output[1] == 5  # foldedFma(C[1]+2) = 1 + (0 + 2) * 2

when isMainModule:
  runTest()
