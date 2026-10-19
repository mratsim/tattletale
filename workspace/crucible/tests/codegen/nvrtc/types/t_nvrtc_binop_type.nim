## gpuBinOp self-carried result type (bType): end-to-end NVRTC test
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/types/t_nvrtc_binop_type.nim
import std/[unittest]
import workspace/crucible

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

  suite "gpuBinOp self-carried bType":
    test "fold do-block pulled into device procs compiles and executes":
      var output: array[2, int32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernel)
      let oDev = engine.allocDevice(int32, output.len)
      engine.copyToDeviceFromHost(oDev, addr output[0], output.len)
      engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr output[0], oDev, output.len)
      check output[0] == 8  # foldedMul(C[0]+8) = 1 * (0 + 8)
      check output[1] == 5  # foldedFma(C[1]+2) = 1 + (0 + 2) * 2

when isMainModule:
  runTest()
