## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL: output buffer pre-init for in-place kernels.
## execOpenCL uploads outputInit / outputInitSize into the output buffer
## before running the kernel. Previously the output buffer was fresh-zeroed,
## which broke in-place kernels that read their own output, such as
## out[i] = out[i] + 1. This test runs the same in-place kernel twice on
## the same buffer, carrying the previous contents forward via outputInit,
## and checks the accumulation.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_output_preinit.nim

import workspace/crucible/src/codegen/cl

const kernelCode = opencl:
  proc incInPlace(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = output[0] + 1'u32

echo kernelCode

block:
  var ctx = initOpenCL()
  defer: ctx.shutdown()

  # First run: the output buffer starts zeroed, so 0 + 1 = 1.
  let r1 = execOpenCL(
    ctx, kernelCode, "incInPlace", outputBytes = 4,
    taggedArgs = newSeq[tuple[data: pointer, size: int, isValue: bool]]()
  )
  let v1 = cast[ptr uint32](r1[0].addr)[]
  doAssert v1 == 1, "first run must see a zeroed output buffer"

  # Second run: seed the output buffer with the previous contents. The
  # in-place kernel then reads 1 and writes 2.
  let r2 = execOpenCL(
    ctx, kernelCode, "incInPlace", outputBytes = 4,
    taggedArgs = newSeq[tuple[data: pointer, size: int, isValue: bool]](),
    outputInit = cast[pointer](r1[0].addr), outputInitSize = 4
  )
  let v2 = cast[ptr uint32](r2[0].addr)[]
  doAssert v2 == 2, "second run must see the previous output contents"
  echo "  OK"
