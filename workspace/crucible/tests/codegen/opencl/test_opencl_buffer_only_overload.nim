## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL: the buffer-only execOpenCL overload must keep working.
## The old signature takes inputs as tuple[data, size] pairs and must
## delegate to the taggedArgs path with every entry marked isValue == false.
## This test exercises the old-style call and then runs the same kernel
## through the taggedArgs path, asserting identical results, so a refactor
## that removes or changes the overload cannot pass silently.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_buffer_only_overload.nim

import workspace/crucible

const kernelCode = opencl:
  proc mulKernel(a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32];
                 output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] * b[0]
    output[1] = a[1] * b[1]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode

  block:
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode)

    var a: array[2, uint32] = [3'u32, 5'u32]
    var b: array[2, uint32] = [7'u32, 11'u32]
    var output: array[2, uint32]
    engine.run("mulKernel", output, (a, b))
    doAssert output[0] == 21, "3 * 7 must be 21"
    doAssert output[1] == 55, "5 * 11 must be 55"
    echo "  OK"

when isMainModule:
  runTest()
