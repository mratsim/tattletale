## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL: output buffer pre-init for in-place kernels.
## The engine uploads the output var's current bytes before launch
## (clCreateBuffer leaves buffer contents spec-undefined, so in-place
## kernels that read their own output, such as out[i] = out[i] + 1, must
## be seeded on the host). This test runs the same in-place kernel twice
## on the same output var, carrying the previous contents forward via the
## pre-launch upload, and checks the accumulation.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_output_preinit.nim

import workspace/crucible

const kernelCode = opencl:
  proc incInPlace(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = output[0] + 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode

  block:
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode)

    # The engine uploads the output var's current bytes before launch:
    # seed the output var and run in place twice.
    var outBuf: array[1, uint32]
    outBuf[0] = 0
    engine.run("incInPlace", outBuf, ())
    doAssert outBuf[0] == 1, "first run must see a zeroed output buffer"

    # Second run: the engine uploads outBuf's current contents (1) and the
    # in-place kernel reads 1 and writes 2.
    engine.run("incInPlace", outBuf, ())
    doAssert outBuf[0] == 2, "second run must see the previous output contents"
    echo "  OK"

when isMainModule:
  runTest()
