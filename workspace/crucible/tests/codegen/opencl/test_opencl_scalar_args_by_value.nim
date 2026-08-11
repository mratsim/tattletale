## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL: scalar kernel arguments must be bound by value.
## execOpenCL takes taggedArgs where entries with isValue == true are bound
## via setKernelArg(index, size, data) instead of being treated as cl_mem
## buffers. Previously scalar coefficients (alpha/beta) were passed as
## cl_mem, which fails with CL_INVALID_ARG_SIZE. This test passes a float32
## and an int32 scalar through taggedArgs with isValue == true alongside a
## buffer argument, and checks the computed result.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_scalar_args_by_value.nim

import workspace/crucible/src/codegen/cl

const kernelCode = opencl:
  proc scaleKernel(a: ptr UncheckedArray[float32];
                   alpha: float32;
                   beta: int32;
                   output: ptr UncheckedArray[float32]) {.global.} =
    output[0] = a[0] * alpha + float32(beta)

echo kernelCode

block:
  var ctx = initOpenCL()
  defer: ctx.shutdown()

  # Scalars must be copied into live vars: the taggedArgs entry points at
  # their address, and the pointer must stay valid for the whole call.
  var a: array[1, float32] = [2.5'f32]
  var alpha = 3.0'f32
  var beta = 4'i32

  let r = execOpenCL(
    ctx, kernelCode, "scaleKernel", outputBytes = 4,
    taggedArgs = @[
      (cast[pointer](a[0].addr), 4, false),
      (cast[pointer](alpha.addr), 4, true),
      (cast[pointer](beta.addr), 4, true)
    ]
  )
  let outVal = cast[ptr float32](r[0].addr)[]
  doAssert outVal == 11.5'f32, "expected 2.5 * 3.0 + 4.0 = 11.5"
  echo "  OK"
