## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL: an object constructor that omits a field must zero-initialize it.
## OpenCL C is C99: `{}` is invalid for a scalar ("scalar initializer cannot
## be empty"). A constructor such as `WorkItem(value: 7)` omits the array
## field and the pointer field. The fix emits `{0}` for array/object fields
## and `0` for scalar/pointer fields, so the object is fully zero-initialized
## and still valid C99. This test executes a kernel that builds such an
## object and checks that the omitted fields are zeroed.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_missing_field_init.nim

import std/strutils
import workspace/crucible/src/codegen/cl

type
  WorkItem = object
    value: uint32
    data: array[4, uint32]
    ptrField: ptr UncheckedArray[uint32]

const kernelCode = opencl:
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    var w = WorkItem(value: 7'u32)
    output[0] = w.value
    output[1] = w.data[0]
    output[2] = w.data[3]
    if cast[uint64](w.ptrField) == 0'u64:
      output[3] = 1'u32
    else:
      output[3] = 0'u32

echo kernelCode

# The omitted array field must be braced ({0}), the omitted pointer field
# must be a plain 0. The empty initializer `{}` is invalid OpenCL C.
doAssert kernelCode.contains("{0}"), "expected a {0} zero-initializer for the omitted array field"
doAssert not kernelCode.contains("){}"), "empty initializer list {} is invalid OpenCL C"

block:
  var ctx = initOpenCL()
  defer: ctx.shutdown()
  let r = execOpenCL(
    ctx, kernelCode, "kernelMain", outputBytes = 16,
    taggedArgs = newSeq[tuple[data: pointer, size: int, isValue: bool]]()
  )
  let out32 = cast[ptr array[4, uint32]](r[0].addr)
  doAssert out32[0] == 7, "set field must keep its value"
  doAssert out32[1] == 0, "omitted array field must be zeroed"
  doAssert out32[2] == 0, "omitted array field must be zeroed"
  doAssert out32[3] == 1, "omitted pointer field must be nil"
  echo "  OK"
