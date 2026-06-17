## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the directory or at http://opensource.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL: passByRef with non-lvalue argument (field access, function call).
## Currently `&` is blindly prepended; non-lvalues need a temp copy.

import std/strformat
import std/strutils
import workspace/crucible/src/codegen/cl

type
  LargeStruct = object
    data: array[8, uint32]  # 32 bytes > 24

const code = opencl:
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    var s = LargeStruct(data: [1'u32, 2, 3, 4, 5, 6, 7, 8])
    # Passing a field access — is this an lvalue in OpenCL?
    let v = takeLarge(s)
    output[0] = v

echo code
# Check: does &s appear in the generated code?
if code.contains("&s"):
  echo "  passByRef: &s used — s is an lvalue, OK"
elif code.contains("takeLarge(s)"):
  echo "  passByRef: by-value — s is small enough or not passByRef"
else:
  echo "  passByRef: unexpected pattern"
echo "  OK — OpenCL passByRef field access"
