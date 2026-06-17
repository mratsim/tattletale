## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL: passByRef call-site lvalue check for `&` prefix.
## When a large struct (>24 bytes) is passed by reference, the call-site
## prepends `&` to the argument. This is only valid for lvalues.

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
    let v = takeLarge(s)
    output[0] = v

doAssert code.contains("&s"), &"Expected &s (pass-by-ref) in:\n{code}"
echo code
echo "  OK — OpenCL passByRef & on lvalue"
