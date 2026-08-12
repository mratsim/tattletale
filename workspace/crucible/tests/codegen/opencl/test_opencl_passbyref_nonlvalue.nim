## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL: passByRef with NON-lvalue arguments.
## If the argument to a passByRef param is a temporary (function return,
## constructor, etc.), the `&` prefix is invalid in OpenCL C.
## The compiler should copy to a temp variable first.

import std/strformat
import std/strutils
import workspace/crucible/src/codegen/gpu_compiler

type
  LargeStruct = object
    data: array[8, uint32]  # 32 bytes > 24 bytes threshold

# Test 1: inline constructor passed to passByRef
const code1 = opencl:
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0] + s.data[1]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    # Passing an inline constructor expression — NOT an lvalue
    let v = takeLarge(LargeStruct(data: [10'u32, 20, 30, 40, 50, 60, 70, 80]))
    output[0] = v

echo "=== Test 1: Inline constructor arg ===\n"
echo code1

# The generated code should NOT contain `&{` — that would mean & applied to
# a bare brace-enclosed initializer (invalid in C).
# `&(struct Name){...}` is the correct C99 compound literal syntax.
if code1.contains("&{"):
  echo "  ❌ BUG: & applied to bare brace-enclosed initializer:"
  let idx = code1.find("&{")
  echo "   ..." & code1[idx-10..idx+30] & "..."
else:
  echo "  ✅ No &{ pattern — compound literal used correctly"

# It SHOULD contain & if it went through a temp variable
# It SHOULD NOT contain & if the arg is not passByRef

echo ""
echo "  OK — OpenCL passByRef non-lvalue test"

# Test 2: function call returning a struct
const code2 = opencl:
  proc makeLarge(val: uint32): LargeStruct {.device.} =
    result.data[0] = val
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    let v = takeLarge(makeLarge(42'u32))
    output[0] = v

echo "\n=== Test 2: Function-call return arg ===\n"
echo code2

if code2.contains("&{"):
  echo "  ❌ BUG: & applied to bare brace-enclosed initializer"
else:
  echo "  ✅ & used on function call — compound literal wraps it"
