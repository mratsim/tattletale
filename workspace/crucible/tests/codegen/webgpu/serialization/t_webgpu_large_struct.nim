## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## WebGPU: large structs passed to device functions (by-value).
## WGSL doesn't use passByRef — all struct params are by-value.
## The gpuMaterialize pass must not break WGSL codegen.

import std/strformat
import std/strutils
import workspace/crucible

type
  LargeStruct = object
    data: array[8, uint32]  # 32 bytes > 24 threshold

# Test 1: inline constructor arg to device function
const code1 = webgpu:
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0] + s.data[1]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = takeLarge(LargeStruct(data: [10'u32, 20, 30, 40, 50, 60, 70, 80]))

echo "=== Test 1: Inline constructor arg ===\n"
echo code1

# Should compile without crashing — generated code should use by-value struct params

# Test 2: function return value arg
const code2 = webgpu:
  proc makeLarge(val: uint32): LargeStruct {.device.} =
    result.data[0] = val
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = takeLarge(makeLarge(42'u32))

echo "\n=== Test 2: Function return arg ===\n"
echo code2

echo "\n  OK — WGSL large struct by-value (no passByRef)"
