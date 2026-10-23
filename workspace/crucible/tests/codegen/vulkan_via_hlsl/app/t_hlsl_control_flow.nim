## HLSL: if/elif/else chain, ternary, plain-data struct, and cast through the real pass bundle.
##
## Test that the `hlsl:` macro emits HLSL for a kernel using an if/elif/else chain, a ternary,
## a brace-initialized plain-data struct with member-wise assignment and an int32-to-uint32 cast,
## compiling clean through glslangValidator and spirv-val with the binding reflection asserted.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_control_flow.nim

import std/strutils
import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

type Pick = object
  lo: uint32
  hi: uint32

const flowHlsl = hlsl:
  proc flowKernel(output: ptr UncheckedArray[uint32], x: ptr UncheckedArray[int32]) {.global.} =
    let bounds = Pick(lo: 100'u32, hi: 200'u32)
    var picked: Pick
    picked = Pick(lo: 7'u32, hi: 9'u32)
    var acc: uint32 = 0
    if x[0] == 0:
      acc = bounds.lo
    elif x[0] == 1:
      acc = bounds.hi
    else:
      acc = 40'u32
    output[0] = acc
    output[1] = if x[1] < 0: 1'u32 else: 2'u32
    output[2] = x[2].uint32
    output[3] = picked.lo + picked.hi

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order: `output` is written, `x` is read-only
  let flowBindings: seq[BindingExpectation] = @[("output", 0, false), ("x", 1, true)]
  discard compileCheck("control_flow", flowHlsl, flowBindings, "else if (")
  doAssert flowHlsl.contains("if ((x[0] == 0))"),
    "the chain must open on a plain if over the int32 buffer read:\n" & flowHlsl
  doAssert flowHlsl.contains(" ? "),
    "the ternary must print as a conditional expression:\n" & flowHlsl
  doAssert flowHlsl.contains("{ 100u, 200u }"),
    "the struct decl must brace-initialize from the object constructor:\n" & flowHlsl
  doAssert flowHlsl.contains("picked.lo = 7u"),
    "the objconstr assignment must expand member-wise:\n" & flowHlsl
  doAssert flowHlsl.contains("uint(x[2])"),
    "the int32-to-uint32 conversion must print an explicit cast:\n" & flowHlsl

when isMainModule:
  runTest()
