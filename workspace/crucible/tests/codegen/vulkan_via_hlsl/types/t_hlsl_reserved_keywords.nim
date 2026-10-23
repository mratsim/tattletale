## HLSL: identifiers colliding with HLSL reserved words rename end to end.
##
## Test that a kernel declaring identifiers that are HLSL reserved words renames
## every collision to `<name>_nim`:
##
## - parameter `uniform`, locals `register` and `precise`
## - no bare reserved word reaches the emitted text
## - the renamed source compiles clean through glslangValidator and spirv-val with the binding reflection asserted
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/types/t_hlsl_reserved_keywords.nim

import std/[re, strutils]
import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const reservedHlsl = hlsl:
  proc reservedKernel(output: ptr UncheckedArray[uint32], uniform: uint32) {.global.} =
    var register = uniform + 1'u32
    var precise = register * 2'u32
    output[0] = precise

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order, `uniform` is a push constant and takes no binding
  let reservedBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("reserved_keywords", reservedHlsl, reservedBindings, "uniform_nim")
  for word in ["uniform", "register", "precise"]:
    # \b does not match between a word char and `_`, so `uniform_nim` never trips the bare-word scan
    doAssert find(reservedHlsl, "\\b" & word & "\\b") == -1,
      "emitted HLSL still carries the bare reserved word `" & word & "`:\n" & reservedHlsl
    doAssert reservedHlsl.contains(word & "_nim"),
      "emitted HLSL is missing the `" & word & "_nim` rename:\n" & reservedHlsl

when isMainModule:
  runTest()
