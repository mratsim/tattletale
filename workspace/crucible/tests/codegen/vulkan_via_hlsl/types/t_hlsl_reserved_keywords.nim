## HLSL: identifiers colliding with HLSL reserved words rename end to end.
##
## Test that fn names and parameters colliding with HLSL reserved words rename
## to `<name>_nim` and local declarations pass through verbatim:
##
## - parameter `uniform` renames in the signature
## - local declarations keep their names, the rename pass does not cover them
## - the renamed source compiles clean through DXC and spirv-val with the binding reflection asserted
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
    var localReg = uniform + 1'u32
    var localPrecise = localReg * 2'u32
    output[0] = localPrecise

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order, `uniform` is a push constant and takes no binding
  let reservedBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("reserved_keywords", reservedHlsl, reservedBindings, "uniform_nim")
  # \b does not match between a word char and `_`, so `uniform_nim` never trips the bare-word scan
  doAssert find(reservedHlsl, "\\buniform\\b") == -1,
    "emitted HLSL still carries the bare reserved word `uniform`:\n" & reservedHlsl
  doAssert reservedHlsl.contains("uniform_nim"),
    "emitted HLSL is missing the `uniform_nim` rename:\n" & reservedHlsl
  for local in ["localReg", "localPrecise"]:
    doAssert reservedHlsl.contains(local),
      "emitted HLSL is missing the local declaration `" & local &
      "`, local declarations pass through verbatim:\n" & reservedHlsl

when isMainModule:
  runTest()
