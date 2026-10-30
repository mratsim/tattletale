## HLSL: identifiers colliding with HLSL reserved words rename end to end.
##
## Test that fn names and parameters colliding with HLSL reserved words rename
## to `<name>_nim` and the renamed kernel executes on the Vulkan descriptor route:
##
## - parameter `uniform` renames in the signature and the push-constant block
## - the multi-read local `localKeep` keeps its name (the rename pass covers only fn names and parameters)
## - readback is byte-exact for the folded single-use arithmetic and the surviving local
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/types/t_hlsl_reserved_keywords.nim

import std/[re, strutils]
import workspace/crucible
import workspace/crucible/src/runtime/backends/vulkan_common
import workspace/crucible/src/runtime/backends/vulkan_descriptor_runtime
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const reservedHlsl = hlsl:
  proc reservedKernel(output: ptr UncheckedArray[uint32], uniform: uint32) {.global.} =
    var localReg = uniform + 1'u32
    var localPrecise = localReg * 2'u32
    output[0] = localPrecise
    var localKeep = uniform
    output[1] = localKeep
    output[2] = localKeep

proc assertU32(got: seq[uint32], want: openArray[uint32], label: string) =
  doAssert got.len == want.len, label & ": " & $got.len & " slots, want " & $want.len
  for i in 0 ..< want.len:
    doAssert got[i] == want[i],
      label & ": slot " & $i & " got " & $got[i] & " want " & $want[i]

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order. The scalar block takes the first
  # ordinal after the buffers: `output` binds 0, `uniform` binds 1
  let reservedBindings: seq[BindingExpectation] = @[("output", 0, false)]
  let reservedWords = compileCheck("reserved_keywords", reservedHlsl, reservedBindings, "uniform_nim")
  # \b does not match between a word char and `_`, so `uniform_nim` never trips the bare-word scan
  doAssert find(reservedHlsl, "\\buniform\\b") == -1,
    "emitted HLSL still carries the bare reserved word `uniform`:\n" & reservedHlsl
  doAssert reservedHlsl.contains("uniform_nim"),
    "emitted HLSL is missing the `uniform_nim` rename:\n" & reservedHlsl
  doAssert reservedHlsl.contains("localKeep"),
    "emitted HLSL is missing the surviving local declaration `localKeep`," &
    " multi-read locals pass through verbatim:\n" & reservedHlsl
  doAssert not reservedHlsl.contains("localReg") and
    not reservedHlsl.contains("localPrecise"),
    "emitted HLSL keeps a once-assigned once-read local the fold should remove:\n" & reservedHlsl

  block: # Vulkan descriptor route: set layout, pool, set, writes, pipeline, dispatch
    var vctx = newVulkanContext()
    var o: array[3, uint32] = [0'u32, 0, 0]
    var u = 1'u32
    var oBuf = vctx.allocBuffer(sizeof(o))
    var uBuf = vctx.allocBuffer(sizeof(uint32))
    vctx.writeBuffer(uBuf, addr u, sizeof(uint32))
    let setLayout = vctx.createDescriptorSetLayout(1, 1)
    let pool = vctx.createDescriptorPool(1, 1)
    let descSet = vctx.allocateDescriptorSet(pool, setLayout)
    vctx.updateDescriptorSet(descSet, [oBuf], 1, [uBuf])
    var sm = vctx.createShaderModule(reservedWords)
    var pipeline = vctx.createPipeline(sm, "main", 0, setLayout)

    vctx.runKernel(pipeline, [256'u32, 1, 1], [256'u32, 1, 1], nil, 0, descSet)
    assertU32(readBuffer[uint32](vctx, oBuf), [4'u32, 1, 1], "vulkan reserved u=1")
    u = 5'u32
    vctx.writeBuffer(uBuf, addr u, sizeof(uint32))
    vctx.runKernel(pipeline, [256'u32, 1, 1], [256'u32, 1, 1], nil, 0, descSet)
    assertU32(readBuffer[uint32](vctx, oBuf), [12'u32, 5, 5], "vulkan reserved u=5")
    echo "  vulkan reserved: u=1 [4, 1, 1], u=5 [12, 5, 5] byte-exact"
    vctx.destroyShaderModule(sm)
    vctx.destroyDescriptorPool(pool)
    vctx.destroyDescriptorSetLayout(setLayout)
    oBuf.dealloc()
    uBuf.dealloc()

when isMainModule:
  runTest()
  echo "t_hlsl_reserved_keywords: all checks passed"
