## Vulkan: descriptor-route execution of the real-bundle-emitted HLSL kernels on MoltenVK.
##
## Test that the `hlsl:` macro's kernels execute on the Vulkan device through the descriptor route:
##
## - set layout, pool, set, descriptor writes in parameter order, pipeline, dispatch
## - add kernel: readback [11, 22] with push scalar z = 0, [12, 23] with z = 1
## - coordinate kernel: readback values derived from the dispatch thread index
## - add readback is byte-identical to the same arithmetic through the Metal engine
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_execute.nim

import std/[os, strutils]
import workspace/crucible
import workspace/crucible/src/runtime/backends/vulkan_runtime
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const addHlsl = hlsl:
  proc addKernel(output, a, b: ptr UncheckedArray[uint32], z: uint32) {.global.} =
    output[0] = a[0] + b[0] + z
    output[1] = a[1] + b[1] + z

const coordHlsl = hlsl:
  proc coordKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let i = thread_position_in_grid.x
    output[i] = i.uint32 + 3

const addMsl = metal:
  proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]

proc assertU32(got: seq[uint32], want: openArray[uint32], label: string) =
  ## Byte-exact u32 slot comparison against a known-good literal
  doAssert got.len == want.len, label & ": " & $got.len & " slots, want " & $want.len
  for i in 0 ..< want.len:
    doAssert got[i] == want[i],
      label & ": slot " & $i & " got " & $got[i] & " want " & $want[i]

proc runTest() =
  # Bindings take buffer parameter order, the push scalar z is no buffer resource.
  # `output` is written, `a` and `b` are read-only.
  let addBindings: seq[BindingExpectation] =
    @[("output", 0, false), ("a", 1, true), ("b", 2, true)]
  let addWords = compileCheck("add_execute", addHlsl, addBindings)
  let coordBindings: seq[BindingExpectation] = @[("output", 0, false)]
  let coordWords = compileCheck("coord_execute", coordHlsl, coordBindings)
  echo "  glslang + spirv-val ok, add: ", addWords.len, " words, coord: ", coordWords.len, " words"

  var addReadback: seq[uint32]
  block: # Vulkan descriptor route: set layout, pool, set, writes, pipeline, dispatch
    var vctx = newVulkanContext()
    var a: array[2, uint32] = [10'u32, 20'u32]
    var b: array[2, uint32] = [1'u32, 2'u32]
    var aBuf = vctx.allocBuffer(sizeof(a))
    var bBuf = vctx.allocBuffer(sizeof(b))
    var oBuf = vctx.allocBuffer(sizeof(a))
    vctx.writeBuffer(aBuf, addr a[0], sizeof(a))
    vctx.writeBuffer(bBuf, addr b[0], sizeof(b))

    # One binding per buffer parameter, numbered by buffer-parameter ordinal:
    # the writes enumerate output, a, b in that order, the push scalar z takes no binding
    let setLayout = vctx.createDescriptorSetLayout(3)
    let pool = vctx.createDescriptorPool(3)
    let descSet = vctx.allocateDescriptorSet(pool, setLayout)
    vctx.updateDescriptorSet(descSet, [oBuf, aBuf, bBuf], 3)

    var sm = vctx.createShaderModule(addWords)
    var pipeline = vctx.createPipeline(sm, "main", 4, setLayout)

    var z = 0'u32
    vctx.runKernel(pipeline, [256'u32, 1, 1], [256'u32, 1, 1], z.addr, 4, descSet)
    addReadback = readBuffer[uint32](vctx, oBuf)
    assertU32(addReadback, [11'u32, 22'u32], "vulkan add z=0")
    z = 1'u32
    vctx.runKernel(pipeline, [256'u32, 1, 1], [256'u32, 1, 1], z.addr, 4, descSet)
    assertU32(readBuffer[uint32](vctx, oBuf), [12'u32, 23'u32], "vulkan add z=1")
    echo "  vulkan add: z=0 [11, 22], z=1 [12, 23] byte-exact"
    vctx.destroyShaderModule(sm)

    block: # SV_DispatchThreadID route: output[i] = i + 3 over the dispatch thread index
      var cBuf = vctx.allocBuffer(256 * sizeof(uint32))
      let coordLayout = vctx.createDescriptorSetLayout(1)
      let coordPool = vctx.createDescriptorPool(1)
      let coordSet = vctx.allocateDescriptorSet(coordPool, coordLayout)
      vctx.updateDescriptorSet(coordSet, [cBuf], 1)
      var csm = vctx.createShaderModule(coordWords)
      var cpipeline = vctx.createPipeline(csm, "main", 0, coordLayout)
      vctx.runKernel(cpipeline, [256'u32, 1, 1], [256'u32, 1, 1], nil, 0, coordSet)
      var want: array[256, uint32]
      for i in 0 ..< 256:
        want[i] = uint32(i) + 3
      assertU32(readBuffer[uint32](vctx, cBuf), want, "vulkan coord")
      echo "  vulkan coord: output[i] = i + 3 byte-exact over 256 threads"
      vctx.destroyShaderModule(csm)
      vctx.destroyDescriptorPool(coordPool)
      vctx.destroyDescriptorSetLayout(coordLayout)
      cBuf.dealloc()

    vctx.destroyDescriptorPool(pool)
    vctx.destroyDescriptorSetLayout(setLayout)
    aBuf.dealloc()
    bBuf.dealloc()
    oBuf.dealloc()

  block: # Metal comparison: same arithmetic, byte-identical readback
    var engine = bkMetal.newEngine()
    engine.ingest(addMsl)
    var a: array[2, uint32] = [10'u32, 20'u32]
    var b: array[2, uint32] = [1'u32, 2'u32]
    let aDev = engine.allocDevice(uint32, a.len)
    let bDev = engine.allocDevice(uint32, b.len)
    let oDev = engine.allocDevice(uint32, 2)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
    engine.launchKernel("addKernel", oDev, (aDev, bDev))
    var out32: array[2, uint32]
    engine.copyToHostFromDevice(addr out32[0], oDev, 2)
    doAssert out32[0] == addReadback[0] and out32[1] == addReadback[1],
      "Metal readback [" & $out32[0] & ", " & $out32[1] &
      "] differs from the Vulkan descriptor-route readback [" &
      $addReadback[0] & ", " & $addReadback[1] & "]"
    echo "  metal add: [", out32[0], ", ", out32[1], "] byte-identical to the vulkan readback"

when isMainModule:
  runTest()
  echo "t_hlsl_execute: all checks passed"
