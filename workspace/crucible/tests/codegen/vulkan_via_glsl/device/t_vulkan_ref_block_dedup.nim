## Vulkan (GLSL): buffer-reference block registration dedup
##
## Test that pointer surfaces sharing one pointee register one block entry, in first-surface order
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_ref_block_dedup.nim

import std/[strformat, tables]
import workspace/crucible
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_lowering_glsl

const kernelCode = glsl:
  proc refDedup(outP: ptr UncheckedArray[uint32], sums: ptr UncheckedArray[float32]) {.global.} =
    # uint32 pointee surfaces: the outP parameter, the cast targets,
    # words and bytes locals, one block serves them all
    let words = cast[ptr UncheckedArray[uint32]](outP)
    let bytes = cast[ptr UncheckedArray[uint8]](outP)
    words[1] = words[0] + 1'u32
    sums[0] = float32(words[0])
    bytes[3] = uint8(words[1])
    outP[2] = cast[uint32](sums[0])

proc buildRefBlockModule(): CirModule =
  ## Hand-built CIRv2 module: one kernel whose uint32 pointee has three surfaces, ahead of float32 and uint8.
  let u32 = tUint(32)
  let f32 = tFloat(32)
  let u8 = tUint(8)
  let i32 = tInt(32)
  let outTy = tAptr(caDevice, u32)
  let sumTy = tAptr(caDevice, f32)
  let wordsTy = tAptr(caDevice, u32)
  let bytesTy = tAptr(caDevice, u8)
  let kernel = procD("refDedup", @[param("outP", outTy), param("sums", sumTy)],
    tVoid(), @[
      varD("words", wordsTy, castE(wordsTy, sym("outP", outTy))),
      varD("bytes", bytesTy, castE(bytesTy, sym("outP", outTy))),
      asgnS(patE(sym("words", wordsTy), intLit(1, i32)),
        atE(sym("words", wordsTy), intLit(0, i32))),
      asgnS(patE(sym("bytes", bytesTy), intLit(3, i32)),
        atE(sym("words", wordsTy), intLit(0, i32))),
    ])
  kernel.isKernel = true
  moduleD("refBlockDedup", @[], @[kernel])

proc runRefBlockPass(m: CirModule) =
  ## Runs the registered ref-block pass alone over the module.
  var reg = PassRegistry.new()
  registerGlslRefBlockPass(reg)
  reg.runPasses(m)

proc runTest() =
  block: # pass output: one entry per distinct pointee, first-surface order
    let m = buildRefBlockModule()
    m.runRefBlockPass()
    # registration order: params first (uint32, float32), then the body walk
    # (uint8). All three uint32 surfaces dedup into the first entry
    let expected = ["Ref_uint", "Ref_float", "Ref_uint8_t"]
    var i = 0
    for name in m.refBlocks.keys:
      doAssert i < expected.len and name == expected[i],
        "block " & $i & ": " & name & " != " & expected[i]
      inc i
    doAssert i == expected.len,
      "refBlocks holds " & $i & " entries, expected " & $expected.len
  block:
    var engine = newEngine(runVulkan)
    engine.ingest(kernelCode)
    let oDev = engine.allocDevice(uint32, 3)
    let sDev = engine.allocDevice(float32, 1)
    var input: array[3, uint32] = [7'u32, 0'u32, 0'u32]
    engine.copyToDeviceFromHost(oDev, addr input[0], 3)
    engine.launchKernel("refDedup", oDev, (sDev,))
    var res: array[3, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 3)
    doAssert res[0] == 0x08000007'u32,
      &"byte write landed in the top byte: {res[0]:#x} != 0x08000007"
    doAssert res[1] == 8'u32, &"alias write through words: {res[1]} != 8"
    doAssert res[2] == 0x40E00000'u32,
      &"float pointee readback as bits: {res[2]:#x} != 0x40E00000"
    echo "  OK — one block per distinct pointee, first-surface order, readback matches"

when isMainModule:
  runTest()
