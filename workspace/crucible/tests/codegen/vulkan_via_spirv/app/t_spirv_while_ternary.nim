## SPIR-V: structured control flow from the `spirv:` macro, launched and read back.
##
## Test that conditioned loops, ternaries and branches emit SPIR-V
## accepted by spirv-val and computing the expected results on the device:
##
## - a `while` loop with a ternary body and an `if` branch
## - a loop whose condition is itself a ternary: the `OpSelect` sits inside
##   the header block, before the merge and branch instructions that end it
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_while_ternary.nim

import std/[os, osproc]
import workspace/crucible

const whileTernarySpv = spirv:
  proc whileTernaryKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    var i: uint32 = 0'u32
    while i < 3'u32:
      output[i] = if i < 1'u32: 100'u32 else: i
      i = i + 1'u32
    var j: uint32 = 1'u32
    if j <= 1'u32:
      output[3] = 7'u32
    else:
      output[3] = 9'u32

const ternCondSpv = spirv:
  proc ternCondKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    var i: uint32 = 0'u32
    while (if i < 2'u32: i < 3'u32 else: false):
      output[i] = i
      i = i + 1'u32
    output[2] = i

proc validateSpirv(words: seq[uint32], path: string) =
  ## Writes one module and requires spirv-val to accept it under vulkan1.4,
  ## the structured control-flow rules included.
  doAssert words.len > 0 and words[0] == 0x07230203'u32, "no SPIR-V magic in " & path
  var f = open(path, fmWrite)
  discard f.writeBuffer(addr words[0], words.len * 4)
  f.close()
  let (outp, code) = execCmdEx("spirv-val --target-env vulkan1.4 " & path)
  doAssert code == 0, "spirv-val rejected " & path & ":\n" & outp

proc runTest() =
  createDir("build/tests")
  validateSpirv(whileTernarySpv, "build/tests/t_spirv_while_ternary.spv")
  validateSpirv(ternCondSpv, "build/tests/t_spirv_tern_cond.spv")

  var engine = newEngine(runVulkan)
  engine.ingest(whileTernarySpv)
  let oDev = engine.allocDevice(uint32, 4)
  engine.launchKernel("whileTernaryKernel", oDev, ())
  var res: array[4, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 4)
  doAssert res[0] == 100'u32, "slot 0 takes the ternary then-branch, got " & $res[0]
  doAssert res[1] == 1'u32 and res[2] == 2'u32,
    "slots 1 and 2 take the ternary else-branch, got " & $res[1] & ", " & $res[2]
  doAssert res[3] == 7'u32, "the if branch takes the then body, got " & $res[3]

  engine.ingest(ternCondSpv)
  let o2 = engine.allocDevice(uint32, 3)
  engine.launchKernel("ternCondKernel", o2, ())
  var res2: array[3, uint32]
  engine.copyToHostFromDevice(addr res2[0], o2, 3)
  doAssert res2[0] == 0'u32 and res2[1] == 1'u32,
    "the ternary-condition loop runs its body once per admitted iteration, got " & $res2
  doAssert res2[2] == 2'u32,
    "the ternary-condition loop exits when the ternary yields false, got " & $res2[2]
  echo "  OK - t_spirv_while_ternary"

when isMainModule:
  runTest()
