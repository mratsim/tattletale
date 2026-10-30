## SPIR-V binary emission: bit and shift instruction opcodes
##
## Test that integer bit/shift operators and bool `not` emit the SPIR-V
## core-grammar opcodes and pass `spirv-val --target-env vulkan1.4`:
##
## - OpShiftRightLogical 194, OpShiftRightArithmetic 195, OpShiftLeftLogical 196
## - OpBitwiseOr 197, OpBitwiseXor 198, OpBitwiseAnd 199, OpNot 200
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_bitwise_ops.nim

import std/os
import std/osproc
import std/sets
import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/spirv_shader_lang

proc buildBitOpsModule(): CirModule =
  ## One kernel whose body exercises every integer bit/shift operator
  ## plus the bool `not`, one assignment per emitted instruction
  let i32 = tInt(32)
  let kernel = procD("bitOpsKernel", @[param("a", i32)], tVoid(), @[
    varD("sll", i32, shlE(i32, sym("a", i32), intLit(2, i32))),
    varD("sra", i32, shrE(i32, sym("a", i32), intLit(1, i32))),
    varD("srl", tUint(32), shrE(tUint(32),
      convE(tUint(32), sym("a", i32)), intLit(1, tUint(32)))),
    varD("bor", i32, bitOrE(i32, sym("a", i32), intLit(3, i32))),
    varD("bxor", i32, bitXorE(i32, sym("a", i32), intLit(3, i32))),
    varD("band", i32, bitAndE(i32, sym("a", i32), intLit(3, i32))),
    varD("bnot", i32, bitNotE(i32, sym("a", i32))),
    varD("nb", tBool(), notE(eqE(sym("a", i32), intLit(0, i32)))),
    retS(),
  ])
  kernel.isKernel = true
  moduleD("bitOps", @[], @[kernel])

const emission = static:
  printSpirvModule(buildBitOpsModule())

proc collectOpcodes(words: seq[uint32]): HashSet[uint32] =
  ## Instruction opcodes of one SPIR-V word stream. After the 5-word header
  ## each instruction opens on (wordCount << 16) | opcode:
  ## the walk reads the low 16 bits of the start word, then steps over
  ## the instruction's word count
  var i = 5
  while i < words.len:
    result.incl words[i] and 0xFFFF'u32
    i += int(words[i] shr 16)

const expected = toHashSet([194'u32, 195'u32, 196'u32, 197'u32, 198'u32,
  199'u32, 200'u32, 168'u32])
# never emit 146-149 or 202-204: other core-grammar instructions own those numbers
const wrong = toHashSet([146'u32, 147'u32, 148'u32, 149'u32,
  202'u32, 203'u32, 204'u32])

proc runTest() =
  doAssert emission.err.len == 0,
    "SPIR-V emission refused the bit-ops kernel: " & emission.err
  doAssert emission.words.len > 0 and emission.words[0] == 0x07230203'u32,
    "SPIR-V magic number missing"

  let opcodes = collectOpcodes(emission.words)
  let missing = expected - opcodes
  doAssert missing.len == 0,
    "emitted opcodes miss the bit/shift instructions " & $missing &
    ", full opcode set: " & $opcodes
  let bad = wrong * opcodes
  doAssert bad.len == 0,
    "emitted opcodes use the wrong numbers " & $bad &
    " (OpShiftLeftLogical 196, OpShiftRightLogical 194, OpShiftRightArithmetic 195, OpBitwiseOr 197, OpBitwiseXor 198, OpBitwiseAnd 199, OpNot 200)" &
    ", full opcode set: " & $opcodes

  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_bitwise_ops.spv"
  let words = emission.words
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr words[0], words.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the emitted module:\n" & valOut

when isMainModule:
  runTest()
  echo "t_spirv_bitwise_ops: bit/shift opcodes correct and spirv-valid"
