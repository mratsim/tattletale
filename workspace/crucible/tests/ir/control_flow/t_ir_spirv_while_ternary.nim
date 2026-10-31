## Hand-built CIRv2 through the SPIR-V structured emitter, spirv-val checked.
##
## Test that hand-built while, ternary and if modules emit SPIR-V accepted by spirv-val with OpLoopMerge, OpSelectionMerge and OpSelect.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/control_flow/t_ir_spirv_while_ternary.nim

import std/[os, osproc, strutils]
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/spirv_shader_lang

proc instructionOpcodes(words: seq[uint32]): seq[uint32] =
  ## Opcode of every instruction in one SPIR-V word stream.
  ##
  ## Each instruction starts with a word packing (wordCount shl 16) or opcode.
  ## The first five words are the module header, so the scan begins at word 5
  ## and collects one opcode per instruction.
  var i = 5
  while i < words.len:
    let wc = int(words[i] shr 16)
    doAssert wc >= 1, "invalid instruction word count at word " & $i
    result.add words[i] and 0xFFFF'u32
    i += wc

proc countOpcode(ops: seq[uint32], op: uint32): int =
  ## Occurrences of one opcode in a collected opcode stream.
  for o in ops:
    if o == op:
      result.inc

const spirvWords = block:
  let u32 = tUint(32)
  let outT = tAptr(caDevice, u32)

  # kernel: while i < 3, y[i] = (i < 1 ? 100 : i), an if adds 10 to i early so both structured constructs emit
  let kern = procD("whileTernaryKernel", @[param("y", outT)], tVoid(), @[
    varD("i", u32, intLit(0, u32)),
    whileS(ltE(sym("i", u32), intLit(3, u32)), @[
      asgnS(patE(sym("y", outT), sym("i", u32)),
        ternaryE(u32, ltE(sym("i", u32), intLit(1, u32)),
          intLit(100, u32), sym("i", u32))),
      iteS(leE(sym("i", u32), intLit(1, u32)), @[
        asgnS(sym("i", u32),
          addE(u32, sym("i", u32), intLit(10, u32))),
      ], @[]),
      asgnS(sym("i", u32),
        addE(u32, sym("i", u32), intLit(1, u32))),
    ]),
    retS(),
  ])
  kern.isKernel = true

  let m = moduleD("probe", @[], @[kern])
  let (words, err) = printSpirvModule(m)

  # emitter checks on the hand-built module
  doAssert err.len == 0, "SPIR-V emitter refused the module: " & err
  doAssert words.len > 0, "SPIR-V emitter produced no words"
  doAssert words[0] == 0x07230203'u32, "SPIR-V magic number missing"

  # structured control-flow opcodes must appear, each doAssert names its opcode
  # OpSelect=169, OpLoopMerge=246, OpSelectionMerge=247, OpBranchConditional=250
  let ops = instructionOpcodes(words)
  doAssert countOpcode(ops, 169) >= 1,
    "the ternary must emit OpSelect, got " & $ops
  doAssert countOpcode(ops, 246) >= 1,
    "the conditioned loop must emit OpLoopMerge, got " & $ops
  doAssert countOpcode(ops, 247) >= 1,
    "the if must emit OpSelectionMerge, got " & $ops
  doAssert countOpcode(ops, 250) >= 2,
    "the loop header and the if must each emit OpBranchConditional, got " & $ops
  words

const ternCondWords = block:
  let u32 = tUint(32)
  let outT = tAptr(caDevice, u32)

  # kernel: while ((i < 2) ? (i < 3) : false), i += 1
  # for a ternary loop condition the emitter emits OpSelect before OpLoopMerge in the header block
  let ternKern = procD("ternCondKernel", @[param("y", outT)], tVoid(), @[
    varD("i", u32, intLit(0, u32)),
    whileS(ternaryE(tBool(),
      ltE(sym("i", u32), intLit(2, u32)),
      ltE(sym("i", u32), intLit(3, u32)),
      falseE()), @[
      asgnS(sym("i", u32), addE(u32, sym("i", u32), intLit(1, u32))),
    ]),
    retS(),
  ])
  ternKern.isKernel = true

  let tm = moduleD("probeTernCond", @[], @[ternKern])
  let (twords, terr) = printSpirvModule(tm)

  # emitter checks on the hand-built module
  doAssert terr.len == 0, "SPIR-V emitter refused the ternary-condition module: " & terr
  doAssert twords.len > 0, "SPIR-V emitter produced no words for the ternary-condition module"
  doAssert twords[0] == 0x07230203'u32, "SPIR-V magic number missing"

  # OpSelect=169 must appear in the loop header block
  # OpLoopMerge=246 and OpBranchConditional=250 end the header block
  let tops = instructionOpcodes(twords)
  doAssert countOpcode(tops, 169) >= 1,
    "the ternary condition must emit OpSelect, got " & $tops
  doAssert countOpcode(tops, 246) >= 1,
    "the conditioned loop must emit OpLoopMerge, got " & $tops
  doAssert countOpcode(tops, 250) >= 1,
    "the loop header must emit OpBranchConditional, got " & $tops
  twords

proc runTest() =
  block validation:
    let spirvVal = findExe("spirv-val")
    doAssert spirvVal.len > 0,
      "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
    let spirvDis = findExe("spirv-dis")

    let outDir = "build/tests"
    createDir(outDir)
    let spvPath = outDir / "t_ir_spirv_while_ternary.spv"
    let ternSpvPath = outDir / "t_ir_spirv_while_ternary_cond.spv"
    # defer statements close both files, so the subprocesses read complete modules
    block:
      var f = open(spvPath, fmWrite)
      defer: f.close()
      discard f.writeBuffer(addr spirvWords[0], spirvWords.len * 4)
    block:
      var f = open(ternSpvPath, fmWrite)
      defer: f.close()
      discard f.writeBuffer(addr ternCondWords[0], ternCondWords.len * 4)

    for path in [spvPath, ternSpvPath]:
      let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & path)
      if valCode != 0:
        var msg = "spirv-val rejected the emitted module:\n" & valOut
        if spirvDis.len > 0:
          let (disOut, _) = execCmdEx(spirvDis & " " & path)
          msg.add "\nspirv-dis dump:\n" & disOut
        doAssert false, msg

when isMainModule:
  runTest()
