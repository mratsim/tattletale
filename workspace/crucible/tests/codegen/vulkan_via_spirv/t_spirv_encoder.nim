## SPIR-V encoder core: grammar tables, word emitter, typed builder
##
## Test that the encoder emits grammar-checked tables and word-exact modules:
## - every spvInstTable row matches spirv.core.grammar.json
## - type and constant memoization reuse ids
## - finalize emits the 5-word header with the id bound
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/t_spirv_encoder.nim

import std/os
import std/strutils
import workspace/crucible/src/codegen/backends/spirv_encoder

const grammarPath = currentSourcePath().parentDir() / "fixtures" / "spirv.core.grammar.json"

proc runTest() =
  block grammarTables:
    # loud failure: the grammar fixture locks the opcode table, a missing file
    # would silently void the check
    doAssert fileExists(grammarPath),
      "SPIR-V grammar fixture missing at " & grammarPath &
      ", vendor spirv.core.grammar.json into tests/codegen/vulkan_via_spirv/fixtures/"
    let mismatches = verifyGrammar(readFile(grammarPath))
    doAssert mismatches.len == 0,
      "grammar table mismatches:\n" & mismatches.join("\n")

  block memoization:
    var b = initSpvBuilder()
    let voidTy = b.tyVoid()
    let intTy = b.tyInt(32, false)
    doAssert b.tyVoid() == voidTy, "type memoization reuses ids"
    doAssert b.tyInt(32, false) == intTy
    doAssert b.tyInt(32, true) != intTy, "signedness is part of the memo key"
    doAssert b.constUint(intTy, 7) == b.constUint(intTy, 7), "const memoization"
    doAssert b.constUint(intTy, 7) != b.constUint(intTy, 8)
    doAssert b.constUint(b.tyInt(64, false), 1'u64 shl 40) !=
             b.constUint(b.tyInt(32, false), 0), "64-bit consts take two words"

  block goldenWords:
    var b = initSpvBuilder()
    b.emitCapability(capShader)
    let words = b.module.finalize()
    doAssert words.len == 7
    doAssert words[0] == spvMagic
    doAssert words[1] == spvVersion13
    doAssert words[2] == 0'u32, "generator word"
    doAssert words[3] == 1'u32, "bound word = 1 allocated id"
    doAssert words[4] == 0'u32, "schema word"
    # OpCapability Shader: wordCount 2, opcode 17, operand 1
    doAssert words[5] == 0x00020011'u32
    doAssert words[6] == 1'u32

  block constantWidth:
    # OpConstant literal word count comes from the result type width, not the value.
    # Even a 64-bit zero carries two literal words, one per 32-bit half
    var b = initSpvBuilder()
    let f64Ty = b.tyFloat(64)
    discard b.constUint(f64Ty, 0)
    let words = b.module.finalize()
    # header (5 words) + OpTypeFloat (3 words, id 1, width 64)
    #      + OpConstant (5 words: type id 1, result id 2, two literal words)
    doAssert words.len == 13, "expected 13 words, got " & $words.len
    doAssert words[5] == 0x00030016'u32, "OpTypeFloat 64 header"
    doAssert words[8] == 0x0005002b'u32,
      "OpConstant over a 64-bit type carries two literal words, got header " &
      words[8].toHex()

when isMainModule:
  runTest()
  echo "t_spirv_encoder: all checks passed"
