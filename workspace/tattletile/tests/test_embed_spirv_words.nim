# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Embedded fixtures to SPIR-V words: the layering dividend, structure-checked
##
## Claim under test: the same bridge CIRv2 module the Metal path produces feeds
## crucible's SPIR-V backend unchanged, zero tattletile lowering added.
##
## Both fixtures embed through `embedKernel(backendKind = gbSpirv)`, the words pass `spirv-val --target-env vulkan1.4`:
##
## - the SPIR-V header magic and the LocalSize 4 1 1 kernel entry
## - the one push block and the BDA pointer capability set
## - the loop shape: OpLoopMerge, phi-free, one fused value loop for axpy
## - gemm: the value loop plus the mma k loop, word-stable across two compiles
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --nimcache:nimcache/tattletile --outdir:build/tattletile workspace/tattletile/tests/test_embed_spirv_words.nim

import std/[os, osproc, strutils]

import workspace/tattletile/tattletile

const AxpyWords = embedKernel("../fixtures/vec_add_axpy.tileir", "axpy", threads = 4, backendKind = gbSpirv).words
  ## Axpy kernel, compiled from Tile IR text to SPIR-V words in the Nim VM.

const AxpyWordsAgain = embedKernel("../fixtures/vec_add_axpy.tileir", "axpy", threads = 4, backendKind = gbSpirv).words
  ## Second compile of the same fixture, for word-level stability.

const GemmWords = embedKernel("../fixtures/gemm.tileir", "gemm", threads = 4, backendKind = gbSpirv).words
  ## Gemm kernel, compiled from Tile IR text to SPIR-V words in the Nim VM.

proc disassembly(words: seq[uint32], name: string): string =
  ## spirv-val the module first: a mis-encoded word sequence must fail here,
  ## not as a driver-side opaque launch error. Returns the spirv-dis text.
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let spirvDis = findExe("spirv-dis")
  doAssert spirvDis.len > 0,
    "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tattletile"
  createDir(outDir)
  let spvPath = outDir / (name & ".spv")
  # the file closes at block exit, before the subprocesses read it
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr words[0], words.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  if valCode != 0:
    let (disOut, _) = execCmdEx(spirvDis & " " & spvPath)
    echo "spirv-dis dump:\n", disOut
    doAssert false, "spirv-val rejected " & name & ":\n" & valOut
  let (disOut, disCode) = execCmdEx(spirvDis & " " & spvPath)
  doAssert disCode == 0, "spirv-dis failed on " & name
  disOut

proc expect(dis, fragment: string, what: string) =
  doAssert fragment in dis, what & " misses:\n  " & fragment & "\nemitted:\n" & dis

proc checkAxpyStructure() =
  let dis = disassembly(AxpyWords, "test_embed_spirv_words_axpy")
  doAssert AxpyWords[0] == 0x07230203'u32, "SPIR-V magic number missing"
  # the kernel entry and the launch squadron baked from the workgroup size
  expect(dis, "OpEntryPoint GLCompute", "axpy entry")
  expect(dis, "\"axpy\"", "axpy entry name")
  expect(dis, "LocalSize 4 1 1", "axpy LocalSize")
  # one push block, the launch arguments: three BDA pointers, three i32 shape scalars
  expect(dis, "OpMemoryModel PhysicalStorageBuffer64 GLSL450", "axpy memory model")
  expect(dis, "OpCapability PhysicalStorageBufferAddresses", "axpy BDA capability")
  expect(dis, "OpExtension \"SPV_KHR_physical_storage_buffer\"", "axpy BDA extension")
  expect(dis, "OpTypeStruct %ulong %int %ulong %int %ulong %int", "axpy push block members")
  expect(dis, "OpTypePointer PushConstant", "axpy push block pointer")
  # the fused value loop, phi-free by construction
  doAssert dis.count("OpLoopMerge") == 1, "axpy emits " & $dis.count("OpLoopMerge") &
    " loops, want the one fused value loop"
  doAssert "OpPhi" notin dis, "axpy emits OpPhi, the emission is phi-free by construction"
  expect(dis, "OpFMul %float", "axpy multiply")
  expect(dis, "OpFAdd %float", "axpy add")
  expect(dis, "OpAccessChain %_ptr_PhysicalStorageBuffer_float", "axpy BDA access chain")

proc checkGemmStructure() =
  let dis = disassembly(GemmWords, "test_embed_spirv_words_gemm")
  doAssert GemmWords[0] == 0x07230203'u32, "SPIR-V magic number missing"
  expect(dis, "OpEntryPoint GLCompute", "gemm entry")
  expect(dis, "\"gemm\"", "gemm entry name")
  expect(dis, "LocalSize 4 1 1", "gemm LocalSize")
  # one push block, the launch arguments: three BDA pointers, no shape scalars
  expect(dis, "OpMemoryModel PhysicalStorageBuffer64 GLSL450", "gemm memory model")
  expect(dis, "OpTypeStruct %ulong %ulong %ulong", "gemm push block members")
  expect(dis, "OpTypePointer PushConstant", "gemm push block pointer")
  # the fused value loop plus the per-thread mma k loop, phi-free
  doAssert dis.count("OpLoopMerge") == 2, "gemm emits " & $dis.count("OpLoopMerge") &
    " loops, want the value loop plus the mma k loop"
  doAssert "OpPhi" notin dis, "gemm emits OpPhi, the emission is phi-free by construction"
  expect(dis, "OpFMul %float", "gemm multiply")
  expect(dis, "OpFAdd %float", "gemm accumulate")

proc checkStability() =
  doAssert AxpyWords == AxpyWordsAgain,
    "two compiles of the same fixture disagree on the SPIR-V words"

when isMainModule:
  checkAxpyStructure()
  checkGemmStructure()
  checkStability()
  echo "[OK] embedded fixtures compile to SPIR-V words: spirv-val accepted, " &
    "structure locked, two compiles word-identical"
