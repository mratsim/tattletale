## SPIR-V module parsing: entry point and LocalSize extraction from raw words
##
## Test that spirvSquadron decodes hand-built modules:
## the OpEntryPoint name and id, the LocalSize bound to that id,
## the OpEntryPoint count.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/t_spirv_squadron_parse.nim

import workspace/crucible/src/runtime/backends/vulkan_compile

# Hand-built per the SPIR-V spec: a 5-word header, then instructions.
# Independent truth: the words come from the spec, not from the encoder.

const opEntryPoint = 15'u32
const opExecutionMode = 16'u32
const modeLocalSize = 17'u32
const modeLocalSizeId = 18'u32

proc header(): seq[uint32] =
  @[0x07230203'u32, 0x00010300'u32, 0'u32, 4'u32, 0'u32]

proc entryPointInst(id: uint32, name: string): seq[uint32] =
  # word layout: execution model (5 = GLCompute), then the entry-point id,
  # then the NUL-terminated name padded to the word boundary
  # interface ids come after the name
  let nameWords = (name.len + 1 + 3) div 4
  result = @[(uint32(3 + nameWords) shl 16) or opEntryPoint, 5'u32, id]
  var bytes = name & '\0'
  bytes.setLen(nameWords * 4)
  for i in countup(0, bytes.len - 1, 4):
    result.add uint32(bytes[i]) or uint32(bytes[i + 1]) shl 8 or
               uint32(bytes[i + 2]) shl 16 or uint32(bytes[i + 3]) shl 24

proc localSizeInst(id: uint32, x, y, z: uint32): seq[uint32] =
  @[(6'u32 shl 16) or opExecutionMode, id, modeLocalSize, x, y, z]

proc runTest() =
  block happyPath:
    let spv = header() & entryPointInst(5, "addScalarKernel") &
              localSizeInst(5, 8, 8, 1)
    let sq = spirvSquadron(spv)
    doAssert sq.name == "addScalarKernel"
    doAssert sq.id == 5
    doAssert (sq.localSize.x, sq.localSize.y, sq.localSize.z) == (8, 8, 1)
    doAssert sq.entryPoints == 1

  block stringPadding:
    # a 1-char name carries 3 NUL/pad bytes in its one literal word
    let spv = header() & entryPointInst(2, "k")
    let sq = spirvSquadron(spv)
    doAssert sq.name == "k", "got \"" & sq.name & "\""
    doAssert sq.id == 2

  block noEntryPoint:
    let spv = header()
    let sq = spirvSquadron(spv)
    doAssert sq.name == ""
    doAssert sq.id == 0
    doAssert sq.entryPoints == 0
    doAssert (sq.localSize.x, sq.localSize.y, sq.localSize.z) == (0, 0, 0)

  block noLocalSize:
    let spv = header() & entryPointInst(5, "k")
    let sq = spirvSquadron(spv)
    doAssert sq.name == "k"
    doAssert (sq.localSize.x, sq.localSize.y, sq.localSize.z) == (0, 0, 0)

  block otherExecutionModeSkipped:
    # a LocalSizeId execution mode is skipped, the LocalSize one wins
    let spv = header() & entryPointInst(5, "k") &
              @[(3'u32 shl 16) or opExecutionMode, 5'u32, modeLocalSizeId] &
              localSizeInst(5, 4, 1, 1)
    let sq = spirvSquadron(spv)
    doAssert (sq.localSize.x, sq.localSize.y, sq.localSize.z) == (4, 1, 1)

  block localSizeForOtherIdIgnored:
    # the LocalSize pairs to the parsed entry-point id
    # a mismatched LocalSize must not leak through
    let spv = header() & entryPointInst(5, "k") &
              localSizeInst(9, 8, 8, 1)
    let sq = spirvSquadron(spv)
    doAssert sq.name == "k"
    doAssert (sq.localSize.x, sq.localSize.y, sq.localSize.z) == (0, 0, 0),
      "a LocalSize bound to another entry-point id must be ignored"

when isMainModule:
  runTest()
  echo "t_spirv_squadron_parse: all checks passed"
