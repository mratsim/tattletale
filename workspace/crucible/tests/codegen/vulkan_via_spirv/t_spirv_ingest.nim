## SPIR-V ingest validation: the engine refuses malformed direct modules
##
## Test that ingest(engine, seq[uint32]) refuses loudly: an empty module,
## a module without the SPIR-V magic, a module with two OpEntryPoints.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/t_spirv_ingest.nim

import std/[os, osproc, strutils]

# Hand-built per the SPIR-V spec: a 5-word header, then instructions

const opEntryPoint = 15'u32

proc header(): seq[uint32] =
  @[0x07230203'u32, 0x00010300'u32, 0'u32, 4'u32, 0'u32]

proc entryPointInst(id: uint32, name: string): seq[uint32] =
  let nameWords = (name.len + 1 + 3) div 4
  result = @[(uint32(3 + nameWords) shl 16) or opEntryPoint, 5'u32, id]
  var bytes = name & '\0'
  bytes.setLen(nameWords * 4)
  for i in countup(0, bytes.len - 1, 4):
    result.add uint32(bytes[i]) or uint32(bytes[i + 1]) shl 8 or
               uint32(bytes[i + 2]) shl 16 or uint32(bytes[i + 3]) shl 24

proc wordsLiteral(w: seq[uint32]): string =
  ## Formats the words as a Nim seq literal for the generated check source
  result = "@["
  for i, x in w:
    if i > 0:
      result.add ", "
    result.add $x & "'u32"
  result.add "]"

proc expectIngestRefusal(name, wantMessage, wordsExpr: string) =
  ## Compiles a check that creates the engine and ingests the words
  ## the refusal must quit(1) with the expected message
  let testDir = currentSourcePath().parentDir()
  # tests/codegen/vulkan_via_spirv -> the repo root that carries workspace/
  let projectRoot = testDir / "../../../.."
  let tmpDir = testDir / "nimcache" / "ingest_checks"
  createDir(tmpDir)
  let checkPath = tmpDir / ("check_" & name & ".nim")
  writeFile(checkPath,
    "import workspace/crucible\n" &
    "proc runTest() =\n" &
    "  var engine = newEngine(runVulkan)\n" &
    "  engine.ingest(" & wordsExpr & ")\n" &
    "  echo \"ingested\"\n" &
    "when isMainModule: runTest()\n")
  let (outp, rc) = execCmdEx("nim c -r --hints:off --warnings:off" &
    " --path:" & projectRoot.absolutePath &
    " --outdir:" & tmpDir & " --nimcache:" & tmpDir / "nimcache" &
    " " & checkPath)
  removeFile(checkPath)
  doAssert rc != 0, name & ": ingest must refuse, it accepted the module: " & outp
  doAssert wantMessage in outp,
    name & ": the refusal must name the defect, wanted \"" & wantMessage &
    "\", got: " & outp

proc runTest() =
  block emptyModule:
    expectIngestRefusal("empty", "not a SPIR-V module", "@[]")

  block badMagic:
    expectIngestRefusal("badmagic", "not a SPIR-V module",
      wordsLiteral(@[1'u32, 2'u32, 3'u32, 4'u32, 5'u32]))

  block twoEntryPoints:
    let words = header() & entryPointInst(1, "k1") & entryPointInst(2, "k2")
    expectIngestRefusal("twoentry", "exactly one OpEntryPoint", wordsLiteral(words))

when isMainModule:
  runTest()
  echo "t_spirv_ingest: all checks passed"
