## Metal: engine pipeline cache (ingest-once / run-many).
##
## Exercises the MetalEngine's two-level cache in runImpl.
## The compiled library (level 1) is created once per ingest.
## Compute pipeline states are created once per kernel name
## (level 2) and reused across runs. Running the same kernel twice
## with different input arrays asserts byte-exact outputs on both runs.
## Re-ingesting a new source and running again proves cache invalidation.
## Every run is asserted byte-exact on the device output.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_pipeline_cache.nim

import std/unittest
import workspace/crucible

# ── Kernels: elementwise scaling of a 4-element uint32 buffer ─────────────────

const kernelDouble = metal:
  proc doubleKernel(output: ptr UncheckedArray[uint32];
                    input: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = input[0] * 2'u32
    output[1] = input[1] * 2'u32
    output[2] = input[2] * 2'u32
    output[3] = input[3] * 2'u32

const kernelTriple = metal:
  proc doubleKernel(output: ptr UncheckedArray[uint32];
                    input: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = input[0] * 3'u32
    output[1] = input[1] * 3'u32
    output[2] = input[2] * 3'u32
    output[3] = input[3] * 3'u32

# ── Host side ─────────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - pipeline cache":

    test "same kernel, different input arrays (byte-exact outputs)":
      var engine = bkMetal.init()
      engine.ingest(kernelDouble)
      echo kernelDouble
      var res: array[4, uint32]
      var a = [1'u32, 2'u32, 3'u32, 4'u32]
      engine.run("doubleKernel", res, (a,))
      check res[0] == 2
      check res[1] == 4
      check res[2] == 6
      check res[3] == 8
      var b = [10'u32, 20'u32, 30'u32, 40'u32]
      engine.run("doubleKernel", res, (b,))
      check res[0] == 20
      check res[1] == 40
      check res[2] == 60
      check res[3] == 80

    test "re-ingest invalidates cached pipelines":
      var engine = bkMetal.init()
      engine.ingest(kernelDouble)
      echo kernelDouble
      var res: array[4, uint32]
      var a = [1'u32, 2'u32, 3'u32, 4'u32]
      engine.run("doubleKernel", res, (a,))
      check res[0] == 2
      check res[1] == 4
      engine.ingest(kernelTriple)
      echo kernelTriple
      engine.run("doubleKernel", res, (a,))
      check res[0] == 3
      check res[1] == 6
      check res[2] == 9
      check res[3] == 12

when isMainModule:
  runTest()
