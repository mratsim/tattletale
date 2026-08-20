## Vulkan: shared-memory scratch + barrier with neighbor-slot semantics.
##
## The kernel stages values in `{.smem.}` scratch (GLSL `shared`), runs the Vulkan-idiom
## `barrier()`, then each work-item reads a slot written by a different work-item.
## The neighbor read verifies the staged values are visible after the barrier.
## The scratch is module-level because GLSL allows `shared` only at global scope.
## The emitted-text assertions pin the shared-array emission and the barrier
## lowering; a missing barrier can pass by scheduling luck on this hardware,
## as in test_metal_cross_vocabulary, so the barrier assert is the effective
## barrier check.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/vulkan --nimcache:nimcache/tests/vulkan \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_cross_vocabulary_shared.nim

import std/strutils
import workspace/crucible

const kernelCode = vulkan:
  var scratch {.smem.}: array[8, uint32]
  proc crossVocabSharedKernel(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    let tid = thread_index_in_threadgroup
    scratch[tid] = tid * 3'u32
    barrier()
    # Read a slot written by a different work-item of the same group.
    C[tid] = scratch[(tid + 1) mod 8]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # The shared array must print in the GLSL declaration position:
  # `shared uint scratch[8]`, not `shared uint [8] scratch`.
  doAssert "shared uint scratch[8]" in kernelCode,
    "Vulkan shared-array emission missing:\n" & kernelCode
  doAssert "barrier()" in kernelCode, "Vulkan barrier lowering missing:\n" & kernelCode

  var engine = bkVulkan.init()
  engine.ingest(kernelCode)
  var res: array[8, uint32]
  # 2 work-groups of (4, 2). Each group stages identical values, so the
  # group count is not pinned by this test; the neighbor-slot dependency
  # is within-group.
  engine.run<<(grid: (2, 1), blk: (4, 2))>>("crossVocabSharedKernel", res, ())
  for i in 0 ..< 8:
    let expected = uint32((i + 1) mod 8) * 3'u32
    doAssert res[i] == expected,
      "slot " & $i & ": got " & $res[i] & ", expected " & $expected &
      " (neighbor slot after barrier)"
  echo "  OK — shared-memory scratch + barrier runs with value semantics on Vulkan"

when isMainModule:
  runTest()
