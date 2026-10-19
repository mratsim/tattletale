## Vulkan: shared-memory scratch + barrier with neighbor-slot semantics.
## Work-items stage values in `{.smem.}` scratch (GLSL `shared`, module scope),
## run `barrier()`, then each work-item reads a slot written by a different work-item.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/functions/t_vulkan_cross_vocabulary_shared.nim

import std/strutils
import workspace/crucible

const kernelCode = vulkan:
  var scratch {.smem.}: array[8, uint32]
  proc crossVocabSharedKernel(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    let tid = thread_index_in_threadgroup
    scratch[tid] = uint32(tid) * 3'u32
    barrier()
    # Read a slot written by a different work-item of the same group.
    C[tid] = scratch[(tid + 1) mod 8]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # The shared array must print in the GLSL declaration position:
  # `shared uint scratch[8]`, not `shared uint [8] scratch`.
  doAssert "shared uint scratch[8]" in kernelCode,
    "Vulkan shared-array emission missing:\n" & kernelCode
  doAssert "barrier()" in kernelCode, "Vulkan barrier lowering missing:\n" & kernelCode

  var engine = bkVulkan.newEngine()
  engine.ingest(kernelCode)
  let oDev = engine.allocDevice(uint32, 8)
  # 2 work-groups of (4, 2). Each group stages identical values, so the
  # group count is not pinned by this test; the neighbor-slot dependency
  # is within-group.
  engine.launchKernel<<(grid: (2, 1), blk: (4, 2))>>("crossVocabSharedKernel", oDev, ())
  var res: array[8, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 8)
  for i in 0 ..< 8:
    let expected = uint32((i + 1) mod 8) * 3'u32
    doAssert res[i] == expected,
      "slot " & $i & ": got " & $res[i] & ", expected " & $expected &
      " (neighbor slot after barrier)"
  echo "  OK — shared-memory scratch + barrier runs with value semantics on Vulkan"

when isMainModule:
  runTest()
