## OpenCL: Vulkan-idiom `barrier()` in an OpenCL kernel, with shared
## memory semantics.
##
## The zero-arg `barrier()` (Vulkan idiom) expands to the canonical
## `threadgroup_barrier` call. The OpenCL printer lowers it to `barrier(CLK_LOCAL_MEM_FENCE)`.
## The kernel stages values in `{.shared.}` scratch (`__local`), barriers,
## then each work-item reads a slot written by a different work-item:
## the barrier is what makes the staged values visible.
## The emitted-text assertion is the effective barrier check.
## A missing barrier can pass by scheduling luck, as in test_metal_cross_vocabulary.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/opencl --nimcache:nimcache/tests/opencl \
##     workspace/crucible/tests/codegen/opencl/test_opencl_cross_vocabulary_barrier.nim

import std/strutils
import workspace/crucible

const kernelCode = opencl:
  proc crossVocabBarrierKernel(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    var scratch {.shared.}: array[8, uint32]
    let tid = thread_index_in_threadgroup
    scratch[tid] = tid * 3'u32
    barrier()   # Vulkan-idiom zero-arg barrier in an OpenCL kernel
    # Read a slot written by a different work-item of the same group.
    C[tid] = scratch[(tid + 1) mod 8]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # The zero-arg barrier must lower to the OpenCL flags barrier.
  doAssert "barrier(CLK_LOCAL_MEM_FENCE)" in kernelCode,
    "OpenCL barrier lowering missing:\n" & kernelCode

  var engine = bkOpenCL.init()
  engine.ingest(kernelCode)
  var res: array[8, uint32]
  # 2 work-groups of (4, 2). Each group stages identical values, so the
  # group count is not pinned by this test; the neighbor-slot dependency
  # is within-group.
  engine.run<<(grid: (2, 1), blk: (4, 2))>>("crossVocabBarrierKernel", res, ())
  for i in 0 ..< 8:
    let expected = uint32((i + 1) mod 8) * 3'u32
    doAssert res[i] == expected,
      "slot " & $i & ": got " & $res[i] & ", expected " & $expected &
      " (neighbor slot after barrier)"
  echo "  OK — Vulkan-idiom barrier() runs with shared-memory semantics on OpenCL"

when isMainModule:
  runTest()
