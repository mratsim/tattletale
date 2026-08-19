## WebGPU: cross-vocabulary coordinate + barrier kernel (GEMM-idiom slice).
##
## A single kernel written in foreign idioms runs on the WebGPU backend:
##   - CUDA idiom `blockIdx.x + threadIdx.x` (workgroup + local position).
##     `blockDim` is deliberately absent: `threads_per_threadgroup` has
##     no WGSL builtin and is a compile error on this backend.
##   - `syncthreads()` (CUDA barrier alias) -> `workgroupBarrier()`
##   - `thread_index_in_threadgroup` -> native `local_invocation_index`,
##     injected as `@builtin(local_invocation_index)` when referenced
## The host compares the global flat id and the flat local index.
## The emitted WGSL must never contain `@builtin(workgroup_size)`, which
## is not a WGSL builtin: only the `@workgroup_size` attribute exists.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_cross_vocabulary.nim

import std/[strutils, unittest]
import workspace/crucible

const kernel2d = webgpu:
  proc crossVocab2d(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    # CUDA idiom in a WebGPU kernel. With grid (2,4) × blk (4,2), the flat
    # global id is gy * 8 + gx.
    let gx = blockIdx.x * 4'u32 + threadIdx.x
    let gy = blockIdx.y * 2'u32 + threadIdx.y
    C[gy * 8'u32 + gx] = gy * 8'u32 + gx
    # canonical flat local index -> native local_invocation_index
    C[64'u32 + thread_index_in_threadgroup] = thread_index_in_threadgroup
    syncthreads()

block:
  # `threads_per_threadgroup` is deferred on WGSL: no `workgroup_size`
  # builtin exists, so referencing it in a `webgpu:` kernel is a loud
  # compile error, never a silent wrong emission.
  static:
    doAssert not compiles(block:
      const bad = webgpu:
        proc bad(C: ptr UncheckedArray[uint32]) {.global.} =
          C[0] = threads_per_threadgroup.x
    )
  echo "  OK — threads_per_threadgroup rejected loudly on WGSL"

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel2d

  suite "WebGPU - cross-vocabulary coordinates + barrier":

    test "CUDA idiom + native flat local + barrier execute with correct values":
      var engine = bkWGSL.init()
      engine.ingest(kernel2d)
      let src = engine.getArtifact()
      check "@builtin(workgroup_size)" notin src
      check "@builtin(local_invocation_index)" in src
      check "@builtin(workgroup_id)" in src
      var res: array[64 + 8, uint32]
      engine.run<<(grid: (2, 4), blk: (4, 2))>>("crossVocab2d", res, ())
      for i in 0 ..< 64:
        check res[i] == uint32(i)
      for i in 0 ..< 8:
        check res[64 + i] == uint32(i)

when isMainModule:
  runTest()
