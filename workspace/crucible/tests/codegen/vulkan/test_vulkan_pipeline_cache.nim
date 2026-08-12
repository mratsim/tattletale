## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: pipeline / SPIR-V cache (ingest-once, run-many).
##
## The engine caches the fully-created pipeline by (kernel, ssboCount,
## pushConstSize) and per-kernel SPIR-V by kernel name. This test runs the
## same kernel twice with different input arrays on one engine (proves the
## cached pipeline is reused and its descriptor set rewritten), runs a
## non-entry kernel twice (per-kernel SPIR-V cache), then re-ingests a new
## source with the same kernel name and runs again (a stale cached pipeline
## would return the old result, proving cache invalidation).
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip_vkcache \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_pipeline_cache.nim

import workspace/crucible

# Pointer-only multi-kernel source (pointer-only multi-kernel is supported;
# by-value scalars would trip the multi-kernel guard, see the poc test).
const code = vulkan:
  proc addKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]
  proc subKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] - b[0]
    output[1] = a[1] - b[1]

# Replacement source for the re-ingest (cache invalidation) step: same
# kernel name as the first source, different body. A stale cached pipeline
# would return the old addition result instead of the product.
const codeMul = vulkan:
  proc addKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] * b[0]
    output[1] = a[1] * b[1]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo "=== Vulkan: pipeline / SPIR-V cache ===\n"

  var engine = bkVulkan.init()
  engine.ingest(code)

  block:  # entry kernel, run twice with different data on the same engine
    var a: array[2, uint32] = [10'u32, 20'u32]
    var b: array[2, uint32] = [1'u32, 2'u32]
    var out32: array[2, uint32]
    engine.run("addKernel", out32, (a, b))
    doAssert out32[0] == 11 and out32[1] == 22
    echo "  addKernel run 1: [10,20] + [1,2] = [", out32[0], ", ", out32[1], "]"
    a = [100'u32, 200'u32]
    b = [30'u32, 40'u32]
    engine.run("addKernel", out32, (a, b))
    doAssert out32[0] == 130 and out32[1] == 240
    echo "  addKernel run 2: [100,200] + [30,40] = [", out32[0], ", ", out32[1], "]"
    echo "  OK: cached pipeline reused, descriptor set rewritten"

  block:  # non-entry kernel, run twice (per-kernel SPIR-V cache)
    var a: array[2, uint32] = [10'u32, 20'u32]
    var b: array[2, uint32] = [1'u32, 2'u32]
    var out32: array[2, uint32]
    engine.run("subKernel", out32, (a, b))
    doAssert out32[0] == 9 and out32[1] == 18
    echo "  subKernel run 1: [10,20] - [1,2] = [", out32[0], ", ", out32[1], "]"
    a = [50'u32, 80'u32]
    b = [10'u32, 10'u32]
    engine.run("subKernel", out32, (a, b))
    doAssert out32[0] == 40 and out32[1] == 70
    echo "  subKernel run 2: [50,80] - [10,10] = [", out32[0], ", ", out32[1], "]"
    echo "  OK: non-entry kernel SPIR-V and pipeline cached"

  block:  # re-ingest a new source with the same kernel name, then run
    engine.ingest(codeMul)
    var a: array[2, uint32] = [3'u32, 4'u32]
    var b: array[2, uint32] = [5'u32, 6'u32]
    var out32: array[2, uint32]
    engine.run("addKernel", out32, (a, b))
    doAssert out32[0] == 15 and out32[1] == 24
    echo "  addKernel after re-ingest: [3,4] * [5,6] = [", out32[0], ", ", out32[1], "]"
    echo "  OK: re-ingest invalidated the caches (same kernel name, new body)"

  echo "\n  OK: ingest-once / run-many cache"
  echo "  (engine destroyed at return: cached pipelines released before ctx shutdown)"

when isMainModule:
  runTest()
