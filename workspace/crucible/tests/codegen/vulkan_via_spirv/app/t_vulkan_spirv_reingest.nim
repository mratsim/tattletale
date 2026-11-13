# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan: a failed SPIR-V ingest leaves the live module fully launchable
##
## Run:
##   nim c -r -d:release --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_vulkan_spirv_reingest.nim

import workspace/crucible

const multiEntry = spirv:
  proc bumpKernel(outBuf: ptr UncheckedArray[int32]) {.global.} =
    outBuf[0] = 11
  proc twinKernel(outBuf: ptr UncheckedArray[int32]) {.global.} =
    outBuf[0] = 22

proc runTest() =
  var engine = newEngine(runVulkan)
  engine.ingest(multiEntry)
  let oDev = engine.allocDevice(int32, 1)
  var got: array[1, int32]
  engine.launchKernel("bumpKernel", oDev, ())
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 11
  engine.launchKernel("twinKernel", oDev, ())
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 22

  block:
    var raised = false
    try:
      engine.ingest(@[1'u32, 2, 3, 4, 5])
    except AssertionDefect:
      raised = true
    doAssert raised, "a module without the SPIR-V magic must not ingest"

  # Re-ingesting the live module dedupes, its metadata must be intact
  engine.ingest(multiEntry)
  engine.launchKernel("twinKernel", oDev, ())
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 22, "the non-default entrypoint still launches after a failed ingest"

when isMainModule:
  runTest()
  echo "[OK] Vulkan: a failed SPIR-V ingest leaves the live module launchable"
