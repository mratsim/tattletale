# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan: a failed GLSL ingest raises and leaves the engine live
##
## Run:
##   nim c -r -d:release --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/app/t_vulkan_ingest_failure.nim

import workspace/crucible

const validGlsl = glsl:
  proc bumpKernel(outBuf: ptr UncheckedArray[float32]) {.global.} =
    outBuf[0] = 1.0'f32

const brokenGlsl = "#version 450\nvoid main() { this does not compile"

proc runTest() =
  var engine = newEngine(runVulkan)
  engine.ingest(validGlsl)
  let oDev = engine.allocDevice(float32, 1)
  engine.launchKernel("bumpKernel", oDev, ())
  var got: array[1, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 1.0'f32

  block:
    var raised = false
    try:
      engine.ingest(brokenGlsl)
    except AssertionDefect:
      raised = true
    doAssert raised, "the broken source must not ingest"

  block:
    var raised = false
    try:
      engine.ingest(brokenGlsl)
    except AssertionDefect:
      raised = true
    doAssert raised, "the broken source must raise again, not dedupe to a silent return"

  engine.ingest(validGlsl)
  engine.launchKernel("bumpKernel", oDev, ())
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 1.0'f32, "the valid module still launches after a failed ingest"

when isMainModule:
  runTest()
  echo "[OK] Vulkan: a failed ingest raises again and the valid module stays live"
