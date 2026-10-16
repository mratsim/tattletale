# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: multi-kernel push-constant guard.
## A multi-kernel source unions every kernel's by-value scalars into one file-scope
## KernelParams block, so the engine rejects it loudly at ingest.
## One kernel per source is the supported pattern, pointer-only multi-kernel sources are unaffected.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_multikernel_pushconst_poc.nim

import std/[os, osproc, strformat, strutils]
import workspace/crucible

# Two kernels with different scalar signatures in ONE source: the codegen
# unions both signatures into one KernelParams block (val, a, b), which is
# the configuration the guard rejects.
const codeMulti = vulkan:
  proc kernelA(output: ptr UncheckedArray[uint32];
               val: uint32) {.global.} =
    output[0] = val
  proc kernelB(output: ptr UncheckedArray[uint32];
               a: uint32; b: uint32) {.global.} =
    output[0] = a + b

# Single-kernel control: kernelB alone unions only its own params, so
# packing from offset 0 is correct there.
const codeSingle = vulkan:
  proc kernelB(output: ptr UncheckedArray[uint32];
               a: uint32; b: uint32) {.global.} =
    output[0] = a + b

proc childIngestMulti() =
  ## Child mode: ingesting a multi-kernel scalar source must quit loudly.
  var engine = bkVulkan.init()
  engine.ingest(codeMulti)   # unreachable: ingest quits here

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo "=== Vulkan: multi-kernel push-constant guard ===\n"

  block:  # supported pattern: single-kernel kernelB(3, 4) gives 7
    var engine = bkVulkan.init()
    engine.ingest(codeSingle)
    var gpuOut: array[1, uint32] = [0xDEADBEEF'u32]
    let oDev = engine.allocDevice(sizeof(uint32))
    engine.copyToDeviceFromHost(oDev, addr gpuOut[0], sizeof(uint32))
    engine.launchKernel("kernelB", oDev, (3'u32, 4'u32,))
    engine.copyToHostFromDevice(addr gpuOut[0], oDev, sizeof(uint32))
    engine.freeDevice(oDev)
    echo &"  single-kernel kernelB(3, 4): got {gpuOut[0]}, expected 7"
    doAssert gpuOut[0] == 7

  block:  # multi-kernel + scalars quits loudly, verified in a child process
    let (outp, code) = execCmdEx(getAppFilename() & " --child")
    echo &"  child exit code: {code}"
    doAssert code != 0, "multi-kernel scalar ingest must quit loudly"
    doAssert "multi-kernel source with scalar params" in outp,
             "guard message missing: " & outp

  echo "\n  OK: multi-kernel scalar sources rejected loudly"
  echo "  (single-kernel scalars unaffected)"

when isMainModule:
  if paramCount() > 0 and paramStr(1) == "--child":
    childIngestMulti()
  else:
    runTest()
