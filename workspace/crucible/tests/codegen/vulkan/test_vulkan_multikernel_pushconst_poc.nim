## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: multi-kernel push-constant guard.
##
## A multi-kernel `vulkan:` source unions every kernel's by-value scalar
## params into one file-scope `layout(push_constant) uniform KernelParams`
## block. The runtime packs only the invoked kernel's scalars contiguously
## from offset 0, so a kernel whose params are not first in the union reads
## misaligned offsets and returns silently wrong results (reproduced before
## the guard: kernelB(3, 4) gave 4, expected 7).
##
## The engine enforces the documented contract (AGENTS.md: one kernel per
## source when using scalars) loudly at ingest. Pointer-only multi-kernel
## sources are unaffected (test_vulkan_vec10_multi_kernel).
##
## This test runs the supported single-kernel pattern, then spawns a child
## process that ingests a multi-kernel scalar source and asserts the ingest
## quits with the guard message.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip_p2 \
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
    engine.run("kernelB", gpuOut, (3'u32, 4'u32,))
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
