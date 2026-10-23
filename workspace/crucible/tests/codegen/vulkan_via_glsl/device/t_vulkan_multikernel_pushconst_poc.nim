##
## Vulkan: multi-kernel push-constant guard.
## A multi-kernel source unions every kernel's by-value scalars into one file-scope
## KernelParams block, so the engine rejects it loudly at ingest.
## One kernel per source is the supported pattern, pointer-only multi-kernel sources are unaffected.
## Single-kernel by-value scalars are emitted as push constants (no function params).
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_multikernel_pushconst_poc.nim

import std/[os, osproc, strformat, strutils]
import workspace/crucible

# Two kernels with different scalar signatures in ONE source: the codegen
# unions both signatures into one KernelParams block (val, a, b), which is
# the configuration the guard rejects.
const codeMulti = glsl:
  proc kernelA(output: ptr UncheckedArray[uint32];
               val: uint32) {.global.} =
    output[0] = val
  proc kernelB(output: ptr UncheckedArray[uint32];
               a: uint32; b: uint32) {.global.} =
    output[0] = a + b

# Single-kernel control: kernelB alone unions only its own params, so
# packing from offset 0 is correct there.
const codeSingle = glsl:
  proc kernelB(output: ptr UncheckedArray[uint32];
               a: uint32; b: uint32) {.global.} =
    output[0] = a + b

# Single-kernel by-value scalar: emitted as a push constant block with no
# function parameters (covers the contract from t_vulkan_kernel_pushconst).
const codeWithVal = glsl:
  proc kernelWithVal(output: ptr UncheckedArray[uint32], val: uint32) {.global.} =
    output[0] = val

proc childIngestMulti() =
  ## Child mode: ingesting a multi-kernel scalar source must quit loudly.
  var engine = newEngine(runVulkan)
  engine.ingest(codeMulti)   # unreachable: ingest quits here

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo "=== Vulkan: multi-kernel push-constant guard ===\n"

  block:  # supported pattern: single-kernel kernelB(3, 4) gives 7
    var engine = newEngine(runVulkan)
    engine.ingest(codeSingle)
    var gpuOut: array[1, uint32] = [0xDEADBEEF'u32]
    let oDev = engine.allocDevice(uint32, 1)
    engine.copyToDeviceFromHost(oDev, addr gpuOut[0], 1)
    engine.launchKernel("kernelB", oDev, (3'u32, 4'u32,))
    engine.copyToHostFromDevice(addr gpuOut[0], oDev, 1)
    echo &"  single-kernel kernelB(3, 4): got {gpuOut[0]}, expected 7"
    doAssert gpuOut[0] == 7

  block:  # multi-kernel + scalars quits loudly, verified in a child process
    let (outp, code) = execCmdEx(getAppFilename() & " --child")
    echo &"  child exit code: {code}"
    doAssert code != 0, "multi-kernel scalar ingest must quit loudly"
    doAssert "multi-kernel source with scalar params" in outp,
             "guard message missing: " & outp

  block:  # single-kernel by-value scalar becomes a push constant block
    doAssert codeWithVal.contains("push_constant"),
             &"Expected push_constant block in:\n{codeWithVal}"
    doAssert codeWithVal.contains("void kernelWithVal()"),
             &"Expected void kernelWithVal() in:\n{codeWithVal}"
    doAssert codeWithVal.contains("val"),
             &"Expected `val` in push constants or a storage-buffer descriptor in:\n{codeWithVal}"
    echo "  single-kernel kernelWithVal: push_constant block, no function params"

  echo "\n  OK: multi-kernel scalar sources rejected loudly"
  echo "  (single-kernel scalars unaffected)"

when isMainModule:
  if paramCount() > 0 and paramStr(1) == "--child":
    childIngestMulti()
  else:
    runTest()
