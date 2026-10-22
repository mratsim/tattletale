##
## Vulkan: by-value (scalar) kernel params as push constants, anti-regression.
## By-value scalars pack into the push-constant range, only pointer args
## bind as SSBOs, the output stays at binding 0.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/serialization/t_vulkan_scalar_args.nim

import std/strformat
import workspace/crucible

# One kernel per source: the codegen emits a single file-scope KernelParams
# push-constant block per source = the union of every kernel's by-value params.
# Mixing kernels with different scalar signatures in one source misaligns the
# block (kernel 2's params land after kernel 1's) — a codegen contract caveat;
# one kernel per ingest is the supported pattern.
const code1 = vulkan:
  proc kernelWithVal(output: ptr UncheckedArray[uint32];
                     val: uint32) {.global.} =
    output[0] = val

const code2 = vulkan:
  proc kernelWithTwoVals(output: ptr UncheckedArray[uint32];
                         a: uint32; b: uint32) {.global.} =
    output[0] = a + b

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo "=== Vulkan: scalar by-value param via push constants ===\n"

  var engine = bkVulkan.newEngine()
  engine.ingest(code1)

  let oDev = engine.allocDevice(uint32, 1)
  # Sentinel: with the misbinding regression the kernel never writes this
  # buffer and the runtime reads it back unchanged — silent wrong result.
  var gpuOut: array[1, uint32] = [0xDEADBEEF'u32]
  engine.copyToDeviceFromHost(oDev, addr gpuOut[0], 1)
  engine.launchKernel("kernelWithVal", oDev, (7'u32,))
  engine.copyToHostFromDevice(addr gpuOut[0], oDev, 1)
  doAssert gpuOut[0] == 7, &"push-constant scalar: got {gpuOut[0]:#x}, expected 7"

  echo "  OK — scalar by-value param via push constants"

  # Re-ingest (RAII field replacement) + two scalars: proves packing order
  # matches the KernelParams block member order.
  engine.ingest(code2)
  engine.copyToDeviceFromHost(oDev, addr gpuOut[0], 1)
  engine.launchKernel("kernelWithTwoVals", oDev, (3'u32, 4'u32,))
  engine.copyToHostFromDevice(addr gpuOut[0], oDev, 1)
  doAssert gpuOut[0] == 7, &"two push-constant scalars: got {gpuOut[0]}, expected 7"

  echo "  OK — two scalar by-value params (packing order, re-ingest)"

when isMainModule:
  runTest()
