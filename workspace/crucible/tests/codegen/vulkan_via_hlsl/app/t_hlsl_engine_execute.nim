## Vulkan: descriptor-route execution of `hlsl:` kernels through the engine.
##
## Test that the engine ingests an HLSL source, detects the descriptor route
## from the compiled SPIR-V, and executes its kernels with exact readback:
##
## - add kernel: output[i] = a[i] + b[i] over 256 elements, exact readback
## - coordinate kernel: output[i] = i + 3 over the dispatch thread index
## - both launched via engine.launchKernel, buffers bound as storage-buffer
##   descriptors in parameter order
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_engine_execute.nim

import workspace/crucible

const addHlsl = hlsl:
  proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
    let i = thread_position_in_grid.x
    output[i] = a[i] + b[i]

const coordHlsl = hlsl:
  proc coordKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let i = thread_position_in_grid.x
    output[i] = i.uint32 + 3

proc assertU32(got: openArray[uint32], want: openArray[uint32], label: string) =
  ## Byte-exact u32 slot comparison against a known-good literal
  doAssert got.len == want.len, label & ": " & $got.len & " slots, want " & $want.len
  for i in 0 ..< want.len:
    doAssert got[i] == want[i],
      label & ": slot " & $i & " got " & $got[i] & " want " & $want[i]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  const N = 256

  block: # addKernel: two buffers read, one written, descriptors in parameter order
    var engine = newEngine(runVulkan)
    engine.ingest(addHlsl)
    var a: array[N, uint32]
    var b: array[N, uint32]
    for i in 0 ..< N:
      a[i] = uint32(i)
      b[i] = uint32(N - i)
    let aDev = engine.allocDevice(uint32, N)
    let bDev = engine.allocDevice(uint32, N)
    let oDev = engine.allocDevice(uint32, N)
    engine.copyToDeviceFromHost(aDev, addr a[0], N)
    engine.copyToDeviceFromHost(bDev, addr b[0], N)

    engine.launchKernel("addKernel", oDev, (aDev, bDev))

    var got: array[N, uint32]
    engine.copyToHostFromDevice(addr got[0], oDev, N)
    var want: array[N, uint32]
    for i in 0 ..< N:
      want[i] = uint32(i) + uint32(N - i)
    assertU32(got, want, "vulkan add")
    echo "  vulkan add: a[i] + b[i] = N byte-exact over ", N, " threads"

  block: # coordKernel: output[i] = i + 3 over the dispatch thread index
    var engine = newEngine(runVulkan)
    engine.ingest(coordHlsl)
    let oDev = engine.allocDevice(uint32, N)

    engine.launchKernel("coordKernel", oDev, ())

    var got: array[N, uint32]
    engine.copyToHostFromDevice(addr got[0], oDev, N)
    var want: array[N, uint32]
    for i in 0 ..< N:
      want[i] = uint32(i) + 3
    assertU32(got, want, "vulkan coord")
    echo "  vulkan coord: output[i] = i + 3 byte-exact over ", N, " threads"

when isMainModule:
  runTest()
  echo "t_hlsl_engine_execute: all checks passed"
