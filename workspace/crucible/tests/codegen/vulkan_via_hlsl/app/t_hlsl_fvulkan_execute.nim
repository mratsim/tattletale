## fvVulkan HLSL flavor check: CIRv2 kernels through dxc to raw SPIR-V words,
## executed on the Vulkan engine's zero-descriptor address route.
##
## Checks:
## - the axpy kernel compiles through the validated DXC route, the engine
##   ingests the raw words and dispatches the BDA address route, byte-exact
## - a threadgroups_per_grid read proves the engine-staged dispatch grid
## - the same axpy kernel through the GLSL flavor gives identical bitwise results
## - a device function receives buffer parameters as u64 address values
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_fvulkan_execute.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const axpyHlsl = hlsl(fvVulkan):
  proc kern_axpy(y, x: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global.} =
    let i = thread_position_in_grid.x
    if i < n:
      y[i] = alpha * x[i] + y[i]

const gridHlsl = hlsl(fvVulkan):
  proc kern_grid(dst: ptr UncheckedArray[int32]) {.global.} =
    dst[0] = int32(threadgroups_per_grid.x)
    dst[1] = int32(threadgroups_per_grid.y)
    dst[2] = int32(threadgroups_per_grid.z)

proc axpyAt(y, x: ptr UncheckedArray[float32], alpha: float32, i: int) =
  y[i] = alpha * x[i]

const devFnHlsl = hlsl(fvVulkan):
  proc kern_devfn(y, x: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global.} =
    let i = thread_position_in_grid.x
    if i < n:
      axpyAt(y, x, alpha, i)

const axpyGlsl = glsl:
  proc kern_axpy(y, x: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global.} =
    let i = thread_position_in_grid.x
    if i < n:
      y[i] = alpha * x[i] + y[i]

proc assertF32(got: openArray[float32], want: openArray[float32], label: string) =
  ## Byte-exact f32 slot comparison against a known-good literal
  doAssert got.len == want.len, label & ": " & $got.len & " slots, want " & $want.len
  for i in 0 ..< want.len:
    doAssert got[i] == want[i],
      label & ": slot " & $i & " got " & $got[i] & " want " & $want[i]

proc assertI32(got: openArray[int32], want: openArray[int32], label: string) =
  doAssert got.len == want.len, label & ": " & $got.len & " slots, want " & $want.len
  for i in 0 ..< want.len:
    doAssert got[i] == want[i],
      label & ": slot " & $i & " got " & $got[i] & " want " & $want[i]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  const N = 128

  block: # fvVulkan axpy: two buffers read, one written, everything in the push block
    let words = compileToSpirV("fvulkan_axpy", axpyHlsl, dxilLeg = false)
    var engine = newEngine(runVulkan)
    engine.ingest(words)
    var xHost: array[N, float32]
    var yHost: array[N, float32]
    for i in 0 ..< N:
      xHost[i] = float32(i)
      yHost[i] = float32(10 * (i + 1))
    let xDev = engine.allocDevice(float32, N)
    let yDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(xDev, addr xHost[0], N)
    engine.copyToDeviceFromHost(yDev, addr yHost[0], N)

    engine.launchKernel("main", yDev, (xDev, 2.5'f32, int32(N))) do:
      grid = (1, 1)
      blk = (256, 1)

    var got: array[N, float32]
    engine.copyToHostFromDevice(addr got[0], yDev, N)
    var want: array[N, float32]
    for i in 0 ..< N:
      want[i] = 2.5'f32 * xHost[i] + yHost[i]
    assertF32(got, want, "fvulkan axpy")
    echo "  fvulkan axpy: alpha*x + y byte-exact over ", N, " threads"

  block: # fvVulkan staged grid: the engine stages the dispatch group counts into the push block
    let words = compileToSpirV("fvulkan_grid", gridHlsl, dxilLeg = false)
    var engine = newEngine(runVulkan)
    engine.ingest(words, stagedGrid = true)
    let dstDev = engine.allocDevice(int32, 3)

    engine.launchKernel("main", dstDev, ()) do:
      grid = (2, 3)
      blk = (256, 1)

    var got: array[3, int32]
    engine.copyToHostFromDevice(addr got[0], dstDev, 3)
    assertI32(got, [2'i32, 3, 1], "fvulkan staged grid")
    echo "  fvulkan staged grid: dispatch group counts byte-exact through the push block"

  block: # fvVulkan device function: buffer params pass their u64 address by value
    let words = compileToSpirV("fvulkan_devfn", devFnHlsl, dxilLeg = false)
    var engine = newEngine(runVulkan)
    engine.ingest(words)
    var xHost: array[N, float32]
    var yHost: array[N, float32]
    for i in 0 ..< N:
      xHost[i] = float32(i)
      yHost[i] = float32(10 * (i + 1))
    let xDev = engine.allocDevice(float32, N)
    let yDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(xDev, addr xHost[0], N)
    engine.copyToDeviceFromHost(yDev, addr yHost[0], N)

    engine.launchKernel("main", yDev, (xDev, 2.5'f32, int32(N))) do:
      grid = (1, 1)
      blk = (256, 1)

    var got: array[N, float32]
    engine.copyToHostFromDevice(addr got[0], yDev, N)
    var want: array[N, float32]
    for i in 0 ..< N:
      want[i] = 2.5'f32 * xHost[i]
    assertF32(got, want, "fvulkan device fn")
    echo "  fvulkan device fn: buffer params as u64 address values byte-exact"

  block: # cross-check: the same axpy kernel through the GLSL flavor, identical bitwise
    var engine = newEngine(runVulkan)
    engine.ingest(axpyGlsl)
    var xHost: array[N, float32]
    var yHost: array[N, float32]
    for i in 0 ..< N:
      xHost[i] = float32(i)
      yHost[i] = float32(10 * (i + 1))
    let xDev = engine.allocDevice(float32, N)
    let yDev = engine.allocDevice(float32, N)
    engine.copyToDeviceFromHost(xDev, addr xHost[0], N)
    engine.copyToDeviceFromHost(yDev, addr yHost[0], N)

    engine.launchKernel("kern_axpy", yDev, (xDev, 2.5'f32, int32(N))) do:
      grid = (1, 1)
      blk = (256, 1)

    var got: array[N, float32]
    engine.copyToHostFromDevice(addr got[0], yDev, N)
    var want: array[N, float32]
    for i in 0 ..< N:
      want[i] = 2.5'f32 * xHost[i] + yHost[i]
    assertF32(got, want, "glsl cross-check")
    echo "  glsl cross-check: same kernel, identical bitwise results"

when isMainModule:
  runTest()
  echo "t_hlsl_fvulkan_execute: all checks passed"
