## Vulkan (GLSL): buffer add, external struct type + device fn, and generic device fn instantiation.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/app/t_vulkan_add.nim

import workspace/crucible

# ── External types (defined outside the `vulkan` block) ─────────────────────

type
  Vec2 = object
    x: uint32
    y: uint32

proc vec2Add(a, b: Vec2): Vec2 {.device.} =
  result.x = a.x + b.x
  result.y = a.y + b.y

# ── GLSL generation via `vulkan:` macro ─────────────────────────────────────

const addVk = vulkan:
  proc addKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]

const vec2Vk = vulkan:
  proc vec2AddKernel(output: ptr UncheckedArray[uint32];
                     a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32]) {.global.} =
    let va = Vec2(x: a[0], y: a[1])
    let vb = Vec2(x: b[0], y: b[1])
    let vr = vec2Add(va, vb)
    output[0] = vr.x
    output[1] = vr.y

const maxVk = vulkan:
  proc maxGeneric[T](a, b: T): T {.device.} =
    if a > b: result = a else: result = b

  proc maxKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = maxGeneric(a[0], b[0])

# ── Codegen tests (always runs) ─────────────────────────────────────────────

proc runTest() =   # private — tests run in a proc so engines are destroyed at return

  # ── Execution via Vulkan runtime ────────────────────────────────────────────

  block: # addKernel
    var engine = newEngine(runVulkan)
    engine.ingest(addVk)

    var a: array[2, uint32] = [10'u32, 20'u32]
    var b: array[2, uint32] = [1'u32, 2'u32]
    let aDev = engine.allocDevice(uint32, a.len)
    let bDev = engine.allocDevice(uint32, b.len)
    let oDev = engine.allocDevice(uint32, 2)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len)

    engine.launchKernel("addKernel", oDev, (aDev, bDev))

    var out32: array[2, uint32]
    engine.copyToHostFromDevice(addr out32[0], oDev, 2)

    doAssert out32[0] == 11
    doAssert out32[1] == 22

  block: # vec2AddKernel (external type + fn)
    var engine = newEngine(runVulkan)
    engine.ingest(vec2Vk)

    var a: array[2, uint32] = [100'u32, 200'u32]
    var b: array[2, uint32] = [3'u32, 4'u32]
    let aDev = engine.allocDevice(uint32, a.len)
    let bDev = engine.allocDevice(uint32, b.len)
    let oDev = engine.allocDevice(uint32, 2)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len)

    engine.launchKernel("vec2AddKernel", oDev, (aDev, bDev))

    var out32: array[2, uint32]
    engine.copyToHostFromDevice(addr out32[0], oDev, 2)

    doAssert out32[0] == 103
    doAssert out32[1] == 204

  block: # maxKernel (generic instantiation)
    var engine = newEngine(runVulkan)
    engine.ingest(maxVk)

    var a: array[1, uint32] = [42'u32]
    var b: array[1, uint32] = [17'u32]
    let aDev = engine.allocDevice(uint32, a.len)
    let bDev = engine.allocDevice(uint32, b.len)
    let oDev = engine.allocDevice(uint32, 1)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len)

    engine.launchKernel("maxKernel", oDev, (aDev, bDev))

    var outVal: array[1, uint32]
    engine.copyToHostFromDevice(addr outVal[0], oDev, 1)

    doAssert outVal[0] == 42

when isMainModule:
  runTest()
