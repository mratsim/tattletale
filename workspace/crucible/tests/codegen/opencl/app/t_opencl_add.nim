## OpenCL: add, vec2-add, and generic-max kernels generate OpenCL C
## through the `opencl:` macro and run on the device via `engine.launchKernel()`.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/app/t_opencl_add.nim

import workspace/crucible

# ── External types (defined outside the `opencl` block) ─────────────────────

type
  Vec2 = object
    x: uint32
    y: uint32

proc vec2Add(a, b: Vec2): Vec2 {.device.} =
  result.x = a.x + b.x
  result.y = a.y + b.y

# ── OpenCL C generation via `opencl:` macro ────────────────────────────────

const addCl = opencl:
  proc addKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]

const vec2Cl = opencl:
  proc vec2AddKernel(output: ptr UncheckedArray[uint32];
                     a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32]) {.global.} =
    let va = Vec2(x: a[0], y: a[1])
    let vb = Vec2(x: b[0], y: b[1])
    let vr = vec2Add(va, vb)
    output[0] = vr.x
    output[1] = vr.y

const maxCl = opencl:
  proc maxGeneric[T](a, b: T): T {.device.} =
    if a > b: result = a else: result = b

  proc maxKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = maxGeneric(a[0], b[0])

# ── Codegen tests (always runs) ─────────────────────────────────────────────

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo "=== OpenCL generation tests ===\n"

  echo "--- addKernel ---"
  echo addCl
  echo ""

  echo "--- vec2AddKernel (external type Vec2 + external fn vec2Add) ---"
  echo vec2Cl
  echo ""

  echo "--- maxKernel (generic T -> uint32 instantiation) ---"
  echo maxCl
  echo ""

  # ── Optional execution via OpenCL runtime ───────────────────────────────────

  echo "=== OpenCL execution ===\n"

  block: # addKernel
    var engine = newEngine(runOpenCL)
    engine.ingest(addCl)

    var a: array[2, uint32] = [10'u32, 20'u32]
    var b: array[2, uint32] = [1'u32, 2'u32]
    var out32: array[2, uint32]

    let aDev = engine.allocDevice(uint32, a.len)
    let bDev = engine.allocDevice(uint32, b.len)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
    let oDev = engine.allocDevice(uint32, 2)
    engine.launchKernel("addKernel", oDev, (aDev, bDev))
    engine.copyToHostFromDevice(addr out32[0], oDev, 2)

    echo "  addKernel: [10,20] + [1,2] = [", out32[0], ", ", out32[1], "]"
    doAssert out32[0] == 11
    doAssert out32[1] == 22
    echo "  OK"

  block: # vec2AddKernel (external type + fn)
    var engine = newEngine(runOpenCL)
    engine.ingest(vec2Cl)

    var a: array[2, uint32] = [100'u32, 200'u32]
    var b: array[2, uint32] = [3'u32, 4'u32]
    var out32: array[2, uint32]

    let aDev = engine.allocDevice(uint32, a.len)
    let bDev = engine.allocDevice(uint32, b.len)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
    let oDev = engine.allocDevice(uint32, 2)
    engine.launchKernel("vec2AddKernel", oDev, (aDev, bDev))
    engine.copyToHostFromDevice(addr out32[0], oDev, 2)

    echo "  vec2AddKernel: Vec2(100,200) + Vec2(3,4) = (", out32[0], ", ", out32[1], ")"
    doAssert out32[0] == 103
    doAssert out32[1] == 204
    echo "  OK"

  block: # maxKernel (generic instantiation)
    var engine = newEngine(runOpenCL)
    engine.ingest(maxCl)

    var a: array[1, uint32] = [42'u32]
    var b: array[1, uint32] = [17'u32]
    var outVal: array[1, uint32]

    let aDev = engine.allocDevice(uint32, a.len)
    let bDev = engine.allocDevice(uint32, b.len)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("maxKernel", oDev, (aDev, bDev))
    engine.copyToHostFromDevice(addr outVal[0], oDev, 1)

    echo "  maxKernel: max(42, 17) = ", outVal[0]
    doAssert outVal[0] == 42
    echo "  OK"

  echo "All execution tests passed ✅"

when isMainModule:
  runTest()
