## WebGPU: generate WGSL via `webgpu:` macro, validate syntax, and execute on CPU.
##
## Run with:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_add.nim
##
## For full execution (requires libwgpu_native.so):
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_add.nim

import workspace/crucible

# ── External types (defined outside the `webgpu` block) ─────────────────────

type
  Vec2 = object
    x: uint32
    y: uint32

proc vec2Add(a, b: Vec2): Vec2 {.device.} =
  result.x = a.x + b.x
  result.y = a.y + b.y

# ── WGSL generation via `webgpu:` macro ────────────────────────────────────

const addWgsl = webgpu:
  proc addKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]

const vec2Wgsl = webgpu:
  proc vec2AddKernel(output: ptr UncheckedArray[uint32];
                     a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32]) {.global.} =
    let va = Vec2(x: a[0], y: a[1])
    let vb = Vec2(x: b[0], y: b[1])
    let vr = vec2Add(va, vb)
    output[0] = vr.x
    output[1] = vr.y

const maxWgsl = webgpu:
  proc maxGeneric[T](a, b: T): T {.device.} =
    if a > b: result = a else: result = b

  proc maxKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = maxGeneric(a[0], b[0])

# ── Host code ───────────────────────────────────────────────────────────────

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo "=== WebGPU WGSL generation tests ===\n"

  echo "--- addWgsl ---"
  echo addWgsl
  echo ""

  echo "--- vec2Wgsl (external type Vec2 + external fn vec2Add) ---"
  echo vec2Wgsl
  echo ""

  echo "--- maxWgsl (generic T -> uint32 instantiation) ---"
  echo maxWgsl
  echo ""

  # ── Optional execution via wgpu-native ─────────────────────────────────────

  echo "=== wgpu-native execution ===\n"

  block: # addKernel
    var engine = bkWGSL.init()
    engine.ingest(addWgsl)

    var a: array[2, uint32] = [10'u32, 20'u32]
    var b: array[2, uint32] = [1'u32, 2'u32]
    var out32: array[2, uint32]

    engine.run("addKernel", out32, (a, b))
    echo "  addKernel: [10,20] + [1,2] = [", out32[0], ", ", out32[1], "]"
    doAssert out32[0] == 11
    doAssert out32[1] == 22
    echo "  OK"

  block: # vec2AddKernel (external type + fn)
    var engine = bkWGSL.init()
    engine.ingest(vec2Wgsl)

    var a: array[2, uint32] = [100'u32, 200'u32]
    var b: array[2, uint32] = [3'u32, 4'u32]
    var out32: array[2, uint32]

    engine.run("vec2AddKernel", out32, (a, b))
    echo "  vec2AddKernel: Vec2(100,200) + Vec2(3,4) = (", out32[0], ", ", out32[1], ")"
    doAssert out32[0] == 103
    doAssert out32[1] == 204
    echo "  OK"

  block: # maxKernel (generic instantiation)
    var engine = bkWGSL.init()
    engine.ingest(maxWgsl)

    var a: array[1, uint32] = [42'u32]
    var b: array[1, uint32] = [17'u32]
    var outVal: array[1, uint32]

    engine.run("maxKernel", outVal, (a, b))
    echo "  maxKernel: max(42, 17) = ", outVal[0]
    doAssert outVal[0] == 42
    echo "  OK"

  echo "All execution tests passed ✅"

when isMainModule:
  runTest()
