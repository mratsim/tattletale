## WebGPU: generate WGSL via `webgpu:` macro, validate syntax, and execute on CPU.
##
## Run with:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_add.nim
##
## For full execution (requires libwgpu_native.so):
##   nim c -r -d:webgpu --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_add.nim

import workspace/crucible/src/codegen/wgpu

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
  proc addKernel(a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32];
                 output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]

const vec2Wgsl = webgpu:
  proc vec2AddKernel(a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32];
                     output: ptr UncheckedArray[uint32]) {.global.} =
    let va = Vec2(x: a[0], y: a[1])
    let vb = Vec2(x: b[0], y: b[1])
    let vr = vec2Add(va, vb)
    output[0] = vr.x
    output[1] = vr.y

const maxWgsl = webgpu:
  proc maxGeneric[T](a, b: T): T {.device.} =
    if a > b: result = a else: result = b

  proc maxKernel(a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32];
                 output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = maxGeneric(a[0], b[0])

# ── Host code ───────────────────────────────────────────────────────────────

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
  var ctx = initWgpu()
  defer: ctx.shutdown()

  var a: array[2, uint32] = [10'u32, 20'u32]
  var b: array[2, uint32] = [1'u32, 2'u32]

  let result = execWgpu(
    ctx,
    addWgsl,
    "addKernel",
    outputBytes = 8,  # 2 x uint32
    inputs = [
      (cast[pointer](a[0].addr), 8),
      (cast[pointer](b[0].addr), 8)
    ]
  )

  doAssert result.len == 8
  let out32 = cast[ptr array[2, uint32]](result[0].addr)
  echo "  addKernel: [10,20] + [1,2] = [", out32[0], ", ", out32[1], "]"
  doAssert out32[0] == 11
  doAssert out32[1] == 22
  echo "  OK"

block: # vec2AddKernel (external type + fn)
  var ctx = initWgpu()
  defer: ctx.shutdown()

  var a: array[2, uint32] = [100'u32, 200'u32]
  var b: array[2, uint32] = [3'u32, 4'u32]

  let result = execWgpu(
    ctx,
    vec2Wgsl,
    "vec2AddKernel",
    outputBytes = 8,
    inputs = [
      (cast[pointer](a[0].addr), 8),
      (cast[pointer](b[0].addr), 8)
    ]
  )

  let out32 = cast[ptr array[2, uint32]](result[0].addr)
  echo "  vec2AddKernel: Vec2(100,200) + Vec2(3,4) = (", out32[0], ", ", out32[1], ")"
  doAssert out32[0] == 103
  doAssert out32[1] == 204
  echo "  OK"

block: # maxKernel (generic instantiation)
  var ctx = initWgpu()
  defer: ctx.shutdown()

  var a: array[1, uint32] = [42'u32]
  var b: array[1, uint32] = [17'u32]

  let result = execWgpu(
    ctx,
    maxWgsl,
    "maxKernel",
    outputBytes = 4,
    inputs = [
      (cast[pointer](a[0].addr), 4),
      (cast[pointer](b[0].addr), 4)
    ]
  )

  let outVal = cast[ptr uint32](result[0].addr)[]
  echo "  maxKernel: max(42, 17) = ", outVal
  doAssert outVal == 42
  echo "  OK"

echo "All execution tests passed ✅"

