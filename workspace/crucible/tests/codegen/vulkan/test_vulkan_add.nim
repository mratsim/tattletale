## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_add.nim

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
  proc addKernel(a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32];
                 output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]

const vec2Vk = vulkan:
  proc vec2AddKernel(a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32];
                     output: ptr UncheckedArray[uint32]) {.global.} =
    let va = Vec2(x: a[0], y: a[1])
    let vb = Vec2(x: b[0], y: b[1])
    let vr = vec2Add(va, vb)
    output[0] = vr.x
    output[1] = vr.y

const maxVk = vulkan:
  proc maxGeneric[T](a, b: T): T {.device.} =
    if a > b: result = a else: result = b

  proc maxKernel(a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32];
                 output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = maxGeneric(a[0], b[0])

# ── Codegen tests (always runs) ─────────────────────────────────────────────

echo "=== Vulkan GLSL generation tests ===\n"

echo "--- addKernel ---"
echo addVk
echo ""

echo "--- vec2AddKernel (external type Vec2 + external fn vec2Add) ---"
echo vec2Vk
echo ""

echo "--- maxKernel (generic T -> uint32 instantiation) ---"
echo maxVk
echo ""

# ── Optional execution via Vulkan runtime ───────────────────────────────────

echo "=== Vulkan execution ===\n"

block: # addKernel
  var engine = bkVulkan.init()
  engine.ingest(addVk)

  var a: array[2, uint32] = [10'u32, 20'u32]
  var b: array[2, uint32] = [1'u32, 2'u32]
  var out32: array[2, uint32]

  engine.run("addKernel", out32, (a, b))
  echo "  addKernel: [10,20] + [1,2] = [", out32[0], ", ", out32[1], "]"
  doAssert out32[0] == 11
  doAssert out32[1] == 22
  echo "  OK"

block: # vec2AddKernel (external type + fn)
  var engine = bkVulkan.init()
  engine.ingest(vec2Vk)

  var a: array[2, uint32] = [100'u32, 200'u32]
  var b: array[2, uint32] = [3'u32, 4'u32]
  var out32: array[2, uint32]

  engine.run("vec2AddKernel", out32, (a, b))
  echo "  vec2AddKernel: Vec2(100,200) + Vec2(3,4) = (", out32[0], ", ", out32[1], ")"
  doAssert out32[0] == 103
  doAssert out32[1] == 204
  echo "  OK"

block: # maxKernel (generic instantiation)
  var engine = bkVulkan.init()
  engine.ingest(maxVk)

  var a: array[1, uint32] = [42'u32]
  var b: array[1, uint32] = [17'u32]
  var outVal: array[1, uint32]

  engine.run("maxKernel", outVal, (a, b))
  echo "  maxKernel: max(42, 17) = ", outVal[0]
  doAssert outVal[0] == 42
  echo "  OK"

echo "All execution tests passed ✅"

