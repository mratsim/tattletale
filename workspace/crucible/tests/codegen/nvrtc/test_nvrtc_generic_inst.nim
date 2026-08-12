## NVRTC: generic function instantiation for GPU code
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_generic_inst.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
##
## Tests the PR #565 feature: Nim generic functions defined outside the `cuda`
## block are automatically instantiated for use inside GPU code. One function
## is emitted for each generic instantiation.
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

# ── Generic functions (defined outside the `cuda` block) ────────────────────

proc maxGeneric[T](a, b: T): T {.device.} =
  if a > b:
    result = a
  else:
    result = b

proc minGeneric[T](a, b: T): T {.device.} =
  if a < b:
    result = a
  else:
    result = b

proc clampGeneric[T](val, lo, hi: T): T {.device.} =
  result = minGeneric(maxGeneric(val, lo), hi)

proc sumThree[T](a, b, c: T): T {.device.} =
  result = a + b + c

# ── External type for generic use ───────────────────────────────────────────

type
  Vec3 = object
    x, y, z: uint32

proc addVec3(a, b: Vec3): Vec3 {.device.} =
  result.x = a.x + b.x
  result.y = a.y + b.y
  result.z = a.z + b.z

# ── GPU code ────────────────────────────────────────────────────────────────

const kernelCode = cuda:
  # Generic instantiated with `uint32`
  proc genericInstUint32(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = maxGeneric(10'u32, 20'u32)
    output[1] = minGeneric(30'u32, 5'u32)
    output[2] = clampGeneric(50'u32, 0'u32, 25'u32)
    output[3] = sumThree(1'u32, 2'u32, 3'u32)

  # Generic instantiated with `int32`
  proc genericInstInt32(output: ptr UncheckedArray[int32]) {.global.} =
    output[0] = maxGeneric(1'i32, -5'i32)
    output[1] = minGeneric(-10'i32, 20'i32)
    output[2] = sumThree(100'i32, 200'i32, 300'i32)

# ── Host code: uint32 generic ───────────────────────────────────────────────

var buf32: array[4, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"

engine.run("genericInstUint32", buf32, ())
echo "  maxGeneric[uint32](10,20)       = ", buf32[0]
echo "  minGeneric[uint32](30,5)        = ", buf32[1]
echo "  clampGeneric[uint32](50,0,25)   = ", buf32[2]
echo "  sumThree[uint32](1,2,3)         = ", buf32[3]

doAssert buf32[0] == 20
doAssert buf32[1] == 5
doAssert buf32[2] == 25
doAssert buf32[3] == 6

# ── Host code: int32 generic ────────────────────────────────────────────────

var bufI32: array[3, int32]
engine.run("genericInstInt32", bufI32, ())
echo "  maxGeneric[int32](1,-5)         = ", bufI32[0]
echo "  minGeneric[int32](-10,20)       = ", bufI32[1]
echo "  sumThree[int32](100,200,300)    = ", bufI32[2]

doAssert bufI32[0] == 1
doAssert bufI32[1] == -10
doAssert bufI32[2] == 600

echo "  OK (test_nvrtc_generic_inst)"
