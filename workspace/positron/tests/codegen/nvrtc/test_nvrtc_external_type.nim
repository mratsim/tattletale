## NVRTC: external types pulled from outside the `cuda` block
## Run with: nim cpp -d:cuda -r workspace/positron/tests/nvrtc/test_nvrtc_external_type.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
##
## Tests the PR #565 feature: types defined outside the `cuda` block
## are pulled in automatically when used by GPU code.
import workspace/positron/src/codegen/nvrtc

# ── External types (defined outside the `cuda` block) ───────────────────────

type
  Vec2 = object
    x: uint32
    y: uint32

  Vec4 = object
    a, b, c, d: uint32

# ── External functions using external types ─────────────────────────────────

proc addVec2(a, b: Vec2): Vec2 {.device.} =
  result.x = a.x + b.x
  result.y = a.y + b.y

proc dotVec2(a, b: Vec2): uint32 {.device.} =
  result = a.x * b.x + a.y * b.y

proc sumVec4(v: Vec4): uint32 {.device.} =
  result = v.a + v.b + v.c + v.d

# ── GPU code ────────────────────────────────────────────────────────────────

const kernelCode = cuda:
  proc externalTypeKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Using types and functions defined OUTSIDE the `cuda` block
    let v1 = Vec2(x: 10'u32, y: 20'u32)
    let v2 = Vec2(x: 1'u32, y: 2'u32)
    let sum = addVec2(v1, v2)
    output[0] = sum.x
    output[1] = sum.y

    let d = dotVec2(v1, v2)
    output[2] = d

    let v4 = Vec4(a: 1'u32, b: 2'u32, c: 3'u32, d: 4'u32)
    output[3] = sumVec4(v4)

# ── Host code ───────────────────────────────────────────────────────────────

var buf: array[4, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"

nv.execute("externalTypeKernel", buf, ())
echo "  Vec2(10,20)+Vec2(1,2) = (", buf[0], ", ", buf[1], ")"
echo "  dot(Vec2(10,20), Vec2(1,2)) = ", buf[2]
echo "  sum(Vec4(1,2,3,4))  = ", buf[3]

doAssert buf[0] == 11  # 10 + 1
doAssert buf[1] == 22  # 20 + 2
doAssert buf[2] == 50  # 10*1 + 20*2
doAssert buf[3] == 10  # 1+2+3+4

echo "  OK"
