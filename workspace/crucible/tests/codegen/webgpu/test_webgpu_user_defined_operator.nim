## User-defined operators: basic types (uint32) and struct types (Wrapper)
##
## Basic types: gpuBinOp → `(a + b)` — compiles and runs on all backends.
## Struct types: gpuCall with operator syntax `a + b`.
##
## NOTE: WGSL requires `u32` for shift amounts (`<<`, `>>`), so we use uint32.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_user_defined_operator.nim

import workspace/crucible/src/codegen/wgpu

# ── Basic types — gpuBinOp ──────────────────────────────────────────────────

const basicWgsl = webgpu:
  proc basicKernel(a: ptr UncheckedArray[uint32];
                   b: ptr UncheckedArray[uint32];
                   output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[0] - b[0]
    output[2] = a[0] * b[0]
    output[3] = a[0] div b[0]
    output[4] = a[0] mod b[0]
    output[5] = a[0] shl b[0]
    output[6] = a[0] shr b[0]
    output[7] = a[0] and b[0]
    output[8] = a[0] or b[0]
    output[9] = a[0] xor b[0]

# ── Struct types — operator syntax ──────────────────────────────────────────

type
  Wrapper* = object
    val*: uint32

proc `+`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val + b.val)
proc `-`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val - b.val)
proc `*`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val * b.val)
proc `div`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val div b.val)
proc `mod`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val mod b.val)
proc `shl`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val shl b.val)
proc `shr`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val shr b.val)
proc `and`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val and b.val)
proc `or`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val or b.val)
proc `xor`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val xor b.val)

const structWgsl = webgpu:
  proc structKernel(a: ptr UncheckedArray[uint32];
                    b: ptr UncheckedArray[uint32];
                    output: ptr UncheckedArray[uint32]) {.global.} =
    let x = Wrapper(val: a[0])
    let y = Wrapper(val: b[0])
    output[0] = (x + y).val
    output[1] = (x - y).val
    output[2] = (x * y).val
    output[3] = (x div y).val
    output[4] = (x mod y).val
    output[5] = (x shl y).val
    output[6] = (x shr y).val
    output[7] = (x and y).val
    output[8] = (x or y).val
    output[9] = (x xor y).val

echo "=== WebGPU generation tests ===\n"

echo "--- basicKernel ---"
echo basicWgsl
echo ""

echo "--- structKernel ---"
echo structWgsl
echo ""

echo "=== WebGPU execution ===\n"

block:
  var ctx = initWgpu()
  defer: ctx.shutdown()

  var a: array[1, uint32] = [13'u32]
  var b: array[1, uint32] = [5'u32]

  echo "--- basicKernel ---"
  let result = execWgpu(
    ctx,
    basicWgsl,
    "basicKernel",
    outputBytes = 40,
    inputs = [
      (cast[pointer](a[0].addr), 4),
      (cast[pointer](b[0].addr), 4)
    ]
  )
  doAssert result.len == 40
  let out32 = cast[ptr array[10, uint32]](result[0].addr)
  echo "  [13,5] -> [", out32[0], ", ", out32[1], ", ", out32[2], ", ", out32[3], ", ", out32[4],
       ", ", out32[5], ", ", out32[6], ", ", out32[7], ", ", out32[8], ", ", out32[9], "]"
  doAssert out32[0] == 18
  doAssert out32[1] == 8
  doAssert out32[2] == 65
  doAssert out32[3] == 2
  doAssert out32[4] == 3
  doAssert out32[5] == 416
  doAssert out32[6] == 0
  doAssert out32[7] == 5
  doAssert out32[8] == 13
  doAssert out32[9] == 8
  echo "  OK"

  echo "--- structKernel ---"
  let result2 = execWgpu(
    ctx,
    structWgsl,
    "structKernel",
    outputBytes = 40,
    inputs = [
      (cast[pointer](a[0].addr), 4),
      (cast[pointer](b[0].addr), 4)
    ]
  )
  doAssert result2.len == 40
  let out32b = cast[ptr array[10, uint32]](result2[0].addr)
  echo "  [13,5] -> [", out32b[0], ", ", out32b[1], ", ", out32b[2], ", ", out32b[3], ", ", out32b[4],
       ", ", out32b[5], ", ", out32b[6], ", ", out32b[7], ", ", out32b[8], ", ", out32b[9], "]"
  doAssert out32b[0] == 18
  doAssert out32b[1] == 8
  doAssert out32b[2] == 65
  doAssert out32b[3] == 2
  doAssert out32b[4] == 3
  doAssert out32b[5] == 416
  doAssert out32b[6] == 0
  doAssert out32b[7] == 5
  doAssert out32b[8] == 13
  doAssert out32b[9] == 8
  echo "  OK"

echo "All execution tests passed ✅"
