## User-defined operators: basic types (uint32) and struct types (Wrapper, Wrapper2)
##
## Basic types: gpuBinOp → `(a + b)` — compiles and runs on all backends.
## Struct types: gpuCall with operator syntax `a + b`.
## Wrapper2 tests name collision handling for sanitized operator names.
##
## NOTE: WGSL requires `u32` for shift amounts (`<<`, `>>`), so we use uint32.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_user_defined_operator.nim

import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

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

  Wrapper2* = object
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

proc `+`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val + b.val)
proc `-`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val - b.val)
proc `*`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val * b.val)
proc `div`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val div b.val)
proc `mod`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val mod b.val)
proc `shl`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val shl b.val)
proc `shr`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val shr b.val)
proc `and`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val and b.val)
proc `or`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val or b.val)
proc `xor`*(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val xor b.val)

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

const structWgsl2 = webgpu:
  proc structKernel2(a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32];
                     output: ptr UncheckedArray[uint32]) {.global.} =
    let x = Wrapper2(val: a[0])
    let y = Wrapper2(val: b[0])
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

echo "--- structKernel2 ---"
echo structWgsl2
echo ""

echo "=== WebGPU execution ===\n"

block:
  var engine = bkWGSL.init()
  engine.ingest(basicWgsl)

  var a: array[1, uint32] = [13'u32]
  var b: array[1, uint32] = [5'u32]

  echo "--- basicKernel ---"
  var out32: array[10, uint32]
  engine.run("basicKernel", out32, (a, b))
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
  engine.ingest(structWgsl)
  a[0] = 7
  b[0] = 3
  var out32b: array[10, uint32]
  engine.run("structKernel", out32b, (a, b))
  echo "  [7,3] -> [", out32b[0], ", ", out32b[1], ", ", out32b[2], ", ", out32b[3], ", ", out32b[4],
       ", ", out32b[5], ", ", out32b[6], ", ", out32b[7], ", ", out32b[8], ", ", out32b[9], "]"
  doAssert out32b[0] == 10
  doAssert out32b[1] == 4
  doAssert out32b[2] == 21
  doAssert out32b[3] == 2
  doAssert out32b[4] == 1
  doAssert out32b[5] == 7'u32 shl 3
  doAssert out32b[6] == 7'u32 shr 3
  doAssert out32b[7] == (7'u32 and 3)
  doAssert out32b[8] == (7'u32 or 3)
  doAssert out32b[9] == (7'u32 xor 3)
  echo "  OK"

  echo "--- structKernel2 ---"
  engine.ingest(structWgsl2)
  a[0] = 14
  b[0] = 3
  var out32c: array[10, uint32]
  engine.run("structKernel2", out32c, (a, b))
  echo "  [14,3] -> [", out32c[0], ", ", out32c[1], ", ", out32c[2], ", ", out32c[3], ", ", out32c[4],
       ", ", out32c[5], ", ", out32c[6], ", ", out32c[7], ", ", out32c[8], ", ", out32c[9], "]"
  doAssert out32c[0] == 17
  doAssert out32c[1] == 11
  doAssert out32c[2] == 42
  doAssert out32c[3] == 4
  doAssert out32c[4] == 2
  doAssert out32c[5] == 14'u32 shl 3
  doAssert out32c[6] == 14'u32 shr 3
  doAssert out32c[7] == (14'u32 and 3)
  doAssert out32c[8] == (14'u32 or 3)
  doAssert out32c[9] == (14'u32 xor 3)
  echo "  OK"

echo "All execution tests passed ✅"
