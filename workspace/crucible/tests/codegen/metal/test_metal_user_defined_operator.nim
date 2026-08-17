## Metal: user-defined operators. Basic uint32 operands lower to gpuBinOp
## (C-family `+ - * / % << >> & | ^` in MSL). Struct operands lower
## to gpuCall with operator syntax (`x + y` on a Wrapper). Wrapper2
## covers the sanitized-operator name-collision handling.
## Each overload set gets its own mangled fn name. All three kernels run
## on the device with byte-exact asserts, including the full 10-binop basicKernel set
## that the printer compile gate deliberately left to the execution suite.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_user_defined_operator.nim

import std/unittest
import workspace/crucible

# ── Basic types (gpuBinOp) ──────────────────────────────────────────────────

const basicMsl = metal:
  proc basicKernel(output: ptr UncheckedArray[uint32];
                   a: ptr UncheckedArray[uint32];
                   b: ptr UncheckedArray[uint32]) {.global.} =
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

# ── Struct types (operator syntax) ──────────────────────────────────────────

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

const structMsl = metal:
  proc structKernel(output: ptr UncheckedArray[uint32];
                    a: ptr UncheckedArray[uint32];
                    b: ptr UncheckedArray[uint32]) {.global.} =
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

const structMsl2 = metal:
  proc structKernel2(output: ptr UncheckedArray[uint32];
                     a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32]) {.global.} =
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

# ── Host side ───────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo basicMsl

  suite "Metal - user-defined operators":
    test "basicKernel: the 10-binop set on uint32":
      var engine = bkMetal.init()
      engine.ingest(basicMsl)
      var a: array[1, uint32] = [13'u32]
      var b: array[1, uint32] = [5'u32]
      var out32: array[10, uint32]
      engine.run("basicKernel", out32, (a, b))
      check out32[0] == 18
      check out32[1] == 8
      check out32[2] == 65
      check out32[3] == 2
      check out32[4] == 3
      check out32[5] == 416
      check out32[6] == 0
      check out32[7] == 5
      check out32[8] == 13
      check out32[9] == 8

    test "structKernel: operator syntax on Wrapper":
      var engine = bkMetal.init()
      engine.ingest(structMsl)
      echo structMsl
      var a: array[1, uint32] = [7'u32]
      var b: array[1, uint32] = [3'u32]
      var out32b: array[10, uint32]
      engine.run("structKernel", out32b, (a, b))
      check out32b[0] == 10
      check out32b[1] == 4
      check out32b[2] == 21
      check out32b[3] == 2
      check out32b[4] == 1
      check out32b[5] == 7'u32 shl 3
      check out32b[6] == 7'u32 shr 3
      check out32b[7] == (7'u32 and 3)
      check out32b[8] == (7'u32 or 3)
      check out32b[9] == (7'u32 xor 3)

    test "structKernel2: name-collision handling on Wrapper2":
      var engine = bkMetal.init()
      engine.ingest(structMsl2)
      echo structMsl2
      var a: array[1, uint32] = [14'u32]
      var b: array[1, uint32] = [3'u32]
      var out32c: array[10, uint32]
      engine.run("structKernel2", out32c, (a, b))
      check out32c[0] == 17
      check out32c[1] == 11
      check out32c[2] == 42
      check out32c[3] == 4
      check out32c[4] == 2
      check out32c[5] == 14'u32 shl 3
      check out32c[6] == 14'u32 shr 3
      check out32c[7] == (14'u32 and 3)
      check out32c[8] == (14'u32 or 3)
      check out32c[9] == (14'u32 xor 3)

when isMainModule:
  runTest()
