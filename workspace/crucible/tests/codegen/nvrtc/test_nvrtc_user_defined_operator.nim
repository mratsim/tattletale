## User-defined operators: basic types (int32) and struct types (Wrapper)
##
## Basic types: gpuBinOp → `(a + b)` — compiles and runs on all backends.
## Struct types: gpuCall with operator syntax `a + b`.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_user_defined_operator.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc

# ── Basic types — gpuBinOp ──────────────────────────────────────────────────

const kernelBasic = cuda:
  proc kernelBasic(C: ptr UncheckedArray[int32]) {.global.} =
    let a: int32 = 13
    let b: int32 = 5
    C[0] = a + b
    C[1] = a - b
    C[2] = a * b
    C[3] = a div b
    C[4] = a mod b
    C[5] = a shl b
    C[6] = a shr b
    C[7] = a and b
    C[8] = a or b
    C[9] = a xor b

suite "Operator codegen":
  test "basic types (int32) — all operators on GPU":
    var output: array[10, int32]
    var nv = initNvrtc(kernelBasic)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernelBasic", output, ())
    check output[0] == 18
    check output[1] == 8
    check output[2] == 65
    check output[3] == 2
    check output[4] == 3
    check output[5] == 416
    check output[6] == 0
    check output[7] == 5
    check output[8] == 13
    check output[9] == 8

# ── Struct types — operator syntax ──────────────────────────────────────────

type
  Wrapper* = object
    val*: int32

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

const kernelStruct = cuda:
  proc kernelStruct(C: ptr UncheckedArray[int32]) {.global.} =
    let a = Wrapper(val: 13)
    let b = Wrapper(val: 5)
    C[0] = (a + b).val
    C[1] = (a - b).val
    C[2] = (a * b).val
    C[3] = (a div b).val
    C[4] = (a mod b).val
    C[5] = (a shl b).val
    C[6] = (a shr b).val
    C[7] = (a and b).val
    C[8] = (a or b).val
    C[9] = (a xor b).val

suite "User-defined operators":
  test "struct types (Wrapper) — operator syntax on GPU":
    var output: array[10, int32]
    var nv = initNvrtc(kernelStruct)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernelStruct", output, ())
    check output[0] == 18
    check output[1] == 8
    check output[2] == 65
    check output[3] == 2
    check output[4] == 3
    check output[5] == 416
    check output[6] == 0
    check output[7] == 5
    check output[8] == 13
    check output[9] == 8
