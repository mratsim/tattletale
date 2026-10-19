## User-defined operators: basic types lower through gpuBinOp, struct types
## through gpuCall with operator syntax, Wrapper2 covers sanitized-name collisions.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_user_defined_operator.nim

import std/[unittest]
import workspace/crucible

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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Operator codegen":
    test "basic types (int32) — all operators on GPU":
      var output: array[10, int32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernelBasic)
      let oB = engine.allocDevice(int32, output.len)
      engine.launchKernel<<(1, 1)>>("kernelBasic", oB, ())
      engine.copyToHostFromDevice(addr output[0], oB, output.len)
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
    Wrapper = object
      val: int32

    Wrapper2 = object
      val: int32

  proc `+`(a, b: Wrapper): Wrapper = Wrapper(val: a.val + b.val)
  proc `-`(a, b: Wrapper): Wrapper = Wrapper(val: a.val - b.val)
  proc `*`(a, b: Wrapper): Wrapper = Wrapper(val: a.val * b.val)
  proc `div`(a, b: Wrapper): Wrapper = Wrapper(val: a.val div b.val)
  proc `mod`(a, b: Wrapper): Wrapper = Wrapper(val: a.val mod b.val)
  proc `shl`(a, b: Wrapper): Wrapper = Wrapper(val: a.val shl b.val)
  proc `shr`(a, b: Wrapper): Wrapper = Wrapper(val: a.val shr b.val)
  proc `and`(a, b: Wrapper): Wrapper = Wrapper(val: a.val and b.val)
  proc `or`(a, b: Wrapper): Wrapper = Wrapper(val: a.val or b.val)
  proc `xor`(a, b: Wrapper): Wrapper = Wrapper(val: a.val xor b.val)

  proc `+`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val + b.val)
  proc `-`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val - b.val)
  proc `*`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val * b.val)
  proc `div`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val div b.val)
  proc `mod`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val mod b.val)
  proc `shl`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val shl b.val)
  proc `shr`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val shr b.val)
  proc `and`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val and b.val)
  proc `or`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val or b.val)
  proc `xor`(a, b: Wrapper2): Wrapper2 = Wrapper2(val: a.val xor b.val)

  const kernelStruct = cuda:
    proc kernelStruct(C: ptr UncheckedArray[int32]) {.global.} =
      let a = Wrapper(val: 7)
      let b = Wrapper(val: 3)
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

  const kernelStruct2 = cuda:
    proc kernelStruct2(C: ptr UncheckedArray[int32]) {.global.} =
      let a = Wrapper2(val: 14)
      let b = Wrapper2(val: 3)
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

  const kernelBoth = cuda:
    proc kernelBoth(C: ptr UncheckedArray[int32]) {.global.} =
      let a = Wrapper(val: 7)
      let b = Wrapper(val: 3)
      let c = Wrapper2(val: 14)
      let d = Wrapper2(val: 3)
      C[0] = (a + b).val
      C[1] = (c + d).val

  suite "User-defined operators":
    test "struct types (Wrapper) — operator syntax on GPU":
      var output: array[10, int32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernelStruct)
      let oS = engine.allocDevice(int32, output.len)
      engine.launchKernel<<(1, 1)>>("kernelStruct", oS, ())
      engine.copyToHostFromDevice(addr output[0], oS, output.len)
      check output[0] == 10
      check output[1] == 4
      check output[2] == 21
      check output[3] == 2
      check output[4] == 1
      check output[5] == 56
      check output[6] == 0
      check output[7] == 3
      check output[8] == 7
      check output[9] == 4

    test "struct types (Wrapper2) — operator syntax on GPU":
      var output: array[10, int32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernelStruct2)
      let oS2 = engine.allocDevice(int32, output.len)
      engine.launchKernel<<(1, 1)>>("kernelStruct2", oS2, ())
      engine.copyToHostFromDevice(addr output[0], oS2, output.len)
      check output[0] == 17
      check output[1] == 11
      check output[2] == 42
      check output[3] == 4
      check output[4] == 2
      check output[5] == 112
      check output[6] == 1
      check output[7] == 2
      check output[8] == 15
      check output[9] == 13

    test "struct types (both Wrapper + Wrapper2) — name collision":
      var output: array[2, int32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernelBoth)
      let oBoth = engine.allocDevice(int32, output.len)
      engine.launchKernel<<(1, 1)>>("kernelBoth", oBoth, ())
      engine.copyToHostFromDevice(addr output[0], oBoth, output.len)
      check output[0] == 10
      check output[1] == 17

when isMainModule:
  runTest()
