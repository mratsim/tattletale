## Metal: generate MSL via the `metal:` macro and execute on the device.
## Covers three DSL shapes in one file:
##   - plain buffer add (addKernel)
##   - an external struct type + external device fn (vec2AddKernel)
##   - a generic device fn instantiated to uint32 (maxKernel)
## Every kernel runs through `engine.run()` and the outputs are asserted
## byte-exact, never print-only.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_add.nim

import std/unittest
import workspace/crucible

# ── External types (defined outside the `metal` block) ─────────────────────

type
  Vec2 = object
    x: uint32
    y: uint32

proc vec2Add(a, b: Vec2): Vec2 =
  result.x = a.x + b.x
  result.y = a.y + b.y

# ── MSL generation via the `metal:` macro ──────────────────────────────────

const addMsl = metal:
  proc addKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]

const vec2Msl = metal:
  proc vec2AddKernel(output: ptr UncheckedArray[uint32];
                     a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32]) {.global.} =
    let va = Vec2(x: a[0], y: a[1])
    let vb = Vec2(x: b[0], y: b[1])
    let vr = vec2Add(va, vb)
    output[0] = vr.x
    output[1] = vr.y

const maxMsl = metal:
  proc maxGeneric[T](a, b: T): T {.device.} =
    if a > b: result = a else: result = b

  proc maxKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = maxGeneric(a[0], b[0])

# ── Host code ───────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - add":

    test "addKernel: [10,20] + [1,2] = [11,22]":
      var engine = bkMetal.init()
      engine.ingest(addMsl)
      var a: array[2, uint32] = [10'u32, 20'u32]
      var b: array[2, uint32] = [1'u32, 2'u32]
      var out32: array[2, uint32]
      engine.run("addKernel", out32, (a, b))
      check out32[0] == 11
      check out32[1] == 22

    test "vec2AddKernel (external type + fn)":
      var engine = bkMetal.init()
      engine.ingest(vec2Msl)
      var a: array[2, uint32] = [100'u32, 200'u32]
      var b: array[2, uint32] = [3'u32, 4'u32]
      var out32: array[2, uint32]
      engine.run("vec2AddKernel", out32, (a, b))
      check out32[0] == 103
      check out32[1] == 204

    test "maxKernel (generic instantiation)":
      var engine = bkMetal.init()
      engine.ingest(maxMsl)
      var a: array[1, uint32] = [42'u32]
      var b: array[1, uint32] = [17'u32]
      var outVal: array[1, uint32]
      engine.run("maxKernel", outVal, (a, b))
      check outVal[0] == 42

when isMainModule:
  runTest()
