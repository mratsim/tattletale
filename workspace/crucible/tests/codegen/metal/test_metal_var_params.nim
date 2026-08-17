## Metal: `var T` params in device fns (thread-space pointers).
## `setPair` writes two fields through a `var Pair`; `swap` exchanges two
## `var uint32` locals. Both exercise the printer's `thread` address-space
## emission for implicit `var T` params. Runs through `engine.run()` and
## asserts the output byte-exact.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_var_params.nim

import std/unittest
import workspace/crucible

const varParamMsl = metal:
  type Pair = object
    x: uint32
    y: uint32
  proc setPair(p: var Pair; vx, vy: uint32) {.device.} =
    p.x = vx
    p.y = vy
  proc swap(a, b: var uint32) {.device.} =
    let tmp = a
    a = b
    b = tmp
  proc varParamKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var p: Pair
    setPair(p, 10'u32, 20'u32)
    output[0] = p.x
    output[1] = p.y
    var a, b: uint32 = 1
    b = 2
    swap(a, b)
    output[2] = a
    output[3] = b

proc runTest() =
  suite "Metal - var T params":
    test "setPair + swap round-trip":
      var engine = bkMetal.init()
      engine.ingest(varParamMsl)
      var res: array[4, uint32]
      engine.run("varParamKernel", res, ())
      check res[0] == 10'u32
      check res[1] == 20'u32
      check res[2] == 2'u32
      check res[3] == 1'u32

when isMainModule:
  runTest()
