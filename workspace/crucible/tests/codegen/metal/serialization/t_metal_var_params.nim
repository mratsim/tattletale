## Metal: `var T` params in device fns (thread-space pointers).
## `setPair` writes two fields through a `var Pair`, and `swap` exchanges
## two `var uint32` locals, exercising the printer's `thread` address-space
## emission for implicit `var T` params.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_var_params.nim

import std/[strutils, unittest]
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
      var engine = bkMetal.newEngine()
      engine.ingest(varParamMsl)
      let msl = engine.getArtifact()
      # Unannotated locals emit no address-space qualifier: register (thread)
      # storage is the default declaration form. Only implicit `var T` params
      # carry an explicit `thread` qualifier.
      check "uint a = 1;" in msl
      check "uint b = 1;" in msl
      check "Pair p;" in msl
      check "thread uint a" notin msl
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel("varParamKernel", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 10'u32
      check res[1] == 20'u32
      check res[2] == 2'u32
      check res[3] == 1'u32

when isMainModule:
  runTest()
