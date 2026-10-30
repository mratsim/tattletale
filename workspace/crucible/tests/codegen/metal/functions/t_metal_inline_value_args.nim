## Metal: inline device fn taking value arguments, including a struct.
##
## Test that the spliced call keeps plain value-argument passing correct:
## the caller's variables feed the call and stay unchanged after it.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_inline_value_args.nim

import std/unittest
import workspace/crucible

type Pack = object
  atom: uint32
  tag: int32

const valueArgsMsl = metal:
  proc weight(a: int32, p: Pack): int32 {.device, inline.} =
    return a * p.tag + int32(p.atom)

  proc scaleKernel(output: ptr UncheckedArray[int32]) {.global.} =
    var n: int32 = 3
    var v = Pack(atom: 7'u32, tag: 5)
    output[0] = weight(n, v)
    output[1] = n
    output[2] = v.tag

proc runTest() =
  suite "Metal - inline value arguments pass by fresh temps":
    test "the call reads the caller's values, the caller's variables keep theirs":
      var engine = newEngine(runMetal)
      engine.ingest(valueArgsMsl)
      let oDev = engine.allocDevice(int32, 3)
      engine.launchKernel("scaleKernel", oDev, ())
      var res: array[3, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 3)
      check res[0] == 22
      check res[1] == 3
      check res[2] == 5

when isMainModule:
  runTest()
