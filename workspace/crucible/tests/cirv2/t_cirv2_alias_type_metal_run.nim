## CIRv2 Metal launch of an alias-typed kernel parameter against an expansion-typed callee
##
## Test that an alias-typed kernel parameter launches on Metal through an expansion-typed callee
##
## Run:
##   nim c -r -d:release --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/t2name/t_cirv2_alias_type_metal_run \
##     workspace/crucible/tests/cirv2/t_cirv2_alias_type_metal_run.nim

import workspace/crucible

type
  Vec2[T] = object
    x, y: T
  V2F = Vec2[float32]

proc combine(p: Vec2[float32], s: float32): float32 =
  p.x * s + p.y * s

const kernelSrc = metal:
  proc aliasParamKernel(o: ptr UncheckedArray[float32], p: V2F) {.global.} =
    o[0] = combine(p, 3.0'f32)

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(kernelSrc)
  let oDev = engine.allocDevice(float32, 1)
  let p = V2F(x: 2.5'f32, y: 0.5'f32)
  engine.launchKernel("aliasParamKernel", oDev, (p,))
  var res: array[1, float32]
  engine.copyToHostFromDevice(addr res[0], oDev, 1)
  doAssert res[0] == 9.0'f32, "combine(p, 3) over (2.5, 0.5) must be 9.0, got " & $res[0]

when isMainModule:
  runTest()
