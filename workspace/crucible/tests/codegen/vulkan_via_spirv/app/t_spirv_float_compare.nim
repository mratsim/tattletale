## SPIR-V direct: float comparisons through the OpFOrd family
##
## Test that ==, !=, <, <=, >, >= over float32 operands compile through the SPIR-V pipeline and compute exact results on MoltenVK
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_float_compare.nim

import workspace/crucible

const cmpSpv = spirv:
  proc cmpKernel(o: ptr UncheckedArray[uint32], x: float32) {.global.} =
    var r = 0'u32
    r = 0'u32
    if x < 2.5'f32:
      r = 1'u32
    o[0] = r
    r = 0'u32
    if x <= 2.5'f32:
      r = 2'u32
    o[1] = r
    r = 0'u32
    if x > 2.5'f32:
      r = 3'u32
    o[2] = r
    r = 0'u32
    if x >= 2.5'f32:
      r = 4'u32
    o[3] = r
    r = 0'u32
    if x == 2.5'f32:
      r = 5'u32
    o[4] = r
    r = 0'u32
    if x != 2.5'f32:
      r = 6'u32
    o[5] = r

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(cmpSpv)
  let oDev = engine.allocDevice(uint32, 6)

  # x = 1.5: <, <=, != fire; >, >=, == do not
  engine.launchKernel("cmpKernel", oDev, (1.5'f32,))
  var out32: array[6, uint32]
  engine.copyToHostFromDevice(addr out32[0], oDev, 6)
  doAssert out32[0] == 1, "1.5 < 2.5 got " & $out32[0] & ", want 1"
  doAssert out32[1] == 2, "1.5 <= 2.5 got " & $out32[1] & ", want 2"
  doAssert out32[2] == 0, "1.5 > 2.5 got " & $out32[2] & ", want 0"
  doAssert out32[3] == 0, "1.5 >= 2.5 got " & $out32[3] & ", want 0"
  doAssert out32[4] == 0, "1.5 == 2.5 got " & $out32[4] & ", want 0"
  doAssert out32[5] == 6, "1.5 != 2.5 got " & $out32[5] & ", want 6"
  echo "  x=1.5: [", out32, "] exact on MoltenVK"

  # x = 2.5: <=, >=, == fire; <, >, != do not
  engine.launchKernel("cmpKernel", oDev, (2.5'f32,))
  engine.copyToHostFromDevice(addr out32[0], oDev, 6)
  doAssert out32[0] == 0, "2.5 < 2.5 got " & $out32[0] & ", want 0"
  doAssert out32[1] == 2, "2.5 <= 2.5 got " & $out32[1] & ", want 2"
  doAssert out32[2] == 0, "2.5 > 2.5 got " & $out32[2] & ", want 0"
  doAssert out32[3] == 4, "2.5 >= 2.5 got " & $out32[3] & ", want 4"
  doAssert out32[4] == 5, "2.5 == 2.5 got " & $out32[4] & ", want 5"
  doAssert out32[5] == 0, "2.5 != 2.5 got " & $out32[5] & ", want 0"
  echo "  x=2.5: [", out32, "] exact on MoltenVK"

  # x = 4.0: >, >=, != fire; <, <=, == do not
  engine.launchKernel("cmpKernel", oDev, (4.0'f32,))
  engine.copyToHostFromDevice(addr out32[0], oDev, 6)
  doAssert out32[0] == 0, "4.0 < 2.5 got " & $out32[0] & ", want 0"
  doAssert out32[1] == 0, "4.0 <= 2.5 got " & $out32[1] & ", want 0"
  doAssert out32[2] == 3, "4.0 > 2.5 got " & $out32[2] & ", want 3"
  doAssert out32[3] == 4, "4.0 >= 2.5 got " & $out32[3] & ", want 4"
  doAssert out32[4] == 0, "4.0 == 2.5 got " & $out32[4] & ", want 0"
  doAssert out32[5] == 6, "4.0 != 2.5 got " & $out32[5] & ", want 6"
  echo "  x=4.0: [", out32, "] exact on MoltenVK"

when isMainModule:
  runTest()
  echo "t_spirv_float_compare: all checks passed"
