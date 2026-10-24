## SPIR-V direct: signed integer % with dividend-sign remainder
##
## Test that signed % through the SPIR-V pipeline computes Nim semantics, dividend-sign remainder
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_signed_mod.nim

import workspace/crucible

const modSpv = spirv:
  proc modKernel(o: ptr UncheckedArray[int32], a: int32, b: int32) {.global.} =
    o[0] = a mod b
    o[1] = (-a) mod b

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(modSpv)
  let oDev = engine.allocDevice(int32, 2)

  # Nim: -7 % 3 == -1 (dividend sign). An OpSMod lowering gives 2 (divisor sign).
  engine.launchKernel("modKernel", oDev, ((-7'i32), 3'i32))
  var out32: array[2, int32]
  engine.copyToHostFromDevice(addr out32[0], oDev, 2)
  doAssert out32[0] == -1, "-7 % 3 got " & $out32[0] & ", want -1 (dividend sign)"
  doAssert out32[1] == 1, "7 % 3 got " & $out32[1] & ", want 1"
  echo "  signed %: (-7 % 3) = ", out32[0], ", (7 % 3) = ", out32[1], " exact on MoltenVK"

  # Nim: 7 % -3 == 1 (dividend sign). An OpSMod lowering gives -2.
  engine.launchKernel("modKernel", oDev, (7'i32, (-3'i32)))
  engine.copyToHostFromDevice(addr out32[0], oDev, 2)
  doAssert out32[0] == 1, "7 % -3 got " & $out32[0] & ", want 1 (dividend sign)"
  doAssert out32[1] == -1, "-7 % -3 got " & $out32[1] & ", want -1"
  echo "  signed %: (7 % -3) = ", out32[0], ", (-7 % -3) = ", out32[1], " exact on MoltenVK"

when isMainModule:
  runTest()
  echo "t_spirv_signed_mod: all checks passed"
