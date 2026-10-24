## SPIR-V direct: compound assignments over pure-expression indices
##
## Test that `out[i*n + j] += x` and `a[int32(k)] += y` compile through the SPIR-V pipeline and compute exact results on MoltenVK
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_compound_assign_index_expr.nim

import workspace/crucible

const matSpv = spirv:
  proc matKernel(o: ptr UncheckedArray[float32], x: float32) {.global.} =
    let i = thread_position_in_grid.x
    for j in 0 ..< 2:
      o[i * 2 + j] += x

const convSpv = spirv:
  proc convKernel(o: ptr UncheckedArray[float32], y: float32, k: uint32) {.global.} =
    o[int32(k)] += y

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block binopIndex:
    # 2x2 row-major: slot i*2 + j gains x, one += per element
    var engine = newEngine(runVulkan)
    engine.ingest(matSpv)
    let oDev = engine.allocDevice(float32, 4)
    var oHost: array[4, float32] = [0.0'f32, 1.0, 2.0, 3.0]
    engine.copyToDeviceFromHost(oDev, addr oHost[0], oHost.len)

    engine.launchKernel("matKernel", oDev, (0.5'f32,))

    engine.copyToHostFromDevice(addr oHost[0], oDev, oHost.len)
    for j in 0 ..< 4:
      doAssert oHost[j] == float32(j) + 0.5'f32,
        "slot " & $j & " got " & $oHost[j] & ", want the input + 0.5"
    echo "  arithmetic index: o[i*n + j] += x exact on MoltenVK"

  block convIndex:
    # the int32(k) conversion index stays a pure expression, one += at slot k
    var engine = newEngine(runVulkan)
    engine.ingest(convSpv)
    let oDev = engine.allocDevice(float32, 4)
    var oHost: array[4, float32] = [10.0'f32, 20.0, 30.0, 40.0]
    engine.copyToDeviceFromHost(oDev, addr oHost[0], oHost.len)

    engine.launchKernel("convKernel", oDev, (5.0'f32, 2'u32))

    engine.copyToHostFromDevice(addr oHost[0], oDev, oHost.len)
    doAssert oHost[2] == 35.0'f32, "slot 2 got " & $oHost[2] & ", want 30 + 5"
    doAssert oHost[0] == 10.0'f32 and oHost[1] == 20.0'f32 and oHost[3] == 40.0'f32,
      "the untouched slots stay exact"
    echo "  conversion index: a[int32(k)] += y exact on MoltenVK"

when isMainModule:
  runTest()
  echo "t_spirv_compound_assign_index_expr: all checks passed"
