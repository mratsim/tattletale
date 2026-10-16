## WebGPU: float16 by-value scalar argument binds through the scalar scratch buffer.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_f16_scalar_arg.nim

import workspace/crucible

const f16ScalarWgsl = webgpu:
  proc f16ScaleKernel(output, a: ptr UncheckedArray[uint32], x: float16) {.global.} =
    output[0] = a[0] + uint32(toFp32(x))

proc runTest() =   # private, tests run in a proc so engines are destroyed at return
  block: # f16ScaleKernel (float16 scalar by value)
    var engine = bkWGSL.init()
    engine.ingest(f16ScalarWgsl)

    var a: array[1, uint32] = [10'u32]
    var out32: array[1, uint32]

    let aDev = engine.allocDevice(a.len * sizeof(uint32))
    let oDev = engine.allocDevice(out32.len * sizeof(uint32))
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
    # 0x4400 is the f16 bit pattern of 4.0
    let x = cast[float16](0x4400'u16)
    engine.launchKernel("f16ScaleKernel", oDev, (aDev, x))
    engine.copyToHostFromDevice(addr out32[0], oDev, out32.len * sizeof(uint32))
    engine.freeDevice(aDev)
    engine.freeDevice(oDev)

    doAssert out32[0] == 14, "f16ScaleKernel: " & $out32[0] & ", expected 14"
    echo "  OK: float16 scalar arg (4.0) added to 10 = 14"

when isMainModule:
  runTest()
