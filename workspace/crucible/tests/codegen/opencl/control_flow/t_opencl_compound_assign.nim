## OpenCL: compound-assign `a[i] += v` desugars to read-modify-write
## and executes on the device (indexed write and a kernel-local accumulator).
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/control_flow/t_opencl_compound_assign.nim

import workspace/crucible

# ── OpenCL C generation via `opencl:` macro ────────────────────────────────

const accCl = opencl:
  proc accKernel(output, a: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 2:
      var acc = uint32(0)
      for k in 0 ..< 4:
        acc += uint32(k)
      output[i] = a[i]
      output[i] += acc
      output[i] += 1

# ── Execution via OpenCL runtime ───────────────────────────────────────────

proc runTest() =   # private, tests run in a proc so engines are destroyed at return
  echo "=== OpenCL compound-assign execution ===\n"

  block: # accKernel: output[i] = a[i] + acc(0..3) + 1, acc(k) = 0+1+2+3 = 6
    var engine = bkOpenCL.init()
    engine.ingest(accCl)

    var a: array[2, uint32] = [10'u32, 20'u32]
    var out32: array[2, uint32]

    let aDev = engine.allocDevice(uint32, a.len)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    let oDev = engine.allocDevice(uint32, 2)
    engine.launchKernel("accKernel", oDev, (aDev,))
    engine.copyToHostFromDevice(addr out32[0], oDev, 2)

    echo "  accKernel: output = [", out32[0], ", ", out32[1], "]"
    doAssert out32[0] == 17  # 10 + 6 + 1
    doAssert out32[1] == 27  # 20 + 6 + 1
    echo "  OK"

  echo "All compound-assign execution tests passed"

when isMainModule:
  runTest()
