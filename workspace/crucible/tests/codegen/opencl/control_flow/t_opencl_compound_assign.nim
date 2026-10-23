## OpenCL: compound-assign `+=` / `*=` executes on the device, including side-effecting-index and block-expression lvalues
##
## Test that the value accumulates in the stored element and the LHS index expression is evaluated exactly once
##
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

const hazardCl = opencl:
  proc tick(x: ptr UncheckedArray[int32]): int32 {.device.} =
    result = x[0]
    x[0] = result + 1

  proc hazardKernel(x: ptr UncheckedArray[int32]) {.global.} =
    x[tick(x) + 1] += 7

const blitCl = opencl:
  proc blitKernel(x: ptr UncheckedArray[int32]) {.global.} =
    for i in 0 ..< 2:
      x[block:
          let j = i * 2
          j] += 3

# ── Execution via OpenCL runtime ───────────────────────────────────────────

proc runTest() =   # private, tests run in a proc so engines are destroyed at return
  echo "=== OpenCL compound-assign execution ===\n"

  block: # accKernel: output[i] = a[i] + acc(0..3) + 1, acc(k) = 0+1+2+3 = 6
    var engine = newEngine(runOpenCL)
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

  block: # hazardKernel: x[tick(x) + 1] += 7, the index side effect fires exactly once
    var engine = newEngine(runOpenCL)
    engine.ingest(hazardCl)

    var x: array[4, int32] = [0'i32, 0, 0, 0]
    let xDev = engine.allocDevice(int32, x.len)
    engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
    engine.launchKernel("hazardKernel", xDev, ())
    engine.copyToHostFromDevice(addr x[0], xDev, x.len)

    echo "  hazardKernel: x = [", x[0], ", ", x[1], ", ", x[2], ", ", x[3], "]"
    doAssert x[0] == 1  # the index expression was evaluated exactly once
    doAssert x[1] == 7  # stored back into the element the single index selected
    doAssert x[2] == 0  # a second index evaluation would have stored here
    echo "  OK"

  block: # blitKernel: x[block-expr] += 3, the block index is blitted and accumulated
    var engine = newEngine(runOpenCL)
    engine.ingest(blitCl)

    var x: array[4, int32] = [0'i32, 0, 0, 0]
    let xDev = engine.allocDevice(int32, x.len)
    engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
    engine.launchKernel("blitKernel", xDev, ())
    engine.copyToHostFromDevice(addr x[0], xDev, x.len)

    echo "  blitKernel: x = [", x[0], ", ", x[1], ", ", x[2], ", ", x[3], "]"
    doAssert x[0] == 3  # i = 0: x[0] += 3
    doAssert x[1] == 0
    doAssert x[2] == 3  # i = 1: x[2] += 3
    doAssert x[3] == 0
    echo "  OK"

  echo "All compound-assign execution tests passed"

when isMainModule:
  runTest()
