## CIRv2 verifier flag default
##
## Test that pass-boundary verification defaults off without -d:CIR_verify and on with it: the metal: kernel runs clean in both compilations.
## moduleD, which the frontend path uses as its module constructor, defaults verifyIR accordingly. Both run commands must exit 0.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:/tmp/m4f-out --nimcache:/tmp/m4f-cache \
##     workspace/crucible/tests/verify/t_cirv2_verify_flag.nim
##   nim c -r --hints:off --warnings:off -d:CIR_verify --outdir:/tmp/m4f-out-v --nimcache:/tmp/m4f-cache-v \
##     workspace/crucible/tests/verify/t_cirv2_verify_flag.nim

import workspace/crucible
import workspace/crucible/src/codegen/ir/cirv2_build

const addSrc = metal:
  proc addKernel(
      output: ptr UncheckedArray[uint32],
      a: ptr UncheckedArray[uint32],
      b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]

proc runTest() =
  when defined(CIR_verify):
    doAssert moduleD("flag_probe", @[], @[]).verifyIR,
      "expected verifyIR to default on with -d:CIR_verify"
  else:
    doAssert not moduleD("flag_probe", @[], @[]).verifyIR,
      "expected verifyIR to default off without -d:CIR_verify"

  block metalKernel:
    var engine = newEngine(runMetal)
    engine.ingest(addSrc)
    var a: array[1, uint32] = [7'u32]
    var b: array[1, uint32] = [5'u32]
    let aDev = engine.allocDevice(uint32, a.len)
    let bDev = engine.allocDevice(uint32, b.len)
    let oDev = engine.allocDevice(uint32, 1)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
    engine.launchKernel("addKernel", oDev, (aDev, bDev))
    var out32: array[1, uint32]
    engine.copyToHostFromDevice(addr out32[0], oDev, 1)
    doAssert out32[0] == 12, "expected 7 + 5 computed on the device, got: " & $out32[0]

when isMainModule:
  runTest()
