## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## NVRTC: passByRef with non-lvalue arguments, a large struct passed by const&
## accepts temporaries at call sites.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/serialization/t_nvrtc_passbyref_nonlvalue.nim

import std/strformat
import workspace/crucible

type
  LargeStruct = object
    data: array[8, uint32]  # 32 bytes > 24 threshold

# Test 1: inline constructor arg
const kernelInline = cuda:
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0] + s.data[1]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = takeLarge(LargeStruct(data: [10'u32, 20, 30, 40, 50, 60, 70, 80]))

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  block:
    var buf: array[1, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelInline)
    let o1 = engine.allocDevice(uint32, buf.len)
    engine.launchKernel("kernelMain", o1, ())
    engine.copyToHostFromDevice(addr buf[0], o1, buf.len)
    doAssert buf[0] == 30, &"inline constructor: expected 30, got {buf[0]}"
    echo "  ✅ Test 1 — inline constructor arg: ", buf[0]

  # Test 2: function return value arg
  const kernelFnRet = cuda:
    proc makeLarge(val: uint32): LargeStruct {.device.} =
      result.data[0] = val
    proc takeLarge(s: LargeStruct): uint32 {.device.} =
      result = s.data[0]
    proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
      output[0] = takeLarge(makeLarge(42'u32))

  block:
    var buf: array[1, uint32]
    var engine = bkCuda.init()
    engine.ingest(kernelFnRet)
    let o2 = engine.allocDevice(uint32, buf.len)
    engine.launchKernel("kernelMain", o2, ())
    engine.copyToHostFromDevice(addr buf[0], o2, buf.len)
    doAssert buf[0] == 42, &"fn return: expected 42, got {buf[0]}"
    echo "  ✅ Test 2 — function return arg: ", buf[0]

  echo ""
  echo "  OK — all NVRTC passByRef non-lvalue execution tests pass"

when isMainModule:
  runTest()
