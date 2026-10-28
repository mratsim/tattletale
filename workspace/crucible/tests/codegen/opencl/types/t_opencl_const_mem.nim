## `{.const_mem.}` locals referenced from nested scopes and device functions
##
## One kernel covers the three declaration sites:
##
## - kernel top level: `cmBias`, read in both `if` branches
## - nested scope: `cmMult`, declared inside the taken branch
## - device function: `cmK`, declared in `cmScale`
##
## Contract:
##
## - the outputs assert byte-exact values against the host reference values,
##   never print-only
## - ## - before the const_mem relocation fix the declarations dropped and the OpenCL compiler
##   rejected the source with undeclared identifiers
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/types/t_opencl_const_mem.nim

import std/unittest
import workspace/crucible

proc cmScale(a: float32): float32 =
  var cmK {.const_mem.}: float32 = 3.0'f32
  result = a * cmK

const constMemCl = opencl:
  proc constMemKernel(output, a: ptr UncheckedArray[float32]) {.global.} =
    var cmBias {.const_mem.}: float32 = 1.0'f32
    if a[0] > 0'f32:
      var cmMult {.const_mem.}: float32 = 2.0'f32
      output[0] = a[0] * cmMult + cmBias
    else:
      output[0] = cmBias
    output[1] = cmScale(a[1])

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "OpenCL - const_mem locals":

    test "kernel-top-level, nested-scope, and device-fn const_mem resolve":
      var engine = newEngine(runOpenCL)
      engine.ingest(constMemCl)
      var a: array[2, float32] = [2.0'f32, 5.0'f32]
      let aDev = engine.allocDevice(float32, a.len)
      let oDev = engine.allocDevice(float32, 2)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      engine.launchKernel("constMemKernel", oDev, (aDev,))
      var out32: array[2, float32]
      engine.copyToHostFromDevice(addr out32[0], oDev, 2)
      # a[0] > 0: 2 * 2 + 1 = 5, device fn: 5 * 3 = 15
      check out32[0] == 5.0'f32
      check out32[1] == 15.0'f32

    test "the nested-scope else branch reads the top-level constant":
      var engine = newEngine(runOpenCL)
      engine.ingest(constMemCl)
      var a: array[2, float32] = [-1.0'f32, 5.0'f32]
      let aDev = engine.allocDevice(float32, a.len)
      let oDev = engine.allocDevice(float32, 2)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      engine.launchKernel("constMemKernel", oDev, (aDev,))
      var out32: array[2, float32]
      engine.copyToHostFromDevice(addr out32[0], oDev, 2)
      # else branch: cmBias alone = 1, device fn: 5 * 3 = 15
      check out32[0] == 1.0'f32
      check out32[1] == 15.0'f32

when isMainModule:
  runTest()
