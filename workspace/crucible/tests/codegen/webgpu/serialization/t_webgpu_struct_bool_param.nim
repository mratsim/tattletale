## WebGPU: a bool field on a struct kernel param binds as the NIM_BOOL word
##
## Test that a value struct with a bool field marshals through the WGSL
## storage binding: the field declares NIM_BOOL, body reads coerce to bool,
## and a local struct value with a bool field keeps working
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/serialization/t_webgpu_struct_bool_param.nim

import std/[unittest, strformat]
import workspace/crucible

type
  Flags = object
    count: uint32
    on: bool

const kernelStructBool = webgpu:
  proc structBoolKernel(output: ptr UncheckedArray[uint32], f: Flags) {.global.} =
    if f.on:
      output[0] = f.count
    else:
      output[0] = 0'u32

const kernelLocalBool = webgpu:
  proc localBoolKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var f = Flags(count: 9'u32, on: false)
    f.on = true
    if f.on:
      output[0] = f.count

proc runTest() =
  suite "WebGPU - struct bool field marshalling":

    test "bool field on a struct kernel param (u32 word in storage)":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelStructBool)
      echo kernelStructBool
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("structBoolKernel", oDev, (Flags(count: 42'u32, on: true),))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42
      engine.launchKernel("structBoolKernel", oDev, (Flags(count: 42'u32, on: false),))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 0

    test "local struct value with a bool field":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelLocalBool)
      echo kernelLocalBool
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("localBoolKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 9

when isMainModule:
  runTest()
