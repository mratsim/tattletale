## WebGPU: generic device fn instantiation derives the arg's address space
## from the authoritative map, not from a symbol-kind mirror.
##
## Covered:
## - a `{.smem.}` var arg specializes the generic param as
##   `ptr<workgroup, u32>` and encodes the space in the instantiation name
##   (`_wmut_l` suffix), so distinct spaces yield distinct instantiations
## - a local (`function`-space) arg specializes the same generic as
##   `ptr<function, u32>` (`_lmut_l` suffix) and runs on wgpu-native
##
## The smem instantiation is text-asserted on the macro output: wgpu-native
## (naga) rejects `ptr<workgroup, ...>` function parameters, a WGSL
## limitation independent of this derivation, and a shader module carrying
## such a function reports its error on the next engine run.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_generic_smem_arg.nim

import std/[strutils, unittest]
import workspace/crucible

const smemGenericWgsl = webgpu:
  var scratch {.smem.}: array[4, uint32]

  proc bump[T](p: T, v: uint32): uint32 {.device.} =
    result = p[] + v

  proc smemGenericKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    scratch[0] = 5'u32
    output[0] = bump(addr scratch[0], 1'u32)
    output[1] = scratch[0]

const localGenericWgsl = webgpu:
  proc bump[T](p: T, v: uint32): uint32 {.device.} =
    result = p[] + v

  proc localGenericKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    var x: uint32 = 7
    output[0] = bump(addr x, 1'u32)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "WebGPU - generic instantiation address space":
    test "smem arg specializes the generic param as ptr<workgroup, ...>":
      check "var<workgroup> scratch: array<u32, 4>;" in smemGenericWgsl
      # The instantiation name encodes the arg space (`w` = workgroup), so a
      # smem call and a local call never alias one instantiation.
      check "_wmut_l(" in smemGenericWgsl
      check "(p: ptr<workgroup, u32>, v: u32)" in smemGenericWgsl
    test "local arg specializes as ptr<function, ...> and runs":
      check "_lmut_l(" in localGenericWgsl
      var engine = bkWGSL.init()
      engine.ingest(localGenericWgsl)
      let src = engine.getArtifact()
      check "(p: ptr<function, u32>, v: u32)" in src
      var res: array[1, uint32]
      engine.run("localGenericKernel", res, ())
      check res[0] == 8'u32

when isMainModule:
  runTest()
