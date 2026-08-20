## WebGPU: address-space tagging of locals and pointer aliases.
##
## Covered:
## - an unannotated local emits no address-space template: `var<private>` is
##   invalid in function scope, `function` is the default scope for locals
## - a `let` alias of a storage param keeps a `ptr<storage, ...>` type
## - a `let` alias of a `{.smem.}` var keeps a `ptr<workgroup, ...>` type
## - the `{.smem.}` var itself lifts to a module-scope `var<workgroup>`
##
## Every kernel runs through `engine.run()` and the outputs are asserted.
## Source-text asserts read the generated WGSL from `engine.getArtifact()`.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_address_space.nim

import std/[strutils, unittest]
import workspace/crucible

const addressSpaceWgsl = webgpu:
  var scratch {.smem.}: array[4, uint32]

  proc plainLocalKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    var x: uint32 = 7
    var arr: array[4, uint32]
    arr[0] = x
    output[0] = arr[0]

  proc paramAliasKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    var x: uint32 = 7
    let p = output
    output[0] = x
    output[1] = p[0]

  proc smemAliasKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 1, 1).} =
    let tid = thread_position_in_threadgroup.x
    scratch[tid] = tid * 3'u32
    let p = addr scratch[tid]
    output[tid] = p[]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "WebGPU - address-space tagging":
    test "unannotated locals emit no address-space template and run":
      var engine = bkWGSL.init()
      engine.ingest(addressSpaceWgsl)
      let src = engine.getArtifact()
      check "<private>" notin src
      check "var x: u32 = 7;" in src
      check "var arr: array<u32, 4>;" in src
      var res: array[1, uint32]
      engine.run("plainLocalKernel", res, ())
      check res[0] == 7'u32
    test "let alias of a storage param keeps ptr<storage, ...> and runs":
      var engine = bkWGSL.init()
      engine.ingest(addressSpaceWgsl)
      let src = engine.getArtifact()
      check "let p: ptr<storage, array<u32>, read_write> = (&output);" in src
      var res: array[2, uint32]
      engine.run("paramAliasKernel", res, ())
      check res[0] == 7'u32
      check res[1] == 7'u32
    test "let alias of a smem var keeps ptr<workgroup, ...> and runs":
      var engine = bkWGSL.init()
      engine.ingest(addressSpaceWgsl)
      let src = engine.getArtifact()
      check "var<workgroup> scratch: array<u32, 4>;" in src
      check "let p: ptr<workgroup, u32> = (&scratch[" in src
      var res: array[4, uint32]
      engine.run("smemAliasKernel", res, ())
      for i in 0 ..< 4:
        check res[i] == uint32(i * 3)

when isMainModule:
  runTest()
