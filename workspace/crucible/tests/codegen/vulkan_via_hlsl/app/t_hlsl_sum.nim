## HLSL: sum kernel with a counted for loop over grid-offset indices, through the real pass bundle.
##
## Test that the `hlsl:` macro emits HLSL for a kernel accumulating buffer elements
## into output[0] with a counted for loop over grid-offset indices,
## compiling clean through DXC and spirv-val with the binding reflection asserted.
##
## The launch stays compile-only: the descriptor route serializes buffer
## bindings and scalar kernel arguments have no uniform-block layout yet.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_sum.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const sumHlsl = hlsl:
  proc sumKernel(output, x: ptr UncheckedArray[uint32], n: uint32) {.global.} =
    let first = thread_position_in_grid.x.uint32
    var acc: uint32 = 0
    for i in 0 ..< n:
      acc += x[first + i]
    output[0] = acc

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order: `output` is written, `x` is read-only
  let sumBindings: seq[BindingExpectation] = @[("output", 0, false), ("x", 1, true)]
  discard compileCheck("sum", sumHlsl, sumBindings, "gid.x")

  echo "  OK - t_hlsl_sum"

when isMainModule:
  runTest()
