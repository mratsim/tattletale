## Metal: multi-kernel source, one `metal:` block emits two kernels
## (vec10_add, vec10_mul). One ingest compiles the whole library.
## Each run gets its own pipeline state from the engine's level-2 cache,
## keyed by (kernel, argSizes). The same engine dispatches both kernels,
## and the 10-element outputs are asserted byte-exact.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_vec10_multi_kernel.nim

import std/unittest
import workspace/crucible

const code = metal:
  proc vec10_add(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] + b[i]
  proc vec10_mul(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] * b[i]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo code

  var a: array[10, uint32] = [1'u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  var b: array[10, uint32] = [10'u32, 20, 30, 40, 50, 60, 70, 80, 90, 100]

  suite "Metal - multi-kernel source":
    test "vec10_add and vec10_mul from one library":
      var engine = bkMetal.init()
      engine.ingest(code)
      var res: array[10, uint32]
      engine.run("vec10_add", res, (a, b))
      for i in 0 ..< 10:
        check res[i] == a[i] + b[i]
      engine.run("vec10_mul", res, (a, b))
      for i in 0 ..< 10:
        check res[i] == a[i] * b[i]

when isMainModule:
  runTest()
