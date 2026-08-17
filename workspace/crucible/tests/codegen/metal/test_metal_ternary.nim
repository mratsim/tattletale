## Metal: ternary expression lowering. The DSL funnels `if cond: a else: b`
## in an expression slot through the ternary IR shape. The MSL printer
## emits the C-style `(cond ? a : b)` form. The kernel covers a compile-time constant condition
## and a data-dependent condition, so the device executes a real branch,
## and both outputs are asserted byte-exact.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_ternary.nim

import std/unittest
import workspace/crucible

const code = metal:
  proc ternaryKernel(output: ptr UncheckedArray[uint32];
                     a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = if a[0] > b[0]: a[0] else: b[0]
    output[1] = if true: 1'u32 else: 0'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo code

  suite "Metal - ternary lowering":
    test "data-dependent and constant ternaries":
      var engine = bkMetal.init()
      engine.ingest(code)
      var res: array[2, uint32]
      var a = [7'u32]
      var b = [3'u32]
      engine.run("ternaryKernel", res, (a, b))
      check res[0] == 7
      check res[1] == 1
      a[0] = 2
      b[0] = 9
      engine.run("ternaryKernel", res, (a, b))
      check res[0] == 9
      check res[1] == 1

when isMainModule:
  runTest()
