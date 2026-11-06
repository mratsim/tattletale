## SPIR-V: unary negation in a while condition's replayed splice
##
## Test that a statement the while-condition rewrite replays at the loop's back edge
## emits its unary negation at the operand's type.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_while_neg_splice.nim

import workspace/crucible

proc negSplice(output: ptr UncheckedArray[int32]) =
  var i: int32 = 0
  var n: int32 = 0
  while (block: n = -n + 1; i < 2):
    i = i + 1
  output[0] = n
  output[1] = i

proc runTest() =
  var engine = newEngine(runVulkan)
  let oDev = engine.allocDevice(int32, 2)
  engine.launchDeviceFn negSplice(oDev) do:
    lang = spirv
    grid = (1, 1)
    blk = (1, 1)
  var got: array[2, int32]
  engine.copyToHostFromDevice(addr got[0], oDev, 2)
  # splice runs before every test: n counts 1, 0, 1, ending at 1
  doAssert got[0] == 1, "n: got " & $got[0] & ", want 1"
  doAssert got[1] == 2, "i: got " & $got[1] & ", want 2"

when isMainModule:
  runTest()
  echo "[OK] SPIR-V: the while condition's replayed splice keeps its negation operand type"
