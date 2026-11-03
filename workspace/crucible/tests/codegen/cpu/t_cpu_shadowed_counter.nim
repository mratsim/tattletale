## CPU engine: a kernel local named after a generated launch counter.
##
## Test that a kernel declaring locals named `crucibleGx` and `crucibleWg`
## compiles and that the coordinate reads still evaluate to the true
## threadgroup coordinates:
##
## - the launch appends a per-launch numeric suffix to each generated
##   counter (`crucibleGx_7`), a kernel local can never collide with one
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc shadowKernel(outBuf: ptr UncheckedArray[int32]) =
  var crucibleGx: int32 = 99
  var crucibleWg: int32 = 77
  outBuf[int(threadgroup_position_in_grid.x)] =
    int32(threadgroup_position_in_grid.x) + crucibleGx + crucibleWg

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var buf: array[4, int32]
  engine.launchCpuFn shadowKernel(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (3, 1, 1)
    blk = (1, 1, 1)
  for i in 0 ..< 3:
    doAssert buf[i] == int32(i) + 99 + 77,
      "workgroup " & $i & " must read its true threadgroup coordinate, got " & $buf[i]
  echo "[OK] launchCpuFn: kernel locals named crucibleGx and crucibleWg shadow nothing"

when isMainModule:
  runTest()
