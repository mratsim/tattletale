## CPU engine: the per-axis grid*block schedule guard.
##
## Test that a launch whose per-axis grid*block product exceeds int32 refuses:
##
## - thread_position_in_grid computes grid*blk + thread per axis in int32
##   in the emitted C, a product at or beyond 2^31 overflows at the last
##   threads and kernels indexing memory by the coordinate read wrong
##   elements
## - grid (65536, 1, 1) with blk (65536, 1, 1) passes the per-dim bounds
##   and the workgroup-total bound while the x axis product is 2^32
##
## Run:
##   nim test_crucible_cpu
import std/strutils
import workspace/crucible

proc probe(outBuf: ptr UncheckedArray[int32]) =
  outBuf[0] = 1

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var o: array[1, int32]
  var refusal = ""
  try:
    engine.launchCpuFn probe(cast[ptr UncheckedArray[int32]](addr o[0])) do:
      grid = (65536, 1, 1)
      blk = (65536, 1, 1)
  except AssertionDefect as e:
    refusal = e.msg
  doAssert refusal.len > 0,
    "grid 65536 times blk 65536 on the x axis is 2^32, beyond int32 the emitted thread_position_in_grid arithmetic overflows and the launch must refuse loudly"
  doAssert o[0] == 0, "a refused launch must not run the kernel"
  echo "[OK] launchCpuFn: a per-axis grid*block product beyond int32 refuses"

runTest()
