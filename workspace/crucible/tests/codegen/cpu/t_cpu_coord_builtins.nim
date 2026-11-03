## CPU engine: coordinate builtins rewrite to launch-site expressions.
##
## Test that every canonical coordinate builtin read produces the value the launch schedule assigns, over grid (2, 1) x blk (4, 1).
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc coordProbe(C: ptr UncheckedArray[uint32]) =
  let tid = int(thread_position_in_threadgroup.x)
  let gid = int(thread_position_in_grid.x)
  C[gid] = uint32(gid * 100 + tid)
  C[gid + 8] = uint32(threadgroup_position_in_grid.x)
  C[gid + 16] = uint32(thread_index_in_threadgroup)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  const N = 24
  var host: array[N, uint32]

  engine.launchCpuFn coordProbe(cast[ptr UncheckedArray[uint32]](addr host[0])) do:
    grid = (2, 1)
    blk = (4, 1)

  for tg in 0 ..< 2:
    for tid in 0 ..< 4:
      let gid = tg * 4 + tid
      doAssert host[gid] == uint32(gid * 100 + tid),
        "position: gid " & $gid & " got " & $host[gid] & " want " & $(gid * 100 + tid)
      doAssert host[gid + 8] == uint32(tg),
        "threadgroup: gid " & $gid & " got " & $host[gid + 8] & " want " & $tg
      doAssert host[gid + 16] == uint32(tid),
        "thread index: gid " & $gid & " got " & $host[gid + 16] & " want " & $tid
  echo "[OK] launchCpuFn: coordinate builtins rewrite to data-struct member expressions"

runTest()
