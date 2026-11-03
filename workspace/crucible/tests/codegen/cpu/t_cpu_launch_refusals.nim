## CPU engine: launch refusals for schedules the emitted C cannot index.
##
## Test that a launch refuses loudly when a grid dim or the grid workgroup
## total exceeds int32. The emitted C stores the dims and the workgroup
## total in int32, a truncated value would silently run zero workgroups
## behind a successful launch.
##
## Run:
##   nim test_crucible_cpu
import std/strutils
import workspace/crucible

proc probe(outBuf: ptr UncheckedArray[uint32]) =
  outBuf[0] = 1

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runCpu)

  block:   # a grid workgroup total beyond int32 truncates to a negative count
    var outBuf: array[1, uint32]
    var refusal = ""
    try:
      engine.launchCpuFn probe(cast[ptr UncheckedArray[uint32]](addr outBuf[0])) do:
        grid = (2, 2, 600000000)
        blk = (1, 1, 1)
    except AssertionDefect as e:
      refusal = e.msg
    doAssert refusal.len > 0,
      "grid (2, 2, 600000000) has a workgroup total of 2400000000, beyond int32 the launch must refuse loudly"
    doAssert outBuf[0] == 0, "a refused launch must not run the kernel"
    echo "[OK] refusal: the grid workgroup total beyond int32 refuses with the failing value"

  block:   # a single dim beyond int32 truncates at the data-struct fill
    var outBuf: array[1, uint32]
    var refusal = ""
    try:
      engine.launchCpuFn probe(cast[ptr UncheckedArray[uint32]](addr outBuf[0])) do:
        grid = (1'i64, 1'i64, 2147483648'i64)
        blk = (1, 1, 1)
    except AssertionDefect as e:
      refusal = e.msg
    doAssert refusal.len > 0,
      "grid z = 2147483648 does not fit int32, the launch must refuse loudly"
    doAssert outBuf[0] == 0, "a refused launch must not run the kernel"
    echo "[OK] refusal: a single grid dim beyond int32 refuses with the failing dim"

  echo "[OK] launchCpuFn: schedules the emitted C launch cannot index refuse loudly"

runTest()
