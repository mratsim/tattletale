## CPU engine: launch refusals for schedules the emitted C cannot index.
##
## Test that a launch refuses loudly when the schedule exceeds int32:
##
## - a grid dim or the grid workgroup total beyond int32 truncates,
##   the launch would silently run zero workgroups behind a successful return
## - a non-literal workgroupsPerTask refuses at compile time in serial mode,
##   the serial launch never evaluates the knob
##
## Run:
##   nim test_crucible_cpu
import std/strutils
import workspace/crucible

proc probe(outBuf: ptr UncheckedArray[uint32]) =
  outBuf[0] = 1

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()

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

  block:   # a non-literal knob runs the pooled launch, serial mode refuses it at compile time
    var outBuf: array[8, uint32]
    when compileOption("threads"):
      var k = 4
      engine.launchCpuFn probe(cast[ptr UncheckedArray[uint32]](addr outBuf[0])) do:
        grid = (4, 1)
        blk = (1, 1)
        workgroupsPerTask = k
      doAssert outBuf[0] == 1, "a pooled launch with a non-literal knob must run"
      echo "[OK] workgroupsPerTask: a pooled launch takes a non-literal knob and validates it at runtime"
    else:
      engine.launchCpuFn probe(cast[ptr UncheckedArray[uint32]](addr outBuf[0])) do:
        grid = (4, 1)
        blk = (1, 1)
        workgroupsPerTask = 3
      doAssert outBuf[0] == 1, "a literal knob is accepted but ignored, the launch covers the whole grid"
      echo "[OK] workgroupsPerTask: the serial launch ignores a literal knob, the non-literal refusal is verified by manual_cpu_serial_knob_refused.nim"

runTest()
