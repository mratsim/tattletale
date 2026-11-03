## CPU engine: the chevron-less derived schedule.
##
## Test that a launch without a schedule runs a coordinate-free kernel on (1, 1, 1) x (1, 1, 1).
## The derived launch matches the explicit single-workgroup schedule.
##
## Run:
##   nim test_crucible_cpu
import std/strformat
import workspace/crucible

proc scaleAll(outBuf: ptr UncheckedArray[float32], factor: float32) =
  for i in 0 ..< 8:
    outBuf[i] = outBuf[i] * factor

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()

  block:   # derived launch, coordinate-free kernel
    var buf: array[8, float32]
    for i in 0 ..< 8:
      buf[i] = float32(i + 1)
    engine.launchCpuFn scaleAll(cast[ptr UncheckedArray[float32]](addr buf[0]), 3.0'f32)
    for i in 0 ..< 8:
      doAssert buf[i] == float32(3 * (i + 1)),
        &"derived [{i}]: got {buf[i]} want {3 * (i + 1)}"
    echo "[OK] chevron-less derived launch (1, 1, 1) x (1, 1, 1)"

  block:   # derived launch matches the explicit single-workgroup schedule
    var bufA: array[8, float32]
    var bufB: array[8, float32]
    for i in 0 ..< 8:
      bufA[i] = float32(i + 1)
      bufB[i] = float32(i + 1)
    engine.launchCpuFn scaleAll(cast[ptr UncheckedArray[float32]](addr bufA[0]), 3.0'f32)
    engine.launchCpuFn scaleAll(cast[ptr UncheckedArray[float32]](addr bufB[0]), 3.0'f32) do:
      grid = (1, 1)
      blk = (1, 1)
    for i in 0 ..< 8:
      doAssert bufA[i] == bufB[i],
        &"derived vs explicit [{i}]: {bufA[i]} vs {bufB[i]}"
    echo "[OK] derived launch matches the explicit single-workgroup schedule"

runTest()
echo "t_cpu_derived_schedule: all checks passed"
