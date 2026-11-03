## CPU engine: whole-array runtime parameter.
##
## Test that a kernel taking a whole Nim array as a runtime parameter
## reads the copied array in the launch data struct:
##
## - the data struct member is a C array, a scalar cast into it does not
##   compile
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc probe(arr: array[4, int32], outBuf: ptr UncheckedArray[int32]) =
  outBuf[0] = arr[2]
  outBuf[1] = arr[3]

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var a: array[4, int32] = [1, 2, 3, 4]
  var o: array[4, int32]
  engine.launchCpuFn probe(a, cast[ptr UncheckedArray[int32]](addr o[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert o[0] == 3, "arr[2] must reach the kernel, got " & $o[0]
  doAssert o[1] == 4, "arr[3] must reach the kernel, got " & $o[1]
  echo "[OK] launchCpuFn: a whole-array runtime parameter copies into the data struct"

runTest()
