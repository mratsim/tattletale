## CPU engine: nested arrays in a kernel.
##
## Test that `array[N, array[M, T]]` locals and runtime parameters declare
## with every dimension stacked in C (`float name[3][4]`) and copy whole:
##
## - a one-dimension declaration or `sizeof` breaks subscripts and copies part of the array
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc probe(outBuf: ptr UncheckedArray[int32]) =
  var m, n: array[3, array[4, float32]]
  n[0][0] = 1.5'f32
  n[2][3] = 2.5'f32
  m = n
  n[2][3] = 99.0'f32
  outBuf[0] = int32(m[0][0] * 2)
  outBuf[1] = int32(m[2][3] * 4)
  outBuf[2] = int32(n[2][3])

proc probeParam(arr: array[3, array[4, float32]], outBuf: ptr UncheckedArray[int32]) =
  outBuf[3] = int32(arr[1][2] * 2)

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var buf: array[4, int32]
  var a: array[3, array[4, float32]]
  a[1][2] = 3.5'f32
  engine.launchCpuFn probe(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  engine.launchCpuFn probeParam(a, cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert buf[0] == 3, "m[0][0] must read 1.5 * 2 after the copy, got " & $buf[0]
  doAssert buf[1] == 10, "m[2][3] must read 2.5 * 4 after the copy, got " & $buf[1]
  doAssert buf[2] == 99, "n[2][3] must read the write after the copy, got " & $buf[2]
  doAssert buf[3] == 7, "arr[1][2] must reach the kernel, 3.5 * 2, got " & $buf[3]
  echo "[OK] launchCpuFn: nested arrays declare with stacked dimensions and copy whole"

runTest()
