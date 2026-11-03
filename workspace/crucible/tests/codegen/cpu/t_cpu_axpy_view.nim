## CPU engine: an axpy kernel over a struct-with-pointer View runs in-process.
##
## Test that a View kernel proc, pointer + length in one object like the OpenCL sibling, runs on the CPU engine with host values.
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

type
  AxpyView = object
    p: ptr UncheckedArray[float32]
    n: int

proc axpy(y: AxpyView, alpha: float32, x: AxpyView) =
  for i in 0 ..< y.n:
    y.p[i] = alpha * x.p[i] + y.p[i]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  const N = 7
  var xHost: array[N, float32]
  var yHost: array[N, float32]
  var expected: array[N, float32]
  for i in 0 ..< N:
    xHost[i] = float32(i + 1)
    yHost[i] = float32(10 * (i + 1))
    expected[i] = 2.5'f32 * xHost[i] + yHost[i]

  engine.launchCpuFn axpy(AxpyView(p: cast[ptr UncheckedArray[float32]](addr yHost[0]), n: N),
                          2.5'f32,
                          AxpyView(p: cast[ptr UncheckedArray[float32]](addr xHost[0]), n: N)) do:
    grid = (1, 1)
    blk = (1, 1)

  for i in 0 ..< N:
    doAssert yHost[i] == expected[i],
      "axpy: element " & $i & " got " & $yHost[i] & " want " & $expected[i]
  echo "[OK] launchCpuFn: View kernel runs through the emitted-C launch on the CPU engine"

runTest()
