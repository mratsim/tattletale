## CPU engine: a ptr array[N, T] runtime parameter.
##
## Test that a kernel indexing a pointer-to-array parameter compiles and reads
## the pointee elements:
##
## - the C declaration prints the pointer-to-array declarator `T (*p)[N]`,
##   `T* p` would reject the element indexing at C compile time
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc probe(p: ptr array[4, int32], outBuf: ptr UncheckedArray[int32]) =
  outBuf[0] = p[0]
  outBuf[1] = p[2]

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var a: array[4, int32] = [11, 22, 33, 44]
  var o: array[4, int32]
  engine.launchCpuFn probe(addr a, cast[ptr UncheckedArray[int32]](addr o[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert o[0] == 11, "p[0] must read the first element 11, got " & $o[0]
  doAssert o[1] == 33, "p[2] must read the third element 33, got " & $o[1]
  echo "[OK] launchCpuFn: a ptr array parameter declares as T (*p)[N] and indexes the pointee"

runTest()
