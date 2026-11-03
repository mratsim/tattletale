## CPU engine: a reserved-word local renaming into a name the proc already uses.
##
## Test that a kernel declaring both `register` and `register_nim` compiles
## and keeps the two locals apart:
##
## - the reserved-word rename pass maps `register` to `register_nim`,
##   a blind rename redefines the existing `register_nim` at C compile time
##   or silently shadows it
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc probe(x: int32, outBuf: ptr UncheckedArray[int32]) =
  var register_nim = x + 100
  var acc = x
  block:
    var register = x + 1
    acc = int32 register_nim + register
  outBuf[0] = acc

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var o: array[4, int32]
  engine.launchCpuFn probe(1'i32, cast[ptr UncheckedArray[int32]](addr o[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert o[0] == 103,
    "register_nim(101) + register(2) must read 103, a rename collision reads " & $o[0]
  echo "[OK] launchCpuFn: a reserved-word local renames past the names the proc already uses"

runTest()
