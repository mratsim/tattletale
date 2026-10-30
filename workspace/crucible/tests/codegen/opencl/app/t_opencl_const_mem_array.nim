## OpenCL: program-scope `__constant` from a const_mem array local.
##
## Test that a `{.const_mem.}` array local with an array-constructor
## initializer hoists to a compiling program-scope `__constant` declaration
## and the kernel reads the declared values back through the engine.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/app/t_opencl_const_mem_array.nim

import workspace/crucible

proc readConst(o: ptr UncheckedArray[float32]) =
  var tbl {.const_mem.}: array[4, float32] = [1.5'f32, 2.5'f32, 3.5'f32, 4.5'f32]
  for i in 0 ..< 4:
    o[i] = 10.0'f32 * tbl[i]

proc runTest() =
  var engine = newEngine(runOpenCL)
  let oDev = engine.allocDevice(float32, 4)
  engine.launchDeviceFn readConst(oDev) do:
    grid = (1, 1)
    blk = (1, 1)
  var got: array[4, float32]
  engine.copyToHostFromDevice(addr got[0], oDev, 4)
  for i in 0 ..< 4:
    let expected = 10.0'f32 * [1.5'f32, 2.5'f32, 3.5'f32, 4.5'f32][i]
    doAssert got[i] == expected,
      "const_mem array: element " & $i & " got " & $got[i] & " want " & $expected
  echo "[OK] const_mem array local hoists to a compiling __constant declaration"

runTest()
