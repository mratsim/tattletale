## OpenCL: whole-array assignment and array-valued declaration init.
##
## Test that a kernel copying one whole array to another in all three shapes
## (assignment, declaration init, copy nested in a branch) produces byte-exact
## results against a CPU reference, each array element sentinel-checked.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/app/t_opencl_array_assign.nim

import workspace/crucible

const N = 8

const arrayAssignCl = opencl:
  proc copyKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var b: array[8, float32]
    for i in 0 ..< 8:
      b[i] = float32(i) * 1.5'f32 + 0.25'f32
    # whole-array assignment
    var a: array[8, float32]
    a = b
    # declaration init from another array
    var c = b
    # whole-array copy nested in a branch
    var d: array[8, float32]
    let t = thread_position_in_grid.x
    if t == 0:
      d = a
    for i in 0 ..< 8:
      d[i] = d[i] + 1.0'f32
    output[t] = a[3] + c[4] + d[t]

proc runTest() =
  var engine = newEngine(runOpenCL)
  engine.ingest(arrayAssignCl)
  let oDev = engine.allocDevice(float32, N)
  # sentinel: every element must be overwritten by the kernel
  var sentinel: array[N, float32]
  for i in 0 ..< N:
    sentinel[i] = -7.0'f32
  engine.copyToDeviceFromHost(oDev, addr sentinel[0], N)
  engine.launchKernel<<(grid: (N, 1), blk: (1, 1))>>("copyKernel", oDev, ())
  var got: array[N, float32]
  engine.copyToHostFromDevice(addr got[0], oDev, N)
  # CPU reference over the same inputs:
  # b[i] = i*1.5 + 0.25, a = c = b, thread 0 copies a into d,
  # every d element gains 1, output[t] = b[3] + b[4] + d[t]
  for t in 0 ..< N:
    let b_i = float32(t) * 1.5'f32 + 0.25'f32
    let expected = 4.75'f32 + 6.25'f32 + (b_i + 1.0'f32)
    doAssert got[t] == expected,
      "element " & $t & " got " & $got[t] & " want " & $expected
  echo "[OK] whole-array assignment, declaration init, and branch-nested copy are byte-exact"

when isMainModule:
  runTest()
