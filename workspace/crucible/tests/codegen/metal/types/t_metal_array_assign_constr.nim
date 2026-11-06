## Metal: whole-array assignment from a constructor decomposes to `__builtin_memcpy`.
##
## Test that a kernel assigning an array constructor to a whole array reads
## every element back byte-exact against the CPU reference.
## The constructor lists `float32(i) + 1.0` and every `output[i]` reads that value.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_array_assign_constr.nim

import std/unittest
import workspace/crucible

const N = 8

const constrAssignMsl = metal:
  proc kern(output: ptr UncheckedArray[float32]) {.global.} =
    var a: array[8, float32]
    a = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32, 5.0'f32, 6.0'f32, 7.0'f32, 8.0'f32]
    let t = thread_position_in_grid.x
    output[t] = a[t]

proc runTest() =
  suite "Metal - constructor whole-array assignment decomposes to a copy":
    test "constructor assignment is byte-exact":
      var engine = newEngine(runMetal)
      engine.ingest(constrAssignMsl)
      let oDev = engine.allocDevice(float32, N)
      var sentinel: array[N, float32]
      for i in 0 ..< N:
        sentinel[i] = -7.0'f32
      engine.copyToDeviceFromHost(oDev, addr sentinel[0], N)
      engine.launchKernel<<(grid: (N, 1), blk: (1, 1))>>("kern", oDev, ())
      var got: array[N, float32]
      engine.copyToHostFromDevice(addr got[0], oDev, N)
      # CPU reference: the constructor lists i + 1 and the memcpy delivers it intact
      for i in 0 ..< N:
        check got[i] == float32(i) + 1.0'f32

when isMainModule:
  runTest()
