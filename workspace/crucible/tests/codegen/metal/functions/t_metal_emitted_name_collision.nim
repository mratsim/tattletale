## Metal: distinct Nim procs whose emitted names collide at the C level.
## Three collision shapes over in-block device procs and kernels:
## - overloads: same name, different signatures
## - distinct names sanitizing identically (`a!b` vs `a_b`)
## - a device proc sharing a kernel's name: the kernel keeps the plain name,
##   the device proc takes a suffix
## Each shape must emit one distinct C name per proc, all read back.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_emitted_name_collision.nim

import std/unittest
import workspace/crucible

const collisionMsl = metal:
  proc mixv(a: int32): int32 {.device.} =
    a + 1
  proc mixv(a: int32, b: int32): int32 {.device.} =
    a * b
  proc `a!b`(a: int32): int32 {.device.} =
    a + 7
  proc `a_b`(a: int32): int32 {.device.} =
    a * 3

  proc k(output: ptr UncheckedArray[int32]) {.global.} =
    output[0] = mixv(5'i32)
    output[1] = mixv(5'i32, 6'i32)
    output[2] = `a!b`(2'i32)
    output[3] = `a_b`(2'i32)

const kernelNameMsl = metal:
  proc k(a: int32): int32 {.device.} =
    a * 5

  proc k(output: ptr UncheckedArray[int32]) {.global.} =
    output[0] = k(4'i32)

proc runTest() =
  suite "Metal - emitted-name collision":
    test "colliding names distinct after interning":
      var engine = newEngine(runMetal)
      engine.ingest(collisionMsl)
      let oDev = engine.allocDevice(int32, 4)
      engine.launchKernel("k", oDev, ())
      var res: array[4, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 6
      check res[1] == 30
      check res[2] == 9
      check res[3] == 6

    test "device proc sharing a kernel name takes the suffix, kernel launches by plain name":
      var engine = newEngine(runMetal)
      engine.ingest(kernelNameMsl)
      let oDev = engine.allocDevice(int32, 1)
      engine.launchKernel("k", oDev, ())
      var res: array[1, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 20

when isMainModule:
  runTest()
