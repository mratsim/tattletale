## Metal: bit-reinterpreting device casts
## Test that `cast[float32](0xFF7FFFFF'u32)` round-trips its bit pattern,
## emitted on the device as the MSL bit reinterpret `as_type<float>`
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/app/t_metal_bitcast.nim

import std/unittest
import workspace/crucible

# ── MSL generation via the `metal:` macro ──────────────────────────────────

const bitcastMsl = metal:
  proc bitcastKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let negFloor = cast[float32](0xFF7FFFFF'u32)
    output[0] = cast[uint32](negFloor)

# ── Host code ───────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - bitcast":

    test "cast[float32](0xFF7FFFFF'u32) round-trips its bit pattern on the device":
      var engine = newEngine(runMetal)
      engine.ingest(bitcastMsl)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("bitcastKernel", oDev, ())
      var out32: array[1, uint32]
      engine.copyToHostFromDevice(addr out32[0], oDev, 1)
      # -FLT_MAX, not the numeric conversion of 4286578687
      check out32[0] == 0xFF7FFFFF'u32

runTest()
