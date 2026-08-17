## Metal engine end-to-end: `bkMetal.init()` → `ingest()` → `run()`,
## with verified device output (2 + 3 == 5). Also covers ingest-once/run-many cache reuse,
## re-ingest invalidation, grid/blk dispatch,
## scalar args via the packed constant buffer, deviceName, and repeated init/run cycles
## that exercise the RAII `=destroy` path.
##
## Tested ABI (macOS, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_engine.nim

import std/unittest
import workspace/crucible

# ── MSL kernel sources (hand-written, engine-level) ───────────────────────────────────────────────────────────────────

const addMsl = """
#include <metal_stdlib>
using namespace metal;

kernel void addKernel(device uint* output [[buffer(0)]],
                      device const uint* a [[buffer(1)]],
                      device const uint* b [[buffer(2)]])
{
  output[0] = a[0] + b[0];
}
"""

const mulMsl = """
#include <metal_stdlib>
using namespace metal;

kernel void addKernel(device uint* output [[buffer(0)]],
                      device const uint* a [[buffer(1)]],
                      device const uint* b [[buffer(2)]])
{
  output[0] = a[0] * b[0];
}
"""

const vecAddMsl = """
#include <metal_stdlib>
using namespace metal;

kernel void vecAdd(device uint* output [[buffer(0)]],
                   device const uint* a [[buffer(1)]],
                   device const uint* b [[buffer(2)]],
                   uint3 tid [[thread_position_in_threadgroup]],
                   uint3 bid [[threadgroup_position_in_grid]],
                   uint3 bdim [[threads_per_threadgroup]])
{
  uint gid = bid.x * bdim.x + tid.x;
  output[gid] = a[gid] + b[gid];
}
"""

const scalarMsl = """
#include <metal_stdlib>
using namespace metal;

kernel void scalarKernel(device uint* output [[buffer(0)]],
                         constant int& x [[buffer(1)]],
                         constant float& f [[buffer(2)]])
{
  output[0] = uint(x);
  output[1] = uint(f);
}
"""

# ── Host side ─────────────────────────────────────────────────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal engine":

    test "a+b kernel executes on the device (2 + 3 == 5)":
      var engine = bkMetal.init()
      engine.ingest(addMsl)
      echo addMsl
      var res: array[1, uint32]
      engine.run("addKernel", res, ([2'u32], [3'u32]))
      check res[0] == 5

    test "ingest-once run-many keeps producing correct results across runs":
      # "No recompiles" is structural. Ingest is the only compileLibrary call site,
      # and the PSO cache is keyed by (kernel, argSizes).
      var engine = bkMetal.init()
      engine.ingest(addMsl)
      var res: array[1, uint32]
      engine.run("addKernel", res, ([2'u32], [3'u32]))
      check res[0] == 5
      engine.run("addKernel", res, ([10'u32], [20'u32]))
      check res[0] == 30

    test "re-ingest with a changed body invalidates and recompiles":
      var engine = bkMetal.init()
      engine.ingest(addMsl)
      var res: array[1, uint32]
      engine.run("addKernel", res, ([2'u32], [3'u32]))
      check res[0] == 5
      engine.ingest(mulMsl)
      engine.run("addKernel", res, ([2'u32], [3'u32]))
      check res[0] == 6

    test "dispatch with grid/blk via dispatchThreadgroups":
      var engine = bkMetal.init()
      engine.ingest(vecAddMsl)
      var a, b, res: array[128, uint32]
      for i in 0 ..< 128:
        a[i] = uint32(i)
        b[i] = 1'u32
      engine.run<<(2, 64)>>("vecAdd", res, (a, b))
      for i in 0 ..< 128:
        check res[i] == uint32(i) + 1

    test "scalar args pack into the constant buffer at 16-byte slots":
      var engine = bkMetal.init()
      engine.ingest(scalarMsl)
      echo scalarMsl
      var res: array[2, uint32]
      engine.run("scalarKernel", res, (7'i32, 1.5'f32))
      check res[0] == 7
      check res[1] == 1
      engine.run("scalarKernel", res, (10'i32, 2.5'f32))
      check res[0] == 10
      check res[1] == 2

    test "deviceName returns the Metal device name":
      var engine = bkMetal.init()
      let name = engine.deviceName()
      echo "  device: ", name
      check name.len > 0

    test "repeated init/run cycles destroy cleanly (RAII)":
      for i in 0 ..< 5:
        var engine = bkMetal.init()
        engine.ingest(addMsl)
        var res: array[1, uint32]
        engine.run("addKernel", res, ([2'u32], [3'u32]))
        check res[0] == 5

when isMainModule:
  runTest()
