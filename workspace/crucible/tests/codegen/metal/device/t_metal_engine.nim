## Metal engine end-to-end: `bkMetal.init()` → `ingest()` → `run()`,
## with verified device output. Covers ingest-once/run-many cache reuse,
## re-ingest invalidation, grid/blk dispatch, scalar args, deviceName, and repeated
## init/run cycles that exercise the RAII `=destroy` path. The plain add-kernel
## smoke itself is owned by t_metal_add.nim.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_engine.nim

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

    test "ingest-once run-many keeps producing correct results across runs":
      # "No recompiles" is structural. Ingest is the only compileLibrary call site,
      # and the PSO cache is keyed by (kernel, argSizes).
      var engine = bkMetal.init()
      engine.ingest(addMsl)
      var a: array[1, uint32] = [2'u32]
      var b: array[1, uint32] = [3'u32]
      let aDev = engine.allocDevice(sizeof(uint32))
      engine.copyToDeviceFromHost(aDev, addr a[0], sizeof(uint32))
      let bDev = engine.allocDevice(sizeof(uint32))
      engine.copyToDeviceFromHost(bDev, addr b[0], sizeof(uint32))
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("addKernel", oDev, (aDev, bDev))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      check res[0] == 5
      var a2: array[1, uint32] = [10'u32]
      var b2: array[1, uint32] = [20'u32]
      engine.copyToDeviceFromHost(aDev, addr a2[0], sizeof(uint32))
      engine.copyToDeviceFromHost(bDev, addr b2[0], sizeof(uint32))
      engine.launchKernel("addKernel", oDev, (aDev, bDev))
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      check res[0] == 30
      engine.freeDevice(aDev)
      engine.freeDevice(bDev)
      engine.freeDevice(oDev)

    test "re-ingest with a changed body invalidates and recompiles":
      var engine = bkMetal.init()
      engine.ingest(addMsl)
      var a: array[1, uint32] = [2'u32]
      var b: array[1, uint32] = [3'u32]
      let aDev = engine.allocDevice(sizeof(uint32))
      engine.copyToDeviceFromHost(aDev, addr a[0], sizeof(uint32))
      let bDev = engine.allocDevice(sizeof(uint32))
      engine.copyToDeviceFromHost(bDev, addr b[0], sizeof(uint32))
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("addKernel", oDev, (aDev, bDev))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      check res[0] == 5
      engine.ingest(mulMsl)
      engine.launchKernel("addKernel", oDev, (aDev, bDev))
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      check res[0] == 6
      engine.freeDevice(aDev)
      engine.freeDevice(bDev)
      engine.freeDevice(oDev)

    test "dispatch with grid/blk via dispatchThreadgroups":
      var engine = bkMetal.init()
      engine.ingest(vecAddMsl)
      var a, b: array[128, uint32]
      for i in 0 ..< 128:
        a[i] = uint32(i)
        b[i] = 1'u32
      let aDev = engine.allocDevice(128 * sizeof(uint32))
      engine.copyToDeviceFromHost(aDev, addr a[0], 128 * sizeof(uint32))
      let bDev = engine.allocDevice(128 * sizeof(uint32))
      engine.copyToDeviceFromHost(bDev, addr b[0], 128 * sizeof(uint32))
      let oDev = engine.allocDevice(128 * sizeof(uint32))
      engine.launchKernel<<(2, 64)>>("vecAdd", oDev, (aDev, bDev))
      var res: array[128, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 128 * sizeof(uint32))
      engine.freeDevice(aDev)
      engine.freeDevice(bDev)
      engine.freeDevice(oDev)
      for i in 0 ..< 128:
        check res[i] == uint32(i) + 1

    test "scalar args pack into the constant buffer at 16-byte slots":
      var engine = bkMetal.init()
      engine.ingest(scalarMsl)
      echo scalarMsl
      let oDev = engine.allocDevice(2 * sizeof(uint32))
      engine.launchKernel("scalarKernel", oDev, (7'i32, 1.5'f32))
      var res: array[2, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2 * sizeof(uint32))
      check res[0] == 7
      check res[1] == 1
      engine.launchKernel("scalarKernel", oDev, (10'i32, 2.5'f32))
      engine.copyToHostFromDevice(addr res[0], oDev, 2 * sizeof(uint32))
      check res[0] == 10
      check res[1] == 2
      engine.freeDevice(oDev)

    test "deviceName returns the Metal device name":
      var engine = bkMetal.init()
      let name = engine.deviceName()
      echo "  device: ", name
      check name.len > 0

    test "repeated init/run cycles destroy cleanly (RAII)":
      for i in 0 ..< 5:
        var engine = bkMetal.init()
        engine.ingest(addMsl)
        var a: array[1, uint32] = [2'u32]
        var b: array[1, uint32] = [3'u32]
        let aDev = engine.allocDevice(sizeof(uint32))
        engine.copyToDeviceFromHost(aDev, addr a[0], sizeof(uint32))
        let bDev = engine.allocDevice(sizeof(uint32))
        engine.copyToDeviceFromHost(bDev, addr b[0], sizeof(uint32))
        let oDev = engine.allocDevice(sizeof(uint32))
        engine.launchKernel("addKernel", oDev, (aDev, bDev))
        var res: array[1, uint32]
        engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
        engine.freeDevice(aDev)
        engine.freeDevice(bDev)
        engine.freeDevice(oDev)
        check res[0] == 5

when isMainModule:
  runTest()
