## Metal: plain `pointer` launch args bind as buffers, like `ptr T` leaves.
## A raw-byte binding leaves the buffer slot unbound, the output keeps
## its sentinel value.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_pointer_args.nim

import std/unittest
import workspace/crucible

const copyMsl = metal:
  proc ptrCopyKernel(output: ptr UncheckedArray[uint32], a: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0]
    output[1] = a[1]

const rawStructMsl = """
#include <metal_stdlib>
using namespace metal;

struct CopyArgs {
  int n;
};

kernel void structPtrCopyKernel(
    device uint32_t* out [[buffer(0)]],
    const device uint32_t* src [[buffer(1)]],
    const constant CopyArgs& args [[buffer(2)]]) {
  for (int i = 0; i < args.n; i++) {
    out[i] = src[i];
  }
}
"""

type
  PtrArgs = object
    ## Pointer field explodes to a buffer binding, other fields pack as bytes.
    src: pointer
    n: int32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - pointer launch args":

    test "plain pointer output and arg bind as buffers":
      var engine = bkMetal.init()
      engine.ingest(copyMsl)
      var a: array[2, uint32] = [7'u32, 9'u32]
      let aDev = engine.allocDevice(a.len * sizeof(uint32))
      let oDev = engine.allocDevice(2 * sizeof(uint32))
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
      var zero: array[2, uint32] = [0'u32, 0'u32]
      engine.copyToDeviceFromHost(oDev, addr zero[0], 2 * sizeof(uint32))
      engine.run("ptrCopyKernel", cast[pointer](oDev), (cast[pointer](aDev),))
      var out32: array[2, uint32]
      engine.copyToHostFromDevice(addr out32[0], oDev, 2 * sizeof(uint32))
      engine.freeDevice(aDev)
      engine.freeDevice(oDev)
      check out32[0] == 7
      check out32[1] == 9

    test "struct arg with a pointer field explodes to a buffer binding":
      var engine = bkMetal.init()
      engine.ingest(rawStructMsl)
      var a: array[4, uint32] = [3'u32, 1'u32, 4'u32, 1'u32]
      let aDev = engine.allocDevice(a.len * sizeof(uint32))
      let oDev = engine.allocDevice(a.len * sizeof(uint32))
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
      var zero: array[4, uint32] = [0'u32, 0'u32, 0'u32, 0'u32]
      engine.copyToDeviceFromHost(oDev, addr zero[0], a.len * sizeof(uint32))
      engine.run("structPtrCopyKernel", cast[pointer](oDev),
                 (PtrArgs(src: cast[pointer](aDev), n: 4),))
      var out32: array[4, uint32]
      engine.copyToHostFromDevice(addr out32[0], oDev, a.len * sizeof(uint32))
      engine.freeDevice(aDev)
      engine.freeDevice(oDev)
      check out32[0] == 3
      check out32[1] == 1
      check out32[2] == 4
      check out32[3] == 1

runTest()
