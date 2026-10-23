## Metal direct runs: a ptr-free struct arg binds as ONE chunk, at the lowered
## field widths with natural alignment.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_struct_args.nim

import std/unittest
import workspace/crucible

const gemmParamsMsl = """
#include <metal_stdlib>
using namespace metal;

struct GEMMParams {
  int m; int n; int k;
  int lda; int ldb; int ldc;
  int m0; int n0;
  int64_t strideA; int64_t strideB; int64_t strideC;
  int batchA; int batchB; int batchC;
};

kernel void paramsReadKernel(
    device int32_t* out [[buffer(0)]],
    const constant GEMMParams& p [[buffer(1)]]) {
  out[0] = p.m; out[1] = p.n; out[2] = p.k;
  out[3] = (int)(p.strideA / 1000000);
  out[4] = (int)(p.strideB / 1000000);
  out[5] = p.batchA;
}
"""

const gapMsl = """
#include <metal_stdlib>
using namespace metal;

struct GapStruct {
  int a; int b; int c;
  int64_t big;
  float f;
};

kernel void gapReadKernel(
    device int32_t* out [[buffer(0)]],
    const constant GapStruct& g [[buffer(1)]]) {
  out[0] = g.a; out[1] = g.b; out[2] = g.c;
  out[3] = (int)(g.big / 1000000);
  out[4] = (int)g.f;
}
"""

const idxMsl = """
#include <metal_stdlib>
using namespace metal;

kernel void idxReadKernel(
    device int32_t* out [[buffer(0)]],
    const constant int32_t& alpha [[buffer(1)]],
    const device int32_t* in [[buffer(2)]]) {
  out[0] = alpha;
  out[1] = in[0];
}
"""

const mixMsl = """
#include <metal_stdlib>
using namespace metal;

struct MixParams {
  int m; int n; int k; int iters;
};

kernel void mixReadKernel(
    device int32_t* out [[buffer(0)]],
    const device int32_t* in [[buffer(1)]],
    const constant MixParams& p [[buffer(2)]],
    const constant int32_t& iters [[buffer(3)]]) {
  out[0] = p.m; out[1] = p.n; out[2] = p.k; out[3] = iters; out[4] = in[0];
}
"""

type
  GemmParams = object
    ## 8x int + 3x int64 + 3x int, the steel GEMMParams shape: Nim `int`
    ## fields lower to 32-bit, 8x4 + 3x8 + 3x4 = 68 bytes
    m*, n*, k*, lda*, ldb*, ldc*, m0*, n0*: int
    strideA*, strideB*, strideC*: int64
    batchA*, batchB*, batchC*: int

  MixParams = object
    m*, n*, k*, iters*: int32

  GapStruct = object
    ## 3x int32 then an int64 field: the declared MSL layout pads `big`
    ## to its 8-byte alignment, a flat field-order chunk does not
    a*, b*, c*: int32
    big*: int64
    f*: float32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - whole-value struct args (direct runs)":

    test "positional binding: args bind in launch order":
      # hand-written kernels bind positionally, the launch tuple's order
      # is the signature
      var engine = newEngine(runMetal)
      engine.ingest(idxMsl)
      var inVal: array[1, int32] = [10'i32]
      let inDev = engine.allocDevice(int32, 1)
      engine.copyToDeviceFromHost(inDev, addr inVal[0], 1)
      let oDev = engine.allocDevice(int32, 2)
      var zero: array[2, int32] = [0'i32, 0]
      engine.copyToDeviceFromHost(oDev, addr zero[0], 2)
      engine.launchKernel("idxReadKernel", oDev, (5'i32, inDev))
      var out32: array[2, int32]
      engine.copyToHostFromDevice(addr out32[0], oDev, 2)
      check out32 == [5'i32, 10]

    test "a ptr-free struct binds as one chunk at the lowered field widths":
      var engine = newEngine(runMetal)
      engine.ingest(gemmParamsMsl)
      let oDev = engine.allocDevice(int32, 6)
      var zero: array[6, int32] = [0'i32, 0, 0, 0, 0, 0]
      engine.copyToDeviceFromHost(oDev, addr zero[0], 6)
      var p = GemmParams(m: 1, n: 2, k: 3,
                         strideA: 4_000_000'i64, strideB: 5_000_000'i64,
                         batchA: 6)
      engine.launchKernel("paramsReadKernel", oDev, (p,))
      var out32: array[6, int32]
      engine.copyToHostFromDevice(addr out32[0], oDev, 6)
      check out32 == [1'i32, 2, 3, 4, 5, 6]

    test "a struct with an alignment gap packs each field at its declared offset":
      # the MSL struct pads `big` to offset 16, the chunk must too
      var engine = newEngine(runMetal)
      engine.ingest(gapMsl)
      let oDev = engine.allocDevice(int32, 5)
      var zero: array[5, int32] = [0'i32, 0, 0, 0, 0]
      engine.copyToDeviceFromHost(oDev, addr zero[0], 5)
      var g = GapStruct(a: 1, b: 2, c: 3, big: 4_000_000'i64, f: 2.5'f32)
      engine.launchKernel("gapReadKernel", oDev, (g,))
      var out32: array[5, int32]
      engine.copyToHostFromDevice(addr out32[0], oDev, 5)
      check out32 == [1'i32, 2, 3, 4, 2]

    test "mixed pointer, struct and int args bind per the declared signature":
      var engine = newEngine(runMetal)
      engine.ingest(mixMsl)
      var inVal: array[1, int32] = [10'i32]
      let inDev = engine.allocDevice(int32, 1)
      engine.copyToDeviceFromHost(inDev, addr inVal[0], 1)
      let oDev = engine.allocDevice(int32, 5)
      var zero: array[5, int32] = [0'i32, 0, 0, 0, 0]
      engine.copyToDeviceFromHost(oDev, addr zero[0], 5)
      var p = MixParams(m: 1, n: 2, k: 3, iters: 4)
      engine.launchKernel("mixReadKernel", oDev,
                 (inDev, p, 7'i32))
      var out32: array[5, int32]
      engine.copyToHostFromDevice(addr out32[0], oDev, 5)
      check out32 == [1'i32, 2, 3, 7, 10]

runTest()
