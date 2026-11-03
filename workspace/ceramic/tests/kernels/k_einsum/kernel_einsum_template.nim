# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Einsum suite machinery: the recipes as device kernels, one per kernel proc,
## replayed against the python-generated fixtures.

import std/os
import workspace/math
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_sum
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/src/kernels/k_layout_fillwith_cpu
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_einsum/einsum_fixtures
import workspace/ceramic/src/kernels/k_tile_einsum

export int_tuples, layout_algebra, tensors, tile_algebra

# ═══════════════════════════════════════════════════════════════════════
#  The recipes, one proc per einsum statement
# ═══════════════════════════════════════════════════════════════════════

proc kEinsumTranspose*(D, A: distinct TensorView) =
  einsum(A):
    D[j, i] = A[i, j]

proc kEinsumSumTotal*(S, A: distinct TensorView) =
  einsum(A):
    S = A[i, j]

proc kEinsumSumRow*(D, A: distinct TensorView) =
  einsum(A):
    D[i] = A[i, j]

proc kEinsumSumCol*(D, A: distinct TensorView) =
  einsum(A):
    D[j] = A[i, j]

proc kEinsumMatvec*(D, A, v: distinct TensorView) =
  einsum(A, v):
    D[i] = A[i, j] * v[j]

proc kEinsumMm*(D, A, B: distinct TensorView) =
  einsum(A, B):
    D[i, k] = A[i, j] * B[j, k]

proc kEinsumDot*(d, u, v: distinct TensorView) =
  einsum(u, v):
    d = u[i] * v[i]

proc kEinsumHadamard*(D, A, B: distinct TensorView) =
  einsum(A, B):
    D[i, j] = A[i, j] * B[i, j]

proc kEinsumOuter*(D, u, v: distinct TensorView) =
  einsum(u, v):
    D[i, j] = u[i] * v[j]

proc kEinsumContractionAbc*(D, A, B: distinct TensorView) =
  einsum(A, B):
    D[a, c] = A[a, b, c] * B[c, b]

proc kEinsumContraction2Bound*(D, A, B: distinct TensorView) =
  einsum(A, B):
    D[i, l] = A[i, j, k] * B[k, j, l]

proc kEinsumContraction3Op*(D, A, B, C: distinct TensorView) =
  einsum(A, B, C):
    D[a, b] = A[a, b, c] * B[c] * C[b, c]

proc kEinsumContractionBatched*(D, A, B: distinct TensorView) =
  einsum(A, B):
    D[b, i, k] = A[b, i, j] * B[b, j, k]

# ═══════════════════════════════════════════════════════════════════════
#  Replay, one proc per recipe
# ═══════════════════════════════════════════════════════════════════════

proc replayTranspose[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1])
    outShape = (meta.outShape[0], meta.outShape[1])
    outStrides = (meta.outStrides[0], meta.outStrides[1])
    outSize = meta.outShape[0] * meta.outShape[1]
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "o.bin", expect)
  var dView = dHost.view()
  fillWith_cpu(dView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumTranspose(dDev, aDev) do:
    grid = (ceil_div(outSize, 32), 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), msg = "einsum " & meta.name)

proc replaySumTotal[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1])
    outShape = (meta.outShape[0], meta.outShape[1])
    outStrides = (meta.outStrides[0], meta.outStrides[1])
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    sHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "o.bin", expect)
  # k_tile_sum contract: dst starts at zero and the accumulation is atomic
  var sView = sHost.view()
  fillWith_cpu(sView, 0.0'f32)
  let
    aDev = engine.bridge(aHost)
    sDev = engine.bridge(sHost)
  engine.launchDeviceFn kEinsumSumTotal(sDev, aDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr sHost.data[0], sDev.data, sHost.len)
  assertAllClose(sHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replaySumRow[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1])
    outShape = (meta.outShape[0],)
    outStrides = (meta.outStrides[0],)
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "o.bin", expect)
  var dView = dHost.view()
  fillWith_cpu(dView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumSumRow(dDev, aDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replaySumCol[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1])
    outShape = (meta.outShape[0],)
    outStrides = (meta.outStrides[0],)
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "o.bin", expect)
  var dView = dHost.view()
  fillWith_cpu(dView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumSumCol(dDev, aDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replayMatvec[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1])
    vShape = (meta.operandShapes[1][0],)
    vStrides = (meta.operandStrides[1][0],)
    outShape = (meta.outShape[0],)
    outStrides = (meta.outStrides[0],)
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    vHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "v.bin",
      vShape, vStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "v.bin", vHost)
  checkXorhash(meta, "o.bin", expect)
  var dView = dHost.view()
  fillWith_cpu(dView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    vDev = engine.bridge(vHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumMatvec(dDev, aDev, vDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replayMm[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1])
    bShape = (meta.operandShapes[1][0], meta.operandShapes[1][1])
    bStrides = (meta.operandStrides[1][0], meta.operandStrides[1][1])
    outShape = (meta.outShape[0], meta.outShape[1])
    outStrides = (meta.outStrides[0], meta.outStrides[1])
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    bHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "b.bin",
      bShape, bStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "b.bin", bHost)
  checkXorhash(meta, "o.bin", expect)
  var dHostView = dHost.view()
  fillWith_cpu(dHostView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumMm(dDev, aDev, bDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-3,
    msg = "einsum " & meta.name)

proc replayDot[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    uShape = (meta.operandShapes[0][0],)
    uStrides = (meta.operandStrides[0][0],)
    vShape = (meta.operandShapes[1][0],)
    vStrides = (meta.operandStrides[1][0],)
    outShape = (meta.outShape[0], meta.outShape[1])
    outStrides = (meta.outStrides[0], meta.outStrides[1])
  var
    uHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "u.bin",
      uShape, uStrides, float32)
    vHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "v.bin",
      vShape, vStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "u.bin", uHost)
  checkXorhash(meta, "v.bin", vHost)
  checkXorhash(meta, "o.bin", expect)
  var dHostView = dHost.view()
  fillWith_cpu(dHostView, -12345.0'f32) # every D cell must be written
  let
    uDev = engine.bridge(uHost)
    vDev = engine.bridge(vHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumDot(dDev, uDev, vDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replayHadamard[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1])
    bShape = (meta.operandShapes[1][0], meta.operandShapes[1][1])
    bStrides = (meta.operandStrides[1][0], meta.operandStrides[1][1])
    outShape = (meta.outShape[0], meta.outShape[1])
    outStrides = (meta.outStrides[0], meta.outStrides[1])
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    bHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "b.bin",
      bShape, bStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "b.bin", bHost)
  checkXorhash(meta, "o.bin", expect)
  var dHostView = dHost.view()
  fillWith_cpu(dHostView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumHadamard(dDev, aDev, bDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replayOuter[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    uShape = (meta.operandShapes[0][0],)
    uStrides = (meta.operandStrides[0][0],)
    vShape = (meta.operandShapes[1][0],)
    vStrides = (meta.operandStrides[1][0],)
    outShape = (meta.outShape[0], meta.outShape[1])
    outStrides = (meta.outStrides[0], meta.outStrides[1])
    outSize = meta.outShape[0] * meta.outShape[1]
  var
    uHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "u.bin",
      uShape, uStrides, float32)
    vHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "v.bin",
      vShape, vStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "u.bin", uHost)
  checkXorhash(meta, "v.bin", vHost)
  checkXorhash(meta, "o.bin", expect)
  var dHostView = dHost.view()
  fillWith_cpu(dHostView, -12345.0'f32) # every D cell must be written
  let
    uDev = engine.bridge(uHost)
    vDev = engine.bridge(vHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumOuter(dDev, uDev, vDev) do:
    grid = (ceil_div(outSize, 32), 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replayContractionAbc[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1], meta.operandShapes[0][2])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1], meta.operandStrides[0][2])
    bShape = (meta.operandShapes[1][0], meta.operandShapes[1][1])
    bStrides = (meta.operandStrides[1][0], meta.operandStrides[1][1])
    outShape = (meta.outShape[0], meta.outShape[1])
    outStrides = (meta.outStrides[0], meta.outStrides[1])
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    bHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "b.bin",
      bShape, bStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "b.bin", bHost)
  checkXorhash(meta, "o.bin", expect)
  var dHostView = dHost.view()
  fillWith_cpu(dHostView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumContractionAbc(dDev, aDev, bDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replayContraction2Bound[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1], meta.operandShapes[0][2])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1], meta.operandStrides[0][2])
    bShape = (meta.operandShapes[1][0], meta.operandShapes[1][1], meta.operandShapes[1][2])
    bStrides = (meta.operandStrides[1][0], meta.operandStrides[1][1], meta.operandStrides[1][2])
    outShape = (meta.outShape[0], meta.outShape[1])
    outStrides = (meta.outStrides[0], meta.outStrides[1])
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    bHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "b.bin",
      bShape, bStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "b.bin", bHost)
  checkXorhash(meta, "o.bin", expect)
  var dHostView = dHost.view()
  fillWith_cpu(dHostView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumContraction2Bound(dDev, aDev, bDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replayContraction3Op[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1], meta.operandShapes[0][2])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1], meta.operandStrides[0][2])
    bShape = (meta.operandShapes[1][0],)
    bStrides = (meta.operandStrides[1][0],)
    cShape = (meta.operandShapes[2][0], meta.operandShapes[2][1])
    cStrides = (meta.operandStrides[2][0], meta.operandStrides[2][1])
    outShape = (meta.outShape[0], meta.outShape[1])
    outStrides = (meta.outStrides[0], meta.outStrides[1])
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    bHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "b.bin",
      bShape, bStrides, float32)
    cHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "c.bin",
      cShape, cStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "b.bin", bHost)
  checkXorhash(meta, "c.bin", cHost)
  checkXorhash(meta, "o.bin", expect)
  var dHostView = dHost.view()
  fillWith_cpu(dHostView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    cDev = engine.bridge(cHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumContraction3Op(dDev, aDev, bDev, cDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replayContractionBatched[E: HwEngine](engine: E, meta: EinsumMeta) =
  let
    aShape = (meta.operandShapes[0][0], meta.operandShapes[0][1], meta.operandShapes[0][2])
    aStrides = (meta.operandStrides[0][0], meta.operandStrides[0][1], meta.operandStrides[0][2])
    bShape = (meta.operandShapes[1][0], meta.operandShapes[1][1], meta.operandShapes[1][2])
    bStrides = (meta.operandStrides[1][0], meta.operandStrides[1][1], meta.operandStrides[1][2])
    outShape = (meta.outShape[0], meta.outShape[1], meta.outShape[2])
    outStrides = (meta.outStrides[0], meta.outStrides[1], meta.outStrides[2])
  var
    aHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "a.bin",
      aShape, aStrides, float32)
    bHost = loadFixtureTensor(EinsumFixturesDir / meta.name, "b.bin",
      bShape, bStrides, float32)
    dHost = makeHostTensor(float32, outShape, outStrides)
    expect = loadFixtureTensor(EinsumFixturesDir / meta.name, "o.bin",
      outShape, outStrides, float32)
  checkXorhash(meta, "a.bin", aHost)
  checkXorhash(meta, "b.bin", bHost)
  checkXorhash(meta, "o.bin", expect)
  var dHostView = dHost.view()
  fillWith_cpu(dHostView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumContractionBatched(dDev, aDev, bDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  assertAllClose(dHost.view(), expect.view(), rtol = 1e-5, abstol = 1e-4,
    msg = "einsum " & meta.name)

proc replayEinsumFixture*[E: HwEngine](engine: E, name: string) =
  let meta = parseEinsumMeta(EinsumFixturesDir / name)
  case meta.recipe
  of "transpose": engine.replayTranspose(meta)
  of "sum_total": engine.replaySumTotal(meta)
  of "sum_row": engine.replaySumRow(meta)
  of "sum_col": engine.replaySumCol(meta)
  of "matvec": engine.replayMatvec(meta)
  of "mm": engine.replayMm(meta)
  of "dot": engine.replayDot(meta)
  of "hadamard": engine.replayHadamard(meta)
  of "outer": engine.replayOuter(meta)
  of "contraction_abc": engine.replayContractionAbc(meta)
  of "contraction_2bound": engine.replayContraction2Bound(meta)
  of "contraction_3op": engine.replayContraction3Op(meta)
  of "contraction_batched": engine.replayContractionBatched(meta)
  else:
    doAssert false, "unknown recipe " & meta.recipe

# Uncommenting fails to compile: the operand takes 3 indices on a rank-2 tensor
#[ 
proc kEinsumBadRank(D, A: distinct TensorView) =
  einsum(A):
    D[i] = A[i, j, k]
]#
