## Kernel params of structs carrying (nested) pointer fields lower to per-field params, the kernel runs end-to-end,
## byte-exact vs the closed form.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_explode_struct_params.nim

import workspace/crucible

# ── Recreated Ceramic types (crucible-only) ─────────────────────────────────

type
  StoreMask {.size: sizeof(uint32).} = enum
    smStoreNone, smStoreAll, smStoreDiag

  Layout2D = object
    shape: tuple[rows, cols: int32]
    stride: tuple[rowS, colS: int32]

  View[T] = object
    ## TensorView stand-in: device pointer + layout value
    data: ptr UncheckedArray[T]
    layout: Layout2D

  EpiAXPBY[T] = object
    ## D = alpha*x + beta*c, gated by storeMask
    alpha, beta: T
    c: View[T]
    storeMask: StoreMask

func mkLayout(rows, cols, rowStride, colStride: int32): Layout2D =
  Layout2D(shape: (rows, cols), stride: (rowStride, colStride))

const axpbyWgsl = webgpu:
  # x is a plain pointer param: WebGPU's default storage-binding limit is 8
  # per shader stage and 3 views would need 9.
  proc axpbyKernel(dst: View[float32], x: ptr UncheckedArray[float32], epi: EpiAXPBY[float32]) {.global.} =
    let rows = dst.layout.shape.rows
    let cols = dst.layout.shape.cols
    for r in 0 ..< rows:
      for c in 0 ..< cols:
        let i = r * dst.layout.stride.rowS + c * dst.layout.stride.colS
        let ci = r * epi.c.layout.stride.rowS + c * epi.c.layout.stride.colS
        if uint32(epi.storeMask) == 1'u32:
          dst.data[i] = epi.alpha * x[i] + epi.beta * epi.c.data[ci]
        else:
          dst.data[i] = 0.0'f32

# ── Reference (closed form) ─────────────────────────────────────────────────

const N = 6

proc axpbyReference(expected: var array[N, float32], x, c: array[N, float32], alpha, beta: float32, storeAll: bool) =
  for i in 0 ..< N:
    expected[i] = if storeAll: alpha * x[i] + beta * c[i] else: 0.0'f32

# ── Kernel launch over the recreated struct params ──────────────────────────
# Struct params explode to per-field WGSL storage bindings:
#
# | binding | field                        |
# | ------- | ---------------------------- |
# | 0       | dst.data (the output buffer) |
# | 1       | dst.layout                   |
# | 2       | x                            |
# | 3       | epi.alpha                    |
# | 4       | epi.beta                     |
# | 5       | epi.c.data                   |
# | 6       | epi.c.layout                 |
# | 7       | epi.storeMask                |
#
# Each args-tuple entry supplies a device buffer for each pointer field
# and a by-value scalar for each value field, in field declaration order.
# blk must match the shader-baked workgroup size.

proc launchAxpbY[E](engine: var E, dstDev, xDev, cDev: auto, alpha, beta: float32, mask: StoreMask) =
  engine.run << (grid: (1, 1), blk: (64, 1)) >> ("axpbyKernel", dstDev, (
    mkLayout(2, 3, 3, 1),
    xDev,
    alpha, beta,
    cDev,
    mkLayout(2, 3, 3, 1),
    mask))

proc runTest() =
  var engine = bkWGSL.init()
  engine.ingest(axpbyWgsl)

  let x = [1.0'f32, 2, 3, 4, 5, 6]
  let c = [10.0'f32, 20, 30, 40, 50, 60]
  let xDev = engine.allocDevice(N * sizeof(float32))
  engine.copyToDeviceFromHost(xDev, unsafeAddr x[0], N * sizeof(float32))
  let cDev = engine.allocDevice(N * sizeof(float32))
  engine.copyToDeviceFromHost(cDev, unsafeAddr c[0], N * sizeof(float32))

  block storeAll:
    var dst: array[N, float32] = [7.0'f32, 7, 7, 7, 7, 7]
    let dstDev = engine.allocDevice(N * sizeof(float32))
    engine.copyToDeviceFromHost(dstDev, addr dst[0], N * sizeof(float32))
    engine.launchAxpbY(dstDev, xDev, cDev, 2.5'f32, 0.5'f32, smStoreAll)
    engine.copyToHostFromDevice(addr dst[0], dstDev, N * sizeof(float32))
    engine.freeDevice(dstDev)
    var expected: array[N, float32]
    axpbyReference(expected, x, c, 2.5'f32, 0.5'f32, true)
    for i in 0 ..< N:
      let actual = dst[i]
      doAssert actual == expected[i],
        "smStoreAll: actual[" & $i & "]=" & $actual & " expected=" & $expected[i]

  block storeNone:
    var dst: array[N, float32] = [7.0'f32, 7, 7, 7, 7, 7]
    let dstDev = engine.allocDevice(N * sizeof(float32))
    engine.copyToDeviceFromHost(dstDev, addr dst[0], N * sizeof(float32))
    engine.launchAxpbY(dstDev, xDev, cDev, 2.5'f32, 0.5'f32, smStoreNone)
    engine.copyToHostFromDevice(addr dst[0], dstDev, N * sizeof(float32))
    engine.freeDevice(dstDev)
    for i in 0 ..< N:
      let actual = dst[i]
      doAssert actual == 0.0'f32,
        "smStoreNone: actual[" & $i & "]=" & $actual & " expected=0.0"

  engine.freeDevice(xDev)
  engine.freeDevice(cDev)

when isMainModule:
  runTest()
