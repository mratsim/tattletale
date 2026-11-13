## Direct SPIR-V: address of a member chain rooted at a kernel parameter
##
## Test that a kernel parameter's member chain reaches a device call as a by-ref temporary copy and the kernel computes from it
##
## Run:
##   nim c -r -d:release --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/compiletime/t_spirv_byref_member_addr.nim

import workspace/crucible

type
  Cells = object
    a, b, c, d, e, f, g: int32

  Layout = object
    rank: int32
    cells: Cells

  View = object
    offset: int32
    layout: Layout

const memberAddrSpv = spirv:
  proc sumCells(cells: Cells, pos: int32): int32 {.device.} =
    result = cells.a + cells.b + cells.c + cells.d + cells.e + cells.f + cells.g
    result += pos

  proc indexKernel(outBuf: ptr UncheckedArray[int32], v: View) {.global.} =
    outBuf[0] = sumCells(v.layout.cells, 5'i32)

type
  ArrayLayout = object
    shape: array[4, int32]
    strides: array[4, int32]
    rank: int32

  ArrayView = object
    offset: int32
    layout: ArrayLayout

const memberArraySpv = spirv:
  proc flatIndex(shape, strides: array[4, int32], rank, pos: int32): int32 {.device.} =
    result = 0
    for d in 0 ..< rank:
      result += shape[d] * strides[d]
    result += pos

  proc arrayKernel(outBuf: ptr UncheckedArray[int32], v: ArrayView) {.global.} =
    outBuf[0] = flatIndex(v.layout.shape, v.layout.strides, v.layout.rank, 5'i32)

proc runTest() =
  var engine = newEngine(runVulkan)
  engine.ingest(memberAddrSpv)
  let oDev = engine.allocDevice(int32, 1)
  let v = View(offset: 3'i32, layout: Layout(
    rank: 4'i32, cells: Cells(a: 1'i32, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7)))
  engine.launchKernel("indexKernel", oDev, (v,))
  var got: array[1, int32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 33'i32, "got " & $got[0] & ", want 33"

when isMainModule:
  runTest()
  echo "[OK] direct SPIR-V: member-chain address of a kernel parameter"
