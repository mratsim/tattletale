## Metal: the kernel builds a view over a kernel buffer parameter
## and passes `base +% offset` to a device fn that takes a `device` pointer.
## This ensures that the buffer's address space (`device`) is propagated through casts.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_view_field_cast_space.nim

import std/unittest
import workspace/crucible

type
  GlView[T] = object
    ## View over a device buffer: the base pointer plus runtime strides.
    base: ptr UncheckedArray[T]
    strideRow: int32
    strideCol: int32

template `+%`[T](p: ptr UncheckedArray[T], offset: SomeInteger): ptr UncheckedArray[T] =
  ## Returns `p` advanced by `offset` elements (uint64 address arithmetic).
  cast[ptr UncheckedArray[T]](cast[uint64](p) + uint64(offset) * uint64(sizeof(T)))

const viewCastMsl = metal:
  proc readAt[T](p: ptr UncheckedArray[T]; off: uint32): T =
    p[off]

  proc viewCastKernel(Out: ptr UncheckedArray[uint32];
                      A: ptr UncheckedArray[uint16];
                      N, K: int32) {.global.} =
    let glA = GlView[uint16](base: A, strideRow: K, strideCol: 1)
    let baseOff = uint32(2 * glA.strideRow + glA.strideCol)
    Out[0] = uint32(readAt(glA.base +% baseOff, 0))

proc runTest() =
  suite "Metal - address-space propagation through pointer casts":
    test "view base pointer arithmetic keeps the device address space":
      var engine = newEngine(runMetal)
      engine.ingest(viewCastMsl)
      var A = [uint16(10), 20, 30, 40, 50, 60, 70, 80]
      let aDev = engine.allocDevice(uint16, A.len)
      engine.copyToDeviceFromHost(aDev, addr A[0], A.len)
      let oDev = engine.allocDevice(uint32, 1)
      var res: array[1, uint32]
      # baseOff = 2 * strideRow + strideCol = 2 * 3 + 1 = 7
      engine.launchKernel("viewCastKernel", oDev, (aDev, int32(4), int32(3)))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == uint32(A[7])

when isMainModule:
  runTest()
