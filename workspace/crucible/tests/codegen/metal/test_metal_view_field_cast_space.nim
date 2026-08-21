## Metal: the kernel builds a view over a kernel buffer parameter
## and passes `base +% offset` to a device fn that takes a `device` pointer.
## This ensures that the buffer's address space (`device`) is propagated through casts.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_view_field_cast_space.nim

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
      var engine = bkMetal.init()
      engine.ingest(viewCastMsl)
      var A = [uint16(10), 20, 30, 40, 50, 60, 70, 80]
      var res: array[1, uint32]
      # baseOff = 2 * strideRow + strideCol = 2 * 3 + 1 = 7
      engine.run("viewCastKernel", res, (A, int32(4), int32(3)))
      check res[0] == uint32(A[7])

when isMainModule:
  runTest()
