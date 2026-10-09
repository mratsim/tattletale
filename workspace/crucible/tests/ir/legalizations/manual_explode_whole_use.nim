## Manual test: a whole-value use of an exploded kernel param must fail
## the explodePointerStructParams pass loudly.
##
## - only field accesses rewrite onto the exploded leaf params
## - `let w = dst` keeps the bare `dst` ident alive, no whole param
##   remains in the signature to receive it
## - compilation of this file is the test, it must fail with the expected
##   message below
##
# Run (expected FAIL, compile only):
#   nim c --hints:off --warnings:off workspace/crucible/tests/ir/legalizations/manual_explode_whole_use.nim
# Expect: "explodePointerStructParams: kernel param 'dst___...' is used whole, only field accesses rewrite to exploded leaf params"

import workspace/crucible

type
  Layout2D = object
    shape: tuple[rows, cols: int32]
    stride: tuple[rowS, colS: int32]

  View[T] = object
    data: ptr UncheckedArray[T]
    layout: Layout2D

const code = vulkan:
  proc k(output: ptr UncheckedArray[float32], dst: View[float32]) {.global.} =
    let w = dst
    output[0] = dst.data[0] + float32(w.layout.shape.rows)
