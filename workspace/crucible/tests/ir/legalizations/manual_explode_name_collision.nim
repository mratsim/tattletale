## Manual test: an exploded leaf name colliding with a retained kernel param
## must fail the explodePointerStructParams pass loudly.
##
## - the retained param names seed the collision set before any explosion
## - exploding `view` generates the leaf `view_data`, colliding
##   with the retained int32 param of the same name
## - compilation of this file is the test, it must fail with the expected
##   message below
##
# Run (expected FAIL, compile only):
#   nim c --hints:off --warnings:off workspace/crucible/tests/ir/legalizations/manual_explode_name_collision.nim
# Expect: "explodePointerStructParams: exploded param name collision on 'view_data'"

import workspace/crucible

type
  Layout2D = object
    shape: tuple[rows, cols: int32]
    stride: tuple[rowS, colS: int32]

  View[T] = object
    data: ptr UncheckedArray[T]
    layout: Layout2D

const code = vulkan:
  proc k(output: ptr UncheckedArray[float32], view: View[float32], view_data: int32) {.global.} =
    output[0] = view.data[0] + float32(view_data)
