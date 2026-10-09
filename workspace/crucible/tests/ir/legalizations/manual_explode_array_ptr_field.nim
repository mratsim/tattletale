## Manual test: an array field carrying pointer-bearing elements must fail
## the explodePointerStructParams pass loudly.
##
## - containsPtrDeep recurses into array element types
## - the struct is selected for explosion, the array field hits
##   the array branch, which asserts loudly
## - compilation of this file is the test, it must fail with the expected
##   message below
##
# Run (expected FAIL, compile only):
#   nim c --hints:off --warnings:off workspace/crucible/tests/ir/legalizations/manual_explode_array_ptr_field.nim
# Expect: "explodePointerStructParams: array field 'x_boxes' carries pointer-bearing elements"

import workspace/crucible

type
  PtrBox = object
    p: ptr UncheckedArray[float32]
    n: uint32

  Bad = object
    data: ptr UncheckedArray[float32]
    boxes: array[2, PtrBox]

const code = vulkan:
  proc k(output: ptr UncheckedArray[float32], x: Bad) {.global.} =
    output[0] = x.data[0]
