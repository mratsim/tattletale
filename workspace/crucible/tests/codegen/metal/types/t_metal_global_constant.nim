## Globals emit in the constant address space
##
## Test that a top-level var emits as a constant program-scope declaration,
## read-only storage per the Metal spec
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_global_constant.nim

import std/[strutils, unittest]
import workspace/crucible

const globalConstantMsl = metal:
  var counter: int32 = 0
  proc bump(C: ptr UncheckedArray[int32]) {.global.} =
    C[0] = counter

test "the global emits as a constant program-scope declaration":
  check "constant int counter = 0;" in globalConstantMsl
