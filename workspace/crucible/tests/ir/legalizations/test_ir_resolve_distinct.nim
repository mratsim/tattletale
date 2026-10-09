## resolveType maps an ordinary distinct type to its base GPU type.
##
## - getTypeImpl of a distinct returns an nnkDistinctTy wrapper
## - the wrapper's typeKind is ntyDistinct again
## - recursing on the wrapper re-enters the ntyDistinct branch forever, and the VM dies on call depth at macro expansion
## - builtin distincts (float16, bfloat16) resolve by name, never reaching the typeKind switch
##
## Run:
##   nim c -r --hints:off --warnings:off workspace/crucible/tests/ir/legalizations/test_ir_resolve_distinct.nim (from tattletale)

import std/[macros, strutils]
import workspace/crucible
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/ir/resolvers

type
  Kilometers = distinct int32
  NauticalMiles = distinct Kilometers

macro resolveGpuKind(t: typed): string =
  ## Resolves the argument's type through resolveType and names the GpuTypeKind.
  var reg = TypeRegistry()
  let g = resolveType(reg, t.getTypeInst())
  result = newLit($g.kind)

proc runTest() =
  doAssert resolveGpuKind(Kilometers) == "gtInt32",
    "a distinct of int32 must resolve to gtInt32"
  doAssert resolveGpuKind(NauticalMiles) == "gtInt32",
    "a distinct of a distinct must peel one wrapper per resolve and reach gtInt32"

  # End-to-end: a kernel taking a distinct scalar compiles through
  # the public codegen API. Compilation is the smoke test: without
  # the wrapper peel the VM dies on the recursion before emitting code.
  const code = vulkan:
    proc k(output: ptr UncheckedArray[uint32], dist: Kilometers) {.global.} =
      output[0] = uint32(dist)
  doAssert "uint(dist)" in code,
    "distinct param must codegen as its base type, got:\n" & code

when isMainModule:
  runTest()
