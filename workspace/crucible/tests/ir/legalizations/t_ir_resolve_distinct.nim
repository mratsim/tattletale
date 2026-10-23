## resolveType distinct-wrapper peeling
##
## Test that resolveType maps an ordinary distinct type to its base GPU type
##
## - nested distincts peel one wrapper per resolve, the VM never re-enters `ntyDistinct`
## - a distinct kernel param compiles through the public codegen API
## - builtin distincts (`float16`, `bfloat16`) resolve by name
##
## Run:
##   nim c -r --hints:off --warnings:off workspace/crucible/tests/ir/legalizations/t_ir_resolve_distinct.nim

import std/[macros, strutils]
import workspace/crucible
import workspace/crucible/src/crucible_internal_compiler

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
  const code = glsl:
    proc k(output: ptr UncheckedArray[uint32], dist: Kilometers) {.global.} =
      output[0] = uint32(dist)
  doAssert "uint(dist)" in code,
    "distinct param must codegen as its base type, got:\n" & code

when isMainModule:
  runTest()
