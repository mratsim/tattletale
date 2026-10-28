## CIRv2 struct naming: canonical type identity over alias names
##
## Test that a type alias and the instantiation it names share one struct name and distinct instantiations keep distinct names
##
## - an alias-typed kernel parameter verifies against an expansion-typed callee
## - one struct name serves both source forms in the emitted module
## - two distinct instantiations never share a struct name
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/cirv2 --nimcache:nimcache/tests/cirv2 \
##     workspace/crucible/tests/cirv2/t_cirv2_alias_type_naming.nim

import std/strutils
import std/macros
import
  workspace/crucible/src/builtins/builtins_pragmas,
  workspace/crucible/src/codegen/frontend/nim_to_crucible,
  workspace/crucible/tests/cirv2/cirv2_lower_macros

type
  Vec2[T] = object
    x, y: T
  V2F = Vec2[float32]
  V2I = Vec2[int32]

proc combine(p: Vec2[float32], s: float32): float32 =
  p.x * s + p.y * s

proc combineI(p: Vec2[int32], s: int32): int32 =
  p.x * s + p.y * s

proc aliasKernel(o: ptr UncheckedArray[float32], p: V2F) {.global.} =
  o[0] = combine(p, 2.0'f32)

proc aliasI32Kernel(o: ptr UncheckedArray[int32], p: V2I) {.global.} =
  o[0] = combineI(p, 2'i32)

const aliasTrace = lowerTrace:
  aliasKernel

const aliasI32Trace = lowerTrace:
  aliasI32Kernel

func structNames(trace, base: string): seq[string] =
  ## Returns every distinct `:<base>_<suffix>` struct name found in `trace`.
  ## Order: first appearance.
  let prefix = ":" & base & "_"
  var i = 0
  while true:
    i = trace.find(prefix, i)
    if i < 0:
      return
    var j = i + prefix.len
    while j < trace.len and trace[j] in {'A'..'Z', 'a'..'z', '0'..'9', '_'}:
      j.inc
    let name = trace[i ..< j]
    if name notin result:
      result.add(name)
    i = j

static:
  # alias and instantiation agree on one struct name
  let f32Names = structNames(aliasTrace, "Vec2_float32")
  doAssert ":Vec2_float32_" in aliasTrace,
    "the expanded instantiation must keep the readable base: " & aliasTrace
  doAssert ":V2F" notin aliasTrace,
    "the alias name must not survive as a struct name: " & aliasTrace
  doAssert f32Names.len == 1,
    "one canonical struct for both type spellings, got: " & $f32Names

  # two distinct instantiations never share a struct name
  let i32Names = structNames(aliasI32Trace, "Vec2_int32")
  doAssert ":Vec2_int32_" in aliasI32Trace
  doAssert ":V2I" notin aliasI32Trace
  doAssert i32Names.len == 1,
    "one canonical struct for both type spellings, got: " & $i32Names
  doAssert f32Names[0] != i32Names[0],
    "distinct instantiations must not share a struct name"
