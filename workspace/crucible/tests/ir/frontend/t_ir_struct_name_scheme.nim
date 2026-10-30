## Frontend struct-name scheme
##
## Test that struct names emit as a short base plus a 7-character base58 hash, generic arguments never expand into the base.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/frontend/t_ir_struct_name_scheme.nim

import std/macros
import std/strutils
import workspace/crucible
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/frontend/nim_type_resolvers

type
  Shape = object
    dims: array[2, int32]
  Stride = object
    strides: array[2, int32]
  TensorView[Sh, St] = object
    shape: Sh
    stride: St
  Meter[T] = object
    value: T
  MeterF32 = Meter[float32]

func hashToken(name: string): string =
  ## Returns the trailing `_`-separated base58 hash token of `name`.
  let i = name.rfind('_')
  result = name[(i + 1) .. ^1]

func isBase58Token(tok: string): bool =
  result = tok.len == 7
  for c in tok:
    if c notin Base58Alphabet:
      return false

func isTargetIdent(name: string): bool =
  ## True when `name` is a legal identifier on every target (no leading digit, identifier characters only).
  if name.len == 0 or name[0] in Digits:
    return false
  for c in name:
    if c notin {'a'..'z', 'A'..'Z', '0'..'9', '_'}:
      return false
  result = true

macro structNameChecks(body: typed): untyped =
  ## Resolves the type of every statement in `body` through the frontend
  ## resolver and asserts the struct-name contract at compile time.
  var r: CirTypeResolver
  var names: seq[string]
  for e in body:
    # each statement is `discard <value>`, the value's type names the struct
    let expr = if e.kind == nnkDiscardStmt: e[0] else: e
    let t = r.resolveCirType(expr.getTypeInst())
    doAssert r.err.len == 0, "frontend refused: " & r.err
    doAssert t != nil and t.kind == ctStruct, "expected a struct type"
    names.add(t.name)
  for name in names:
    doAssert isTargetIdent(name), "not a target identifier: " & name
    doAssert isBase58Token(hashToken(name)),
      "missing 7-char base58 hash suffix: " & name
  # two or more generic arguments fold into the hash: the base names none
  doAssert names[0] == "TensorView_" & hashToken(names[0]),
    "generic args expanded into the base name: " & names[0]
  # a single scalar argument keeps its short form
  doAssert names[1] == "Meter_f32_" & hashToken(names[1]),
    "single scalar argument lost its short form: " & names[1]
  # tuple types render the short `Tuple` base
  doAssert names[2] == "Tuple_" & hashToken(names[2]),
    "tuple type expanded into the base name: " & names[2]
  # a plain named object keeps its name
  doAssert names[3] == "Shape_" & hashToken(names[3]),
    "plain object lost its name: " & names[3]
  # distinct instantiations sharing one base never collide
  doAssert names[1] != names[4],
    "Meter[float32] and Meter[int64] collided: " & names[1]
  # a type alias and the instantiation it names share one struct
  doAssert names[5] == names[1],
    "alias MeterF32 and Meter[float32] disagree: " & names[5] & " vs " & names[1]
  result = newEmptyNode()

structNameChecks:
  discard TensorView[Shape, Stride](shape: Shape(dims: [1'i32, 2'i32]),
    stride: Stride(strides: [4'i32, 8'i32]))
  discard Meter[float32](value: 1.0'f32)
  discard (1'i32, 2'i32)
  discard Shape(dims: [1'i32, 2'i32])
  discard Meter[int64](value: 1'i64)
  discard MeterF32(value: 1.0'f32)
