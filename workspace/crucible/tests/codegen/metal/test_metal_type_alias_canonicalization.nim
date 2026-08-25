## Metal: the type-alias canonicalization (the two-module probe).
##
## A tile type spelled through an alias and spelled canonically must emit
## the same MSL type name. Module A (this test) declares the tile var
## through the alias and calls a generic apply-style device proc.
## Module B (helper_metal_type_alias_apply) defines that proc
## over the canonical `RtLeft` type. Before the canonicalization,
## the var's type resolved to the alias's base name (`RTileF3232x32`)
## while the apply's instantiated param resolved to the canonical name
## (`RtLeftf32x32x32xMmaAtom_...`), so the emitted call passed
## an incompatible pointer type. The canonicalization expands the alias
## to its RHS at resolve time, so the names agree and the call resolves
## (the emitted MSL ingests and runs).
##
## Run:
##   cd tattletale
##   nim test_crucible_metal
## or directly:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_type_alias_canonicalization.nim

import std/[strutils, unittest]
import workspace/crucible
import ./helper_metal_type_alias_apply

type Pair[T] = object
  ## plain generic object (NOT an alias): must keep its own name
  a, b: T

type MyInt4 = Int[4]
  ## non-generic alias over a generic instantiation: canonicalizes to Int4

type Wrap[R, C: static int] = RTileF32[R, C]
  ## nested alias: RHS is itself an alias application

type NestedInt[R: static int] = MyInt4
  ## nested alias through a named type: RHS is a bare alias sym

type RTileF32DefaultedParam[R, C: static int,
    A: static MmaAtom = APPLE_8x8x8_F32] = RtLeft[float32, R, C, A]
  ## alias with a defaulted atom param: the application omits the atom arg

type TemplateCallAlias[R, C: static int] = rt_l(float32, R, C)
  ## template-call RHS alias: rejected loudly, never silently emitted

type BoolHolder[B: static bool] = object
  flag: bool

type BoolAlias[N: static int, B: static bool = true] = BoolHolder[B]
  ## The omitted bool default substitutes as the bool type, matching
  ## the explicit `BoolAlias[4, true]` spelling.

type Color = enum cBlue, cRed

type ColorHolder[C: static Color] = object
  hue: Color

type ColorAlias[N: static int, C: static Color = cBlue] = ColorHolder[C]
  ## The omitted enum default substitutes as the enum's storage type,
  ## matching the explicit `ColorAlias[4, cBlue]` spelling.

type Vec[N: static int] = array[N, float32]
  ## An array-RHS alias: rejected loudly, never silently emitted.

# The alias-spelled declaration (module A) calling the apply (module B).
const aliasMsl = metal:
  proc aliasKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var d_reg: RTileF32[32, 32]
    applyTile(d_reg)
    C[0] = threadElements(d_reg.frags[0][0].frag, 0'u32)

# The same tile with the atom defaulted via TileConfigFor inside the canonical
# bracket: must emit the identical struct name too.
const defaultedAtomMsl = metal:
  proc defaultedAtomKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var d_reg: RTileF32DefaultedAtom[32, 32]
    threadElements(d_reg.frags[0][0].frag, 0'u32) = 1.0'f32
    C[0] = threadElements(d_reg.frags[0][0].frag, 0'u32)

# The canonical-spelled declaration: must emit the identical tile struct.
const canonicalMsl = metal:
  proc canonicalKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var d_reg: RtLeft[float32, 32, 32, APPLE_8x8x8_F32]
    threadElements(d_reg.frags[0][0].frag, 0'u32) = 1.0'f32
    C[0] = threadElements(d_reg.frags[0][0].frag, 0'u32)

const wrapMsl = metal:
  proc wrapKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var d_reg: Wrap[32, 32]
    threadElements(d_reg.frags[0][0].frag, 0'u32) = 1.0'f32
    C[0] = threadElements(d_reg.frags[0][0].frag, 0'u32)

const pairMsl = metal:
  proc pairKernel(C: ptr UncheckedArray[int32]) {.global.} =
    var p: Pair[int32]
    p.a = 1
    p.b = 2
    C[0] = p.a + p.b

const myIntMsl = metal:
  proc myIntKernel(C: ptr UncheckedArray[int32]) {.global.} =
    var x: MyInt4
    C[0] = 7

const nestedIntMsl = metal:
  proc nestedIntKernel(C: ptr UncheckedArray[int32]) {.global.} =
    var x: NestedInt[8]
    C[0] = 7

const defaultedParamMsl = metal:
  proc defaultedParamKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var d_reg: RTileF32DefaultedParam[16, 16]
    threadElements(d_reg.frags[0][0].frag, 0'u32) = 1.0'f32
    C[0] = threadElements(d_reg.frags[0][0].frag, 0'u32)

const canonical16Msl = metal:
  proc canonical16Kernel(C: ptr UncheckedArray[float32]) {.global.} =
    var d_reg: RtLeft[float32, 16, 16, APPLE_8x8x8_F32]
    threadElements(d_reg.frags[0][0].frag, 0'u32) = 1.0'f32
    C[0] = threadElements(d_reg.frags[0][0].frag, 0'u32)

const boolAliasMsl = metal:
  proc boolAliasKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var b: BoolAlias[4]
    C[0] = 1.0'f32

const boolExplicitMsl = metal:
  proc boolExplicitKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var b: BoolAlias[4, true]
    C[0] = 1.0'f32

const colorAliasMsl = metal:
  proc colorAliasKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var c: ColorAlias[4]
    C[0] = 1.0'f32

const colorExplicitMsl = metal:
  proc colorExplicitKernel(C: ptr UncheckedArray[float32]) {.global.} =
    var c: ColorAlias[4, cBlue]
    C[0] = 1.0'f32

proc varDeclType(msl: string, varName: string): string =
  ## The type token on the `varName;` declaration in the emitted MSL.
  for line in msl.splitLines:
    let idx = line.find(" " & varName & ";")
    if idx >= 0:
      return line[0 ..< idx].strip()

proc runTest() =
  suite "Metal - type-alias canonicalization":
    test "alias-spelled and canonical-spelled tiles emit the same MSL name":
      let aliasType = varDeclType(aliasMsl, "d_reg")
      let canonicalType = varDeclType(canonicalMsl, "d_reg")
      check aliasType.len > 0
      check canonicalType.len > 0
      check aliasType == canonicalType
      # exact canonical-identity pin: the emitted name must match
      # the canonical base and args, never the alias's own base name
      check canonicalType.startsWith("RtLeftf32x32x32xMmaAtom_")
      check "RTileF3232x32" notin aliasMsl # the alias name must not leak
      check varDeclType(defaultedAtomMsl, "d_reg") == canonicalType
      check "RTileF32DefaultedAtom32x32" notin defaultedAtomMsl

    test "the canonicalization never rewrites a non-alias generic":
      # A plain generic object keeps its own emitted name: an over-broad
      # canonicalization (treating every generic as an alias) would fail
      # this exact pin.
      check varDeclType(pairMsl, "p") == "Pairi32"

    test "a non-generic alias emits the exact canonical name":
      check varDeclType(myIntMsl, "x") == "Int4"

    test "a nested alias canonicalizes through the recursion":
      check varDeclType(wrapMsl, "d_reg") == varDeclType(canonicalMsl, "d_reg")

    test "a nested alias through a named type canonicalizes too":
      check varDeclType(nestedIntMsl, "x") == "Int4"

    test "an omitted defaulted param substitutes its declared default":
      check varDeclType(defaultedParamMsl, "d_reg") == varDeclType(canonical16Msl, "d_reg")

    test "an omitted bool default emits the explicit bool arg's name":
      # A literal substitution would name the default's int-backed value
      # and diverge from the explicit `true` arg. The emitted name must
      # converge to the canonical `BoolHolderbool`.
      let omitted = varDeclType(boolAliasMsl, "b")
      let explicit = varDeclType(boolExplicitMsl, "b")
      check omitted == explicit
      check omitted == "BoolHolderbool"

    test "an omitted enum default emits the explicit enum arg's name":
      # A literal substitution would name the default's int-backed value
      # and diverge from the explicit `cBlue` arg. The emitted name must
      # converge to the canonical `ColorHolderu32`.
      let omitted = varDeclType(colorAliasMsl, "c")
      let explicit = varDeclType(colorExplicitMsl, "c")
      check omitted == explicit
      check omitted == "ColorHolderu32"

    test "a template-call RHS alias is rejected loudly":
      # A template-call RHS cannot be expanded at resolve time. The alias
      # must spell the canonical type application instead. The rejection
      # is a compile error, never a silent emission.
      static:
        doAssert not compiles(block:
          const bad = metal:
            proc bad(C: ptr UncheckedArray[float32]) {.global.} =
              var d: TemplateCallAlias[32, 32]
        )

    test "an array-RHS alias is rejected loudly":
      # An alias over an array cannot canonicalize to a named struct.
      # Without the loud rejection it would emit a fieldless declaration
      # and corrupt the emitted code. The rejection is a compile error,
      # never a silent emission.
      static:
        doAssert not compiles(block:
          const bad = metal:
            proc bad(C: ptr UncheckedArray[float32]) {.global.} =
              var v: Vec[4]
        )

    test "the generic apply call resolves across modules":
      # The apply's emitted parameter must be the same struct the kernel
      # declares: `thread RtLeftf32x32x32xMmaAtom_...* d`.
      let aliasType = varDeclType(aliasMsl, "d_reg")
      check ("thread " & aliasType & "* d)") in aliasMsl
      check "applyTile" in aliasMsl # the call is present, not dropped
      # The MSL compiler accepts the call only when the declaration and parameter
      # types agree. Ingest proves the cross-module link.
      var engine = bkMetal.init()
      engine.ingest(aliasMsl)
      var res: array[1, float32]
      engine.run<<(grid: (1, 1), blk: (32, 1))>>("aliasKernel", res, ())
      check res[0] == 7.0'f32

when isMainModule:
  runTest()
