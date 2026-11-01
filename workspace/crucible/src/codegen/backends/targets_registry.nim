# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Schema and data of the printers' static target rows.
##
## A row is a const object of plain data: strings, bools, ints, enums, sets and tables of those.
## No field stores a proc or a placeholder whose value depends on a body walk.
##
## | type       | contents                                      |
## | ---------- | --------------------------------------------- |
## | ScalarRow  | type names by (kind, width) cell              |
## | LitRow     | literal token forms                           |
## | KindRow    | walk policy: loops, else folding, routing set |
## | VectorRule | a target's real vector coverage               |
## | CastRule   | one bit-reinterpret coverage rule             |
##
## The six printer targets (Metal MSL, CUDA C++, OpenCL C, GLSL, WebGPU WGSL, HLSL):
##
## - each states its rows in the `genTargets:` DSL at the bottom of this module
## - the macro in `targets_macros.nim` validates every row against these types
##   during compilation and generates one lookup const per row family
## - a row missing the schema fails the compile at the row definition
## - the printers read the generated lookups, signature text, coordinate
##   builtins and refusal messages stay procedural in the printers

import std/tables
import ../ir/cirv2_lang

type
  FloatLitTokens* = object
    ## Non-finite float literal tokens for one printer target:
    ##
    ## - one field per literal width
    ## - tokens are stated only for the widths the target emits, a width
    ##   without a token is covered by the target's scalar refusal
    ## - Nim's `$` renders the special values `inf`, `-inf` and `nan`, and no
    ##   shader language accepts those as a literal token

    inf16*, negInf16*, nan16*: string
    inf32*, negInf32*, nan32*: string
    inf64*, negInf64*, nan64*: string

  IntLitTokens* = object
    ## Integer literal token chosen by signedness and width:
    ## `prefix & $value & suffix`.
    ##
    ## - 64-bit widths take `u64Prefix`/`u64Suffix` or `i64Prefix`/`i64Suffix`
    ## - 32-bit unsigned takes `u32Suffix`, 32-bit signed prints bare
    ## - narrow (8/16-bit) widths print bare or, per `narrowCast`, as a cast
    ##   to the scalar type name from the target's scalar row
    u64Prefix*, u64Suffix*, u32Suffix*: string
    i64Prefix*, i64Suffix*: string
    narrowCast*: bool

  LoopStyle* = object
    ## Loop text of one printer target:
    ##
    ## - `unrollDirective`: `{.unroll.}` directive text, printed before
    ##   the loop at the loop indentation, "" prints no directive
    ## - `braceSameLine`: the open brace sits on the head line, otherwise it
    ##   prints on its own line under the head (HLSL)
    unrollDirective*: string
    braceSameLine*: bool

type
  ScalarCell* = object
    ## One (kind, width) cell of a scalar row.
    kind*: CirTypeKind
    width*: int
    ## Bit width, 0 for the widthless kinds (void, bool).
    spelling*: string
    ## Type name the target prints.

  ScalarRow* = object
    ## Scalar type names of one target by (kind, width) cell.
    ## A row lists the cells the target names, an omitted cell refuses at the lookup.
    cells*: seq[ScalarCell]
    refusals*: seq[tuple[kind: CirTypeKind, width: int, reason: string]]
    ## One-line refusal reason per omitted cell the row documents.

  LitRow* = object
    ## Literal token forms of one target.
    ## No exponent or hex-float fields: no printer handles either form.
    int*: IntLitTokens
    f16Prefix*, f16Suffix*: string
    f32Suffix*: string
    f64Prefix*, f64Suffix*: string
    nilToken*: string
    nonFinite*: FloatLitTokens

  KindRow* = object
    ## Per-target policy where the shared expression and statement walks branch on target.
    ## No for-range, switch or sizeof field: no printer emits the first two, sizeof routes per target through `printTargetExpr`.
    loop*: LoopStyle
    foldsIteElse*: bool
    ## A single-statement else body folds into an `else if` chain.
    emitOperandRendersExpr*: bool
    ## A `{.emit.}` operand renders as an expression when true, splices its symbol name when false.
    targetExprKinds*: set[CirExprKind]
    ## Optional expression kinds the printer routes through its own `printTargetExpr` proc.
    ## Eight common kinds route unconditionally and are not members of this set.

  VectorNameForm* = enum
    ## Composition of a vector type name from the element's prefix and the lane count.
    vnfPrefixed
    ## `prefix & $laneCount` forms the C-family names (`vec3`, `float4`). An empty prefix composes the element's scalar name from the scalar row.
    vnfGenericElem
    ## `prefix & $laneCount & '<' & elementName & '>'` forms the generic-element names (`vec3<f32>`).

  VectorRule* = object
    ## A target's real vector coverage, as its printer emits it today.
    elems*: seq[tuple[kind: CirTypeKind, width: int, prefix: string]]
    ## Element types the target names, with the vector name's prefix.
    ## The name composes from the prefix and the lane count per `nameForm`.
    ## Empty when the target has no vectors.
    lanes*: seq[int]
    ## Lane counts the target emits.
    nameForm*: VectorNameForm = vnfPrefixed
    ## Composition of the vector type name from the element's prefix and the lane count.

  CastWidth* = range[0..255]
  ## Bit width a cast rule matches.

  CastRule* = object
    ## One bit-cast coverage rule of a target: the first matching rule emits.
    ##
    ## - a rule with `srcAny` matches any source type, otherwise the source
    ##   type must exist and be a scalar leaf of `srcKinds` and, when
    ##   `srcWidths` is non-empty, of `srcWidths`
    ## - the destination must be of `dstKinds` and, when `dstWidths` is
    ##   non-empty, of `dstWidths`, with `sameWidth` also requiring equal
    ##   source and destination widths
    ## - a `spelling` rule emits its text, `$1` stands for the printed
    ##   operand and `$2` for the destination type name
    ## - an empty `spelling` makes the rule a refusal carrying `refusal`,
    ##   `$2` stands for the destination type name
    srcAny*: bool
    srcKinds*, dstKinds*: set[CirTypeKind]
    srcWidths*, dstWidths*: set[CastWidth]
    sameWidth*: bool
    spelling*, refusal*: string

const ScalarCellAxis*: array[13, tuple[tok: string, kind: CirTypeKind, width: int]] = [
  ("void", ctVoid, 0), ("bool", ctBool, 0),
  ("i8", ctInt, 8), ("i16", ctInt, 16), ("i32", ctInt, 32), ("i64", ctInt, 64),
  ("u8", ctUint, 8), ("u16", ctUint, 16), ("u32", ctUint, 32), ("u64", ctUint, 64),
  ("f16", ctFloat, 16), ("f32", ctFloat, 32), ("f64", ctFloat, 64)]
  ## (kind, width) cells with the DSL token naming each cell.
  ## Every scalar row and vector element row keys its entries to this axis.
  ##
  ## - widths name different C types on different hosts, 64-bit `long`
  ##   on LP64 hosts against 32-bit on LLP64 hosts
  ## - each target states its own names, one cell never inherits another
  ##   target's name

# Imported after the schema: `targets_macros` imports this module's schema
# types and `ScalarCellAxis`, so loading it before the schema exists fails
# and the `genTargets` blocks below could not expand against it.
import ./targets_macros

genTargets:
  target msl:
    scalar void "void", bool "bool", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", f32 "float",
      refused f64 "double has no Apple GPU support"
    # `char` and `long` are MSL's 8-bit signed and 64-bit type names.
    # CUDA names the same widths `signed char` and `long long`, each row states its own names
    lit u64Suffix "ULL", u32Suffix "U", i64Suffix "LL", narrowCast true,
      f16Suffix "h", f32Suffix "f", nilToken "nullptr",
      inf16 "INFINITY", negInf16 "-INFINITY", nan16 "NAN",
      inf32 "INFINITY", negInf32 "-INFINITY", nan32 "NAN",
      inf64 "INFINITY", negInf64 "-INFINITY", nan64 "NAN"
    kind unrollDirective "#pragma clang loop unroll(full)", braceSameLine true
    # `{.builtin.}` numeric conversions resolve to the MSL native call text,
    # each entry a format string whose `$1` stands for the call's argument list.
    # `float16` keeps a first-class `half` identity: its conversions are plain native calls.
    # `asFp16`/`asU16` reinterpret through `as_type`.
    # `bfloat16` names no MSL type here, values travel as raw bits.
    # toBf16/toBf32 move the bit pattern through `as_type<ushort>` around the native `(bfloat)` cast.
    # fp8 conversions have no MSL native: the `{.builtin.}` name passes through and refuses at Metal compile time.
    conv toFp16 "half($1)", toFp32 "float($1)", toBf16 "as_type<ushort>((bfloat)($1))",
      toBf32 "float(as_type<bfloat>($1))", asFp16 "as_type<half>($1)", asU16 "as_type<ushort>($1)"
    # Coordinate builtins carry `ivec3`, so the builtin-threading parameters
    # and whole-value reads print this vector name: 32-bit int/uint elements only.
    vector:
      elems i32, u32
      lanes 2, 3, 4
    # Same-width scalar pairs reinterpret through MSL's `as_type<T>`.
    # Pointer casts and width changes keep the conversion form.
    bitcast:
      rule srcKinds {ctInt, ctUint, ctFloat}, srcWidths {8, 16, 32, 64},
        dstKinds {ctInt, ctUint, ctFloat}, sameWidth, "as_type<$2>($1)"

  target cuda:
    scalar void "void", bool "bool", i8 "signed char", i16 "short", i32 "int", i64 "long long",
      u8 "unsigned char", u16 "unsigned short", u32 "unsigned int", u64 "unsigned long long",
      f16 "__half", f32 "float", f64 "double"
    lit u64Suffix "ULL", u32Suffix "U", i64Suffix "LL", narrowCast true,
      f32Suffix "f", nilToken "nullptr",
      inf16 "INFINITY", negInf16 "-INFINITY", nan16 "NAN",
      inf32 "INFINITY", negInf32 "-INFINITY", nan32 "NAN",
      inf64 "INFINITY", negInf64 "-INFINITY", nan64 "NAN"
    kind unrollDirective "#pragma unroll", braceSameLine true
    # CUDA names no vector types, every vector refuses at its declaration.
    vector ()
    # `{.builtin.}` scalar-math calls keyed by (canonical name, operand bit width):
    #
    # - a float operand maps to the CUDA name: CUDA's device `abs` is
    #   integer-only, a bare `abs(-2.5f)` silently computes `-2`
    # - widths absent from the table keep the canonical name, `exp2` over
    #   `double` is a CUDA native
    math abs 32 "fabsf", abs 64 "fabs", exp2 32 "exp2f"
    # `{.builtin.}` fp16 conversions resolve to the cuda_fp16.h native calls:
    # `float16` has a first-class `__half` identity, and both `asFp16`/`asU16` entries map to the half bit builtins.
    # `bfloat16` and fp8 conversions have no entry: cuda_bf16.h is not among this translation unit's includes.
    conv toFp16 "__float2half($1)", toFp32 "__half2float($1)",
      asFp16 "__ushort_as_half($1)", asU16 "__half_as_ushort($1)"
    # Float/integer bit reinterprets print as intrinsics, a C-style cast
    # numerically converts and loses the bit pattern. Pairs of same-width integers
    # use the conversion form, a modular cast preserves every bit.
    # `__half_as_ushort` and `__ushort_as_half` cover the two 16-bit float bit-cast directions.
    # Signed 16-bit reinterprets share the unsigned native, and the implicit
    # C conversion preserves the bit pattern.
    bitcast:
      rule srcKinds {ctFloat}, srcWidths {32}, dstKinds {ctUint}, sameWidth, "__float_as_uint($1)"
      rule srcKinds {ctFloat}, srcWidths {32}, dstKinds {ctInt}, sameWidth, "__float_as_int($1)"
      rule srcKinds {ctFloat}, srcWidths {64}, dstKinds {ctInt, ctUint}, sameWidth, "__double_as_longlong($1)"
      rule srcKinds {ctFloat}, srcWidths {16}, dstKinds {ctInt, ctUint}, sameWidth, "__half_as_ushort($1)"
      rule srcKinds {ctUint}, srcWidths {32}, dstKinds {ctFloat}, sameWidth, "__uint_as_float($1)"
      rule srcKinds {ctInt}, srcWidths {32}, dstKinds {ctFloat}, sameWidth, "__int_as_float($1)"
      rule srcKinds {ctInt, ctUint}, srcWidths {64}, dstKinds {ctFloat}, sameWidth, "__longlong_as_double($1)"
      rule srcKinds {ctInt, ctUint}, srcWidths {16}, dstKinds {ctFloat}, sameWidth, "__ushort_as_half($1)"

  target opencl:
    # OpenCL C is LP64 C: `char` and `long` name the 8-bit and 64-bit signed widths, as on msl.
    # Bool prints as `int` to match the launch ABI.
    scalar void "void", bool "int", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", f32 "float", f64 "double"
    lit u64Suffix "ULL", u32Suffix "U", i64Suffix "LL", narrowCast true,
      f16Prefix "(half)", f32Suffix "f", nilToken "0",
      inf16 "(half)INFINITY", negInf16 "(half)-INFINITY", nan16 "(half)NAN",
      inf32 "INFINITY", negInf32 "-INFINITY", nan32 "NAN",
      inf64 "((double)INFINITY)", negInf64 "((double)-INFINITY)", nan64 "((double)NAN)"
    kind unrollDirective "#pragma unroll", braceSameLine true
    # OpenCL C names vectors by appending the lane count to the scalar name
    # (`float4`, `uchar3`), legal lengths 2, 3, 4, 8, 16.
    vector:
      elems void, bool, i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
      lanes 2, 3, 4, 8, 16
    # A C-style cast between float and integer types converts the value.
    # A same-width pair reinterprets through the OpenCL C `as_` builtin instead.
    bitcast:
      rule srcKinds {ctFloat}, dstKinds {ctInt, ctUint}, sameWidth, "as_$2($1)"
      rule srcKinds {ctInt, ctUint}, dstKinds {ctFloat}, sameWidth, "as_$2($1)"
    # `{.builtin.}` fp16 conversions resolve to the OpenCL C cast forms,
    # `float16` keeps a first-class `half` identity and `asFp16`/`asU16`
    # reinterpret through the `as_` builtins.
    # `bfloat16` and fp8 conversions stay outside: no OpenCL C native exists.
    # `{.builtin.}` names pass through and refuses at OpenCL compile time.
    conv toFp16 "(half)($1)", toFp32 "(float)($1)", asFp16 "as_half($1)", asU16 "as_ushort($1)"

  target glsl:
    # 8, 16 and 64-bit names compile behind the explicit-arithmetic-types extensions the module emits.
    scalar void "void", bool "bool", i8 "int8_t", i16 "int16_t", i32 "int", i64 "int64_t",
      u8 "uint8_t", u16 "uint16_t", u32 "uint", u64 "uint64_t", f16 "float16_t", f32 "float", f64 "double"
    # Wide integer and float literals print as constructor calls, glslang rejects the `LL` suffix.
    # GLSL has no infinity or NaN literal, IEEE-754 bit patterns go through `uintBitsToFloat`.
    lit u64Prefix "uint64_t(", u64Suffix ")", u32Suffix "U",
      i64Prefix "int64_t(", i64Suffix ")", narrowCast true,
      f16Prefix "float16_t(", f16Suffix ")", f32Suffix "f",
      f64Prefix "double(", f64Suffix ")",
      inf16 "float16_t(uintBitsToFloat(0x7f800000u))",
      negInf16 "float16_t(uintBitsToFloat(0xff800000u))",
      nan16 "float16_t(uintBitsToFloat(0x7fc00000u))",
      inf32 "uintBitsToFloat(0x7f800000u)",
      negInf32 "uintBitsToFloat(0xff800000u)",
      nan32 "uintBitsToFloat(0x7fc00000u)",
      inf64 "double(uintBitsToFloat(0x7f800000u))",
      negInf64 "double(uintBitsToFloat(0xff800000u))",
      nan64 "double(uintBitsToFloat(0x7fc00000u))"
    # `{.builtin.}` fp16 conversions resolve to the GLSL native call text,
    # `float16` keeps a first-class `float16_t` identity and `asFp16`/`asU16`
    # reinterpret through the float16 bit-level builtins.
    # `bfloat16` and fp8 conversions stay outside: no GLSL native exists.
    # `{.builtin.}` names pass through and refuses at glslang compile time.
    conv toFp16 "float16_t($1)", toFp32 "float($1)", asFp16 "uint16BitsToFloat16($1)",
      asU16 "float16BitsToUint16($1)"
    kind unrollDirective "[[unroll]]", braceSameLine true,
      targetExprKinds {eNil, eAt, ePat, eDeref, eAddr}
    vector:
      elems f16 "f16vec", f32 "vec", f64 "dvec", i32 "ivec", u32 "uvec", bool "bvec"
      lanes 2, 3, 4
    # Core GLSL floatBits builtins cover the 32-bit float pairs, a wider float
    # pair refuses: a constructor call numerically converts and loses the bits.
    # Same-width integer pairs keep the conversion form.
    bitcast:
      rule srcKinds {ctFloat}, srcWidths {32}, dstKinds {ctUint}, sameWidth, "floatBitsToUint($1)"
      rule srcKinds {ctFloat}, srcWidths {32}, dstKinds {ctInt}, sameWidth, "floatBitsToInt($1)"
      rule srcKinds {ctFloat}, dstKinds {ctInt, ctUint}, sameWidth, refusal "no native 64-bit float bit cast"
      rule srcKinds {ctUint}, srcWidths {32}, dstKinds {ctFloat}, sameWidth, "uintBitsToFloat($1)"
      rule srcKinds {ctInt}, srcWidths {32}, dstKinds {ctFloat}, sameWidth, "intBitsToFloat($1)"
      rule srcKinds {ctInt, ctUint}, dstKinds {ctFloat}, sameWidth, refusal "no native 64-bit float bit cast"

  target wgsl:
    # Absent cells refuse at the lookup (emitWgslScalar): WGSL has no 8/16-bit integer types.
    # 64-bit integer and float cells need the shader-int64 and shader-f64 device features.
    # wgpu-native exposes no WGPUFeatureName to request either.
    scalar void "void", bool "bool", i32 "i32", u32 "u32", f16 "f16", f32 "f32"
    # Bare integer and float literals are abstract and work in any typed context,
    # u32 gains the `u` suffix, f16 the `h`, f32 prints bare.
    # WGSL has no infinity or NaN literal: the 16 and 32-bit IEEE-754 bit patterns
    # go through `bitcast` from the unsigned integer of the same width.
    # Nil literals: the shared walk refuses them.
    lit u32Suffix "u", f16Suffix "h",
      inf16 "bitcast<f16>(0x7c00u)", negInf16 "bitcast<f16>(0xfc00u)", nan16 "bitcast<f16>(0x7e00u)",
      inf32 "bitcast<f32>(0x7f800000u)",
      negInf32 "bitcast<f32>(0xff800000u)",
      nan32 "bitcast<f32>(0x7fc00000u)"
    kind unrollDirective "// {.unroll.} requested but not honored on WebGPU", braceSameLine true,
      targetExprKinds {eNil, eAt, ePat, eTernary, eFloatLit}
    vector:
      elems void "vec", bool "vec", i32 "vec", u32 "vec", f16 "vec", f32 "vec"
      lanes 2, 3, 4
      nameForm vnfGenericElem
    # WGSL's `bitcast` reinterprets any type.
    bitcast:
      rule srcAny, dstKinds {low(CirTypeKind)..high(CirTypeKind)}, "bitcast<$2>($1)"
    # `{.builtin.}` fp16 conversions resolve to the WGSL native call text,
    # `float16` keeps a first-class `f16` identity and `asFp16`/`asU16`
    # reinterpret through `bitcast`.
    # `bfloat16` and fp8 conversions stay outside: no WGSL native exists.
    # `{.builtin.}` names pass through and refuses at wgpu parse time.
    conv toFp16 "f16($1)", toFp32 "f32($1)", asFp16 "bitcast<f16>($1)", asU16 "bitcast<u16>($1)"

  target hlsl:
    # Absent cells refuse at the lookup (emitHlslScalar): the 8/16-bit and 64-bit integer
    # types are frontend parse errors or capability rejections. The float16 cell is
    # outside the subset.
    scalar void "void", bool "bool", i32 "int", u32 "uint", f32 "float", f64 "double"
    # HLSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
    # `asfloat` (32 bit) and `asdouble` (64 bit). Without the suffix the HLSL
    # frontend changes width or signedness. Nil literals: the shared walk refuses them.
    lit u32Suffix "u", u64Suffix "u", f16Suffix "f", f32Suffix "f", f64Suffix "l",
      inf16 "asfloat(0x7f800000)", negInf16 "asfloat(0xff800000)", nan16 "asfloat(0x7fc00000)",
      inf32 "asfloat(0x7f800000)", negInf32 "asfloat(0xff800000)", nan32 "asfloat(0x7fc00000)",
      inf64 "asdouble(0u, 0x7ff00000u)",
      negInf64 "asdouble(0u, 0xfff00000u)",
      nan64 "asdouble(0u, 0x7ff80000u)"
    kind unrollDirective "[unroll]", braceSameLine false, foldsIteElse true,
      emitOperandRendersExpr true, targetExprKinds {eNil, eAt, ePat, eDeref, eAddr}
    # `{.builtin.}` fp16/bf16/fp8 conversion calls refuse: the frontend erases
    # half-width types to 16-bit storage before printing, so the call name is the only signal the printer can match.
    refusals toFp16, toFp32, toBf16, toBf32, toFp8e4m3, toFp8e5m2, asFp16, asU16
    vector:
      elems bool "bool", i32 "int", u32 "uint", f32 "float"
      lanes 2, 3, 4
    # Only the 32-bit asint/asuint/asfloat reinterprets are HLSL bit-cast forms,
    # every other bit cast refuses.
    bitcast:
      rule srcAny, dstKinds {ctInt}, dstWidths {32}, "asint($1)"
      rule srcAny, dstKinds {ctUint}, dstWidths {32}, "asuint($1)"
      rule srcAny, dstKinds {ctFloat}, dstWidths {32}, "asfloat($1)"
      rule srcAny, dstKinds {low(CirTypeKind)..high(CirTypeKind)}, refusal "no bit cast to $2"
