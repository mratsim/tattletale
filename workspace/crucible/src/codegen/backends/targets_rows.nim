# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Rows of the Metal (MSL), CUDA C++, OpenCL C, GLSL, WebGPU WGSL and HLSL targets.
##
## Plain per-target constants, one per row family, read directly by the printer
## that needs them: msl_lang.nim, cuda_lang.nim, opencl_lang.nim, glsl_lang.nim,
## wgsl_lang.nim and hlsl_lang.nim.
## Signature text, coordinate builtins and refusal messages stay procedural in the printers.

import std/tables
import ../ir/cirv2_lang
import ./targets_registry

const mslScalarRow* = ScalarRow(cells: @[
  ScalarCell(kind: ctVoid, width: 0, spelling: "void"),
  ScalarCell(kind: ctBool, width: 0, spelling: "bool"),
  # `char` and `long` are MSL's 8-bit signed and 64-bit type names.
  # CUDA names the same widths `signed char` and `long long`, each row states its own names
  ScalarCell(kind: ctInt, width: 8, spelling: "char"),
  ScalarCell(kind: ctInt, width: 16, spelling: "short"),
  ScalarCell(kind: ctInt, width: 32, spelling: "int"),
  ScalarCell(kind: ctInt, width: 64, spelling: "long"),
  ScalarCell(kind: ctUint, width: 8, spelling: "uchar"),
  ScalarCell(kind: ctUint, width: 16, spelling: "ushort"),
  ScalarCell(kind: ctUint, width: 32, spelling: "uint"),
  ScalarCell(kind: ctUint, width: 64, spelling: "ulong"),
  ScalarCell(kind: ctFloat, width: 16, spelling: "half"),
  ScalarCell(kind: ctFloat, width: 32, spelling: "float"),
  ScalarCell(kind: ctFloat, width: 64)])

const mslLitRow* = LitRow(
  int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
  f16Suffix: "h",
  f32Suffix: "f",
  nilToken: "nullptr",
  nonFinite: FloatLitTokens(
    inf16: "INFINITY", negInf16: "-INFINITY", nan16: "NAN",
    inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
    inf64: "INFINITY", negInf64: "-INFINITY", nan64: "NAN"))

const mslKindRow* = KindRow(
  loop: LoopStyle(
    unrollDirective: "#pragma clang loop unroll(full)",
    braceSameLine: true))

# `{.builtin.}` numeric conversions resolve to the MSL native call text,
# each entry a format string whose `$1` stands for the call's argument list.
# CIRv2 has no fp16 or bf16 type kind,
# `float16` and `bfloat16` are `distinct uint16` over raw bits in `ushort` storage.
# Value-producing conversions move the bit pattern, not the numeric value.
# fp8 conversions have no MSL native: the `{.builtin.}` name passes through and refuses at Metal compile time.
const mslConvBuiltinNames* = {
  "toFp16": "as_type<ushort>(half($1))",
  "toFp32": "float(as_type<half>($1))",
  "toBf16": "as_type<ushort>((bfloat)($1))",
  "toBf32": "float(as_type<bfloat>($1))",
  "asFp16": "$1",
  "asU16": "$1",
}.toTable()

# Coordinate builtins carry `ivec3`, so the builtin-threading parameters
# and whole-value reads print this vector name: 32-bit int/uint elements only.
const mslVectorRule* = VectorRule(
  elems: @[(ctInt, 32, ""), (ctUint, 32, "")],
  lanes: @[2, 3, 4])

# Same-width scalar pairs reinterpret through MSL's `as_type<T>`.
# Pointer casts and width changes keep the conversion form.
const mslCastRules* = [
  CastRule(srcKinds: {ctInt, ctUint, ctFloat}, dstKinds: {ctInt, ctUint, ctFloat},
    srcWidths: {8, 16, 32, 64}, sameWidth: true, spelling: "as_type<$2>($1)")]

const cudaScalarRow* = ScalarRow(cells: @[
  ScalarCell(kind: ctVoid, width: 0, spelling: "void"),
  ScalarCell(kind: ctBool, width: 0, spelling: "bool"),
  ScalarCell(kind: ctInt, width: 8, spelling: "signed char"),
  ScalarCell(kind: ctInt, width: 16, spelling: "short"),
  ScalarCell(kind: ctInt, width: 32, spelling: "int"),
  ScalarCell(kind: ctInt, width: 64, spelling: "long long"),
  ScalarCell(kind: ctUint, width: 8, spelling: "unsigned char"),
  ScalarCell(kind: ctUint, width: 16, spelling: "unsigned short"),
  ScalarCell(kind: ctUint, width: 32, spelling: "unsigned int"),
  ScalarCell(kind: ctUint, width: 64, spelling: "unsigned long long"),
  # CUDA names fp16 `__half` behind cuda_fp16.h. This printer emits no
  # includes and no fp16 type name, so the cell refuses.
  ScalarCell(kind: ctFloat, width: 16),
  ScalarCell(kind: ctFloat, width: 32, spelling: "float"),
  ScalarCell(kind: ctFloat, width: 64, spelling: "double")])

const cudaLitRow* = LitRow(
  int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
  f32Suffix: "f",
  nilToken: "nullptr",
  nonFinite: FloatLitTokens(
    inf16: "INFINITY", negInf16: "-INFINITY", nan16: "NAN",
    inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
    inf64: "INFINITY", negInf64: "-INFINITY", nan64: "NAN"))

const cudaKindRow* = KindRow(
  loop: LoopStyle(
    unrollDirective: "#pragma unroll",
    braceSameLine: true))

const cudaVectorRule* = VectorRule()
## CUDA names no vector types, every vector refuses at its declaration.

# Float/integer bit reinterprets print as intrinsics, a C-style cast
# numerically converts and loses the bit pattern. A 16-bit float pair has no
# intrinsic and refuses. Same-width integer pairs keep the conversion form,
# a modular cast preserves every bit.
const cudaCastRules* = [
  CastRule(srcKinds: {ctFloat}, dstKinds: {ctUint}, srcWidths: {32},
    sameWidth: true, spelling: "__float_as_uint($1)"),
  CastRule(srcKinds: {ctFloat}, dstKinds: {ctInt}, srcWidths: {32},
    sameWidth: true, spelling: "__float_as_int($1)"),
  CastRule(srcKinds: {ctFloat}, dstKinds: {ctInt, ctUint}, srcWidths: {64},
    sameWidth: true, spelling: "__double_as_longlong($1)"),
  CastRule(srcKinds: {ctFloat}, dstKinds: {ctInt, ctUint}, sameWidth: true,
    refusal: "bit cast of a 16-bit float has no CUDA emission"),
  CastRule(srcKinds: {ctUint}, dstKinds: {ctFloat}, srcWidths: {32},
    sameWidth: true, spelling: "__uint_as_float($1)"),
  CastRule(srcKinds: {ctInt}, dstKinds: {ctFloat}, srcWidths: {32},
    sameWidth: true, spelling: "__int_as_float($1)"),
  CastRule(srcKinds: {ctInt, ctUint}, dstKinds: {ctFloat}, srcWidths: {64},
    sameWidth: true, spelling: "__longlong_as_double($1)"),
  CastRule(srcKinds: {ctInt, ctUint}, dstKinds: {ctFloat}, sameWidth: true,
    refusal: "bit cast to a 16-bit float has no CUDA emission")]

const openclScalarRow* = ScalarRow(cells: @[
  # OpenCL C is LP64 C: `char` and `long` name the 8-bit and 64-bit signed widths, as on msl.
  # Bool prints as `int` to match the launch ABI.
  ScalarCell(kind: ctVoid, width: 0, spelling: "void"),
  ScalarCell(kind: ctBool, width: 0, spelling: "int"),
  ScalarCell(kind: ctInt, width: 8, spelling: "char"),
  ScalarCell(kind: ctInt, width: 16, spelling: "short"),
  ScalarCell(kind: ctInt, width: 32, spelling: "int"),
  ScalarCell(kind: ctInt, width: 64, spelling: "long"),
  ScalarCell(kind: ctUint, width: 8, spelling: "uchar"),
  ScalarCell(kind: ctUint, width: 16, spelling: "ushort"),
  ScalarCell(kind: ctUint, width: 32, spelling: "uint"),
  ScalarCell(kind: ctUint, width: 64, spelling: "ulong"),
  ScalarCell(kind: ctFloat, width: 16, spelling: "half"),
  ScalarCell(kind: ctFloat, width: 32, spelling: "float"),
  ScalarCell(kind: ctFloat, width: 64, spelling: "double")])

const openclLitRow* = LitRow(
  int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
  f16Prefix: "(half)",
  f32Suffix: "f",
  nilToken: "0",
  nonFinite: FloatLitTokens(
    inf16: "(half)INFINITY", negInf16: "(half)-INFINITY", nan16: "(half)NAN",
    inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
    inf64: "((double)INFINITY)", negInf64: "((double)-INFINITY)", nan64: "((double)NAN)"))

const openclKindRow* = KindRow(
  loop: LoopStyle(
    unrollDirective: "#pragma unroll",
    braceSameLine: true))

# OpenCL C names vectors by appending the lane count to the scalar name
# (`float4`, `uchar3`), legal lengths 2, 3, 4, 8, 16.
const openclVectorRule* = VectorRule(
  elems: @[
    (ctVoid, 0, ""), (ctBool, 0, ""),
    (ctInt, 8, ""), (ctInt, 16, ""), (ctInt, 32, ""), (ctInt, 64, ""),
    (ctUint, 8, ""), (ctUint, 16, ""), (ctUint, 32, ""), (ctUint, 64, ""),
    (ctFloat, 16, ""), (ctFloat, 32, ""), (ctFloat, 64, "")],
  lanes: @[2, 3, 4, 8, 16])

# A C-style cast between float and integer types converts the value.
# A same-width pair reinterprets through the OpenCL C `as_` builtin instead.
const openclCastRules* = [
  CastRule(srcKinds: {ctFloat}, dstKinds: {ctInt, ctUint}, sameWidth: true,
    spelling: "as_$2($1)"),
  CastRule(srcKinds: {ctInt, ctUint}, dstKinds: {ctFloat}, sameWidth: true,
    spelling: "as_$2($1)")]

const glslScalarRow* = ScalarRow(cells: @[
  # 8, 16 and 64-bit names compile behind the explicit-arithmetic-types extensions the module emits.
  ScalarCell(kind: ctVoid, width: 0, spelling: "void"),
  ScalarCell(kind: ctBool, width: 0, spelling: "bool"),
  ScalarCell(kind: ctInt, width: 8, spelling: "int8_t"),
  ScalarCell(kind: ctInt, width: 16, spelling: "int16_t"),
  ScalarCell(kind: ctInt, width: 32, spelling: "int"),
  ScalarCell(kind: ctInt, width: 64, spelling: "int64_t"),
  ScalarCell(kind: ctUint, width: 8, spelling: "uint8_t"),
  ScalarCell(kind: ctUint, width: 16, spelling: "uint16_t"),
  ScalarCell(kind: ctUint, width: 32, spelling: "uint"),
  ScalarCell(kind: ctUint, width: 64, spelling: "uint64_t"),
  ScalarCell(kind: ctFloat, width: 16, spelling: "float16_t"),
  ScalarCell(kind: ctFloat, width: 32, spelling: "float"),
  ScalarCell(kind: ctFloat, width: 64, spelling: "double")])

# Wide integer and float literals print as constructor calls, glslang rejects the `LL` suffix.
# GLSL has no infinity or NaN literal, IEEE-754 bit patterns go through `uintBitsToFloat`.
const glslLitRow* = LitRow(
  int: IntLitTokens(
    u64Prefix: "uint64_t(", u64Suffix: ")",
    u32Suffix: "U",
    i64Prefix: "int64_t(", i64Suffix: ")",
    narrowCast: true),
  f16Prefix: "float16_t(", f16Suffix: ")",
  f32Suffix: "f",
  f64Prefix: "double(", f64Suffix: ")",
  nonFinite: FloatLitTokens(
    inf16: "float16_t(uintBitsToFloat(0x7f800000u))",
    negInf16: "float16_t(uintBitsToFloat(0xff800000u))",
    nan16: "float16_t(uintBitsToFloat(0x7fc00000u))",
    inf32: "uintBitsToFloat(0x7f800000u)",
    negInf32: "uintBitsToFloat(0xff800000u)",
    nan32: "uintBitsToFloat(0x7fc00000u)",
    inf64: "double(uintBitsToFloat(0x7f800000u))",
    negInf64: "double(uintBitsToFloat(0xff800000u))",
    nan64: "double(uintBitsToFloat(0x7fc00000u))"))

const glslKindRow* = KindRow(
  loop: LoopStyle(
    unrollDirective: "[[unroll]]",
    braceSameLine: true),
  targetExprKinds: {eNil, eAt, ePat, eDeref, eAddr})

const glslVectorRule* = VectorRule(
  elems: @[
    (ctFloat, 16, "f16vec"), (ctFloat, 32, "vec"), (ctFloat, 64, "dvec"),
    (ctInt, 32, "ivec"), (ctUint, 32, "uvec"), (ctBool, 0, "bvec")],
  lanes: @[2, 3, 4])

# Core GLSL floatBits builtins cover the 32-bit float pairs, a wider float
# pair refuses: a constructor call numerically converts and loses the bits.
# Same-width integer pairs keep the conversion form.
const glslCastRules* = [
  CastRule(srcKinds: {ctFloat}, dstKinds: {ctUint}, srcWidths: {32},
    sameWidth: true, spelling: "floatBitsToUint($1)"),
  CastRule(srcKinds: {ctFloat}, dstKinds: {ctInt}, srcWidths: {32},
    sameWidth: true, spelling: "floatBitsToInt($1)"),
  CastRule(srcKinds: {ctFloat}, dstKinds: {ctInt, ctUint}, sameWidth: true,
    refusal: "bit cast of a 64-bit float has no GLSL emission, only the" &
      " 32-bit floatBits builtins"),
  CastRule(srcKinds: {ctUint}, dstKinds: {ctFloat}, srcWidths: {32},
    sameWidth: true, spelling: "uintBitsToFloat($1)"),
  CastRule(srcKinds: {ctInt}, dstKinds: {ctFloat}, srcWidths: {32},
    sameWidth: true, spelling: "intBitsToFloat($1)"),
  CastRule(srcKinds: {ctInt, ctUint}, dstKinds: {ctFloat}, sameWidth: true,
    refusal: "bit cast to a 64-bit float has no GLSL emission, only the" &
      " 32-bit floatBits builtins")]

const wgslScalarRow* = ScalarRow(cells: @[
  # WGSL has no 8/16-bit integer types, 32 bit is its only always-available width.
  # 64-bit integers and floats require the shader-int64 and shader-f64 device features.
  # wgpu-native exposes no WGPUFeatureName for either feature and no request can enable them, so the cells refuse.
  ScalarCell(kind: ctVoid, width: 0, spelling: "void"),
  ScalarCell(kind: ctBool, width: 0, spelling: "bool"),
  ScalarCell(kind: ctInt, width: 8),
  ScalarCell(kind: ctInt, width: 16),
  ScalarCell(kind: ctInt, width: 32, spelling: "i32"),
  ScalarCell(kind: ctInt, width: 64),
  ScalarCell(kind: ctUint, width: 8),
  ScalarCell(kind: ctUint, width: 16),
  ScalarCell(kind: ctUint, width: 32, spelling: "u32"),
  ScalarCell(kind: ctUint, width: 64),
  ScalarCell(kind: ctFloat, width: 16, spelling: "f16"),
  ScalarCell(kind: ctFloat, width: 32, spelling: "f32"),
  ScalarCell(kind: ctFloat, width: 64)])

# Bare integer and float literals are abstract and work in any typed context,
# u32 gains the `u` suffix, f16 the `h`, f32 prints bare.
# WGSL has no infinity or NaN literal: the 16 and 32-bit IEEE-754 bit patterns
# go through `bitcast` from the unsigned integer of the same width.
# 64-bit bit patterns have no WGSL form and the empty tokens refuse.
# `bitcast<f64>` requires the shader-f64 device feature the runtime cannot request.
# A 64-bit literal with the `u` suffix is a parse error besides.
# Nil literals refuse at the site.
const wgslLitRow* = LitRow(
  int: IntLitTokens(u32Suffix: "u"),
  f16Suffix: "h",
  nonFinite: FloatLitTokens(
    inf16: "bitcast<f16>(0x7c00u)", negInf16: "bitcast<f16>(0xfc00u)", nan16: "bitcast<f16>(0x7e00u)",
    inf32: "bitcast<f32>(0x7f800000u)",
    negInf32: "bitcast<f32>(0xff800000u)",
    nan32: "bitcast<f32>(0x7fc00000u)"))

const wgslKindRow* = KindRow(
  loop: LoopStyle(
    # WGSL has no unroll directive: the drop surfaces in the artifact
    unrollDirective: "// {.unroll.} requested but not honored on WebGPU",
    braceSameLine: true),
  # eFloatLit routes through `printTargetExpr`, a 64-bit non-finite float
  # refuses there. The 16 and 32-bit tokens come from the lit row.
  targetExprKinds: {eNil, eAt, ePat, eTernary, eFloatLit})

const wgslVectorRule* = VectorRule(
  elems: @[
    (ctVoid, 0, "vec"), (ctBool, 0, "vec"),
    (ctInt, 32, "vec"), (ctUint, 32, "vec"),
    (ctFloat, 16, "vec"), (ctFloat, 32, "vec")],
  lanes: @[2, 3, 4],
  nameForm: vnfGenericElem)

# WGSL's `bitcast` reinterprets any type.
const wgslCastRules* = [
  CastRule(srcAny: true, dstKinds: {low(CirTypeKind)..high(CirTypeKind)},
    spelling: "bitcast<$2>($1)")]

const hlslScalarRow* = ScalarRow(cells: @[
  # 8/16-bit integers and int64_t are parse errors in the HLSL frontend,
  # uint64_t pulls the SPIR-V Int64 capability, float16 is outside the subset.
  ScalarCell(kind: ctVoid, width: 0, spelling: "void"),
  ScalarCell(kind: ctBool, width: 0, spelling: "bool"),
  ScalarCell(kind: ctInt, width: 8),
  ScalarCell(kind: ctInt, width: 16),
  ScalarCell(kind: ctInt, width: 32, spelling: "int"),
  ScalarCell(kind: ctInt, width: 64),
  ScalarCell(kind: ctUint, width: 8),
  ScalarCell(kind: ctUint, width: 16),
  ScalarCell(kind: ctUint, width: 32, spelling: "uint"),
  ScalarCell(kind: ctUint, width: 64),
  ScalarCell(kind: ctFloat, width: 16),
  ScalarCell(kind: ctFloat, width: 32, spelling: "float"),
  ScalarCell(kind: ctFloat, width: 64, spelling: "double")])

# HLSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
# `asfloat` (32 bit) and `asdouble` (64 bit). Without the suffix the HLSL
# frontend changes width or signedness. Nil literals refuse at the site.
const hlslLitRow* = LitRow(
  int: IntLitTokens(u32Suffix: "u", u64Suffix: "u"),
  f16Suffix: "f",
  f32Suffix: "f",
  f64Suffix: "l",
  nonFinite: FloatLitTokens(
    inf16: "asfloat(0x7f800000)", negInf16: "asfloat(0xff800000)", nan16: "asfloat(0x7fc00000)",
    inf32: "asfloat(0x7f800000)", negInf32: "asfloat(0xff800000)", nan32: "asfloat(0x7fc00000)",
    inf64: "asdouble(0u, 0x7ff00000u)",
    negInf64: "asdouble(0u, 0xfff00000u)",
    nan64: "asdouble(0u, 0x7ff80000u)"))

const hlslKindRow* = KindRow(
  loop: LoopStyle(
    unrollDirective: "[unroll]",
    braceSameLine: false),
  foldsIteElse: true,
  emitOperandRendersExpr: true,
  targetExprKinds: {eNil, eAt, ePat, eDeref, eAddr})

# `{.builtin.}` fp16/bf16/fp8 conversion calls refuse: the frontend erases
# half-width types to 16-bit storage before printing, so the call name is the only signal the printer can match.
const hlslConvFpRefusals* = [
  "toFp16", "toFp32", "toBf16", "toBf32", "toFp8e4m3", "toFp8e5m2",
  "asFp16", "asU16"]

const hlslVectorRule* = VectorRule(
  elems: @[
    (ctBool, 0, "bool"), (ctInt, 32, "int"), (ctUint, 32, "uint"),
    (ctFloat, 32, "float")],
  lanes: @[2, 3, 4])

# Only the 32-bit asint/asuint/asfloat reinterprets are HLSL bit-cast forms,
# every other bit cast refuses.
const hlslCastRules* = [
  CastRule(srcAny: true, dstKinds: {ctInt}, dstWidths: {32}, spelling: "asint($1)"),
  CastRule(srcAny: true, dstKinds: {ctUint}, dstWidths: {32}, spelling: "asuint($1)"),
  CastRule(srcAny: true, dstKinds: {ctFloat}, dstWidths: {32}, spelling: "asfloat($1)"),
  CastRule(srcAny: true, dstKinds: {low(CirTypeKind)..high(CirTypeKind)},
    refusal: "bit cast to $2 has no HLSL emission, only the" &
      " 32-bit asint/asuint/asfloat reinterprets")]

static:
  checkScalarRow(mslScalarRow, "msl")
  checkScalarRow(cudaScalarRow, "cuda")
  checkScalarRow(openclScalarRow, "opencl")
  checkScalarRow(glslScalarRow, "glsl")
  checkScalarRow(wgslScalarRow, "wgsl")
  checkScalarRow(hlslScalarRow, "hlsl")
  checkScalarRowsTogether([
    ("msl", mslScalarRow), ("cuda", cudaScalarRow),
    ("opencl", openclScalarRow), ("glsl", glslScalarRow),
    ("wgsl", wgslScalarRow), ("hlsl", hlslScalarRow)])
