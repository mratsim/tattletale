# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Rows of the Metal (MSL), CUDA C++, OpenCL C, GLSL, WebGPU WGSL and HLSL targets.
##
## Static target data of the Metal (MSL), CUDA C++, OpenCL C, GLSL, WebGPU
## WGSL and HLSL targets, in the `genTargets` block DSL.
##
## - the macro in `targets_macros.nim` validates every row at compile time
##   and generates one lookup const per target and family
## - a missing or mismatched row is a compile error at its line here
## - the printers read the generated lookups, signature text, coordinate
##   builtins and refusal messages stay procedural in the printers

import std/tables
import ../ir/cirv2_lang
import ./targets_registry
import ./targets_macros

genTargets:
  target msl:
    scalar void "void", bool "bool", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", f32 "float",
      refused f64 "double has no Apple GPU support"
    # `char` and `long` are MSL's 8-bit signed and 64-bit type names.
    # CUDA names the same widths `signed char` and `long long`, each row lists its own names
    lit u64Suffix "ULL", u32Suffix "U", i64Suffix "LL", narrowCast true,
      f16Suffix "h", f32Suffix "f", nilToken "nullptr",
      inf16 "INFINITY", negInf16 "-INFINITY", nan16 "NAN",
      inf32 "INFINITY", negInf32 "-INFINITY", nan32 "NAN",
      inf64 "INFINITY", negInf64 "-INFINITY", nan64 "NAN"
    kind unrollDirective "#pragma clang loop unroll(full)", braceSameLine true
    # `{.builtin.}` numeric conversions resolve to the MSL native call text,
    # each entry a format string whose `$1` stands for the call's argument list.
    # float16 has a first-class `half` identity: its conversions are plain native calls.
    # `asFp16`/`asU16` reinterpret through `as_type`.
    # no MSL type exists for bfloat16 here, the value emits as raw bits.
    # toBf16/toBf32 move the bit pattern through `as_type<ushort>` around the native `(bfloat)` cast.
    # fp8 conversions have no MSL native: the `{.builtin.}` name passes through and the Metal
    # compiler refuses it at compile time.
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
      f32 "float", f64 "double",
      refused f16 "the __half names live behind cuda_fp16.h and this printer emits no includes"
    lit u64Suffix "ULL", u32Suffix "U", i64Suffix "LL", narrowCast true,
      f32Suffix "f", nilToken "nullptr",
      inf16 "INFINITY", negInf16 "-INFINITY", nan16 "NAN",
      inf32 "INFINITY", negInf32 "-INFINITY", nan32 "NAN",
      inf64 "INFINITY", negInf64 "-INFINITY", nan64 "NAN"
    kind unrollDirective "#pragma unroll", braceSameLine true
    # CUDA has no vector types, the printer refuses every vector at its declaration.
    vector ()
    # `{.builtin.}` scalar-math calls keyed by (canonical name, operand bit width):
    #
    # - a float operand maps to the CUDA name: CUDA's device `abs` is
    #   integer-only, a bare `abs(-2.5f)` silently computes `-2`
    # - widths absent from the table keep the canonical name, `exp2` over
    #   `double` is a CUDA native
    math abs 32 "fabsf", abs 64 "fabs", exp2 32 "exp2f"
    # Float/integer bit reinterprets print as intrinsics, a C-style cast
    # numerically converts and loses the bit pattern. The printer keeps the conversion
    # form for pairs of same-width integers, a modular cast preserves every bit.
    #
    # a 16-bit float pair: the printer refuses it on the wiring state.
    # `__ushort_as_half`/`__half_as_ushort` exist behind cuda_fp16.h,
    # no printer include reaches them.
    bitcast:
      rule srcKinds {ctFloat}, srcWidths {32}, dstKinds {ctUint}, sameWidth, "__float_as_uint($1)"
      rule srcKinds {ctFloat}, srcWidths {32}, dstKinds {ctInt}, sameWidth, "__float_as_int($1)"
      rule srcKinds {ctFloat}, srcWidths {64}, dstKinds {ctInt, ctUint}, sameWidth, "__double_as_longlong($1)"
      rule srcKinds {ctFloat}, dstKinds {ctInt, ctUint}, sameWidth, refusal "bit cast of a 16-bit float is unwired, the __half_as_ushort form lives behind cuda_fp16.h and this printer emits no includes"
      rule srcKinds {ctUint}, srcWidths {32}, dstKinds {ctFloat}, sameWidth, "__uint_as_float($1)"
      rule srcKinds {ctInt}, srcWidths {32}, dstKinds {ctFloat}, sameWidth, "__int_as_float($1)"
      rule srcKinds {ctInt, ctUint}, srcWidths {64}, dstKinds {ctFloat}, sameWidth, "__longlong_as_double($1)"
      rule srcKinds {ctInt, ctUint}, dstKinds {ctFloat}, sameWidth, refusal "bit cast to a 16-bit float is unwired, the __ushort_as_half form lives behind cuda_fp16.h and this printer emits no includes"

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
    # float16 has a first-class `half` identity and `asFp16`/`asU16`
    # reinterpret through the `as_` builtins.
    # `bfloat16` and fp8 conversions stay outside: no OpenCL C native exists.
    # `{.builtin.}` names pass through and the OpenCL compiler refuses them at compile time.
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
    # float16 has a first-class `float16_t` identity and `asFp16`/`asU16`
    # reinterpret through the float16 bit-level builtins.
    # `bfloat16` and fp8 conversions stay outside: no GLSL native exists.
    # `{.builtin.}` names pass through and the glslang compiler refuses them at compile time.
    conv toFp16 "float16_t($1)", toFp32 "float($1)", asFp16 "uint16BitsToFloat16($1)",
      asU16 "float16BitsToUint16($1)"
    kind unrollDirective "[[unroll]]", braceSameLine true,
      targetExprKinds {eNil, eAt, ePat, eDeref, eAddr}
    vector:
      elems f16 "f16vec", f32 "vec", f64 "dvec", i32 "ivec", u32 "uvec", bool "bvec"
      lanes 2, 3, 4
    # Core GLSL floatBits builtins cover the pairs of 32-bit floats, a wider float
    # pair meets the printer's refusal: a constructor call numerically converts and loses the bits.
    # for pairs of same-width integers the printer keeps the conversion form.
    bitcast:
      rule srcKinds {ctFloat}, srcWidths {32}, dstKinds {ctUint}, sameWidth, "floatBitsToUint($1)"
      rule srcKinds {ctFloat}, srcWidths {32}, dstKinds {ctInt}, sameWidth, "floatBitsToInt($1)"
      rule srcKinds {ctFloat}, dstKinds {ctInt, ctUint}, sameWidth, refusal "bit cast of a 64-bit float has no GLSL emission, only the 32-bit floatBits builtins"
      rule srcKinds {ctUint}, srcWidths {32}, dstKinds {ctFloat}, sameWidth, "uintBitsToFloat($1)"
      rule srcKinds {ctInt}, srcWidths {32}, dstKinds {ctFloat}, sameWidth, "intBitsToFloat($1)"
      rule srcKinds {ctInt, ctUint}, dstKinds {ctFloat}, sameWidth, refusal "bit cast to a 64-bit float has no GLSL emission, only the 32-bit floatBits builtins"

  target wgsl:
    # Absent cells refuse at the lookup (emitWgslScalar): WGSL has no 8/16-bit integer types.
    # 64-bit integer and float cells need the shader-int64 and shader-f64 device features.
    # wgpu-native exposes no WGPUFeatureName to request either.
    scalar void "void", bool "bool", i32 "i32", u32 "u32", f16 "f16", f32 "f32"
    # Bare integer and float literals are abstract and work in any typed context,
    # u32 gains the `u` suffix, f16 the `h`, f32 prints bare.
    # WGSL has no infinity or NaN literal: the 16 and 32-bit IEEE-754 bit patterns
    # go through `bitcast` from the unsigned integer of the same width.
    # nil literals: the printer refuses them at the site.
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
    # float16 has a first-class `f16` identity and `asFp16`/`asU16`
    # reinterpret through `bitcast`.
    # `bfloat16` and fp8 conversions stay outside: no WGSL native exists.
    # `{.builtin.}` names pass through and the wgpu parser refuses them at parse time.
    conv toFp16 "f16($1)", toFp32 "f32($1)", asFp16 "bitcast<f16>($1)", asU16 "bitcast<u16>($1)"

  target hlsl:
    # Absent cells refuse at the lookup (emitHlslScalar): the 8/16-bit and 64-bit integer
    # types are frontend parse errors or capability rejections. The float16 cell is
    # outside the subset.
    scalar void "void", bool "bool", i32 "int", u32 "uint", f32 "float", f64 "double"
    # HLSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
    # `asfloat` (32 bit) and `asdouble` (64 bit). Without the suffix the HLSL
    # frontend changes width or signedness. The printer refuses nil literals at the site.
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
    # every other bit cast: the printer refuses it.
    bitcast:
      rule srcAny, dstKinds {ctInt}, dstWidths {32}, "asint($1)"
      rule srcAny, dstKinds {ctUint}, dstWidths {32}, "asuint($1)"
      rule srcAny, dstKinds {ctFloat}, dstWidths {32}, "asfloat($1)"
      rule srcAny, dstKinds {low(CirTypeKind)..high(CirTypeKind)}, refusal "bit cast to $2 has no HLSL emission, only the 32-bit asint/asuint/asfloat reinterprets"
