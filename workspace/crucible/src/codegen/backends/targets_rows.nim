# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Registry rows of the Metal (MSL), CUDA C++, OpenCL C, GLSL, WebGPU WGSL and HLSL targets.
##
## The static target data the six printers read through the registry lookups:
## msl_lang.nim, cuda_lang.nim, opencl_lang.nim, glsl_lang.nim, wgsl_lang.nim and hlsl_lang.nim.
## Signature text, coordinate builtins and refusal messages stay procedural in the printers.

import std/tables
import ../ir/cirv2_lang
import ./targets_registry
import ./targets_macros

genTargets:
  target msl:
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      # `char` and `long` are MSL's 8-bit signed and 64-bit type names.
      # CUDA names the same widths `signed char` and `long long`, each row states its own names
      ScalarCell(kind: ctInt, width: 8, spelling: "char", supported: true),
      ScalarCell(kind: ctInt, width: 16, spelling: "short", supported: true),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "long", supported: true),
      ScalarCell(kind: ctUint, width: 8, spelling: "uchar", supported: true),
      ScalarCell(kind: ctUint, width: 16, spelling: "ushort", supported: true),
      ScalarCell(kind: ctUint, width: 32, spelling: "uint", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "ulong", supported: true),
      ScalarCell(kind: ctFloat, width: 16, spelling: "half", supported: true),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, supported: false)])
    lit: LitRow(
      int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
      f16Suffix: "h",
      f32Suffix: "f",
      nilToken: "nullptr",
      nonFinite: FloatLitTokens(
        inf16: "INFINITY", negInf16: "-INFINITY", nan16: "NAN",
        inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
        inf64: "INFINITY", negInf64: "-INFINITY", nan64: "NAN"))
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "#pragma clang loop unroll(full)",
        infiniteHead: "for (;;) {\n",
        braceSameLine: true))
    # `{.builtin.}` numeric conversions resolve to the MSL native call text,
    # each entry a format string whose `$1` stands for the call's argument list.
    # CIRv2 has no fp16 or bf16 type kind,
    # `float16` and `bfloat16` are `distinct uint16` over raw bits in `ushort` storage.
    # Value-producing conversions move the bit pattern, not the numeric value.
    # fp8 conversions have no MSL native: the `{.builtin.}` name passes through and refuses at Metal compile time.
    conv: ConvRow(
      builtinNames: {
        "toFp16": "as_type<ushort>(half($1))",
        "toFp32": "float(as_type<half>($1))",
        "toBf16": "as_type<ushort>((bfloat)($1))",
        "toBf32": "float(as_type<bfloat>($1))",
        "asFp16": "$1",
        "asU16": "$1",
      }.toTable())
    # Coordinate builtins carry `ivec3`, so the builtin-threading parameters
    # and whole-value reads print this vector name: 32-bit int/uint elements only.
    vector: VectorRule(
      context: vcCoordinate,
      elems: @[(ctInt, 32, ""), (ctUint, 32, "")],
      lanes: @[2, 3, 4])

  target cuda:
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      ScalarCell(kind: ctInt, width: 8, spelling: "signed char", supported: true),
      ScalarCell(kind: ctInt, width: 16, spelling: "short", supported: true),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "long long", supported: true),
      ScalarCell(kind: ctUint, width: 8, spelling: "unsigned char", supported: true),
      ScalarCell(kind: ctUint, width: 16, spelling: "unsigned short", supported: true),
      ScalarCell(kind: ctUint, width: 32, spelling: "unsigned int", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "unsigned long long", supported: true),
      # CUDA names fp16 `__half` behind cuda_fp16.h. This printer emits no
      # includes and no fp16 type name, so the cell refuses.
      ScalarCell(kind: ctFloat, width: 16, supported: false),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "double", supported: true)])
    lit: LitRow(
      int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
      f32Suffix: "f",
      nilToken: "nullptr",
      nonFinite: FloatLitTokens(
        inf16: "INFINITY", negInf16: "-INFINITY", nan16: "NAN",
        inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
        inf64: "INFINITY", negInf64: "-INFINITY", nan64: "NAN"))
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "#pragma unroll",
        infiniteHead: "for (;;) {\n",
        braceSameLine: true))
    conv: ConvRow()
    vector: VectorRule(context: vcNone)

  target opencl:
    # OpenCL C is LP64 C: `char` and `long` name the 8-bit and 64-bit signed widths, as on msl.
    # Bool prints as `int` to match the launch ABI.
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 8, spelling: "char", supported: true),
      ScalarCell(kind: ctInt, width: 16, spelling: "short", supported: true),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "long", supported: true),
      ScalarCell(kind: ctUint, width: 8, spelling: "uchar", supported: true),
      ScalarCell(kind: ctUint, width: 16, spelling: "ushort", supported: true),
      ScalarCell(kind: ctUint, width: 32, spelling: "uint", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "ulong", supported: true),
      ScalarCell(kind: ctFloat, width: 16, spelling: "half", supported: true),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "double", supported: true)])
    lit: LitRow(
      int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
      f16Prefix: "(half)",
      f32Suffix: "f",
      nilToken: "0",
      nonFinite: FloatLitTokens(
        inf16: "(half)INFINITY", negInf16: "(half)-INFINITY", nan16: "(half)NAN",
        inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
        inf64: "((double)INFINITY)", negInf64: "((double)-INFINITY)", nan64: "((double)NAN)"))
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "#pragma unroll",
        infiniteHead: "for (;;) {\n",
        braceSameLine: true))
    conv: ConvRow()
    # OpenCL C names vectors by appending the lane count to the scalar name
    # (`float4`, `uchar3`), legal lengths 2, 3, 4, 8, 16.
    vector: VectorRule(
      context: vcEverywhere,
      elems: @[
        (ctVoid, 0, ""), (ctBool, 0, ""),
        (ctInt, 8, ""), (ctInt, 16, ""), (ctInt, 32, ""), (ctInt, 64, ""),
        (ctUint, 8, ""), (ctUint, 16, ""), (ctUint, 32, ""), (ctUint, 64, ""),
        (ctFloat, 16, ""), (ctFloat, 32, ""), (ctFloat, 64, "")],
      lanes: @[2, 3, 4, 8, 16])

  target glsl:
    # 8, 16 and 64-bit names compile behind the explicit-arithmetic-types extensions the module emits.
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      ScalarCell(kind: ctInt, width: 8, spelling: "int8_t", supported: true),
      ScalarCell(kind: ctInt, width: 16, spelling: "int16_t", supported: true),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "int64_t", supported: true),
      ScalarCell(kind: ctUint, width: 8, spelling: "uint8_t", supported: true),
      ScalarCell(kind: ctUint, width: 16, spelling: "uint16_t", supported: true),
      ScalarCell(kind: ctUint, width: 32, spelling: "uint", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "uint64_t", supported: true),
      ScalarCell(kind: ctFloat, width: 16, spelling: "float16_t", supported: true),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "double", supported: true)])
    # Wide integer and float literals print as constructor calls, glslang rejects the `LL` suffix.
    # GLSL has no infinity or NaN literal, IEEE-754 bit patterns go through `uintBitsToFloat`.
    lit: LitRow(
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
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "[[unroll]]",
        infiniteHead: "for (;;) {\n",
        braceSameLine: true),
      targetExprKinds: {eNil, eAt, ePat, eDeref, eAddr})
    conv: ConvRow()
    vector: VectorRule(
      context: vcEverywhere,
      elems: @[
        (ctFloat, 16, "f16vec"), (ctFloat, 32, "vec"), (ctFloat, 64, "dvec"),
        (ctInt, 32, "ivec"), (ctUint, 32, "uvec"), (ctBool, 0, "bvec")],
      lanes: @[2, 3, 4])


  target wgsl:
    # WGSL has no 8/16-bit integer types: 32 and 64 bit are its only integer widths.
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      ScalarCell(kind: ctInt, width: 8, supported: false),
      ScalarCell(kind: ctInt, width: 16, supported: false),
      ScalarCell(kind: ctInt, width: 32, spelling: "i32", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "i64", supported: true),
      ScalarCell(kind: ctUint, width: 8, supported: false),
      ScalarCell(kind: ctUint, width: 16, supported: false),
      ScalarCell(kind: ctUint, width: 32, spelling: "u32", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "u64", supported: true),
      ScalarCell(kind: ctFloat, width: 16, spelling: "f16", supported: true),
      ScalarCell(kind: ctFloat, width: 32, spelling: "f32", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "f64", supported: true)])
    # Bare integer and float literals are abstract and work in any typed context,
    # u32 gains the `u` suffix and f16 the `h`, f32 and f64 print bare.
    # WGSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
    # `bitcast` from the unsigned integer of the same width. Nil literals refuse
    # at the site.
    lit: LitRow(
      int: IntLitTokens(u32Suffix: "u"),
      f16Suffix: "h",
      nonFinite: FloatLitTokens(
        inf16: "bitcast<f16>(0x7c00u)", negInf16: "bitcast<f16>(0xfc00u)", nan16: "bitcast<f16>(0x7e00u)",
        inf32: "bitcast<f32>(0x7f800000u)",
        negInf32: "bitcast<f32>(0xff800000u)",
        nan32: "bitcast<f32>(0x7fc00000u)",
        inf64: "bitcast<f64>(0x7ff0000000000000u)",
        negInf64: "bitcast<f64>(0xfff0000000000000u)",
        nan64: "bitcast<f64>(0x7ff8000000000000u)"))
    kind: KindRow(
      loop: LoopStyle(
        # WGSL has no unroll directive: the drop surfaces in the artifact
        unrollDirective: "// {.unroll.} requested but not honored on WebGPU",
        infiniteHead: "loop {\n",
        braceSameLine: true),
      targetExprKinds: {eNil, eAt, ePat, eTernary})
    conv: ConvRow()
    vector: VectorRule(
      context: vcEverywhere,
      elems: @[
        (ctVoid, 0, "vec"), (ctBool, 0, "vec"),
        (ctInt, 32, "vec"), (ctInt, 64, "vec"),
        (ctUint, 32, "vec"), (ctUint, 64, "vec"),
        (ctFloat, 16, "vec"), (ctFloat, 32, "vec"), (ctFloat, 64, "vec")],
      lanes: @[2, 3, 4],
      nameForm: vnfGenericElem)

  target hlsl:
    # 8/16-bit integers and int64_t are parse errors in the HLSL frontend,
    # uint64_t pulls the SPIR-V Int64 capability, float16 is outside the subset.
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      ScalarCell(kind: ctInt, width: 8, supported: false),
      ScalarCell(kind: ctInt, width: 16, supported: false),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, supported: false),
      ScalarCell(kind: ctUint, width: 8, supported: false),
      ScalarCell(kind: ctUint, width: 16, supported: false),
      ScalarCell(kind: ctUint, width: 32, spelling: "uint", supported: true),
      ScalarCell(kind: ctUint, width: 64, supported: false),
      ScalarCell(kind: ctFloat, width: 16, supported: false),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "double", supported: true)])
    # HLSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
    # `asfloat` (32 bit) and `asdouble` (64 bit). Without the suffix the HLSL
    # frontend changes width or signedness. Nil literals refuse at the site.
    lit: LitRow(
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
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "[unroll]",
        infiniteHead: "for (;;)\n",
        braceSameLine: false),
      foldsIteElse: true,
      emitOperandRendersExpr: true,
      targetExprKinds: {eNil, eAt, ePat, eDeref, eAddr})
    # `{.builtin.}` fp16/bf16/fp8 conversion calls refuse: the frontend erases
    # half-width types to 16-bit storage before printing, so the call name is the only signal the printer can match.
    conv: ConvRow(
      fpRefusals: @[
        "toFp16", "toFp32", "toBf16", "toBf32", "toFp8e4m3", "toFp8e5m2",
        "asFp16", "asU16"])
    vector: VectorRule(
      context: vcEverywhere,
      elems: @[
        (ctBool, 0, "bool"), (ctInt, 32, "int"), (ctUint, 32, "uint"),
        (ctFloat, 32, "float")],
      lanes: @[2, 3, 4])
