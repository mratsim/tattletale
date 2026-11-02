# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Data of the printers' static target registry.
##
## Each const object is plain data: strings, bools, ints, enums, sets and tables of those.
## No field stores a proc or a placeholder whose value depends on a body walk.
##
## Schema, macro and vocabulary types live in the builtins layer
## (`builtins_decl_helpers.nim`).
## The per-backend data lives in `builtins_gpu_types.nim` keyed by BackendKind.
##
## The six printer targets (Metal MSL, CUDA C++, OpenCL C, GLSL, WebGPU WGSL, HLSL):
##
## - each declares its kind, math and vector sections in the `genTargets:` block below
## - the macro validates every section during compilation and generates
##   one lookup const per family
## - a section that misses the schema fails the compile at the section definition
## - the printers read the generated lookups, signature text, coordinate
##   builtins and refusal messages stay procedural in the printers

import std/tables
import ../ir/cirv2_lang
import ../../builtins/builtins_decl_helpers

genTargets:
  target msl:
    # `char` and `long` are MSL's 8-bit signed and 64-bit type names.
    # CUDA names the same widths `signed char` and `long long`, each target declares its own names
    kind unrollDirective "#pragma clang loop unroll(full)", braceSameLine true
    # `{.builtin.}` numeric conversions resolve to the MSL native call text,
    # each entry a format string whose `$1` stands for the call's argument list.
    # `float16` has a first-class `half` identity: its conversions are plain native calls.
    # `asFp16`/`asU16` reinterpret through `as_type`.
    # `bfloat16` has a first-class `bfloat` identity (Metal 3.1+), its conversions are plain casts.
    # fp8 conversions have no MSL native, an unspelled `{.builtin.}` name emits unchanged
    # and Metal's compiler rejects it
    vector:
      elems i32, u32
      lanes 2, 3, 4

  target cuda:
    kind unrollDirective "#pragma unroll", braceSameLine true
    # CUDA has no vector types, the printer refuses a vector at its declaration.
    vector ()
    # `{.builtin.}` scalar-math calls keyed by (canonical name, operand bit width):
    #
    # - a float operand maps to the CUDA name: CUDA's device `abs` is
    #   integer-only, an unsuffixed `abs(-2.5f)` silently computes `-2`
    # - the printer keeps the canonical name for a width absent from the table,
    #   `exp2` over `double` is a CUDA native
    math abs 32 "fabsf", abs 64 "fabs", exp2 32 "exp2f"

  target opencl:
    # OpenCL C is LP64 C: `char` and `long` name the 8-bit and 64-bit signed widths, as on msl.
    # This row prints bool as `int` to match the launch ABI.
    kind unrollDirective "#pragma unroll", braceSameLine true
    # OpenCL C names vectors by appending the lane count to the scalar name
    # (`float4`, `uchar3`), legal lengths 2, 3, 4, 8, 16.
    vector:
      elems void, bool, i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
      lanes 2, 3, 4, 8, 16
    # `{.builtin.}` scalar-math calls map to the f-prefixed OpenCL natives
    # for float operands, v1's NimGpuNumericBuiltinsFnNames data.
    math max 32 "fmax", max 64 "fmax", min 32 "fmin", min 64 "fmin",
      abs 32 "fabs", abs 64 "fabs"
    # fp16 and bf16 values store their 16-bit bit pattern in a `ushort`,
    # OpenCL C has no native half storage outside the fp16 extension.
    # `{.builtin.}` conversion names emit unchanged, the caller's module
    # defines their bodies.

  target glsl:
    # 8, 16 and 64-bit names compile behind the explicit-arithmetic-types extensions the module emits.
    # Wide integer and float literals print as constructor calls, glslang rejects the `LL` suffix.
    # GLSL has no infinity or NaN literal, IEEE-754 bit patterns go through `uintBitsToFloat`.
    # `{.builtin.}` fp16 conversions resolve to the GLSL native call text,
    # `float16` has a first-class `float16_t` identity and `asFp16`/`asU16`
    # reinterpret through the float16 bit-level builtins.
    # `bfloat16` and fp8 conversions are outside the table: no GLSL native exists.
    # `{.builtin.}` names outside the table emit unchanged and fail at glslang compile time.
    kind unrollDirective "[[unroll]]", braceSameLine true,
      targetExprKinds {eNil, eAt, ePat, eDeref, eAddr}
    vector:
      elems f16 "f16vec", f32 "vec", f64 "dvec", i32 "ivec", u32 "uvec", bool "bvec"
      lanes 2, 3, 4

  target wgsl:
    # rsqrt maps to the WGSL native, the scalar-math dispatch reads the width
    math rsqrt 32 "inverseSqrt"
    # emitWgslScalar refuses an absent cell, WGSL has no 8/16-bit integer types.
    # 64-bit integer and float cells need the shader-int64 and shader-f64 device features.
    # wgpu-native exposes no WGPUFeatureName to request either.
    # Integer and float literals without a suffix are abstract and work in any typed context.
    # u32 needs the `u` suffix, f16 the `h` suffix, f32 unsuffixed.
    # WGSL has no infinity or NaN literal: the 16 and 32-bit IEEE-754 bit patterns
    # go through `bitcast` from the unsigned integer of the same width.
    # Nil literals: the shared walk refuses them.
    kind unrollDirective "// {.unroll.} requested but not honored on WebGPU", braceSameLine true,
      targetExprKinds {eNil, eAt, ePat, eTernary, eFloatLit}
    vector:
      elems void "vec", bool "vec", i32 "vec", u32 "vec", f16 "vec", f32 "vec"
      lanes 2, 3, 4
      nameForm vnfGenericElem
    # `{.builtin.}` fp16 conversions resolve to the WGSL native call text,
    # `float16` has a first-class `f16` identity and `asFp16`/`asU16`
    # reinterpret through `bitcast`.
    # `bfloat16` and fp8 conversions are outside the table: no WGSL native exists.
    # `{.builtin.}` names outside the table emit unchanged and fail at wgpu parse time.

  target hlsl:
    # emitHlslScalar refuses an absent cell, the 8/16-bit and 64-bit integer
    # types are frontend parse errors or capability rejections. The float16 cell is
    # outside the subset.
    # HLSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
    # `asfloat` (32 bit) and `asdouble` (64 bit). Without the suffix the HLSL
    # frontend changes width or signedness. Nil literals: the shared walk refuses them.
    kind unrollDirective "[unroll]", braceSameLine false, foldsIteElse true,
      emitOperandRendersExpr true, targetExprKinds {eNil, eAt, ePat, eDeref, eAddr}
    # printBuiltinCall refuses the fp16/bf16/fp8 conversion builtins (the frontend erases half-width types, the call name is the only signal).
    vector:
      elems bool "bool", i32 "int", u32 "uint", f32 "float"
      lanes 2, 3, 4
