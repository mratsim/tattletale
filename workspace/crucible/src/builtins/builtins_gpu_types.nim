# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, tables]
import builtins_decl_helpers
import ../codegen/ir/ast_datatypes
import ../codegen/ir/cirv2_lang
import ./builtins_pragmas

type
  float16* {.builtin.} = distinct uint16
    ## IEEE 754 binary16: 1 sign, 5 exponent, 10 mantissa bits.
  bfloat16* {.builtin.} = distinct uint16
    ## bfloat16: the top 16 bits of an f32, 1 sign, 8 exponent, 7 mantissa.
  fp8e4m3* {.builtin.} = distinct uint8
    ## OCP FP8 E4M3fn, finite, no infinity, 1 sign, 4 exponent, 3 mantissa.
  fp8e5m2* {.builtin.} = distinct uint8
    ## OCP FP8 E5M2, IEEE-like, 1 sign, 5 exponent, 2 mantissa.

template fp16Op(op: untyped): untyped =
  func op*(a, b: float16): float16 {.inline.} =
    raise newException(ValueError, "float16 arithmetic is GPU-side only")
  func op*(a, b: bfloat16): bfloat16 {.inline.} =
    raise newException(ValueError, "bfloat16 arithmetic is GPU-side only")

fp16Op(`+`)
fp16Op(`-`)
fp16Op(`*`)
fp16Op(`/`)

func `+`*(a: float16): float16 {.inline.} =
  raise newException(ValueError, "float16 arithmetic is GPU-side only")
func `-`*(a: float16): float16 {.inline.} =
  raise newException(ValueError, "float16 arithmetic is GPU-side only")
func `+`*(a: bfloat16): bfloat16 {.inline.} =
  raise newException(ValueError, "bfloat16 arithmetic is GPU-side only")
func `-`*(a: bfloat16): bfloat16 {.inline.} =
  raise newException(ValueError, "bfloat16 arithmetic is GPU-side only")

import ./builtins_type_conversions_fallback

# High-level element conversion, thin dispatch over the primitives above.
template to*(src: float32, T: typedesc[float16]): float16 = toFp16(src)
template to*(src: float32, T: typedesc[bfloat16]): bfloat16 = toBf16(src)
template to*(src: float32, T: typedesc[fp8e4m3]): fp8e4m3 = toFp8e4m3(src)
template to*(src: float32, T: typedesc[fp8e5m2]): fp8e5m2 = toFp8e5m2(src)
template to*(src: float32, T: typedesc[float32]): float32 = src
template to*(src: float32, T: typedesc[float64]): float64 = float64(src)

template to*(src: int or int32 or int64, T: typedesc[float16]): float16 = toFp16(float32(src))
template to*(src: int or int32 or int64, T: typedesc[bfloat16]): bfloat16 = toBf16(float32(src))
template to*(src: int or int32 or int64, T: typedesc[fp8e4m3]): fp8e4m3 = toFp8e4m3(float32(src))
template to*(src: int or int32 or int64, T: typedesc[fp8e5m2]): fp8e5m2 = toFp8e5m2(float32(src))
template to*(src: int or int32 or int64, T: typedesc[float32]): float32 = float32(src)
template to*(src: int or int32 or int64, T: typedesc[float64]): float64 = float64(src)

template to*(src: float16, T: typedesc[float32]): float32 = toFp32(src)
template to*(src: bfloat16, T: typedesc[float32]): float32 = toBf32(src)
template to*(src: fp8e4m3, T: typedesc[float32]): float32 = toFp32(src)
template to*(src: fp8e5m2, T: typedesc[float32]): float32 = toFp32(src)

template to*(src: float16, T: typedesc[float64]): float64 = float64(toFp32(src))
template to*(src: bfloat16, T: typedesc[float64]): float64 = float64(toBf32(src))
template to*(src: fp8e4m3, T: typedesc[float64]): float64 = float64(toFp32(src))
template to*(src: fp8e5m2, T: typedesc[float64]): float64 = float64(toFp32(src))

func asFp16*(x: uint16): float16 {.builtin.} =
  ## Bit-pattern reinterpret u16 → fp16.
  raise newException(ValueError, "fp16 reinterpret is GPU-side only")

func asU16*(x: float16): uint16 {.builtin.} =
  ## Bit-pattern reinterpret fp16 → u16.
  raise newException(ValueError, "fp16 reinterpret is GPU-side only")
proc builtinGpuTypeKind*(n: NimNode): GpuTypeKind =
  if n.kind != nnkSym:
    return gtVoid
  let impl = n.getImpl()
  if impl.kind != nnkTypeDef or impl[0].kind != nnkPragmaExpr:
    return gtVoid
  let pragmaNode = impl[0][1]
  if pragmaNode.kind != nnkPragma:
    return gtVoid
  var hasTag = false
  for p in pragmaNode:
    if p.kind in {nnkSym, nnkIdent} and p.strVal == "builtin":
      hasTag = true
  if not hasTag:
    return gtVoid
  case n.strVal
  of "float16":   gtFloat16
  of "bfloat16":  gtBf16
  of "fp8e4m3":   gtF8E4M3
  of "fp8e5m2":   gtF8E5M2
  else:           gtVoid

# ═══════════════════════════════════════════════════════════════════════
#  The conversion mapping: builtins drive, the backends read
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  The conversion mapping data
# ═══════════════════════════════════════════════════════════════════════

convBuiltin:
  bkMetal:
    f32->f16: "half"
    f16->f32: "float"
    f32->bf16: "(bfloat)"
    bf16->f32: "float"
    u16->f16: "as_type<half>"
    f16->u16: "as_type<ushort>"
  bkCuda:
    f32->f16: "__float2half_rn"
    f16->f32: "__half2float"
    u16->f16: "__ushort_as_half"
    f16->u16: "__half_as_ushort"
    f32->bf16: "__float2bfloat16"
    bf16->f32: "__bfloat162float"
  bkGlsl:
    f32->f16: "float16_t"
    f16->f32: "float"
    u16->f16: "uint16BitsToFloat16"
    f16->u16: "float16BitsToUint16"
  bkWebGPU:
    f32->f16: "f16"
    f16->f32: "f32"
    u16->f16: "bitcast<f16>"

targetVocab:
  bkMetal:
    scalar void "void", bool "bool", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", bf16 "bfloat", f32 "float",
      string "const char*"
    lit u64Suffix "ULL", u32Suffix "U", i64Suffix "LL", narrowCast true,
      f16Suffix "h", f32Suffix "f", nilToken "nullptr",
      inf16 "INFINITY", negInf16 "-INFINITY", nan16 "NAN",
      inf32 "INFINITY", negInf32 "-INFINITY", nan32 "NAN",
      inf64 "INFINITY", negInf64 "-INFINITY", nan64 "NAN"
    bitcast:
      any->any:
        "as_type<$2>($1)"
        sameWidth

  bkCuda:
    scalar void "void", bool "bool", i8 "signed char", i16 "short", i32 "int", i64 "long long",
      u8 "unsigned char", u16 "unsigned short", u32 "unsigned int", u64 "unsigned long long",
      f16 "__half", bf16 "__nv_bfloat16", f32 "float", f64 "double", string "const char*"
    lit u64Suffix "ULL", u32Suffix "U", i64Suffix "LL", narrowCast true,
      f32Suffix "f", nilToken "nullptr",
      inf16 "INFINITY", negInf16 "-INFINITY", nan16 "NAN",
      inf32 "INFINITY", negInf32 "-INFINITY", nan32 "NAN",
      inf64 "INFINITY", negInf64 "-INFINITY", nan64 "NAN"
    bitcast:
      f32->u32: "__float_as_uint($1)"
      f32->i32: "__float_as_int($1)"
      f64->i64: "__double_as_longlong($1)"
      f64->u64: "__double_as_longlong($1)"
      f16->i16: "__half_as_ushort($1)"
      f16->u16: "__half_as_ushort($1)"
      i32->f32: "__int_as_float($1)"
      u32->f32: "__uint_as_float($1)"
      i64->f64: "__longlong_as_double($1)"
      u64->f64: "__longlong_as_double($1)"
      i16->f16: "__ushort_as_half($1)"
      u16->f16: "__ushort_as_half($1)"
      i16->bf16: "__ushort_as_bfloat16($1)"
      u16->bf16: "__ushort_as_bfloat16($1)"
      bf16->i16: "__bfloat16_as_ushort($1)"
      bf16->u16: "__bfloat16_as_ushort($1)"
  bkOpenCL:
    scalar void "void", bool "int", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "ushort", bf16 "ushort",
      f32 "float", f64 "double", string "const char*"
    lit u64Suffix "ULL", u32Suffix "U", i64Suffix "LL", narrowCast true,
      f16Prefix "(half)", f32Suffix "f", nilToken "0",
      inf16 "(half)INFINITY", negInf16 "(half)-INFINITY", nan16 "(half)NAN",
      inf32 "INFINITY", negInf32 "-INFINITY", nan32 "NAN",
      inf64 "((double)INFINITY)", negInf64 "((double)-INFINITY)", nan64 "((double)NAN)"
    bitcast:
      f16->i16: "as_$2($1)"
      f16->u16: "as_$2($1)"
      f32->i32: "as_$2($1)"
      f32->u32: "as_$2($1)"
      f64->i64: "as_$2($1)"
      f64->u64: "as_$2($1)"
      i16->f16: "as_$2($1)"
      u16->f16: "as_$2($1)"
      i32->f32: "as_$2($1)"
      u32->f32: "as_$2($1)"
      i64->f64: "as_$2($1)"
      u64->f64: "as_$2($1)"

  bkGlsl:
    scalar void "void", bool "bool", i8 "int8_t", i16 "int16_t", i32 "int", i64 "int64_t",
      u8 "uint8_t", u16 "uint16_t", u32 "uint", u64 "uint64_t", f16 "float16_t", f32 "float", f64 "double",
      string "const char*"
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
    bitcast:
      f32->i32: "floatBitsToInt($1)"
      f32->u32: "floatBitsToUint($1)"
      i32->f32: "intBitsToFloat($1)"
      u32->f32: "uintBitsToFloat($1)"
      f16->i16: nil
      f16->u16: nil
      i16->f16: nil
      u16->f16: nil
      f64->i64: nil
      f64->u64: nil
      i64->f64: nil
      u64->f64: nil

  bkWebGPU:
    scalar void "void", bool "bool", i32 "i32", u32 "u32", f16 "f16", f32 "f32"
    lit u32Suffix "u", f16Suffix "h",
      inf16 "bitcast<f16>(0x7c00u)", negInf16 "bitcast<f16>(0xfc00u)", nan16 "bitcast<f16>(0x7e00u)",
      inf32 "bitcast<f32>(0x7f800000u)",
      negInf32 "bitcast<f32>(0xff800000u)",
      nan32 "bitcast<f32>(0x7fc00000u)"
    bitcast:
      all->all: "bitcast<$2>($1)"

  bkHlsl:
    scalar void "void", bool "bool", i32 "int", u32 "uint", f32 "float", f64 "double"
    lit u32Suffix "u", u64Suffix "u", f16Suffix "f", f32Suffix "f", f64Suffix "l",
      inf16 "asfloat(0x7f800000)", negInf16 "asfloat(0xff800000)", nan16 "asfloat(0x7fc00000)",
      inf32 "asfloat(0x7f800000)", negInf32 "asfloat(0xff800000)", nan32 "asfloat(0x7fc00000)",
      inf64 "asdouble(0u, 0x7ff00000u)",
      negInf64 "asdouble(0u, 0xfff00000u)",
      nan64 "asdouble(0u, 0x7ff80000u)"
    bitcast:
      all->i32: "asint($1)"
      all->u32: "asuint($1)"
      all->f32: "asfloat($1)"
      all->all: nil
