# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
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

# ═══════════════════════════════════════════════════════════════════════
#  Low-precision float conversions: host fallback bodies and dispatch
# ═══════════════════════════════════════════════════════════════════════
#
# builtins_type_conversions_fallback provides the host fallback bodies,
# re-exported here. Device calls lower to the conv tables' native conversions.

import ./builtins_type_conversions_fallback
export builtins_type_conversions_fallback

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
