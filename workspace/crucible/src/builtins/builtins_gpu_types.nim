# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ../runtime/errors
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
    gpuBuiltinHostStub("float16 arithmetic is GPU-side only")
  func op*(a, b: bfloat16): bfloat16 {.inline.} =
    gpuBuiltinHostStub("bfloat16 arithmetic is GPU-side only")

fp16Op(`+`)
fp16Op(`-`)
fp16Op(`*`)
fp16Op(`/`)

func `+`*(a: float16): float16 {.inline.} =
  gpuBuiltinHostStub("float16 arithmetic is GPU-side only")
func `-`*(a: float16): float16 {.inline.} =
  gpuBuiltinHostStub("float16 arithmetic is GPU-side only")
func `+`*(a: bfloat16): bfloat16 {.inline.} =
  gpuBuiltinHostStub("bfloat16 arithmetic is GPU-side only")
func `-`*(a: bfloat16): bfloat16 {.inline.} =
  gpuBuiltinHostStub("bfloat16 arithmetic is GPU-side only")

# ═══════════════════════════════════════════════════════════════════════
#  The fp16 ↔ fp32 conversion primitives
# ═══════════════════════════════════════════════════════════════════════

func toFp16*(x: float32): float16 {.builtin.} =
  ## Numeric fp32 → fp16 conversion (RNE).
  gpuBuiltinHostStub("fp16 conversion is GPU-side only")


func toFp32*(x: float16): float32 {.builtin.} =
  ## Numeric fp16 → fp32 conversion (exact widening).
  gpuBuiltinHostStub("fp16 conversion is GPU-side only")

func asFp16*(x: uint16): float16 {.builtin.} =
  ## Bit-pattern reinterpret u16 → fp16.
  gpuBuiltinHostStub("fp16 reinterpret is GPU-side only")

func asU16*(x: float16): uint16 {.builtin.} =
  ## Bit-pattern reinterpret fp16 → u16.
  gpuBuiltinHostStub("fp16 reinterpret is GPU-side only")
