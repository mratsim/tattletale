# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/tables
import ../codegen/ir/ast_datatypes
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
#  The fp16 ↔ fp32 conversion primitives
# ═══════════════════════════════════════════════════════════════════════
func toFp16*(x: float32): float16 {.builtin.} =
  ## - Returns: the fp16 value nearest to `x`, round-to-nearest-even
  ## - The {.builtin.} pragma lowers the device call to the backend
  ##   native conversion and the body runs on the host
  let u = cast[uint32](x)
  let sign = (uint16(u shr 16) and 0x8000'u16)
  let exp32 = int32((u shr 23) and 0xFF'u32)
  let man32 = u and 0x007FFFFF'u32
  if exp32 == 255:
    if man32 != 0:
      return float16(sign or 0x7E00'u16)
    return float16(sign or 0x7C00'u16)
  var e = exp32 - 112
  var m = man32
  if e >= 31:
    return float16(sign or 0x7C00'u16)
  if e <= 0:
    if e < -24:
      return float16(sign)
    m = m or 0x00800000'u32
    let shift = uint32(14 - e)
    let roundBit = (m shr (shift - 1)) and 1
    let sticky = (m and ((1'u32 shl (shift - 1)) - 1)) != 0
    var man16 = m shr shift
    if roundBit == 1 and (sticky or (man16 and 1) == 1):
      man16 += 1
    return float16(sign or uint16(man16))
  var man16 = m shr 13
  let roundBit = (m shr 12) and 1
  let sticky = (m and ((1'u32 shl 12) - 1)) != 0
  if roundBit == 1 and (sticky or (man16 and 1) == 1):
    man16 += 1
  if man16 == 0x0400:
    man16 = 0
    e += 1
    if e >= 31:
      return float16(sign or 0x7C00'u16)
  return float16(sign or (uint16(e) shl 10) or uint16(man16))

func toFp32*(x: float16): float32 {.builtin.} =
  ## - Returns: the fp32 value of `x`, exact widening
  ## - The {.builtin.} pragma lowers the device call to the backend
  ##   native conversion and the body runs on the host
  let h = cast[uint16](x)
  let sign = uint32(h and 0x8000'u16) shl 16
  let exp16 = int32((h shr 10) and 0x1F'u16)
  let man16 = uint32(h and 0x03FF'u16)
  if exp16 == 31:
    let man = if man16 != 0: 0x00400000'u32 else: 0'u32
    return cast[float32](sign or 0x7F800000'u32 or man)
  if exp16 == 0:
    if man16 == 0:
      return cast[float32](sign)
    var k = 9
    while (man16 and (1'u16 shl k)) == 0:
      k -= 1
    let man = (man16 shl (23 - k)) and 0x007FFFFF'u32
    return cast[float32](sign or (uint32(103 + k) shl 23) or man)
  return cast[float32](sign or (uint32(exp16 + 112) shl 23) or (man16 shl 13))

func toBf16*(x: float32): bfloat16 {.builtin.} =
  ## - Returns: the bf16 value nearest to `x`, round-to-nearest-even
  ## - The {.builtin.} pragma lowers the device call to the backend
  ##   native conversion and the body runs on the host
  let u = cast[uint32](x)
  if (u and 0x7F800000'u32) == 0x7F800000'u32 and (u and 0x007FFFFF'u32) != 0:
    return bfloat16(uint16(u shr 16) or 0x0040'u16)
  var hi = uint16(u shr 16)
  if ((u shr 15) and 1'u32) == 1 and
      ((u and 0x00007FFF'u32) != 0 or (hi and 1'u16) == 1):
    hi = uint16(hi + 1)
  bfloat16(hi)

func toBf32*(x: bfloat16): float32 {.builtin.} =
  ## - Returns: the fp32 value of `x`, exact widening
  ## - The {.builtin.} pragma lowers the device call to the backend
  ##   native conversion and the body runs on the host
  cast[float32](uint32(cast[uint16](x)) shl 16)

func toFp8e4m3*(x: float32): fp8e4m3 {.builtin.} =
  ## - Returns: the fp8e4m3 (OCP OFP8 E4M3fn) value nearest to `x`,
  ##   round-to-nearest-even, overflow saturates to the max finite
  ## - Backends with a native fp8 type lower the device call there,
  ##   elsewhere the body is the conversion
  let u = cast[uint32](x)
  let sign = uint8((u shr 24) and 0x80'u32)
  let exp32 = int32((u shr 23) and 0xFF'u32)
  let man32 = u and 0x007FFFFF'u32
  if exp32 == 255:
    if man32 != 0:
      return fp8e4m3(sign or 0x7F'u8)
    return fp8e4m3(sign or 0x7E'u8)
  var e = exp32 - 120
  var m = man32
  if e >= 16:
    return fp8e4m3(sign or 0x7E'u8)
  if e <= 0:
    if e < -3:
      return fp8e4m3(sign)
    m = m or 0x00800000'u32
    let shift = uint32(21 - e)
    let roundBit = (m shr (shift - 1)) and 1
    let sticky = (m and ((1'u32 shl (shift - 1)) - 1)) != 0
    var man8 = m shr shift
    if roundBit == 1 and (sticky or (man8 and 1) == 1):
      man8 += 1
    return fp8e4m3(sign or uint8(man8))
  var man8 = m shr 20
  let roundBit = (m shr 19) and 1
  let sticky = (m and 0x0007FFFF'u32) != 0
  if roundBit == 1 and (sticky or (man8 and 1) == 1):
    man8 += 1
  if man8 == 8:
    man8 = 0
    e += 1
  if e >= 16 or (e == 15 and man8 == 7):
    return fp8e4m3(sign or 0x7E'u8)
  return fp8e4m3(sign or (uint8(e) shl 3) or uint8(man8))

func toFp8e5m2*(x: float32): fp8e5m2 {.builtin.} =
  ## - Returns: the fp8e5m2 (OCP OFP8 E5M2) value nearest to `x`,
  ##   round-to-nearest-even
  ## - Backends with a native fp8 type lower the device call there,
  ##   elsewhere the body is the conversion
  let u = cast[uint32](x)
  let sign = uint8((u shr 24) and 0x80'u32)
  let exp32 = int32((u shr 23) and 0xFF'u32)
  let man32 = u and 0x007FFFFF'u32
  if exp32 == 255:
    if man32 != 0:
      return fp8e5m2(sign or 0x7D'u8)
    return fp8e5m2(sign or 0x7C'u8)
  var e = exp32 - 112
  var m = man32
  if e >= 31:
    return fp8e5m2(sign or 0x7C'u8)
  if e <= 0:
    if e < -9:
      return fp8e5m2(sign)
    m = m or 0x00800000'u32
    let shift = uint32(22 - e)
    let roundBit = (m shr (shift - 1)) and 1
    let sticky = (m and ((1'u32 shl (shift - 1)) - 1)) != 0
    var man8 = m shr shift
    if roundBit == 1 and (sticky or (man8 and 1) == 1):
      man8 += 1
    return fp8e5m2(sign or uint8(man8))
  var man8 = m shr 21
  let roundBit = (m shr 20) and 1
  let sticky = (m and 0x001FFFFF'u32) != 0
  if roundBit == 1 and (sticky or (man8 and 1) == 1):
    man8 += 1
  if man8 == 4:
    man8 = 0
    e += 1
  if e >= 31:
    return fp8e5m2(sign or 0x7C'u8)
  return fp8e5m2(sign or (uint8(e) shl 2) or uint8(man8))

func toFp32*(x: fp8e4m3): float32 {.builtin.} =
  ## - Returns: the fp32 value of `x`, exact widening
  ## - Backends with a native fp8 type lower the device call there,
  ##   elsewhere the body is the conversion
  let b = cast[uint8](x)
  let sign = uint32(b and 0x80'u8) shl 24
  let exp8 = int32((b shr 3) and 0xF'u8)
  let man8 = uint32(b and 0x7'u8)
  if exp8 == 15 and man8 == 7:
    return cast[float32](sign or 0x7FC00000'u32)
  if exp8 == 0:
    if man8 == 0:
      return cast[float32](sign)
    var i = 2
    while (man8 and (1'u32 shl i)) == 0:
      i -= 1
    let man = (man8 shl (23 - i)) and 0x007FFFFF'u32
    return cast[float32](sign or (uint32(118 + i) shl 23) or man)
  return cast[float32](sign or (uint32(exp8 + 120) shl 23) or (man8 shl 20))

func toFp32*(x: fp8e5m2): float32 {.builtin.} =
  ## - Returns: the fp32 value of `x`, exact widening
  ## - Backends with a native fp8 type lower the device call there,
  ##   elsewhere the body is the conversion
  let b = cast[uint8](x)
  let sign = uint32(b and 0x80'u8) shl 24
  let exp8 = int32((b shr 2) and 0x1F'u8)
  let man8 = uint32(b and 0x3'u8)
  if exp8 == 31:
    let man = if man8 != 0: 0x00400000'u32 else: 0'u32
    return cast[float32](sign or 0x7F800000'u32 or man)
  if exp8 == 0:
    if man8 == 0:
      return cast[float32](sign)
    var i = 1
    while (man8 and (1'u32 shl i)) == 0:
      i -= 1
    let man = (man8 shl (23 - i)) and 0x007FFFFF'u32
    return cast[float32](sign or (uint32(111 + i) shl 23) or man)
  return cast[float32](sign or (uint32(exp8 + 112) shl 23) or (man8 shl 21))

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

const ConvCellTokens* = ["f16", "bf16", "f32", "f64", "i16", "i32", "i64", "u16", "u32", "u64"]

macro convBuiltin*(body: untyped) =
  ## Declarative conversion mapping: one `bkX:` section per backend,
  ## one arrow row per mapping, `f32->f16: "half"`.
  ##
  ## - the sections key on the BackendKind members
  ## - the rows key on the (source, target) cell tokens shared with
  ##   the codegen ScalarCellAxis vocabulary
  ## - a row's value names the native call, the emission appends
  ##   the argument list in parentheses
  ## - an absent backend section or cell pair lets the canonical name
  ##   pass through
  var pairs: seq[NimNode]
  for section in body:
    if section.kind != nnkCall or section[0].kind != nnkIdent or
        section[1].kind != nnkStmtList:
      error("convBuiltin: expected a 'bkX:' section", section)
    let backend = $section[0]
    var known = false
    for member in BackendKind:
      if $member == backend:
        known = true
        break
    if not known:
      error("convBuiltin: unknown backend '" & backend & "'", section)
    for row in section[1]:
      if row.kind != nnkInfix or $row[0] != "->" or
          row[1].kind != nnkIdent or row[2].kind != nnkIdent or
          row[3].kind != nnkStmtList or row[3].len != 1 or row[3][0].kind != nnkStrLit:
        error("convBuiltin: expected a 'fromCell->toCell: \"native\"' row in the " & backend & " section", row)
      for cellTok in [row[1], row[2]]:
        if $cellTok notin ConvCellTokens:
          error("convBuiltin: unknown cell '" & $cellTok & "' in the " & backend & " section", row)
      let key = nnkPar.newTree(ident backend, newLit $row[1], newLit $row[2])
      pairs.add nnkExprColonExpr.newTree(key, row[3][0])
  result = nnkConstSection.newTree(nnkConstDef.newTree(
    nnkPostfix.newTree(ident"*", ident"ConvBuiltinNames"),
    newEmptyNode(),
    nnkCall.newTree(nnkDotExpr.newTree(
      nnkTableConstr.newTree(pairs), ident"toTable"))))

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
