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
