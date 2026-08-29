# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/tables
import ../ir/gpu_types
import ./builtins_pragmas

## Builtin GPU scalar types: distinct integer storage tagged as float formats.
## Resolver maps them by name plus the `{.builtin.}` pragma.

type
  float16* {.builtin.} = distinct uint16
    ## IEEE 754 binary16: 1 sign, 5 exponent, 10 mantissa bits.
  bfloat16* {.builtin.} = distinct uint16
    ## bfloat16: the top 16 bits of an f32, 1 sign, 8 exponent, 7 mantissa.

# Arithmetic operators: the DSL lowers them to the target's native ops.
# Compile-time guards (error pragma, static assert) fire at DSL resolution
# too — the typed macro must resolve these calls. Bodies raise instead:
# codegen never reaches them, host use fails with a clear error.

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
#  Convention: `to` converts numerically, `as` reinterprets the bit pattern.
#  The narrowing rounds to nearest even (RNE). fp16 tiles store their bits as u16.
#  Declared `{.builtin.}` so crucible forwards the call name-only
#  to the per-backend spelling (the table below).
#  Host-side calls raise as the conversions are GPU-side only.

func toFp16*(x: float32): float16 {.builtin.} =
  ## Numeric fp32 → fp16 conversion (RNE).
  raise newException(ValueError, "fp16 conversion is GPU-side only")

func toBf16*(x: float32): bfloat16 {.builtin.} =
  ## Numeric fp32 → bf16 conversion (RNE).
  raise newException(ValueError, "bf16 conversion is GPU-side only")

func toFp32*(x: float16): float32 {.builtin.} =
  ## Numeric fp16 → fp32 conversion (exact widening).
  raise newException(ValueError, "fp16 conversion is GPU-side only")

func toFp32*(x: bfloat16): float32 {.builtin.} =
  ## Numeric bf16 → fp32 conversion (exact widening).
  raise newException(ValueError, "bf16 conversion is GPU-side only")

func asFp16*(x: uint16): float16 {.builtin.} =
  ## Bit-pattern reinterpret u16 → fp16.
  raise newException(ValueError, "fp16 reinterpret is GPU-side only")

func asU16*(x: float16): uint16 {.builtin.} =
  ## Bit-pattern reinterpret fp16 → u16.
  raise newException(ValueError, "fp16 reinterpret is GPU-side only")

let NimGpuFp16ConversionBuiltins* {.compileTime.}: Table[(BackendKind, string), string] =
  # The per-backend spellings of the conversion builtins above, consulted
  # by lang_utils.getFnName. The value is emitted as `name(args)`, so MSL's
  # `as_type<half>` (a template call) is reachable. The CUDA and WGSL
  # type printers lack the half (CUDA) and u16 (WGSL) scalar types, so
  # those spellings cannot be emitted until the types land.
  {(bkMetal,  "toFp16"): "half",
   (bkMetal,  "toFp32"): "float",
   (bkMetal,  "toBf16"): "bfloat",
   (bkMetal,  "asFp16"): "as_type<half>",
   (bkMetal,  "asU16"):  "as_type<ushort>",
   (bkVulkan, "toFp16"): "float16_t",
   (bkVulkan, "toFp32"): "float",
   (bkVulkan, "asFp16"): "uint16BitsToFloat16",
   (bkVulkan, "toBf16"): "bfloat16_t",
   (bkVulkan, "asU16"):  "float16BitsToUint16",
   (bkCuda,   "toFp16"): "__float2half_rn",
   (bkCuda,   "toFp32"): "__half2float",
   (bkCuda,   "asFp16"): "__ushort_as_half",
   (bkCuda,   "asU16"):  "__half_as_ushort",
   (bkCuda,   "toBf16"): "__float2bfloat16",
   (bkWGSL,   "toFp16"): "f16",
   (bkWGSL,   "toFp32"): "f32",
   (bkWGSL,   "asFp16"): "bitcast<f16>"}.toTable()

proc builtinGpuTypeKind*(n: NimNode): GpuTypeKind =
  ## Maps a builtin GPU type symbol to its GpuTypeKind, or gtVoid otherwise.
  ## Recognized by name plus the `{.builtin.}` pragma.
  ## Consulted by the type resolver before the typeKind switch.
  ## Distinct types have no nty* GPU mapping of their own.
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
  else:           gtVoid
