# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
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
