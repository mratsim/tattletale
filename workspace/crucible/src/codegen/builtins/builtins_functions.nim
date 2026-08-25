# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Built-in device functions (exp2/rsqrt) and the Nim-operator/function
## name tables the codegen consults.

import std/tables
import ../ir/gpu_types
import ./builtins_pragmas

# ═════════════════════════════════════════════════════════════════════════
#  The scalar math builtins: the element ops of the tile/col-vec maps
# ═════════════════════════════════════════════════════════════════════════
#
#  Declared here so they resolve before the tile_ops map templates (which
#  pass their own names as map ops): a same-named tile template declared
#  earlier would capture the reference.

proc exp2*(x: float32): float32 {.builtin.} = discard
  ## Returns `2^x`, declared `{.builtin.}` so the DSL forwards
  ## the backend's native `exp2` (used by the online-softmax maps,
  ## exact for integer exponents).

proc rsqrt*(x: float32): float32 {.builtin.} = discard
  ## Returns `1/sqrt(x)`: the DSL forwards the backend's native `rsqrt`.

# ═══════════════════════════════════════════════════════════════════════
#  Built-in Nim functions and operators that collide with or must be
#  mapped to GPU backend equivalents (compile-time tables, codegen-only)
# ═══════════════════════════════════════════════════════════════════════

let NimGpuNumericBuiltinsOperators* {.compileTime.} = {
  # Nim numeric operators and their GPU backend equivalents.
  # These have `{.magic.}` pragmas and may appear as nnkCall (not nnkInfix)
  # when operands have non-basic types (e.g. Int[V] structs). The codegen
  # must recognize them as builtins rather than attempting to register them
  # as external procs.
  "+": "+", "-": "-", "*": "*", "/": "/",
  "div": "/", "mod": "%",
  "shl": "<<", "shr": ">>", "ashr": ">>",
  "and": "&", "or": "|", "xor": "^",
  "not": "~"
}.toTable()

let NimGpuBooleanOperators* {.compileTime.} = {
  # Boolean operator mappings. Set apart from numeric operators because
  # `and`/`or`/`not` differ between boolean (&&/||/!) and bitwise (&/|/~).
  "and": "&&", "or": "||", "not": "!"
}.toTable()

let NimGpuFnBuiltins* {.compileTime.} = ["toOpenArray", "len"]
  # Function-style builtins with `{.magic.}` that are called as function
  # calls (not operators). When these reach registerGenericInstOrExternalProc
  # they must be registered as builtins without parsing their bodies.

let NimGpuNumericBuiltinsFunctions* {.compileTime.} = ["min", "max", "abs"]
  # Functions (not operators) that are {.magic.} in Nim
  # and have dedicated lowering in GPU backends.

let NimGpuNumericBuiltinsFnNames {.compileTime.}: Table[(BackendKind, string, GpuTypeKind), string] =
  # Per-backend per-type mapping for builtin functions.
  {(bkOpenCL, "max", gtFloat32): "fmax",
   (bkOpenCL, "max", gtFloat64): "fmax",
   (bkOpenCL, "min", gtFloat32): "fmin",
   (bkOpenCL, "min", gtFloat64): "fmin",
   (bkOpenCL, "abs", gtFloat32): "fabs",
   (bkOpenCL, "abs", gtFloat64): "fabs",
   (bkCuda,   "abs", gtFloat32): "fabsf",
   (bkCuda,   "abs", gtFloat64): "fabs"}.toTable()

proc getMaxMinAbsBuiltinFnName*(backend: BackendKind; builtin: string;
                             tKind: GpuTypeKind): string =
  ## Backend function name for an ambiguous builtin call whose operands have
  ## type `tKind`.
  ## Combos absent from NimGpuAmbiguousBuiltinFn keep the plain name.
  NimGpuNumericBuiltinsFnNames.getOrDefault((backend, builtin, tKind), builtin)
