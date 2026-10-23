# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Built-in device functions (exp2/rsqrt) and the Nim-operator/function
## name tables the codegen consults.

import std/[tables, math, strutils]
import ../codegen/ir/ast_datatypes
import ./builtins_pragmas

# ═════════════════════════════════════════════════════════════════════════
#  The scalar math builtins
# ═════════════════════════════════════════════════════════════════════════

proc exp2*(x: float32): float32 {.builtin.} = pow(2'f32, x)
  ## Returns 2ˣ
  ##
  ## Declared {.builtin.} so lowers to the backen native function
  ## or the body on host

proc rsqrt*(x: float32): float32 {.builtin.} = 1 / sqrt(x)
  ## Returns 1/√x
  ##
  ## Declared {.builtin.} so lowers the backen native function
  ## or the body on host


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
  "and": "&&", "or": "||", "not": "!",
  "<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "=="
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
   (bkCuda,   "abs", gtFloat64): "fabs",
   (bkHlsl,   "max", gtInt32): "max",
   (bkHlsl,   "max", gtUint32): "max",
   (bkHlsl,   "max", gtSize_t): "max",
   (bkHlsl,   "max", gtFloat32): "max",
   (bkHlsl,   "max", gtFloat64): "max",
   (bkHlsl,   "min", gtInt32): "min",
   (bkHlsl,   "min", gtUint32): "min",
   (bkHlsl,   "min", gtSize_t): "min",
   (bkHlsl,   "min", gtFloat32): "min",
   (bkHlsl,   "min", gtFloat64): "min",
   (bkHlsl,   "abs", gtInt32): "abs",
   (bkHlsl,   "abs", gtUint32): "abs",
   (bkHlsl,   "abs", gtSize_t): "abs",
   (bkHlsl,   "abs", gtFloat32): "abs",
   (bkHlsl,   "abs", gtFloat64): "abs"}.toTable()

proc hasMaxMinAbsBuiltinFnName*(backend: BackendKind, builtin: string, tKind: GpuTypeKind): bool =
  ## Returns true when NimGpuNumericBuiltinsFnNames declares an emitted name
  ## for the (`backend`, `builtin`, `tKind`) combo. A declared combo prints
  ## the intrinsic call and drops the materialized body.
  (backend, builtin, tKind) in NimGpuNumericBuiltinsFnNames

proc getMaxMinAbsBuiltinFnName*(backend: BackendKind; builtin: string;
                             tKind: GpuTypeKind): string =
  ## Backend function name for an ambiguous builtin call whose operands have
  ## type `tKind`.
  ## Combos absent from NimGpuAmbiguousBuiltinFn keep the plain name.
  NimGpuNumericBuiltinsFnNames.getOrDefault((backend, builtin, tKind), builtin)

func isBase58Chunk(s: string): bool =
  ## npHashSuffix tail shape: exactly 7 base58 chars.
  if s.len != 7:
    return false
  for c in s:
    if Base58.find(c) < 0:
      return false
  result = true

func isNimMangleChunk(s: string): bool =
  ## Nim instantiation-mangle chunk shape (`signatureHash` in an iSym):
  ## an alphanumeric run carrying at least one digit. The digit requirement
  ## and the 16-char floor keep lowercase words like `helper` from matching.
  if s.len < 16:
    return false
  var hasDigit = false
  for c in s:
    if c notin {'a'..'z', 'A'..'Z', '0'..'9'}:
      return false
    if c in {'0'..'9'}:
      hasDigit = true
  result = hasDigit

func isMangledHashRemainder(rest: string): bool =
  ## True when `rest`, the part of an fnTab display name after a numeric
  ## builtin stem and its first underscore, parses entirely as hash chunks
  ## separated by underscores:
  ## - the npHashSuffix tail, exactly 7 base58 chars
  ## - a Nim instantiation-mangle chunk
  ##
  ## Any other chunk, e.g. the word in `max_helper`, means the name belongs
  ## to a user function and the parse fails.
  if rest.len == 0:
    return false
  var i = 0
  while i < rest.len:
    while i < rest.len and rest[i] == '_':
      inc i
    if i >= rest.len:
      return false
    let start = i
    while i < rest.len and rest[i] in {'a'..'z', 'A'..'Z', '0'..'9'}:
      inc i
    let chunk = rest[start ..< i]
    if not isBase58Chunk(chunk) and not isNimMangleChunk(chunk):
      return false
  result = true

proc numericBuiltinDisplayName*(name: string): string =
  ## Returns the `min`/`max`/`abs` builtin behind an fnTab display name, seeing through the mangle suffixes generic instantiations carry:
  ## - `max_RJ1UXDu` -> `max`, the npHashSuffix tail
  ## - `max___ZE4xeJFcdj6wQZpDx9bBlvQ_SStiky4` -> `max`, a Nim instantiation mangle under the tail
  ##
  ## Returns "" when `name` is neither a plain nor a mangled numeric builtin.
  ## Everything after the stem must parse as hash chunks, so a user function
  ## like `max_helper` never matches.
  ##
  ## A user-defined generic `max` mangles identically to the builtin, so the call resolves to the intrinsic.
  ## The intrinsic's materialized body computes the same value.
  if name in NimGpuNumericBuiltinsFunctions:
    return name
  for builtin in NimGpuNumericBuiltinsFunctions:
    if name.startsWith(builtin) and name.len > builtin.len and
        name[builtin.len] == '_' and
        isMangledHashRemainder(name[builtin.len + 1 .. ^1]):
      return builtin
  result = ""
