# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Built-in Nim functions and operators that collide with or must be
## mapped to GPU backend equivalents.
## This file is not reexported by builtins.nim, it is only used by the compiler.

import std/tables
import ../ir/gpu_types

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

let NimGpuFnBuiltins* {.compileTime.} = ["toOpenArray", "len",
  # OpenCL work-item dimension helpers (dummy procs in opencl_builtins —
  # calls forward to the OpenCL C builtins by name)
  "get_global_id", "get_local_id", "get_group_id",
  "get_local_size", "get_global_size", "get_num_groups"]
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
