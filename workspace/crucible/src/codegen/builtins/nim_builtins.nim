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

let NimGpuAmbiguousBuiltins* {.compileTime.} = ["min", "max", "abs"]
  ## Names that collide between Nim stdlib (system.min etc.) and GPU backends.
  ## The codegen registers these directly from the concrete type without parsing
  ## their bodies, because system.nim stores only `{.inline.}` (not `{.magic.}`)
  ## in getImpl(), causing the nnkIfExpr handler to crash on statement-style branches.

let NimGpuNumericOperators* {.compileTime.} = {
  "+": "+", "-": "-", "*": "*", "/": "/",
  "div": "/", "mod": "%",
  "shl": "<<", "shr": ">>", "ashr": ">>",
  "and": "&", "or": "|", "xor": "^",
  "not": "~"
}.toTable()
  ## Nim numeric operators and their GPU backend equivalents.
  ## These have `{.magic.}` pragmas and may appear as nnkCall (not nnkInfix)
  ## when operands have non-basic types (e.g. Int[V] structs). The codegen
  ## must recognize them as builtins rather than attempting to register them
  ## as external procs.

let NimGpuBooleanOperators* {.compileTime.} = {
  "and": "&&", "or": "||", "not": "!"
}.toTable()
  ## Boolean operator mappings. Set apart from numeric operators because
  ## `and`/`or`/`not` differ between boolean (&&/||/!) and bitwise (&/|/~).

# NimGpuFnBuiltins* = ["swap", "addr", "sizeof"]
#   ## Function-style builtins with `{.magic.}` that are called as function
#   ## calls (not operators). TODO: handling these might be necessary if
#   ## they reach registerGenericInstOrExternalProc with non-basic arg types.
