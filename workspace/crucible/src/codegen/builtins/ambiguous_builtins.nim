# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Ambiguous builtins between Nim and GPU backend.
# This file is not supposed to be reexported by builtins.nim, it is only used by the compiler

let NimGpuAmbiguousBuiltins* {.compileTime.} = ["min", "max", "abs"]
  ## Names that collide between Nim stdlib (system.min etc.) and GPU backends.
  ## The codegen registers these directly from the concrete type without parsing
  ## their bodies, because system.nim stores only `{.inline.}` (not `{.magic.}`)
  ## in getImpl(), causing the nnkIfExpr handler to crash on statement-style branches.
