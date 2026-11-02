# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Nim-language side of the builtin vocabulary: the membership data
## consulted while lowering, keyed by the system `{.magic: "..."}` names
## that Nim's standard library attaches to its primitive operators.

import std/tables

const NimGpuNumericBuiltinsOps* = {
  "AddI": "+", "AddU": "+", "AddF32": "+", "AddF64": "+",
  "SubI": "-", "SubU": "-", "SubF32": "-", "SubF64": "-",
  "MulI": "*", "MulU": "*", "MulF32": "*", "MulF64": "*",
  "DivI": "/", "DivU": "/", "DivF32": "/", "DivF64": "/",
  "ModI": "mod", "ModU": "mod",
  "ShlI": "shl", "ShrI": "shr",
}.toTable()
