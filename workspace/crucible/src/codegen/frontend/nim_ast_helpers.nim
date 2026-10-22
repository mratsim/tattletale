# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Small AST helpers over NimNode only: unwrapping and shape tests.
## GPU context, type registry, and GpuAst construction
## belong to nim_to_crucible.nim and the type resolvers.

import std/macros

proc unwrapSingleStmt*(n: NimNode): NimNode =
  ## Callers keep their own multi-statement constraints.
  if n.kind == nnkStmtList and n.len == 1:
    n[0]
  else:
    n

proc isTypeDescNode*(n: NimNode): bool =
  result = n.getTypeInst().typeKind == ntyTypeDesc

proc isGenericProc*(node: NimNode): bool =
  # An instantiated generic keeps its params in an `nnkBracket` at `impl[5]`.
  let impl = node.getImpl()
  if impl.kind notin [nnkProcDef, nnkFuncDef]:
    return false
  if impl[2].kind == nnkGenericParams:
    return true
  result = impl.len > 5 and impl[5].kind == nnkBracket and
    impl[5].len > 1 and impl[5][1].kind == nnkGenericParams
