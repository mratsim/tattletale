# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Helpers shared by the transform passes.

import ../ir/ast_datatypes

func unwrapSingleStmtBlocks*(n: GpuAst): GpuAst =
  ## Returns `n` with nested single-statement block wrappers stripped.
  ##
  ## toCrucibleAst shapes an if-expression branch statement as nested
  ## single-statement blocks:
  ##
  ##   `if c: (block: (t = 1.0; t + 1))`
  ##
  ## becomes block(block(assign, expr)).
  result = n
  while result.kind == gpuBlock and result.statements.len == 1:
    result = result.statements[0]

func isExprOnlyBranch*(n: GpuAst): bool =
  ## True when an if-expression branch is a bare expression after
  ## unwrapping single-statement block wrappers.
  unwrapSingleStmtBlocks(n).kind != gpuBlock
