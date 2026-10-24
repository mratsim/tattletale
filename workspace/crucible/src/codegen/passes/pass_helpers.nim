# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Helpers shared by the transform passes.

import std/sets
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

proc containsReturnStmt*(n: GpuAst): bool =
  ## True when a return statement sits anywhere in `n`
  if n.kind == gpuReturn:
    return true
  for ch in n:
    if ch.containsReturnStmt():
      return true

proc collectIdentNames*(n: GpuAst, names: var HashSet[string]) =
  ## Every identifier name appearing in `n`, declaration positions included.
  ## Fresh names must not collide with any name the body already uses.
  ##
  ## Args:
  ## - `names` receives the collected names
  ##
  ## The AST iterator yields value positions only, the dedicated fields carry
  ## declaration names (`var` names, call targets, loop variables).
  case n.kind
  of gpuIdent:
    if n.symbol != nil:
      names.incl n.symbol.name
    return
  of gpuVar:
    names.incl n.vName.symbol.name
  of gpuCall:
    if n.cName.symbol != nil:
      names.incl n.cName.symbol.name
  of gpuFor:
    names.incl n.fVar.symbol.name
  else:
    discard
  for ch in n:
    ch.collectIdentNames(names)

proc collectParamISyms*(n: GpuAst, paramISyms: HashSet[string], found: var HashSet[string], writeTargetsOnly = false) =
  ## Collects the iSyms of kernel parameters referenced by the tree:
  ##
  ## - default: every parameter-rooted identifier
  ## - `writeTargetsOnly`: only assignment targets rooted at a parameter index chain, the written parameters
  if n.isNil:
    return
  if writeTargetsOnly:
    if n.kind == gpuAssign and n.aLeft.kind == gpuIndex:
      var root = n.aLeft.iArr
      while root.kind == gpuDeref:
        root = root.dOf
      if root.kind == gpuIdent and root.symbol != nil and
          root.symbol.iSym in paramISyms:
        found.incl root.symbol.iSym
  elif n.kind == gpuIdent and n.symbol != nil and n.symbol.iSym in paramISyms:
    found.incl n.symbol.iSym
  for ch in n.items:
    collectParamISyms(ch, paramISyms, found, writeTargetsOnly)
