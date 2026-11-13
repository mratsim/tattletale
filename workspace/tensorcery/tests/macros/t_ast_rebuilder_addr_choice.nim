# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Rebuild of a call whose head is a symbol choice resolving to `addr`
##
## Run:
##   nim c -r -d:release --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/tensorcery/tests/macros/t_ast_rebuilder_addr_choice.nim

import std/[macros, os]
import workspace/tensorcery/src/macros/ast_rebuilder

macro addrChoiceRebuilds(): bool =
  ## `addr x` with the callee still a resolved symbol choice
  let node = nnkCall.newTree(
    nnkClosedSymChoice.newTree(bindSym"addr", bindSym"addr"),
    ident"x")
  let rebuilt = rebuildUntypedAst(node)
  doAssert rebuilt.kind == nnkAddr,
    "the addr choice rebuilds to nnkAddr, got " & $rebuilt.kind
  doAssert rebuilt[0].kind == nnkIdent and rebuilt[0].strVal == "x",
    "the operand survives the rebuild, got " & $rebuilt[0].repr
  result = newLit(true)

proc runTest() =
  doAssert addrChoiceRebuilds()
  echo "[OK] tensorcery ast_rebuilder: the addr symbol choice rebuilds with its operand"

when isMainModule:
  runTest()
