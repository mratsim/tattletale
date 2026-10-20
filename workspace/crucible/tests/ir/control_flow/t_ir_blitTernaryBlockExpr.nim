## blitBlockExprs on if-expression block branches
##
## Test that a block-expression branch of an if-expression lowers to a blit
## scope block plus temp before codegen, with the ternary reading the temp.
##
## - `cuda:` macro compile is the check, no expression block survives to codegen
## - The emitted ternary reads the blit temp, matching `((0U < x) ? _blit_0 : 7U)`
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_blitTernaryBlockExpr.nim

import std/strutils
import workspace/crucible

const kernelCode = cuda:
  proc reproTernaryBlock(C: ptr UncheckedArray[uint32]) {.global.} =
    let x = C[0]
    C[1] = if x > 0'u32:
      block:
        let y = x + 1'u32
        y * 2'u32
    else:
      7'u32

when isMainModule:
  # The block-expression branch must be blitted into a scope before the ternary:
  # the emitted ternary reads the blit temp, never an inline block.
  doAssert "{ // _blit_" in kernelCode,
    "block-expression branch must be hoisted into a blit scope:\n" & kernelCode
  doAssert "? _blit_" in kernelCode,
    "ternary must consume the blit temp, not the inline block:\n" & kernelCode
  doAssert "((0U < x) ? _blit_0 : 7U)" in kernelCode,
    "ternary value semantics (block value in the then-branch):\n" & kernelCode
  echo "  OK — if-expr block branch blitted into scope + ternary temp"
