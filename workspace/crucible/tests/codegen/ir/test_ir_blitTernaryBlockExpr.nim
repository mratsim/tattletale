## Blit pass: block-expression branches of an if-expression must be
## lowered before codegen.
##
## An if-expression `if cond: block: ... else: ...` lowers to a gpuTernary
## during normalization. A block-expression branch of that ternary must be
## blitted into a scope block + blit temp before codegen. ensureNoExprBlocks
## rejects any block that survives to codegen, and the ceramic gemm_cta
## tile-view templates expand into if-expr chains whose branches are block
## expressions, so this case is required for gemm_cta entries.
##
## The kernel must compile through the `cuda:` macro, whose run is the compile check.
## The emitted text must show the block hoisted out of the ternary into a blit scope.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_blitTernaryBlockExpr.nim

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
