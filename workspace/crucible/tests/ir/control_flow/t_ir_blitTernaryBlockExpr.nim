## if-expression with a block-expression branch lowers through a result temp
##
## Test that a block-expression branch of an if-expression lowers through one
## result temp: each branch assigns the temp and the statement reads it.
## No inline block survives to codegen.
##
## Run:
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
  # Each branch assigns the block or literal value to a shared result temp.
  # Codegen then reads that temp, never an inline block.
  # Execution equivalence on Metal: x=5 -> 12, x=1 -> 4, x=0 -> 7.
  doAssert "unsigned int iftmp;" in kernelCode,
    "if-expression lowers through one result temp:\n" & kernelCode
  doAssert "iftmp = ((x + 1U) * 2U);" in kernelCode,
    "then branch assigns the block-expression value to the temp:\n" & kernelCode
  doAssert "iftmp = 7U;" in kernelCode,
    "else branch assigns its literal to the temp:\n" & kernelCode
  doAssert "C[1] = iftmp;" in kernelCode,
    "the statement reads the temp, never an inline block:\n" & kernelCode
  echo "  OK — if-expr block branch lowers through the iftmp result temp"
