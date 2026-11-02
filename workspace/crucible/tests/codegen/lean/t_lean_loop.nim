# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Loop-kernel emission
##
## Test that one `for` over a compile-time constant `[0, n)` range with one
## indexed write emits a Lean module keeping the loop as a loop:
##
## - the loop module elaborates under warningAsError, sorry-free
## - loop kernels outside the supported shape are refused loudly
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[os, strutils]
import lean_testutils

const genDir = leanProjectDir & "/generated"

proc runTest() =
  block: # loop-kernel emission + Lean elaboration (warningAsError, sorry-free)
    let ir = leanIr:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    doAssert ir.refusal.len == 0, "unexpected refusal: " & ir.refusal
    let hasLoopDef = "def loopAddLoop " in ir.text
    let hasEntryDef = "def loopAddBody " in ir.text
    let hasLoopLemma = "theorem loopAddLoop_eq " in ir.text
    let hasWriteHalf = "theorem loopAddBody_writes_own_index" in ir.text
    let hasKeepHalf = "theorem loopAddBody_preserves_others" in ir.text
    let noTerminationBy = "\ntermination_by" notin ir.text
    doAssert hasLoopDef, "the loop stays a recursive def in Lean"
    doAssert hasEntryDef, "entry-point def emitted"
    doAssert hasLoopLemma, "loop-state lemma emitted"
    doAssert hasWriteHalf, "per-index postcondition half"
    doAssert hasKeepHalf, "out-of-range preservation half"
    doAssert noTerminationBy,
      "structural recursion on the remaining count needs no termination_by clause"
    createDir(genDir)
    writeFile(genDir / "loopAdd.autogen.lean", ir.text)

    let (outp, code) = elaborate("generated/loopAdd.autogen.lean")
    doAssert code == 0, "Lean rejected the emitted loop module:\n" & outp
    echo "  loop-kernel emission + elaboration (warningAsError): ok"

  block: # loud refusals on loop shapes outside the supported shape
    let nonLiteralBound = leanIr:
      proc nonLiteralBound(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< thread_position_in_grid.x:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported loop - bound is not a compile-time literal",
      nonLiteralBound)

    let nonzeroStart = leanIr:
      proc nonzeroStart(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 1 ..< 4:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported loop - start must be the literal 0 - got '1'",
      nonzeroStart)

    let inclusiveRange = leanIr:
      proc inclusiveRange(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 .. 3:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported loop - condition is not of the shape" &
      " `i < literal`", inclusiveRange)

    let stepped = leanIr:
      proc stepped(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in countup(0, 6, 2):
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported loop - condition is not of the shape" &
      " `i < literal`", stepped)

    let nested = leanIr:
      proc nested(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          for j in 0 ..< 4:
            output[j] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported loop - body is not a single write" &
      " plus the counter increment", nested)

    let nonLoopVarWrite = leanIr:
      proc nonLoopVarWrite(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[0] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported assignment - write index is not the" &
      " loop variable", nonLoopVarWrite)

    let nonLoopVarRead = leanIr:
      proc nonLoopVarRead(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[0] + b[i]
    expectLeanRefusal("lean_lang: unsupported expression ePat - index expression" &
      " is not the index variable", nonLoopVarRead)

    let stmtBefore = leanIr:
      proc stmtBefore(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        output[0] = 1
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported statement sAsgn - statement at" &
      " kernel-body top level", stmtBefore)

    let stmtAfter = leanIr:
      proc stmtAfter(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
        output[0] = 1
    expectLeanRefusal("lean_lang: unsupported statement sVar - local binding whose" &
      " initializer is not `thread_position_in_grid.x`", stmtAfter)

    let twoWrites = leanIr:
      proc twoWrites(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
          output[i] = a[i]
    expectLeanRefusal("lean_lang: unsupported loop - body is not a single write" &
      " plus the counter increment", twoWrites)
    echo "  refusals on loop shapes outside the supported shape: ok"

  block: # loud refusals carried over from the guarded-thread shape
    let unsupportedOp = leanIr:
      proc unsupportedOp(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] * b[i]
    expectLeanRefusal("lean_lang: unsupported expression eMul - the multiply is" &
      " in scope for the axpy shape only", unsupportedOp)

    let zeroBound = leanIr:
      proc zeroBound(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 0:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported loop - bound must be positive (got 0)",
      zeroBound)

    let wrongParams = leanIr:
      proc wrongParams(dst, x, y: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          dst[i] = x[i] + y[i]
    expectLeanRefusal("lean_lang: unsupported kernel - buffer parameters must be" &
      " exactly `a` and `b` (read) plus `output` (written)", wrongParams)

    let subKernel = leanIr:
      proc subKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] - b[i]
    expectLeanRefusal("lean_lang: unsupported expression eSub", subKernel)
    echo "  refusals carried over from the guarded-thread shape: ok"

  block: # refusal on a loop variable colliding with the emitted update binder
    let loopVarJ = leanIr:
      proc loopVarJKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for j in 0 ..< 4:
          output[j] = a[j] + b[j]
    expectLeanRefusal("lean_lang: unsupported loop - loop variable 'j' collides" &
      " with the emitted update binder 'j'", loopVarJ)
    echo "  refusal on a loop variable colliding with the emitted binder: ok"

  block: # refusal on a written buffer other than output in the loop shape
    let writesA = leanIr:
      proc writesAKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          a[i] = output[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported assignment - the written buffer" &
      " must be `output` - got 'a'", writesA)
    echo "  refusal on a written buffer other than output in the loop shape: ok"

when isMainModule:
  runTest()
  echo "t_lean_loop: all checks passed"
