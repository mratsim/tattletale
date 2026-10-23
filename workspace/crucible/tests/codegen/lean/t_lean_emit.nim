# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim_to_lean emitter, end-to-end
##
## Test that the emitter produces the add kernel's Lean verification module:
##
## - the module elaborates via `lake env lean -DwarningAsError=true` (a sorry fails)
## - the emitted def plus spec theorems are byte-identical to the hand-written
##   `AddKernel.lean` statements
## - kernels outside the supported shape are refused loudly
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[assertions, os, osproc, strutils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/frontend/lean_pragmas
import workspace/crucible/src/codegen/frontend/nim_to_lean
import lean_testutils

const genRelPath = leanProjectDir & "/generated/add.lean"

proc statementTail(path: string): string =
  ## Everything from the per-thread def doc comment onward: the def, the spec
  ## theorems and the worked examples, byte-exact
  let text = readFile(path)
  let idx = text.find("/-- Per-thread body")
  doAssert idx >= 0, "per-thread def doc comment not found in " & path
  return text[idx .. ^1]

proc runTest() =
  block: # emission + Lean elaboration + statement match against the P1 module
    let ir = irLean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]

    let emitted = emitLeanModule(ir.statements[0], genRelPath)
    doAssert "def addBody " in emitted, "per-thread def emitted"
    doAssert fileExists(genRelPath), "emitted module written to disk"

    let (outp, code) = elaborate("generated/add.lean")
    doAssert code == 0, "Lean rejected the emitted module:\n" & outp
    doAssert "sorry" notin outp, "no sorry under warningAsError"

    let reference = statementTail(leanProjectDir & "/AddKernel.lean")
    let generated = statementTail(genRelPath)
    if reference != generated:
      let refLines = reference.splitLines()
      let genLines = generated.splitLines()
      for i in 0 ..< max(refLines.len, genLines.len):
        let expected = if i < refLines.len: refLines[i] else: "<missing>"
        let actual = if i < genLines.len: genLines[i] else: "<missing>"
        if expected != actual:
          echo "line ", i + 1, ":\n  P1: ", expected, "\n  P2: ", actual
      doAssert false, "emitted statements differ from the hand-written P1 module"
    echo "  emission + elaboration (warningAsError) + P1 statement match: ok"

  block: # loud refusal on an out-of-scope construct (while loop)
    # a `for` over a compile-time constant range is the P4 loop shape, tested
    # by t_p4_loop.nim. A while loop stays out of scope
    let ir = irLean:
      proc whileAdd(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        var i = 0
        while i < 4:
          output[i] = a[i] + b[i]
          i = i + 1
    doAssertRaises(ValueError):
      discard emitLeanModule(ir.statements[0], genRelPath)
    echo "  refusal on out-of-scope construct: ok"

  block: # loud refusal on a buffer-parameter set other than the add shape
    let copy = irLean:
      proc copyKernel(dst, src: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          dst[i] = src[i]
    doAssertRaises(ValueError):
      discard emitLeanModule(copy.statements[0], genRelPath)

    let renamed = irLean:
      proc renamedKernel(dst, x, y: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          dst[i] = x[i] + y[i]
    doAssertRaises(ValueError):
      discard emitLeanModule(renamed.statements[0], genRelPath)

    let unreadInput = irLean:
      proc copyA(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i]
    doAssertRaises(ValueError):
      discard emitLeanModule(unreadInput.statements[0], genRelPath)
    echo "  refusal on kernels outside the add-shape buffer parameters: ok"

  block: # loud refusal on a non-positive guard literal
    let zeroBound = irLean:
      proc zeroBoundKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 0:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModule(zeroBound.statements[0], genRelPath)

    let negativeBound = irLean:
      proc negativeBoundKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < -1:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModule(negativeBound.statements[0], genRelPath)
    echo "  refusal on non-positive guard literals: ok"

  block: # loud refusal on nested indexing
    # the model covers reads at the thread's own index only. A nested index
    # (`a[a[i]]`) would otherwise render an unelaboratable Lean term
    let nested = irLean:
      proc nestedKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[a[i]] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModule(nested.statements[0], genRelPath)
    echo "  refusal on nested indexing: ok"

when isMainModule:
  runTest()
  echo "t_lean_emit: all checks passed"
