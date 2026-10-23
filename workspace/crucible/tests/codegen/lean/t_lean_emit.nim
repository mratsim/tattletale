# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## lean_lang emitter, end-to-end
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
import workspace/crucible/src/codegen/backends/lean_lang
import workspace/crucible/src/codegen/frontend/nim_to_lean
import lean_testutils

const genRelPath = leanProjectDir & "/generated/add.lean"

proc statementTail(path: string): string =
  ## Everything from the per-thread def doc comment onward, byte-exact:
  ## the def, the spec theorems and the worked examples
  let text = readFile(path)
  let idx = text.find("/-- Per-thread body")
  doAssert idx >= 0, "per-thread def doc comment not found in " & path
  return text[idx .. ^1]

proc runTest() =
  block: # emission + Lean elaboration + statement match against the hand-written reference
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
          echo "line ", i + 1, ":\n  reference: ", expected, "\n  generated: ", actual
      doAssert false, "emitted statements differ from the hand-written reference module"
    echo "  emission + elaboration (warningAsError) + reference statement match: ok"

  block: # loud refusal on an out-of-scope construct (while loop)
    # a `for` over a compile-time constant range is the loop shape.
    # t_lean_loop.nim tests it. A while loop stays out of scope
    let ir = irLean:
      proc whileAdd(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        var i = 0
        while i < 4:
          output[i] = a[i] + b[i]
          i = i + 1
    expectRefusal("lean_lang: unsupported GpuAst node gpuVar - " &
      "let-binding whose initializer is not `thread_position_in_grid.x`", proc() =
      discard emitLeanModule(ir.statements[0], genRelPath))
    echo "  refusal on out-of-scope construct: ok"

  block: # loud refusal on a buffer-parameter set other than the add shape
    let copy = irLean:
      proc copyKernel(dst, src: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          dst[i] = src[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuProc - buffer parameters " &
        "must be exactly `a` and `b` (read) plus `output` (written)", proc() =
      discard emitLeanModule(copy.statements[0], genRelPath))

    let renamed = irLean:
      proc renamedKernel(dst, x, y: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          dst[i] = x[i] + y[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuProc - buffer parameters " &
        "must be exactly `a` and `b` (read) plus `output` (written)", proc() =
      discard emitLeanModule(renamed.statements[0], genRelPath))

    let unreadInput = irLean:
      proc copyA(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuProc - the kernel body " &
        "must read both input buffers `a` and `b`", proc() =
      discard emitLeanModule(unreadInput.statements[0], genRelPath))
    echo "  refusal on kernels outside the add-shape buffer parameters: ok"

  block: # loud refusal on a non-positive guard literal
    let zeroBound = irLean:
      proc zeroBoundKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 0:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuIf - guard literal must " &
        "be positive (got 0)", proc() =
      discard emitLeanModule(zeroBound.statements[0], genRelPath))

    let negativeBound = irLean:
      proc negativeBoundKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < -1:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuIf - guard literal must " &
        "be positive (got -1)", proc() =
      discard emitLeanModule(negativeBound.statements[0], genRelPath))
    echo "  refusal on non-positive guard literals: ok"

  block: # loud refusal on nested indexing
    # the model covers reads at the thread's own index only. A nested index
    # (`a[a[i]]`) would otherwise render an unelaboratable Lean term
    let nested = irLean:
      proc nestedKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[a[i]] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuIndex - index expression " &
        "is not the index variable", proc() =
      discard emitLeanModule(nested.statements[0], genRelPath))
    echo "  refusal on nested indexing: ok"


  block: # refusal on a kernel body with no buffer write
    let noWrite = irLean:
      proc noWriteKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
    expectRefusal("lean_lang: unsupported GpuAst node gpuAssign - no buffer write found in the kernel body", proc() =
          discard emitLeanModule(noWrite.statements[0], genRelPath))
    echo "  refusal on a kernel body with no buffer write: ok"

  block: # refusal on a guard outside the `tidVar < literal` shape
    let leGuard = irLean:
      proc leGuardKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i <= 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuIf - guard is not of the shape `tidVar < literal`", proc() =
          discard emitLeanModule(leGuard.statements[0], genRelPath))
    echo "  refusal on a guard outside the `tidVar < literal` shape: ok"

  block: # refusal on more than one write per thread
    let twoWrites = irLean:
      proc twoWritesKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
          output[i] = b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuAssign - more than one write per thread", proc() =
          discard emitLeanModule(twoWrites.statements[0], genRelPath))
    echo "  refusal on more than one write per thread: ok"

  block: # refusal on a second local binding
    let twoBindings = irLean:
      proc twoBindingsKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        let j = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuVar - more than one local binding", proc() =
          discard emitLeanModule(twoBindings.statements[0], genRelPath))
    echo "  refusal on a second local binding: ok"

  block: # refusal on a write index other than the thread id
    let wrongIndex = irLean:
      proc wrongIndexKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i + 0] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuAssign - write index is not the thread id", proc() =
          discard emitLeanModule(wrongIndex.statements[0], genRelPath))
    echo "  refusal on a write index other than the thread id: ok"

  block: # refusal on a non-assign statement inside the guarded branch
    let nestedIf = irLean:
      proc nestedIfKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          if i < 1:
            output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuIf - statement inside the guarded branch", proc() =
          discard emitLeanModule(nestedIf.statements[0], genRelPath))
    echo "  refusal on a non-assign statement inside the guarded branch: ok"

  block: # refusal on an if before the thread-id binding
    let earlyIf = irLean:
      proc earlyIfKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        if 0 < 1:
          output[0] = a[0] + b[0]
        let i = thread_position_in_grid.x
    expectRefusal("lean_lang: unsupported GpuAst node gpuIf - if before the thread-id binding", proc() =
          discard emitLeanModule(earlyIf.statements[0], genRelPath))
    echo "  refusal on an if before the thread-id binding: ok"

  block: # refusal on elif/else branches
    let withElse = irLean:
      proc withElseKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
        else:
          output[i] = a[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuIf - elif/else branches (the tracer kernel is a bare if)", proc() =
          discard emitLeanModule(withElse.statements[0], genRelPath))
    echo "  refusal on elif/else branches: ok"

  block: # refusal on a let binding that is not the thread id
    let plainLet = irLean:
      proc plainLetKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = 4
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuVar - let-binding whose initializer is not `thread_position_in_grid.x`", proc() =
          discard emitLeanModule(plainLet.statements[0], genRelPath))
    echo "  refusal on a let binding that is not the thread id: ok"

  block: # refusal on a discard statement at kernel-body top level
    let noBinding = irLean:
      proc noBindingKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        discard
    expectRefusal("lean_lang: unsupported GpuAst node gpuDiscard - statement at kernel-body top level", proc() =
          discard emitLeanModule(noBinding.statements[0], genRelPath))
    echo "  refusal on a discard statement at kernel-body top level: ok"

when isMainModule:
  runTest()
  echo "t_lean_emit: all checks passed"
