# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## P2 test: end-to-end check of the nim_to_lean emitter
##
## Runs the emitter on the tracer add kernel (the GlobalInvocationID shape)
## and asserts:
## - the emitted module elaborates via `lake env lean -DwarningAsError=true` (a sorry fails)
## - the emitted def plus spec theorems are byte-identical to the hand-written P1 statements
## - a for-loop kernel, outside the emitter's scope, is refused loudly
##
## Run:
##   cd workspace/crucible
##   nim c -r --hints:off --warnings:off \
##     -o:build/wip/t_p2_emit --nimcache:nimcache/wip poc/lean/t_p2_emit.nim

import std/[assertions, macros, os, osproc, strutils]
import workspace/crucible/src/crucible_internal_compiler
import nim_to_lean

const leanPocDir = "poc/lean"
const genRelPath = leanPocDir & "/generated/add.lean"

proc elaborate(relPath: string): tuple[output: string, exitCode: int] =
  ## Elaborates one emitted Lean module through the lake environment,
  ## warnings escalated to errors so a sorry fails
  execCmdEx("lake env lean -DwarningAsError=true " & relPath,
    workingDir = leanPocDir)

proc statementTail(path: string): string =
  ## Everything from the per-thread def doc comment onward: the def, the spec
  ## theorems and the worked examples, byte-exact
  let text = readFile(path)
  let idx = text.find("/-- Per-thread body")
  doAssert idx >= 0, "per-thread def doc comment not found in " & path
  return text[idx .. ^1]

proc runTest() =
  block: # emission + Lean elaboration + statement match against the P1 module
    let ir = irVulkan:
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

    let reference = statementTail(leanPocDir & "/AddKernel.lean")
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

  block: # loud refusal on an out-of-scope construct (for-loop)
    let ir = irVulkan:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModule(ir.statements[0], genRelPath)
    echo "  refusal on out-of-scope construct: ok"

when isMainModule:
  runTest()
  echo "t_p2_emit: all checks passed"
