# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Loop-kernel emission and loop-invariant scaffolds
##
## Test that the loop kernel shape, one `gpuFor` over a compile-time constant
## range `[0, N)` with one indexed write, emits a Lean module keeping the loop
## as a loop, with the `{.invariant.}` pipeline extended to it:
##
## - the loop module elaborates under warningAsError, sorry-free
## - loop kernels outside the supported shape are refused loudly
## - a loop-guard invariant is proved, an unprovable-by-the-scaffold one downgrades to `sorry`
## - precondition/postcondition pragmas on the loop shape are refused loudly
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[assertions, json, os, strutils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/backends/lean_lang
import lean_testutils

const genLoopRelPath = leanProjectDir & "/generated/loop_add.lean"
const genLoopInvRelPath = leanProjectDir & "/generated/loop_inv.lean"
const genLoopInvWrongRelPath = leanProjectDir & "/generated/loop_inv_wrong.lean"

proc scaffoldEntry(manifest: JsonNode, kind: string): JsonNode =
  ## Manifest scaffold entry of the given kind
  for e in manifest["scaffolds"].items:
    if e["kind"].getStr() == kind:
      return e
  doAssert false, "no " & kind & " scaffold in the manifest"

proc runTest() =
  block: # loop-kernel emission + Lean elaboration (warningAsError, sorry-free)
    let ir = irLean:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]

    let emitted = emitLeanModule(ir.statements[0], genLoopRelPath)
    doAssert "def loopAddLoop " in emitted, "the loop stays a recursive def in Lean"
    doAssert "def loopAddBody " in emitted, "entry-point def emitted"
    doAssert "theorem loopAddLoop_eq " in emitted, "loop-state lemma emitted"
    doAssert "theorem loopAddBody_writes_own_index" in emitted, "per-index postcondition half"
    doAssert "theorem loopAddBody_preserves_others" in emitted, "out-of-range preservation half"
    doAssert "theorem loopAddBody_only_writes_in_range" in emitted, "bounds theorem"
    doAssert "theorem loopAddBody_post" in emitted, "composed postcondition"
    doAssert "\ntermination_by" notin emitted,
      "structural recursion on the remaining count needs no termination_by clause"
    doAssert fileExists(genLoopRelPath), "emitted module written to disk"

    let (outp, code) = elaborate("generated/loop_add.lean")
    doAssert code == 0, "Lean rejected the emitted loop module:\n" & outp
    doAssert "sorry" notin outp, "no sorry under warningAsError"
    echo "  loop-kernel emission + elaboration (warningAsError): ok"

  block: # loud refusals on loop shapes outside the tracer's scope
    let nonLiteralBound = irLean:
      proc nonLiteralBound(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< thread_position_in_grid.x:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuFor - loop bound is not a compile-time literal", proc() =
          discard emitLeanModule(nonLiteralBound.statements[0], genLoopRelPath))

    let nonzeroStart = irLean:
      proc nonzeroStart(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 1 ..< 4:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuFor - loop start must be the literal 0", proc() =
          discard emitLeanModule(nonzeroStart.statements[0], genLoopRelPath))

    let inclusiveRange = irLean:
      proc inclusiveRange(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 .. 3:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuFor - loop range is not `[0 ..< n)` exclusive", proc() =
          discard emitLeanModule(inclusiveRange.statements[0], genLoopRelPath))

    let stepped = irLean:
      proc stepped(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in countup(0, 6, 2):
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuFor - loop range is not `[0 ..< n)` exclusive", proc() =
          discard emitLeanModule(stepped.statements[0], genLoopRelPath))

    let nested = irLean:
      proc nested(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          for j in 0 ..< 4:
            output[j] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuFor - statement inside the loop body", proc() =
          discard emitLeanModule(nested.statements[0], genLoopRelPath))

    let nonLoopVarWrite = irLean:
      proc nonLoopVarWrite(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[0] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuAssign - write index is not the loop variable", proc() =
          discard emitLeanModule(nonLoopVarWrite.statements[0], genLoopRelPath))

    let nonLoopVarRead = irLean:
      proc nonLoopVarRead(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[0] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuIndex - index expression is not the index variable", proc() =
          discard emitLeanModule(nonLoopVarRead.statements[0], genLoopRelPath))

    let stmtBefore = irLean:
      proc stmtBefore(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        output[0] = 1
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuAssign - statement at kernel-body top level", proc() =
          discard emitLeanModule(stmtBefore.statements[0], genLoopRelPath))

    let stmtAfter = irLean:
      proc stmtAfter(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
        output[0] = 1
    expectRefusal("lean_lang: unsupported GpuAst node gpuFor - statement at kernel-body top level", proc() =
          discard emitLeanModule(stmtAfter.statements[0], genLoopRelPath))

    let twoWrites = irLean:
      proc twoWrites(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
          output[i] = a[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuFor - loop body is not a single-statement block", proc() =
          discard emitLeanModule(twoWrites.statements[0], genLoopRelPath))

    echo "  refusals on loop shapes outside the supported shape: ok"

  block: # loud refusals carried over from the guarded-thread shape
    let unsupportedOp = irLean:
      proc unsupportedOp(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] * b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuBinOp - binary operator '*'", proc() =
          discard emitLeanModule(unsupportedOp.statements[0], genLoopRelPath))

    let zeroBound = irLean:
      proc zeroBound(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 0:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuFor - loop bound must be positive (got 0)", proc() =
          discard emitLeanModule(zeroBound.statements[0], genLoopRelPath))

    let wrongParams = irLean:
      proc wrongParams(dst, x, y: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          dst[i] = x[i] + y[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuProc - buffer parameters must be exactly `a` and `b` (read) plus " &
            "`output` (written)", proc() =
          discard emitLeanModule(wrongParams.statements[0], genLoopRelPath))
    echo "  refusals carried over from the guarded-thread shape: ok"

  block: # right loop-guard invariant: proved, manifest `proven: true`, exit 0
    let k = irLeanSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "i < 4".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]

    let emitted = emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, genLoopInvRelPath)
    doAssert "theorem loopAddLoop_invariant" in emitted, "loop-invariant scaffold emitted"
    doAssert "sorry" notin emitted, "no scaffold needed the downgrade"

    let manifest = parseJson(readFile(genLoopInvRelPath.changeFileExt("manifest.json")))
    doAssert manifest["kernel"].getStr() == "loopAdd", "manifest names the kernel"
    let inv = scaffoldEntry(manifest, "invariant")
    doAssert inv["payload"].getStr() == "i < 4"
    doAssert inv["name"].getStr() == "loopAddLoop_invariant"
    doAssert inv["proven"].getBool(), "loop-guard invariant proved"
    doAssert inv["error"].kind == JNull

    let (outp, code) = elaborate("generated/loop_inv.lean")
    doAssert code == 0, "Lean rejected the proven loop module:\n" & outp
    doAssert "sorry" notin outp, "no sorry under warningAsError"
    echo "  right loop invariant: proven manifest + warningAsError elaboration: ok"

  block: # unprovable-by-the-scaffold loop invariant: sorry downgrade, unproven manifest, warningAsError rejection
    # the payload `i ≤ 4` is true over the loop counter, but the fixed proof
    # attempt `exact h` of the scaffold cannot close it, so the proof downgrades
    # to `sorry` and the manifest records unproven, a proof-status verdict
    # that never judges the truth of the payload
    let k = irLeanSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "i ≤ 4".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]

    discard emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, genLoopInvWrongRelPath)
    let manifest = parseJson(readFile(genLoopInvWrongRelPath.changeFileExt("manifest.json")))
    let inv = scaffoldEntry(manifest, "invariant")
    doAssert inv["proven"].getBool() == false, "unprovable-by-the-scaffold loop invariant recorded unproven"
    doAssert inv["error"].getStr().len > 0, "Lean error excerpt recorded"
    echo "  unprovable-by-the-scaffold loop invariant manifest entry: ", inv.pretty

    let emitted = readFile(genLoopInvWrongRelPath)
    doAssert "sorry" in emitted, "proof downgraded to a sorry-backed scaffold"
    let (_, code) = elaborate("generated/loop_inv_wrong.lean")
    doAssert code != 0, "the sorry-backed module must fail under warningAsError"
    echo "  unprovable-by-the-scaffold loop invariant: unproven manifest + warningAsError rejection: ok"

  block: # loud refusals on loop-shape pragma payloads
    let withPrecondition = irLeanSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "tid < n".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported precondition payload 'tid < n' - the loop shape has no thread-id launch precondition", proc() =
          discard emitLeanModuleWithSpecs(withPrecondition.gpu.statements[0],
            withPrecondition.specs, genLoopInvRelPath))

    let withPostcondition = irLeanSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "tid < n".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported postcondition payload 'tid < n' - the " &
            "loop postcondition is the composed", proc() =
          discard emitLeanModuleWithSpecs(withPostcondition.gpu.statements[0],
            withPostcondition.specs, genLoopInvRelPath))

    let bufferInvariant = irLeanSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "a tid < 4".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported invariant payload 'a tid < 4' - the " &
            "loop-shape invariant must mention the loop counter 'i'", proc() =
          discard emitLeanModuleWithSpecs(bufferInvariant.gpu.statements[0],
            bufferInvariant.specs, genLoopInvRelPath))

    let boundInvariant = irLeanSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "i < n".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported invariant payload 'i < n' - the loop-shape " &
            "invariant may only mention the loop counter 'i'", proc() =
          discard emitLeanModuleWithSpecs(boundInvariant.gpu.statements[0],
            boundInvariant.specs, genLoopInvRelPath))
    echo "  refusals on loop-shape pragma payloads: ok"


  block: # refusal on an unsupported binary operator in the loop shape
    let subKernel = irLean:
      proc subKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          output[i] = a[i] - b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuBinOp - binary operator '-'", proc() =
          discard emitLeanModule(subKernel.statements[0], genLoopRelPath))
    echo "  refusal on an unsupported binary operator in the loop shape: ok"

  block: # refusal on a loop variable colliding with a buffer parameter
    let loopVarA = irLean:
      proc loopVarAKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for a in 0 ..< 4:
          output[a] = b[a] + b[a]
    expectRefusal("lean_lang: unsupported GpuAst node gpuFor - loop variable 'a' collides with a kernel buffer parameter", proc() =
          discard emitLeanModule(loopVarA.statements[0], genLoopRelPath))
    echo "  refusal on a loop variable colliding with a buffer parameter: ok"

  block: # refusal on a written buffer other than output in the loop shape
    let writesA = irLean:
      proc writesAKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 4:
          a[i] = output[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuAssign - the written buffer must be `output` - got 'a'", proc() =
          discard emitLeanModule(writesA.statements[0], genLoopRelPath))
    echo "  refusal on a written buffer other than output in the loop shape: ok"

when isMainModule:
  runTest()
  echo "t_lean_loop: all checks passed"
