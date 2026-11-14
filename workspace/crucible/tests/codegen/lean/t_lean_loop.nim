## Loop-kernel emission
##
## Test that one constant-range `for` with one indexed write emits the loop as a structural recursion
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[json, os, strutils]
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

  block: # loud refusal on a loop bound of zero
    let zeroBound = leanIr:
      proc zeroBound(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        for i in 0 ..< 0:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported loop - bound must be positive (got 0)",
      zeroBound)

    echo "  refusal on a loop bound of zero: ok"

  block: # right loop-guard invariant: proved, manifest `proven: true`, exit 0
    let k = leanIrSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "i < 4".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    doAssert k.refusal.len == 0, "unexpected refusal: " & k.refusal
    let emitted = settleModule(k.text, k.scaffolds, genDir, "loop_inv.lean")
    let hasInvScaffold = "theorem loopAddLoop_invariant" in emitted
    let noDowngrade = "sorry" notin emitted
    doAssert hasInvScaffold, "loop-invariant scaffold emitted"
    doAssert noDowngrade, "no scaffold needed the downgrade"

    let manifest = parseJson(readFile((genDir / "loop_inv.lean").changeFileExt("manifest.json")))
    let namesKernel = manifest["kernel"].getStr() == "loopAdd"
    doAssert namesKernel, "manifest names the kernel"
    let inv = scaffoldEntry(manifest, "invariant")
    let payloadOk = inv["payload"].getStr() == "i < 4"
    let nameOk = inv["name"].getStr() == "loopAddLoop_invariant"
    doAssert payloadOk
    doAssert nameOk
    doAssert inv["proven"].getBool(), "loop-guard invariant proved"
    doAssert inv["error"].kind == JNull

    let (outp, code) = elaborate("generated/loop_inv.lean")
    doAssert code == 0, "Lean rejected the proven loop module:\n" & outp
    echo "  right loop invariant: proven manifest + warningAsError elaboration: ok"

  block: # unprovable-by-the-scaffold loop invariant: sorry downgrade, unproven manifest
    # payload `i ≤ 4` is true over the loop counter, but the fixed proof
    # attempt `exact h` of the scaffold cannot close it, so the proof downgrades
    # to `sorry` and the manifest records unproven, a proof-status verdict
    # that never judges the truth of the payload
    let k = leanIrSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "i ≤ 4".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    doAssert k.refusal.len == 0, "unexpected refusal: " & k.refusal
    discard settleModule(k.text, k.scaffolds, genDir, "loop_inv_wrong.lean")
    let manifest = parseJson(readFile((genDir / "loop_inv_wrong.lean").changeFileExt("manifest.json")))
    let inv = scaffoldEntry(manifest, "invariant")
    doAssert inv["proven"].getBool() == false, "unprovable-by-the-scaffold loop invariant recorded unproven"
    doAssert inv["error"].getStr().len > 0, "Lean error excerpt recorded"

    let emitted = readFile(genDir / "loop_inv_wrong.lean")
    let downgraded = "sorry" in emitted
    doAssert downgraded, "proof downgraded to a sorry-backed scaffold"
    let (_, code) = elaborate("generated/loop_inv_wrong.lean")
    doAssert code != 0, "the sorry-backed module must fail under warningAsError"
    echo "  unprovable-by-the-scaffold loop invariant: unproven manifest + warningAsError rejection: ok"

  block: # loud refusals on loop-shape pragma payloads
    let withPrecondition = leanIrSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "tid < n".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported precondition payload 'tid < n' - the" &
      " loop shape has no thread-id launch precondition", withPrecondition)

    let withPostcondition = leanIrSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "tid < n".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported postcondition payload 'tid < n' - the " &
      "loop postcondition is the composed", withPostcondition)

    let bufferInvariant = leanIrSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "a tid < 4".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported invariant payload 'a tid < 4' - the " &
      "loop-shape invariant must mention the loop counter 'i'", bufferInvariant)

    let boundInvariant = leanIrSpec:
      proc loopAdd(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "i < n".} =
        for i in 0 ..< 4:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported invariant payload 'i < n' - the loop-shape " &
      "invariant may only mention the loop counter 'i'", boundInvariant)
    echo "  refusals on loop-shape pragma payloads: ok"

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
