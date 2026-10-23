# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Verification pragma extraction to Lean spec scaffolds
##
## Test that pragma payloads extract into Lean scaffolds with an honest
## proven/unproven verdict:
##
## - correct properties elaborate under warningAsError, recorded proven
## - a wrong property is downgraded to a sorry-backed scaffold, recorded unproven
##   in the manifest with its Lean error
## - wrong modules fail the warningAsError elaboration, out-of-scope payloads are refused loudly
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[assertions, json, os, osproc, strutils, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/frontend/lean_pragmas
import workspace/crucible/src/codegen/backends/lean_lang
import workspace/crucible/src/codegen/frontend/nim_to_lean
import lean_testutils

const genSpecRelPath = leanProjectDir & "/generated/add_spec.lean"
const genWrongPostRelPath = leanProjectDir & "/generated/add_spec_wrong_post.lean"
const genWrongInvRelPath = leanProjectDir & "/generated/add_spec_wrong_inv.lean"

proc scaffoldEntry(manifest: JsonNode, kind: string): JsonNode =
  ## Manifest scaffold entry of the given kind
  for e in manifest["scaffolds"].items:
    if e["kind"].getStr() == kind:
      return e
  doAssert false, "no " & kind & " scaffold in the manifest"

proc runTest() =
  block: # correct pragmas: proven manifest, module elaborates under warningAsError
    let k = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "tid < n", invariant: "tid < n", postcondition: "output tid = a tid + b tid".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]

    let emitted = emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, genSpecRelPath)
    doAssert "theorem addBody_invariant" in emitted, "invariant scaffold emitted"
    doAssert "theorem addBody_postcondition" in emitted, "postcondition scaffold emitted"
    doAssert "sorry" notin emitted, "no scaffold needed the downgrade"

    let manifest = parseJson(readFile(genSpecRelPath.changeFileExt("manifest.json")))
    doAssert manifest["kernel"].getStr() == "addKernel", "manifest names the kernel"
    doAssert manifest["module"].getStr() == "generated/add_spec.lean"
    let pre = scaffoldEntry(manifest, "precondition")
    doAssert pre["payload"].getStr() == "tid < n"
    doAssert pre["proven"].kind == JNull, "a precondition is assumed, not proved"
    let inv = scaffoldEntry(manifest, "invariant")
    doAssert inv["name"].getStr() == "addBody_invariant"
    doAssert inv["proven"].getBool(), "guard invariant proved by the guard"
    doAssert inv["error"].kind == JNull
    let post = scaffoldEntry(manifest, "postcondition")
    doAssert post["payload"].getStr() == "output tid = a tid + b tid"
    doAssert post["name"].getStr() == "addBody_postcondition"
    doAssert post["proven"].getBool(), "add-shape postcondition proved by simp"

    let (outp, code) = elaborate("generated/add_spec.lean")
    doAssert code == 0, "Lean rejected the proven module:\n" & outp
    doAssert "sorry" notin outp, "no sorry under warningAsError"
    echo "  correct pragmas: proven manifest + warningAsError elaboration: ok"

  block: # wrong postcondition: sorry downgrade, unproven manifest, warningAsError rejection
    let k = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "tid < n", postcondition: "output tid = a tid + b tid + 1".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]

    discard emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, genWrongPostRelPath)
    let manifest = parseJson(readFile(genWrongPostRelPath.changeFileExt("manifest.json")))
    let post = scaffoldEntry(manifest, "postcondition")
    doAssert post["proven"].getBool() == false, "wrong postcondition recorded unproven"
    doAssert post["error"].getStr().len > 0, "Lean error excerpt recorded"
    echo "  wrong postcondition manifest entry: ", post.pretty

    let emitted = readFile(genWrongPostRelPath)
    doAssert "sorry" in emitted, "proof downgraded to a sorry-backed scaffold"
    let (_, code) = elaborate("generated/add_spec_wrong_post.lean")
    doAssert code != 0, "the sorry-backed module must fail under warningAsError"
    echo "  wrong postcondition: unproven manifest + warningAsError rejection: ok"

  block: # wrong invariant (not implied by the guard): same downgrade + rejection
    let k = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid ≤ n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]

    discard emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, genWrongInvRelPath)
    let manifest = parseJson(readFile(genWrongInvRelPath.changeFileExt("manifest.json")))
    let inv = scaffoldEntry(manifest, "invariant")
    doAssert inv["proven"].getBool() == false, "wrong invariant recorded unproven"
    doAssert inv["error"].getStr().len > 0, "Lean error excerpt recorded"
    let (_, code) = elaborate("generated/add_spec_wrong_inv.lean")
    doAssert code != 0, "the sorry-backed module must fail under warningAsError"
    echo "  wrong invariant: unproven manifest + warningAsError rejection: ok"

  block: # the `≠` comparator round-trips through the payload grammar
    let e = parseSpecPayload("output tid ≠ a tid + b tid")
    doAssert e.kind == seBin and e.op == "≠", "`≠` is recognized as the comparator"
    doAssert e.left.kind == seApp and e.left.fn == "output"
    doAssert e.right.kind == seBin and e.right.op == "+"
    doAssert specRefs(e) == @["output", "a", "b"]
    var nameMap: Table[string, string]
    nameMap["output"] = "addBody tid n a b out"
    nameMap["a"] = "a"
    nameMap["b"] = "b"
    doAssert renderSpec(e, nameMap) == "addBody tid n a b out tid ≠ a tid + b tid"
    echo "  `≠` comparator round-trip through the payload grammar: ok"

  block: # payload grammar: ∧ chain, uint literals, buffer applications, specRefs
    let e = parseSpecPayload("n ∧ tid < 3 ∧ a tid ≤ b tid + 7")
    doAssert e.kind == seBin and e.op == "∧", "the ∧ chain is one left-associative tree"
    doAssert e.left.op == "∧" and e.left.left.kind == seIdent and e.left.left.name == "n"
    doAssert e.left.right.op == "<" and e.left.right.left.name == "tid"
    doAssert e.left.right.right.kind == seUint and e.left.right.right.value == "3"
    doAssert e.right.op == "≤" and e.right.left.kind == seApp and e.right.left.fn == "a"
    doAssert e.right.right.op == "+"
    doAssert e.right.right.left.kind == seApp and e.right.right.left.fn == "b"
    doAssert e.right.right.right.kind == seUint and e.right.right.right.value == "7"
    doAssert specRefs(e) == @["n", "tid", "a", "b"], "refs in first-occurrence order"
    echo "  payload grammar: ∧ chain, uint literals, applications: ok"

  block: # payload grammar precedence: sum over comparator over ∧, left-associative sums
    let chain = parseSpecPayload("tid < 3 ∧ 4 ≤ n")
    doAssert chain.kind == seBin and chain.op == "∧", "comparators bind tighter than ∧"
    doAssert chain.left.op == "<" and chain.left.left.name == "tid"
    doAssert chain.left.right.kind == seUint and chain.left.right.value == "3"
    doAssert chain.right.op == "≤" and chain.right.left.value == "4"
    doAssert chain.right.right.name == "n"
    let cmp = parseSpecPayload("a tid + 7 ≤ b tid")
    doAssert cmp.kind == seBin and cmp.op == "≤", "the sum binds tighter than the comparator"
    doAssert cmp.left.op == "+" and cmp.left.left.fn == "a" and cmp.left.right.value == "7"
    doAssert cmp.right.kind == seApp and cmp.right.fn == "b"
    let sum = parseSpecPayload("a tid + 7 + 9")
    doAssert sum.op == "+" and sum.left.op == "+" and sum.left.right.value == "7"
    doAssert sum.right.value == "9", "left-associative sums"
    echo "  payload grammar precedence: cmp < ∧, sum < cmp, left-assoc: ok"

  block: # renderSpec maps the loop counter through bareMap, buffer reads stay `tid` applications
    var bareMap: Table[string, string]
    bareMap["i"] = "4 - k"
    var nameMap: Table[string, string]
    nameMap["a"] = "a"
    nameMap["b"] = "b"
    let e = parseSpecPayload("i < 3 ∧ a tid ≤ b tid + 7")
    doAssert renderSpec(e, nameMap, bareMap) == "4 - k < 3 ∧ a tid ≤ b tid + 7"
    echo "  renderSpec: loop counter over the remaining count, reads at `tid`: ok"

  block: # payload grammar refusals: stray character, trailing input, unrenderable terms
    expectRefusal("lean_lang: invalid spec payload 'tid < n - 1' - unexpected character '-'", proc() =
      discard parseSpecPayload("tid < n - 1"))
    expectRefusal("lean_lang: invalid spec payload 'tid < n n' - trailing input at token 'n'", proc() =
      discard parseSpecPayload("tid < n n"))
    var nameMap: Table[string, string]
    nameMap["a"] = "a"
    expectRefusal("lean_lang: invalid spec payload - 'a' must be applied to `tid`", proc() =
      discard renderSpec(parseSpecPayload("a"), nameMap))
    expectRefusal("lean_lang: invalid spec payload - 'x' is neither `tid`, `n` nor a kernel buffer parameter", proc() =
      discard renderSpec(parseSpecPayload("x"), nameMap))
    echo "  payload grammar refusals: stray character, trailing input, bare terms: ok"

  block: # a position-less lean error line refuses the settling, it cannot be misattributed
    # a lean run reporting one scaffold error plus one `file: error:` line
    # with no file:line position would otherwise:
    # - downgrade the scaffold
    # - keep the position-less error in the final module
    # - record the other scaffolds `proven` over an unelaborating module
    # A fake `lake` on PATH reproduces the diagnostic mix
    let negDir = leanProjectDir & "/generated/negativelake"
    createDir(negDir)
    writeFile(negDir & "/lakefile.lean", readFile(leanProjectDir & "/lakefile.lean"))
    writeFile(negDir & "/lean-toolchain", readFile(leanProjectDir & "/lean-toolchain"))
    let shimDir = negDir & "/bin"
    createDir(shimDir)
    let shim = shimDir & "/lake"
    writeFile(shim, """#!/bin/sh
file="$3"
line=$(grep -n "theorem .*_invariant" "$file" | head -1 | cut -d: -f1)
echo "$file:$line:3: error: tactic 'exact h' failed"
echo "$file: error: unknown constant 'add_spec_neg.settling'"
exit 1
""")
    setFilePermissions(shim, {fpUserRead, fpUserWrite, fpUserExec,
      fpGroupRead, fpGroupExec, fpOthersRead, fpOthersExec})
    let savedPath = getenv("PATH")
    putEnv("PATH", shimDir & ":" & savedPath)
    let negOutPath = negDir & "/generated/add_spec_neg.lean"
    removeFile(negOutPath)
    removeFile(negOutPath.changeFileExt("settling.lean"))
    removeFile(negOutPath.changeFileExt("manifest.json"))
    let k = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: proof settling failed - 1 lean error line(s) carry no " &
            "file:line position", proc() =
          discard emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, negOutPath))
    putEnv("PATH", savedPath)
    doAssert not fileExists(negOutPath),
      "no unverified first-pass module may stay on disk after the refusal"
    doAssert not fileExists(negOutPath.changeFileExt("settling.lean")),
      "no settling module may stay on disk after the refusal"
    doAssert not fileExists(negOutPath.changeFileExt("manifest.json")),
      "no manifest may record scaffold verdicts over an unsettled module"
    echo "  proof settling refusal on a position-less lean error line: ok"

  block: # the lean diagnostic extractor separates positioned from position-less errors
    let diags = parseLeanDiagnostics("""
generated/add_spec.lean:40:3: error: tactic 'exact h' failed
generated/add_spec.lean: error: unknown constant 'foo'
"""")
    doAssert diags.len == 2, "both error lines extracted"
    doAssert diags[0].line == 40 and diags[0].excerpt == "tactic 'exact h' failed"
    doAssert diags[1].line == -1, "a `file: error:` line has no position"
    doAssert diags[1].excerpt == "unknown constant 'foo'"
    echo "  diagnostic extractor: positioned and position-less errors: ok"

  block: # proof settling refuses loudly when lean fails without attributable errors
    # a broken toolchain makes `lake env lean` exit nonzero, printing a lake-level error with no `file:line:col: error:` line
    # no error is attributable to a scaffold, so the proofs were never checked
    # and no manifest may record a `proven` verdict
    let negDir = leanProjectDir & "/generated/negativelake"
    createDir(negDir)
    writeFile(negDir & "/lakefile.lean", readFile(leanProjectDir & "/lakefile.lean"))
    writeFile(negDir & "/lean-toolchain", "leanprover/lean4:v0.0.0-notinstalled")
    let negOutPath = negDir & "/generated/add_spec_neg.lean"
    removeFile(negOutPath)
    removeFile(negOutPath.changeFileExt("settling.lean"))
    removeFile(negOutPath.changeFileExt("manifest.json")) # stale runs
    let k = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: proof settling failed - lean exited with 1 but no error was " &
            "attributable to a scaffold: ", proc() =
          discard emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, negOutPath))
    doAssert not fileExists(negOutPath),
      "no unverified first-pass module may stay on disk after the refusal"
    doAssert not fileExists(negOutPath.changeFileExt("settling.lean")),
      "no settling module may stay on disk after the refusal"
    doAssert not fileExists(negOutPath.changeFileExt("manifest.json")),
      "no manifest may record a proven verdict for unchecked proofs"
    echo "  proof settling refusal on an unattributable lean failure: ok"

  block: # loud refusals on out-of-scope payloads
    let unparseable = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "output tid =".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: invalid spec payload - expected an identifier or a uint " &
            "literal, got ''", proc() =
          discard emitLeanModuleWithSpecs(unparseable.gpu.statements[0], unparseable.specs, genSpecRelPath))

    let unknownIdent = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "bogus tid = a tid".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: invalid spec payload - 'bogus tid' is not a read of a kernel " &
            "buffer parameter", proc() =
          discard emitLeanModuleWithSpecs(unknownIdent.gpu.statements[0], unknownIdent.specs, genSpecRelPath))

    let badPrecondition = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "n < tid".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported precondition payload 'n < tid' - the tracer " &
            "model's launch precondition is `tid < n`", proc() =
          discard emitLeanModuleWithSpecs(badPrecondition.gpu.statements[0], badPrecondition.specs, genSpecRelPath))

    let bufferInvariant = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "a tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported invariant payload 'a tid < n' - the tracer " &
            "invariant must be implied by the guard `tid < n` and mention both " &
            "`tid` and `n`", proc() =
          discard emitLeanModuleWithSpecs(bufferInvariant.gpu.statements[0], bufferInvariant.specs, genSpecRelPath))

    let plain = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported spec pragma 'sidecondition'", proc() =
          discard emitLeanModuleWithSpecs(plain.gpu.statements[0],
            @[RawSpec(name: "sidecondition", payload: "tid < n")], genSpecRelPath))
    echo "  refusals on out-of-scope payloads: ok"

  block: # a pragma-less proc (no `{.global.}`) is refused loudly
    # a proc without `{.global.}` is outside the tracer's scope: the macro hands
    # crucible an empty pragma node and `analyzeKernel` refuses through the documented channel
    let plain = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported GpuAst node gpuProc - kernel is not `{.global.}`", proc() =
          discard emitLeanModuleWithSpecs(plain.gpu.statements[0], plain.specs, genSpecRelPath))
    echo "  refusal on a pragma-less (not `{.global.}`) kernel proc: ok"

  block: # a multi-kernel block is refused loudly at expansion time
    # only the first proc of a statement list would be converted,
    # silently dropping the others and their verification pragmas
    static:
      doAssert not compiles(block:
        let k = irLeanSpec:
          proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
            let i = thread_position_in_grid.x
            if i < 2:
              output[i] = a[i] + b[i]
          proc mulKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
            let i = thread_position_in_grid.x
            if i < 2:
              output[i] = a[i] + b[i]
      )
    echo "  refusal on a multi-kernel block: ok"

  block: # a statement-level verification pragma is refused loudly at expansion time
    # crucible's frontend discards statement pragmas, so without the refusal
    # the payload would silently vanish from the emitted module.
    # `compiles` cannot check the refusal message, only that expansion fails.
    static:
      doAssert not compiles(block:
        let k = irLeanSpec:
          proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
            let i = thread_position_in_grid.x
            {.invariant: "tid < n".}
            if i < 2:
              output[i] = a[i] + b[i]
      )
    echo "  refusal on a statement-level verification pragma: ok"


  block: # refusal on a duplicate pragma of one kind
    let dup = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: more than one 'invariant' pragma on the kernel", proc() =
          discard emitLeanModuleWithSpecs(dup.gpu.statements[0], @[
            RawSpec(name: "invariant", payload: "tid < n"),
            RawSpec(name: "invariant", payload: "tid < n")], genSpecRelPath))
    echo "  refusal on a duplicate pragma of one kind: ok"

  block: # refusal on a postcondition not mentioning the written buffer
    let badPost = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: unsupported postcondition payload 'tid < n' - it must " &
            "mention the written buffer 'output'", proc() =
          discard emitLeanModuleWithSpecs(badPost.gpu.statements[0], badPost.specs, genSpecRelPath))
    echo "  refusal on a postcondition not mentioning the written buffer: ok"

  block: # refusal on an emission outside any lake project
    let orphanDir = getTempDir() & "leanlang_orphan/generated"
    createDir(orphanDir)
    let orphan = irLeanSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectRefusal("lean_lang: no lakefile.lean above the emitted module", proc() =
          discard emitLeanModuleWithSpecs(orphan.gpu.statements[0], orphan.specs,
            orphanDir & "/add_orphan.lean"))
    echo "  refusal on an emission outside any lake project: ok"

when isMainModule:
  runTest()
  echo "t_lean_pragmas: all checks passed"
