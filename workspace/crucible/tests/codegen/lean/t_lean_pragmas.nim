## Verification pragma extraction to Lean spec scaffolds
##
## Test that pragma payloads extract into scaffolds with an honest proven or unproven verdict
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[json, os, strutils, tables]
import lean_testutils

const genDir = leanProjectDir & "/generated"

proc expectField(node: JsonNode, want: string, what: string) =
  ## Asserts the manifest string field against the expected value
  let got = node.getStr()
  doAssert got == want, what & ": expected '" & want & "', got '" & got & "'"

proc runTest() =
  block: # correct pragmas: proven manifest, module elaborates under warningAsError
    let k = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "tid < n", invariant: "tid < n", postcondition: "output tid = a tid + b tid".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssert k.refusal.len == 0, "unexpected refusal: " & k.refusal
    let emitted = settleModule(k.text, k.scaffolds, genDir, "add_spec.lean")
    let hasInvariant = "theorem addBody_invariant" in emitted
    let hasPostcondition = "theorem addBody_postcondition" in emitted
    let noDowngrade = "sorry" notin emitted
    doAssert hasInvariant, "invariant scaffold emitted"
    doAssert hasPostcondition, "postcondition scaffold emitted"
    doAssert noDowngrade, "no scaffold needed the downgrade"

    let manifest = parseJson(readFile(genDir / "add_spec.lean".changeFileExt("manifest.json")))
    expectField(manifest["kernel"], "addKernel", "manifest names the kernel")
    expectField(manifest["module"], "generated/add_spec.lean", "module recorded")
    let pre = scaffoldEntry(manifest, "precondition")
    expectField(pre["payload"], "tid < n", "precondition payload")
    doAssert pre["proven"].kind == JNull, "a precondition is assumed, not proved"
    let inv = scaffoldEntry(manifest, "invariant")
    expectField(inv["name"], "addBody_invariant", "invariant name")
    doAssert inv["proven"].getBool(), "guard invariant proved by the guard"
    doAssert inv["error"].kind == JNull
    let post = scaffoldEntry(manifest, "postcondition")
    expectField(post["payload"], "output tid = a tid + b tid", "postcondition payload")
    expectField(post["name"], "addBody_postcondition", "postcondition name")
    doAssert post["proven"].getBool(), "add-shape postcondition proved by simp"

    let (outp, code) = elaborate("generated/add_spec.lean")
    doAssert code == 0, "Lean rejected the proven module:\n" & outp
    echo "  correct pragmas: proven manifest + warningAsError elaboration: ok"

  block: # wrong postcondition: sorry downgrade, unproven manifest, warningAsError rejection
    let k = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "tid < n", postcondition: "output tid = a tid + b tid + 1".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssert k.refusal.len == 0, "unexpected refusal: " & k.refusal
    discard settleModule(k.text, k.scaffolds, genDir, "add_spec_wrong_post.lean")
    let manifest = parseJson(readFile(genDir / "add_spec_wrong_post.lean".changeFileExt("manifest.json")))
    let post = scaffoldEntry(manifest, "postcondition")
    doAssert post["proven"].getBool() == false, "wrong postcondition recorded unproven"
    doAssert post["error"].getStr().len > 0, "Lean error excerpt recorded"

    let emitted = readFile(genDir / "add_spec_wrong_post.lean")
    let downgraded = "sorry" in emitted
    doAssert downgraded, "proof downgraded to a sorry-backed scaffold"
    let (_, code) = elaborate("generated/add_spec_wrong_post.lean")
    doAssert code != 0, "the sorry-backed module must fail under warningAsError"
    echo "  wrong postcondition: unproven manifest + warningAsError rejection: ok"

  block: # wrong invariant (not implied by the guard): same downgrade + rejection
    let k = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid ≤ n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssert k.refusal.len == 0, "unexpected refusal: " & k.refusal
    discard settleModule(k.text, k.scaffolds, genDir, "add_spec_wrong_inv.lean")
    let manifest = parseJson(readFile(genDir / "add_spec_wrong_inv.lean".changeFileExt("manifest.json")))
    let inv = scaffoldEntry(manifest, "invariant")
    doAssert inv["proven"].getBool() == false, "wrong invariant recorded unproven"
    doAssert inv["error"].getStr().len > 0, "Lean error excerpt recorded"
    let (_, code) = elaborate("generated/add_spec_wrong_inv.lean")
    doAssert code != 0, "the sorry-backed module must fail under warningAsError"
    echo "  wrong invariant: unproven manifest + warningAsError rejection: ok"

  block: # `≠` comparator round-trip through the payload grammar
    let e = parseSpecPayload("output tid ≠ a tid + b tid")
    let check111 = e.kind == seBin and e.op == "≠"
    doAssert check111, "`≠` is recognized as the comparator"
    let check113 = e.left.kind == seApp and e.left.fn == "output"
    doAssert check113
    let check115 = e.right.kind == seBin and e.right.op == "+"
    doAssert check115
    doAssert specRefs(e) == @["output", "a", "b"]
    var nameMap: Table[string, string]
    nameMap["output"] = "addBody tid n a b out"
    nameMap["a"] = "a"
    nameMap["b"] = "b"
    let check122 = renderSpec(e, nameMap) == "addBody tid n a b out tid ≠ a tid + b tid"
    doAssert check122
    echo "  `≠` comparator round-trip through the payload grammar: ok"

  block: # payload grammar: ∧ chain, uint literals, buffer applications, specRefs
    let e = parseSpecPayload("n ∧ tid < 3 ∧ a tid ≤ b tid + 7")
    let check128 = e.kind == seBin and e.op == "∧"
    doAssert check128, "the ∧ chain is one left-associative tree"
    let check130 = e.left.op == "∧" and e.left.left.kind == seIdent and e.left.left.name == "n"
    doAssert check130
    let check132 = e.left.right.op == "<" and e.left.right.left.name == "tid"
    doAssert check132
    let check134 = e.left.right.right.kind == seUint and e.left.right.right.value == "3"
    doAssert check134
    let check136 = e.right.op == "≤" and e.right.left.kind == seApp and e.right.left.fn == "a"
    doAssert check136
    let check138 = e.right.right.op == "+"
    doAssert check138
    let check140 = e.right.right.left.kind == seApp and e.right.right.left.fn == "b"
    doAssert check140
    let check142 = e.right.right.right.kind == seUint and e.right.right.right.value == "7"
    doAssert check142
    doAssert specRefs(e) == @["n", "tid", "a", "b"], "refs in first-occurrence order"
    echo "  payload grammar: ∧ chain, uint literals, applications: ok"

  block: # payload grammar precedence: sum over comparator over ∧, left-associative sums
    let chain = parseSpecPayload("tid < 3 ∧ 4 ≤ n")
    let check149 = chain.kind == seBin and chain.op == "∧"
    doAssert check149, "comparators bind tighter than ∧"
    let check151 = chain.left.op == "<" and chain.left.left.name == "tid"
    doAssert check151
    let check153 = chain.left.right.kind == seUint and chain.left.right.value == "3"
    doAssert check153
    let check155 = chain.right.op == "≤" and chain.right.left.value == "4"
    doAssert check155
    let check157 = chain.right.right.name == "n"
    doAssert check157
    let cmp = parseSpecPayload("a tid + 7 ≤ b tid")
    let check160 = cmp.kind == seBin and cmp.op == "≤"
    doAssert check160, "the sum binds tighter than the comparator"
    let check162 = cmp.left.op == "+" and cmp.left.left.fn == "a" and cmp.left.right.value == "7"
    doAssert check162
    let check164 = cmp.right.kind == seApp and cmp.right.fn == "b"
    doAssert check164
    let sum = parseSpecPayload("a tid + 7 + 9")
    let check167 = sum.op == "+" and sum.left.op == "+" and sum.left.right.value == "7"
    doAssert check167
    let check169 = sum.right.value == "9"
    doAssert check169, "left-associative sums"
    echo "  payload grammar precedence: cmp < ∧, sum < cmp, left-assoc: ok"

  block: # renderSpec maps the loop counter through bareMap, buffer reads stay `tid` applications
    var bareMap: Table[string, string]
    bareMap["i"] = "4 - k"
    var nameMap: Table[string, string]
    nameMap["a"] = "a"
    nameMap["b"] = "b"
    let e = parseSpecPayload("i < 3 ∧ a tid ≤ b tid + 7")
    let check180 = renderSpec(e, nameMap, bareMap) == "4 - k < 3 ∧ a tid ≤ b tid + 7"
    doAssert check180
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
    let k = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssert k.refusal.len == 0, "unexpected refusal: " & k.refusal
    expectRefusal("lean_lang: proof settling failed - 1 lean error line(s) carry no " &
            "file:line position", proc() =
          discard settleModule(k.text, k.scaffolds, negDir & "/generated", "add_spec_neg.lean"))
    putEnv("PATH", savedPath)
    doAssert not fileExists(negOutPath),
      "no unverified first-pass module may stay on disk after the refusal"
    doAssert not fileExists(negOutPath.changeFileExt("settling.lean")),
      "no settling module may stay on disk after the refusal"
    doAssert not fileExists(negOutPath.changeFileExt("manifest.json")),
      "no manifest may record scaffold verdicts over an unsettled module"
    echo "  proof settling refusal on a position-less lean error line: ok"

  block: # lean diagnostic extractor, positioned vs position-less errors
    let diags = parseLeanDiagnostics("""
generated/add_spec.lean:40:3: error: tactic 'exact h' failed
generated/add_spec.lean: error: unknown constant 'foo'
""")
    doAssert diags.len == 2, "both error lines extracted"
    let positionedOk = diags[0].line == 40 and diags[0].excerpt == "tactic 'exact h' failed"
    doAssert positionedOk
    doAssert diags[1].line == -1, "a `file: error:` line has no position"
    let positionlessOk = diags[1].excerpt == "unknown constant 'foo'"
    doAssert positionlessOk
    echo "  diagnostic extractor: positioned and position-less errors: ok"

  block: # proof settling refuses loudly when lean fails without attributable errors
    # a broken toolchain makes `lake env lean` exit nonzero.
    # one lake-level error prints, carrying no `file:line:col: error:` position.
    # no scaffold owns an error, the proofs went unchecked, and the manifest records no `proven` verdict
    let negDir = leanProjectDir & "/generated/negativelake"
    createDir(negDir)
    writeFile(negDir & "/lakefile.lean", readFile(leanProjectDir & "/lakefile.lean"))
    writeFile(negDir & "/lean-toolchain", "leanprover/lean4:v0.0.0-notinstalled")
    let negOutPath = negDir & "/generated/add_spec_neg.lean"
    removeFile(negOutPath)
    removeFile(negOutPath.changeFileExt("settling.lean"))
    removeFile(negOutPath.changeFileExt("manifest.json")) # stale runs
    let k = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssert k.refusal.len == 0, "unexpected refusal: " & k.refusal
    expectRefusal("lean_lang: proof settling failed - lean exited with 1 but no error was " &
            "attributable to a scaffold: ", proc() =
          discard settleModule(k.text, k.scaffolds, negDir & "/generated", "add_spec_neg.lean"))
    doAssert not fileExists(negOutPath),
      "no unverified first-pass module may stay on disk after the refusal"
    doAssert not fileExists(negOutPath.changeFileExt("settling.lean")),
      "no settling module may stay on disk after the refusal"
    doAssert not fileExists(negOutPath.changeFileExt("manifest.json")),
      "no manifest may record a proven verdict for unchecked proofs"
    echo "  proof settling refusal on an unattributable lean failure: ok"

  block: # loud refusals on out-of-scope payloads
    let unparseable = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "output tid =".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: invalid spec payload - expected an identifier or a uint " &
      "literal, got ''", unparseable)

    let unknownIdent = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "bogus tid = a tid".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: invalid spec payload - 'bogus tid' is not a read of a kernel " &
      "buffer parameter", unknownIdent)

    let badPrecondition = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "n < tid".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported precondition payload 'n < tid' - the " &
      "model's launch precondition is `tid < n`", badPrecondition)

    let bufferInvariant = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "a tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported invariant payload 'a tid < n' - the " &
      "invariant must be implied by the guard `tid < n` and mention both " &
      "`tid` and `n`", bufferInvariant)

    let badPost = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported postcondition payload 'tid < n' - it must " &
      "mention the written buffer 'output'", badPost)
    echo "  refusals on out-of-scope payloads: ok"

  block: # a pragma-less proc (no `{.global.}`) is refused loudly
    let plain = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported module - expected exactly one `{.global.}`" &
      " kernel, got 0", plain)
    echo "  refusal on a pragma-less (not `{.global.}`) kernel proc: ok"

  block: # capture refusals: multi-kernel block, statement-level pragma, duplicate pragma
    # only the first proc of a statement list would be converted, the others and their verification pragmas drop silently
    # crucible's frontend discards statement pragmas, the refusal keeps the payload from silently vanishing
    let multi = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
      proc mulKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: expected exactly one kernel proc in the" &
      " guarded block, got 2 statements", multi)

    let misplaced = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        {.invariant: "tid < n".}
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported statement-level 'invariant' pragma" &
      " inside the kernel body", misplaced)

    let dup = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid < n", invariant: "tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: more than one 'invariant' pragma on the kernel", dup)
    echo "  capture refusals: multi-kernel block, statement pragma, duplicate: ok"

  block: # refusal on an emission outside any lake project
    let orphanDir = getTempDir() & "leanlang_orphan/generated"
    createDir(orphanDir)
    let orphan = leanIrSpec:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssert orphan.refusal.len == 0, "unexpected refusal: " & orphan.refusal
    expectRefusal("lean_lang: no lakefile.lean above the emitted module", proc() =
      discard settleModule(orphan.text, orphan.scaffolds,
        orphanDir & "/add_orphan.lean".parentDir, "add_orphan.lean"))
    echo "  refusal on an emission outside any lake project: ok"

when isMainModule:
  runTest()
  echo "t_lean_pragmas: all checks passed"
