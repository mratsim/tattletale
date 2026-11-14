## Lean backend emitter, end-to-end
##
## Test that the emitter byte-matches the hand-written AddKernel.lean statements and elaborates green
##
## Run:
##   cd tattletale && nim test_crucible_lean

import std/[os, strutils]
import workspace/crucible/src/codegen/gpu_compiler
import lean_testutils

const genDir = leanProjectDir & "/generated"

proc statementTail(path: string): string =
  ## Everything from the per-thread def doc comment onward, byte-exact:
  ## the def and the postcondition theorems
  let text = readFile(path)
  let idx = text.find("/-- Per-thread body")
  doAssert idx >= 0, "per-thread def doc comment not found in " & path
  result = text[idx .. ^1]
  result = strip(result, leading = false)

proc referenceStatements(path: string): string =
  ## Def and postcondition theorems of the hand-written reference module,
  ## byte-exact. The emitted module stops after the two postcondition theorems.
  ## The reference module continues with the derived contract theorems
  let text = readFile(path)
  let start = text.find("/-- Per-thread body")
  let stop = text.find("/-- The per-thread postcondition")
  doAssert start >= 0 and stop > start, "reference markers not found in " & path
  result = text[start ..< stop]
  result = strip(result, leading = false)

proc runTest() =
  block: # emission + Lean elaboration + statement match against the hand-written reference
    let ir = leanIr:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssert ir.refusal.len == 0, "unexpected refusal: " & ir.refusal
    let hasBodyDef = "def addBody " in ir.text
    doAssert hasBodyDef, "per-thread def emitted"
    createDir(genDir)
    writeFile(genDir / "add.autogen.lean", ir.text)

    let (outp, code) = elaborate("generated/add.autogen.lean")
    doAssert code == 0, "Lean rejected the emitted module:\n" & outp
    let noSorry = "sorry" notin outp
    doAssert noSorry, "no sorry under warningAsError"

    let expected = referenceStatements(leanProjectDir & "/AddKernel.lean")
    let generated = statementTail(genDir / autogenModule("addKernel"))
    if expected != generated:
      let refLines = expected.splitLines()
      let genLines = generated.splitLines()
      for i in 0 ..< max(refLines.len, genLines.len):
        let want = if i < refLines.len: refLines[i] else: "<missing>"
        let got = if i < genLines.len: genLines[i] else: "<missing>"
        if want != got:
          echo "line ", i + 1, ":\n  reference: ", want, "\n  generated: ", got
      doAssert false, "emitted statements differ from the hand-written reference module"
    echo "  emission + elaboration (warningAsError) + reference statement match: ok"

  block: # `lean:` macro and leanIr test path emit the same text
    const macroModule = lean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    let macroMatchesFile = macroModule == readFile(genDir / autogenModule("addKernel"))
    doAssert macroMatchesFile,
      "the lean: macro emits the same module text as the leanIr test path"
    echo "  lean: macro path matches the runtime test path: ok"

  block: # loud refusal on an out-of-scope construct (while loop)
    # a `for` over a compile-time constant range is the loop shape.
    # t_lean_loop.nim tests it. A while loop stays out of scope
    let ir = leanIr:
      proc whileAdd(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        var i = 0
        while i < 4:
          output[i] = a[i] + b[i]
          i = i + 1
    expectLeanRefusal("lean_lang: unsupported statement sVar - local binding whose" &
      " initializer is not `thread_position_in_grid.x`",
      ir)
    echo "  refusal on out-of-scope construct: ok"

  block: # loud refusal on a buffer-parameter set other than the add shape
    let copy = leanIr:
      proc copyKernel(dst, src: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          dst[i] = src[i]
    expectLeanRefusal("lean_lang: unsupported kernel - buffer parameters must be exactly" &
        " `a` and `b` (read) plus `output` (written)",
      copy)

    let renamed = leanIr:
      proc renamedKernel(dst, x, y: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          dst[i] = x[i] + y[i]
    expectLeanRefusal("lean_lang: unsupported kernel - buffer parameters must be exactly" &
        " `a` and `b` (read) plus `output` (written)",
      renamed)

    let unreadInput = leanIr:
      proc copyA(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i]
    expectLeanRefusal("lean_lang: unsupported kernel - the kernel body must read" &
        " both input buffers `a` and `b`",
      unreadInput)
    echo "  refusal on kernels outside the add-shape buffer parameters: ok"

  block: # loud refusal on a non-positive guard literal
    let zeroBound = leanIr:
      proc zeroBoundKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 0:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported guard - literal must be positive (got 0)",
      zeroBound)

    let negativeBound = leanIr:
      proc negativeBoundKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < -1:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported guard - literal must be positive (got -1)",
      negativeBound)
    echo "  refusal on non-positive guard literals: ok"

  block: # loud refusal on nested indexing
    # Model scope: reads at the thread's own index only. A nested index
    # (`a[a[i]]`) would otherwise produce an unelaboratable Lean term
    let nested = leanIr:
      proc nestedKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[a[i]] + b[i]
    expectLeanRefusal("lean_lang: unsupported expression ePat - index expression is" &
        " not the index variable",
      nested)
    echo "  refusal on nested indexing: ok"

  block: # refusal on a kernel body with no buffer write
    let noWrite = leanIr:
      proc noWriteKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
    expectLeanRefusal("lean_lang: unsupported assignment - no buffer write found in" &
      " the kernel body",
      noWrite)
    echo "  refusal on a kernel body with no buffer write: ok"

  block: # refusal on a guard outside the `tidVar < literal` shape
    let leGuard = leanIr:
      proc leGuardKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i <= 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported guard - not of the shape `tidVar < literal`",
      leGuard)
    echo "  refusal on a guard outside the `tidVar < literal` shape: ok"

  block: # refusal on more than one write per thread
    let twoWrites = leanIr:
      proc twoWritesKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
          output[i] = b[i]
    expectLeanRefusal("lean_lang: unsupported assignment - more than one write per thread",
      twoWrites)
    echo "  refusal on more than one write per thread: ok"

  block: # refusal on a second local binding
    let twoBindings = leanIr:
      proc twoBindingsKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        let j = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported statement sVar - more than one local binding",
      twoBindings)
    echo "  refusal on a second local binding: ok"

  block: # refusal on a write index other than the thread id
    let wrongIndex = leanIr:
      proc wrongIndexKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i + 0] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported assignment - write index is not the thread id",
      wrongIndex)
    echo "  refusal on a write index other than the thread id: ok"

  block: # refusal on a non-assign statement inside the guarded branch
    let nestedIf = leanIr:
      proc nestedIfKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          if i < 1:
            output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported statement sIte - statement inside the" &
      " guarded branch",
      nestedIf)
    echo "  refusal on a non-assign statement inside the guarded branch: ok"

  block: # refusal on an if before the thread-id binding
    let earlyIf = leanIr:
      proc earlyIfKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        if 0 < 1:
          output[0] = a[0] + b[0]
        let i = thread_position_in_grid.x
    expectLeanRefusal("lean_lang: unsupported statement sIte - if before the thread-id" &
      " binding",
      earlyIf)
    echo "  refusal on an if before the thread-id binding: ok"

  block: # refusal on elif/else branches
    let withElse = leanIr:
      proc withElseKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
        else:
          output[i] = a[i]
    expectLeanRefusal("lean_lang: unsupported statement sIte - elif/else branches" &
      " (the model kernel is a bare if)",
      withElse)
    echo "  refusal on elif/else branches: ok"

  block: # refusal on a let binding that is not the thread id
    let plainLet = leanIr:
      proc plainLetKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = 4
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported statement sVar - local binding whose" &
      " initializer is not `thread_position_in_grid.x`",
      plainLet)
    echo "  refusal on a let binding that is not the thread id: ok"

  block: # refusal on a discard statement at kernel-body top level
    let noBinding = leanIr:
      proc noBindingKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        discard
    expectLeanRefusal("lean_lang: unsupported statement sDiscard - statement at" &
      " kernel-body top level",
      noBinding)
    echo "  refusal on a discard statement at kernel-body top level: ok"

  block: # refusal on a scalar parameter on the guarded-thread shape
    let scalar = leanIr:
      proc scalarKernel(output, a, b: ptr UncheckedArray[uint32], k: uint32) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported kernel - scalar parameter 'k' on the" &
      " guarded-thread shape",
      scalar)
    echo "  refusal on a scalar parameter on the guarded-thread shape: ok"

  block: # refusal on a written buffer other than output
    let writesA = leanIr:
      proc writesAKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          a[i] = a[i] + b[i]
    expectLeanRefusal("lean_lang: unsupported assignment - the written buffer" &
      " must be `output` - got 'a'", writesA)
    echo "  refusal on a written buffer other than output: ok"

  block: # refusal on arithmetic outside the supported operators
    let sub = leanIr:
      proc subKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] - b[i]
    expectLeanRefusal("lean_lang: unsupported expression eSub",
      sub)

    let mul = leanIr:
      proc mulKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] * b[i]
    expectLeanRefusal("lean_lang: unsupported expression eMul - the multiply is in" &
      " scope for the axpy shape only",
      mul)
    echo "  refusal on arithmetic outside the supported operators: ok"

when isMainModule:
  runTest()
  echo "t_lean_emit: all checks passed"
