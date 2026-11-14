## Shared helpers for the Lean verification suites: compile-time CIRv2 capture,
## `lake env lean` elaboration and execution, refusal assertions

import std/[macros, os, osproc, strutils]
import workspace/crucible/src/builtins/builtins
import workspace/crucible/src/builtins/builtins_compilermagic
import workspace/crucible/src/codegen/frontend/nim_to_crucible
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/lang_lean
import workspace/crucible/src/codegen/backends/print_common

export cirv2_lang, lang_lean, builtins

const leanProjectDir* = currentSourcePath().parentDir()

type LeanIr* = tuple[text: string, refusal: string]
  ## Compile result of `leanIr`: the emitted module text, or the caught
  ## `lean_lang: ...` refusal message with empty text

macro leanIr*(body: typed): LeanIr =
  ## Lowers the body through the Lean pass pipeline and emits its module text
  ## at compile time. A shape refusal surfaces as the `refusal` field instead
  ## of a compile error, so the test suite can assert on refusal messages.
  ##
  ## Returns `(text, refusal)`: exactly one field is non-empty
  crucibleCompileTarget = ctLean
  var reg = PassRegistry()
  reg.registerLeanPasses()
  let m = lowerCirv2(body)
  reg.runPasses(m)
  try:
    result = newLit((text: codegen(m), refusal: ""))
  except ValueError as e:
    result = newLit((text: "", refusal: e.msg))

type LeanIrSpec* = tuple[text: string, refusal: string, scaffolds: seq[ScaffoldData]]
  ## Compile result of `leanIrSpec`: base module text plus prepared scaffolds,
  ## or the caught `lean_lang: ...` refusal message with empty text

proc statementSpecPragma(node: NimNode): string =
  ## Name of the first verification pragma at statement level inside `node`,
  ## or "" when none.
  ##
  ## Captured at proc level only:
  ## - crucible's frontend maps a statement pragma to a discarded statement
  ## - a misplaced `.invariant.` would vanish from the emitted module
  case node.kind
  of nnkPragma, nnkPragmaExpr:
    let pragma = if node.kind == nnkPragmaExpr: node[0] else: node
    for item in pragma:
      let nameNode = if item.kind in {nnkExprColonExpr, nnkCall}: item[0] else: item
      if nameNode.kind in {nnkIdent, nnkSym} and
          nameNode.strVal in specPragmaNames:
        return nameNode.strVal
    if node.kind == nnkPragmaExpr:
      result = statementSpecPragma(node[1])
  else:
    for child in node:
      let found = statementSpecPragma(child)
      if found != "":
        return found

macro leanIrSpec*(body: typed): LeanIrSpec =
  ## Like `leanIr`, plus capture of the verification pragmas on the kernel proc.
  ## Evaluates to the tuple `(text, refusal, scaffolds)`.
  ##
  ## - `scaffolds` contains the prepared scaffold data for `settleModule`
  ## - crucible's frontend refuses verification pragmas, so the capture strips them before the conversion
  ##
  ## Verification pragmas sit beside `global` on the kernel proc, each with a string payload.
  ## A multi-proc block, a misplaced statement-level pragma, a non-string payload
  ## and a duplicated pragma surface as the `refusal` field, so the checks can
  ## assert on each message.
  var refusal = ""
  var procNode: NimNode
  if body.kind == nnkStmtList and body.len != 1:
    refusal = "lean_lang: expected exactly one kernel proc" &
      " in the guarded block, got " & $body.len & " statements"
  else:
    procNode = if body.kind == nnkStmtList: body[0] else: body
    if procNode.kind != nnkProcDef:
      refusal = "lean_lang: expected a kernel proc, got node kind " & $procNode.kind
    else:
      let misplaced = statementSpecPragma(procNode.body)
      if misplaced != "":
        refusal = "lean_lang: unsupported statement-level '" &
          misplaced & "' pragma inside the kernel body - verification pragmas" &
          " are captured at proc level only"
  if refusal == "":
    var specs: seq[RawSpec]
    var kept = newNimNode(nnkPragma, procNode.pragma)
    for item in procNode.pragma:
      if item.kind in {nnkExprColonExpr, nnkCall} and
          item[0].kind in {nnkIdent, nnkSym} and
          item[0].strVal in specPragmaNames:
        if item[1].kind != nnkStrLit:
          refusal = "lean_lang: pragma '" &
            item[0].strVal & "' payload must be a string literal"
          break
        var dup = false
        for existing in specs:
          if existing.name == item[0].strVal:
            refusal = "lean_lang: more than one '" &
              item[0].strVal & "' pragma on the kernel"
            dup = true
        if dup:
          break
        specs.add RawSpec(name: item[0].strVal, payload: item[1].strVal)
      else:
        kept.add item
    if refusal == "":
      let stripped = copyNimTree(procNode)
      # an empty nnkPragma trips crucible's `pragma.len > 0` assert at expansion.
      # keeping one empty pragma node also keeps a missing-`{.global.}` refusal loud
      stripped.pragma = if kept.len == 0: newEmptyNode() else: kept
      crucibleCompileTarget = ctLean
      var reg = PassRegistry()
      reg.registerLeanPasses()
      let m = lowerCirv2(newStmtList(stripped))
      reg.runPasses(m)
      try:
        let kernels = m.getModuleGlobalFns()
        let text = codegen(m)
        let scaffolds = prepareScaffolds(kernels[0], specs)
        result = newLit((text: text, refusal: "", scaffolds: scaffolds))
      except ValueError as e:
        result = newLit((text: "", refusal: e.msg, scaffolds: newSeq[ScaffoldData](0)))
      return
  result = newLit((text: "", refusal: refusal, scaffolds: newSeq[ScaffoldData](0)))

proc expectRefusal*(prefix: string, body: proc) =
  ## Runs `body`, expects the `lean_lang: ...` ValueError whose message
  ## starts with `prefix`.
  ##
  ## - Precondition: `prefix` is the refusal's documented reason, a refusal
  ##   raised for the wrong reason fails with expected and got in the message
  ## - Raises: AssertionError when no ValueError is raised, with the expected
  ##   prefix in the message
  try:
    body()
  except ValueError as e:
    doAssert e.msg.startsWith(prefix),
      "wrong refusal reason, expected '" & prefix & "', got: " & e.msg
    return
  doAssert false, "expected a refusal: " & prefix

proc expectLeanRefusal*(prefix: string, ir: LeanIr) =
  ## Expects the compile-time `lean_lang: ...` refusal captured in `ir`,
  ## asserting the message starts with `prefix`
  doAssert ir.refusal.startsWith(prefix),
    "wrong refusal reason, expected '" & prefix & "', got: " & ir.refusal
  doAssert ir.text.len == 0, "a refused kernel emits no module text"

proc expectLeanRefusal*(prefix: string, ir: LeanIrSpec) =
  ## Expects the compile-time `lean_lang: ...` refusal captured in `ir`,
  ## asserting the message starts with `prefix`
  doAssert ir.refusal.startsWith(prefix),
    "wrong refusal reason, expected '" & prefix & "', got: " & ir.refusal
  doAssert ir.text.len == 0, "a refused kernel emits no module text"
  doAssert ir.scaffolds.len == 0, "a refused kernel prepares no scaffolds"

proc elaborate*(relPath: string): tuple[output: string, exitCode: int] =
  ## Elaborates one emitted Lean module through the lake environment,
  ## warnings escalated to errors so a sorry fails
  execCmdEx("lake env lean -DwarningAsError=true " & relPath,
    workingDir = leanProjectDir)

proc execute*(relPath: string): tuple[output: string, exitCode: int] =
  ## Runs one emitted Lean module's `main` through the lake environment
  execCmdEx("lake env lean --run " & relPath,
    workingDir = leanProjectDir)
