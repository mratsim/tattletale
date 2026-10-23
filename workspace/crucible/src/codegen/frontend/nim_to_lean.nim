# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim_to_lean: Nim kernel proc -> post-pass GpuAst for the Lean verification printer
##
##   Nim kernel proc -> irLean/irLeanSpec -> post-pass GpuAst -> backends/lean_lang.nim
##
## Runs the common pass set plus explicit pointer derefs over the kernel body,
## the canonical buffer-access form of the analysis model.
##
## Backend bundles are excluded on purpose. Each backend bundle renames
## source identifiers to satisfy its target's reserved words, so the GLSL
## bundle renames a parameter the kernel names `output` to `output_nim`.
## The model reasons over the kernel's semantic names, not renamed ones.

import std/macros
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/backends/lean_lang

proc toPostPassAst(body: NimNode): GpuAst {.compileTime.} =
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerInsertPointerDeref()
  var typeReg = TypeRegistry(types: ctx.types)
  result = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  ctx.ingestTopLevel(result, typesFirst = true)
  runPasses(ctx, reg)

macro irLean*(body: typed): GpuAst =
  ## Evaluates to the post-pass GpuAst the Lean emitter consumes
  result = newLit(toPostPassAst(body))

proc statementSpecPragma(node: NimNode): string =
  ## Name of the first verification pragma at statement level inside `node`,
  ## or "" when none.
  ##
  ## Only proc-level pragmas are captured:
  ## - crucible's frontend maps a statement pragma to a discarded statement
  ## - a misplaced `.invariant.` would vanish from the emitted module
  case node.kind
  of nnkPragma, nnkPragmaExpr:
    let pragma = if node.kind == nnkPragmaExpr: node[0] else: node
    for item in pragma:
      let nameNode = if item.kind in {nnkExprColonExpr, nnkCall}: item[0] else: item
      if nameNode.kind in {nnkIdent, nnkSym} and
          nameNode.strVal in ["precondition", "invariant", "postcondition"]:
        return nameNode.strVal
    if node.kind == nnkPragmaExpr:
      result = statementSpecPragma(node[1])
  else:
    for child in node:
      let found = statementSpecPragma(child)
      if found != "":
        return found

macro irLeanSpec*(body: typed): untyped =
  ## Like `irLean`, plus capture of the verification pragmas on the kernel proc.
  ## Evaluates to the tuple `(specs, gpu)`.
  ##
  ## - `specs` carries the raw pragma payloads
  ## - `gpu` is the post-pass GpuAst of the same proc
  ## - Crucible's frontend refuses verification pragmas, so they are stripped
  ##   before the conversion
  ##
  ## The call shape is in `t_lean_pragmas.nim`.
  ## The verification pragmas sit beside `global` on the kernel proc, each carrying a string payload.
  ## `emitLeanModuleWithSpecs` turns them into Lean scaffolds.
  ##
  ## Payloads that are not string literals, and the same pragma twice,
  ## are refused loudly at expansion time.
  if body.kind == nnkStmtList and body.len != 1:
    error "nim_to_lean: expected exactly one kernel proc" &
      " in the guarded block, got " & $body.len & " statements"
  let procNode = if body.kind == nnkStmtList: body[0] else: body
  if procNode.kind != nnkProcDef:
    error "nim_to_lean: expected a kernel proc, got node kind " & $procNode.kind
  let misplaced = statementSpecPragma(procNode.body)
  if misplaced != "":
    error "nim_to_lean: unsupported statement-level '" &
      misplaced & "' pragma inside the kernel body - verification pragmas" &
      " are captured at proc level only"
  var specs: seq[RawSpec]
  var kept = newNimNode(nnkPragma, procNode.pragma)
  for item in procNode.pragma:
    if item.kind in {nnkExprColonExpr, nnkCall} and
        item[0].kind in {nnkIdent, nnkSym} and
        item[0].strVal in ["precondition", "invariant", "postcondition"]:
      if item[1].kind != nnkStrLit:
        error "nim_to_lean: pragma '" &
          item[0].strVal & "' payload must be a string literal"
      for existing in specs:
        if existing.name == item[0].strVal:
          error "nim_to_lean: more than one '" &
            item[0].strVal & "' pragma on the kernel"
      specs.add RawSpec(name: item[0].strVal, payload: item[1].strVal)
    else:
      kept.add item
  let stripped = copyNimTree(procNode)
  # an empty nnkPragma trips crucible's `pragma.len > 0` assert at expansion.
  # The empty node also keeps `analyzeKernel`'s own `{.global.}` refusal loud
  stripped.pragma = if kept.len == 0: newEmptyNode() else: kept
  let astIn = if body.kind == nnkStmtList: newStmtList(stripped) else: stripped
  let gpuAst = toPostPassAst(astIn)
  result = nnkPar.newTree(
    nnkExprColonExpr.newTree(ident"specs", newLit(specs)),
    nnkExprColonExpr.newTree(ident"gpu", newLit(gpuAst)))
