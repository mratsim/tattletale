# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim_to_lean: emit a Lean 4 verification module from crucible's post-pass GpuAst
##
## Walks the GpuAst of one `{.global.}` kernel and emits the per-thread body def plus
## derived spec theorems, in the exact shape of the hand-written `AddKernel.lean` (P1).
## Tracer POC scope: anything else raises a loud `nim_to_lean: unsupported ...` ValueError.
##
## Supported kernel shape (everything else is refused):
## - buffer params exactly `a`, `b` (read) plus `output` (written), the emitted `(a b out : Nat → UInt32)` binders
## - let-binding of `thread_position_in_grid.x` (emitted as the Lean `tid`)
## - one `if` without elif/else, guard `tidVar < literal` with a positive literal (the buffer length `n`)
## - uint32 `+`, comparisons `<` and `==`, uint32 literals,
##   indexed reads and writes at the thread's own index
##
## Spec theorems come from a write-set analysis of the guarded branch:
## - one write at the thread's own index yields the writes-own-index, preserves-others, composed-post and only-writes-tid theorems
## - each theorem carries the `simp`-style proof the P1 module proved by hand
## - a worked `decide` example is emitted only for the add shape
##
## - `irVulkanLean` captures the P3 verification pragmas (payloads in `lean_pragmas.nim`), `emitLeanModuleWithSpecs` emits the scaffolds
## - `{.precondition.}` - an assumption, carried by the scaffolds as hypothesis `h`
## - `{.invariant.}` - implied by the guard, proved by `exact h`
## - `{.postcondition.}` - stated against the per-thread body, attempted with the derived `simp` scripts
##
## A failed scaffold proof is downgraded to a `sorry`-backed scaffold, recorded unproven in a JSON manifest beside the module.

import std/[json, macros, os, osproc, sets, strutils, tables]
import workspace/crucible/src/crucible_internal_compiler
import lean_pragmas

proc toGpuAstVulkan(body: NimNode): GpuAst {.compileTime.} =
  ## Runs the post-pass GpuAst conversion, the same common + Vulkan pass set
  ## as the production `codegenVulkan` macro
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerVulkanPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  result = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)

macro irVulkan*(body: typed): GpuAst =
  ## Evaluates to the post-pass GpuAst, the same common + Vulkan pass set
  ## as the production `codegenVulkan` macro
  result = newLit(toGpuAstVulkan(body))

# ── Emission analysis ────────────────────────────────────────────────────────

type
  KernelModel = object
    kernelName: string   ## crucible kernel proc name, e.g. `addKernel`
    bodyName: string     ## emitted Lean def name, e.g. `addBody`
    readBufs: seq[string]  ## buffer params never written, in declaration order
    writeBuf: string     ## the buffer param written by the per-thread body
    value: string        ## emitted Lean text of the stored value, in terms of `tid`
    nValue: int          ## the guard literal, emitted as the buffer length `n`
    isAddShape: bool     ## stored value is `<buf1> tid + <buf2> tid`, enables examples

proc unsupported(kind: GpuNodeKind, detail: string): ref ValueError =
  ## Loud refusal for every construct outside the tracer's scope
  result = newException(ValueError,
    "nim_to_lean: unsupported GpuAst node " & $kind & " - " & detail)

proc emitExpr(n: GpuAst, tidName: string, bufNames: HashSet[string]): string =
  ## Renders one GpuAst expression as Lean text, rewriting the thread-id
  ## variable to `tid` and an indexed buffer read to a function application
  case n.kind
  of gpuIdent:
    let name = n.symbol.name
    if name == tidName:
      return "tid"
    if name in bufNames:
      return name
    raise unsupported(n.kind, "identifier '" & name &
      "' is not a kernel parameter or the thread id")
  of gpuLit:
    if n.lType.kind == gtUint32:
      return "(" & n.lValue & " : UInt32)"
    raise unsupported(n.kind, "literal of type " & $n.lType.kind &
      " in value position (only uint32 literals are supported)")
  of gpuBinOp:
    let op = n.bOp.symbol.name
    case op
    of "+", "<", "==":
      let leanOp = if op == "==": "=" else: op
      let l = emitExpr(n.bLeft, tidName, bufNames)
      let r = emitExpr(n.bRight, tidName, bufNames)
      let lp = if n.bLeft.kind == gpuBinOp: "(" & l & ")" else: l
      let rp = if n.bRight.kind == gpuBinOp: "(" & r & ")" else: r
      return lp & " " & leanOp & " " & rp
    else:
      raise unsupported(n.kind, "binary operator '" & op & "'")
  of gpuIndex:
    if n.iArr.kind == gpuDeref and n.iArr.dOf.kind == gpuIdent and
        n.iArr.dOf.symbol.name in bufNames:
      if n.iIndex.kind != gpuIdent or n.iIndex.symbol.name != tidName:
        # a nested index like `a[a[i]]` would render a bogus Lean application
        raise unsupported(n.kind,
          "index expression is not the thread id (the model covers reads at" &
          " the thread's own index only)")
      return n.iArr.dOf.symbol.name & " " & emitExpr(n.iIndex, tidName, bufNames)
    raise unsupported(n.kind, "indexing into anything but a kernel buffer parameter")
  else:
    raise unsupported(n.kind, "expression")

proc emitGuard(cond: GpuAst, tidName: string): int =
  ## Recognizes the tracer guard `tidVar < literal` and returns the literal,
  ## which becomes the buffer length parameter `n` of the emitted def.
  ## The literal must be positive: the worked examples run thread 0 of a non-empty buffer of length `n`
  if cond.kind == gpuBinOp and cond.bOp.symbol.name == "<" and
      cond.bLeft.kind == gpuIdent and cond.bLeft.symbol.name == tidName and
      cond.bRight.kind == gpuLit and
      cond.bRight.lType.kind in {gtInt32, gtUint32, gtSize_t}:
    let n = parseInt(cond.bRight.lValue)
    if n <= 0:
      raise unsupported(gpuIf, "guard literal must be positive (got " &
        cond.bRight.lValue & ")")
    return n
  raise unsupported(gpuIf, "guard is not of the shape `tidVar < literal`")

proc isUint32Buffer(p: GpuParam): bool =
  ## A kernel buffer: `ptr UncheckedArray[uint32]`
  p.typ.kind == gtPtr and p.typ.to.kind == gtUA and p.typ.to.uaTo.kind == gtUint32

proc isAddOfTidReads(v: GpuAst, tidName: string, readBufs: seq[string]): bool =
  ## Whether the stored value is `<buf1> tid + <buf2> tid` over two distinct buffers,
  ## the shape the hand-written P1 worked examples check
  if v.kind != gpuBinOp or v.bOp.symbol.name != "+":
    return false
  for side in [v.bLeft, v.bRight]:
    if side.kind != gpuIndex or side.iArr.kind != gpuDeref or
        side.iArr.dOf.kind != gpuIdent or side.iArr.dOf.symbol.name notin readBufs or
        side.iIndex.kind != gpuIdent or side.iIndex.symbol.name != tidName:
      return false
  result = v.bLeft.iArr.dOf.symbol.name != v.bRight.iArr.dOf.symbol.name

proc indexedReadBufs(n: GpuAst, bufNames: HashSet[string], found: var HashSet[string]) =
  ## Collects the kernel buffer parameters the expression reads by indexing
  case n.kind
  of gpuIndex:
    if n.iArr.kind == gpuDeref and n.iArr.dOf.kind == gpuIdent and
        n.iArr.dOf.symbol.name in bufNames:
      found.incl n.iArr.dOf.symbol.name
      indexedReadBufs(n.iIndex, bufNames, found)
  of gpuBinOp:
    indexedReadBufs(n.bLeft, bufNames, found)
    indexedReadBufs(n.bRight, bufNames, found)
  else:
    discard

proc leanBodyName(kernelName: string): string =
  ## `addKernel` -> `addBody`, the naming of the hand-written P1 module
  if kernelName.endsWith("Kernel"):
    kernelName[0 ..^ 7] & "Body"
  else:
    kernelName & "Body"

proc analyzeKernel(kernel: GpuAst): KernelModel =
  ## Walks one post-pass global kernel and derives the per-thread model:
  ## thread id, guard bound, write set, and the stored value.
  ## Enforces the demonstrated shape: buffer parameters exactly `a` and `b`
  ## (both read) plus `output` (written), the binders of the emitted Lean def
  if kernel.kind != gpuProc:
    raise unsupported(kernel.kind, "expected a gpuProc kernel")
  if attGlobal notin kernel.pAttributes:
    raise unsupported(gpuProc, "kernel is not `{.global.}`")

  var paramNames: seq[string]
  var bufNames: HashSet[string]
  for p in kernel.pParams:
    if not isUint32Buffer(p):
      raise unsupported(gpuProc, "parameter '" & p.ident.symbol.name &
        "' is not a ptr UncheckedArray[uint32] buffer")
    paramNames.add p.ident.symbol.name
    bufNames.incl p.ident.symbol.name
  if kernel.pParams.len != 3 or bufNames != ["a", "b", "output"].toHashSet():
    raise unsupported(gpuProc, "buffer parameters must be exactly `a` and `b`" &
      " (read) plus `output` (written), the binders of the emitted Lean def - got '" &
      paramNames.join(", ") & "'")

  if kernel.pBody.kind != gpuBlock:
    raise unsupported(kernel.pBody.kind, "kernel body is not a statement block")

  var tidName = ""
  var valueAst: GpuAst = nil
  for stmt in kernel.pBody.statements:
    case stmt.kind
    of gpuVar:
      if tidName != "":
        raise unsupported(gpuVar, "more than one local binding")
      let init = stmt.vInit
      if init.kind == gpuDot and init.dParent.kind == gpuIdent and
          init.dParent.symbol.symKind == gsBuiltin and
          init.dParent.symbol.coordBuiltin == gbkThreadPositionInGrid and
          init.dField.symbol.name == "x":
        tidName = stmt.vName.symbol.name
      else:
        raise unsupported(gpuVar,
          "let-binding whose initializer is not `thread_position_in_grid.x`")
    of gpuIf:
      if tidName == "":
        raise unsupported(gpuIf, "if before the thread-id binding")
      if stmt.ifIsExpr or stmt.ifElifs.len > 0 or stmt.ifElse.kind != gpuDiscard:
        raise unsupported(gpuIf, "elif/else branches (the tracer kernel is a bare if)")
      result.nValue = emitGuard(stmt.ifCond, tidName)
      if stmt.ifThen.kind != gpuBlock:
        raise unsupported(gpuIf, "then-branch is not a statement block")
      for w in stmt.ifThen.statements:
        if w.kind != gpuAssign:
          raise unsupported(w.kind, "statement inside the guarded branch")
        let lhs = w.aLeft
        if lhs.kind != gpuIndex or lhs.iArr.kind != gpuDeref or
            lhs.iArr.dOf.kind != gpuIdent:
          raise unsupported(gpuAssign, "assignment target is not a buffer element")
        let buf = lhs.iArr.dOf.symbol.name
        if buf notin bufNames:
          raise unsupported(gpuAssign,
            "assignment to '" & buf & "' which is not a kernel parameter")
        if lhs.iIndex.kind != gpuIdent or lhs.iIndex.symbol.name != tidName:
          raise unsupported(gpuAssign,
            "write index is not the thread id (write-set analysis covers the own-index shape only)")
        if result.writeBuf != "":
          raise unsupported(gpuAssign, "more than one write per thread")
        result.writeBuf = buf
        valueAst = w.aRight
        result.value = emitExpr(w.aRight, tidName, bufNames)
    else:
      raise unsupported(stmt.kind, "statement at kernel-body top level")

  if tidName == "":
    raise unsupported(gpuVar, "no `thread_position_in_grid.x` binding found")
  if result.writeBuf == "":
    raise unsupported(gpuAssign, "no buffer write found in the guarded branch")
  if result.writeBuf != "output":
    raise unsupported(gpuAssign, "the written buffer must be `output` - got '" &
      result.writeBuf & "'")
  var readBufSet: HashSet[string]
  indexedReadBufs(valueAst, bufNames, readBufSet)
  if readBufSet != ["a", "b"].toHashSet():
    var readNames: seq[string]
    for r in readBufSet.items:
      readNames.add r
    raise unsupported(gpuProc, "the guarded branch must read both input buffers" &
      " `a` and `b`, the read binders of the emitted Lean def - got '" &
      (if readNames.len == 0: "none" else: readNames.join(", ")) & "'")

  result.kernelName = kernel.pName.symbol.name
  result.bodyName = leanBodyName(result.kernelName)
  for p in kernel.pParams:
    if p.ident.symbol.name != result.writeBuf:
      result.readBufs.add p.ident.symbol.name
  result.isAddShape = isAddOfTidReads(valueAst, tidName, result.readBufs)

# ── Lean emission ────────────────────────────────────────────────────────────

const leanHeader = """
/- Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
-/
"""

proc emitModuleDoc(model: KernelModel): string =
  ## Module docstring of the emitted Lean module
  const doc = """
/-!
# Emitted @KERNEL@ verification module (P2)

Produced by `nim_to_lean.nim` from the post-pass GpuAst of the crucible kernel
`@KERNEL@`. `poc/lean/t_p2_emit.nim` regenerates this file; edit the emitter,
not this file.

The model matches `AddKernel.lean`: buffers are total functions `Nat → UInt32`,
writes are functional updates, and `UInt32.add` wraps modulo 2^32 like the
GPU arithmetic the kernel emits.
-/
"""
  result = doc.replace("@KERNEL@", model.kernelName)

proc emitBodyDef(model: KernelModel): string =
  ## Per-thread body def, in the shape of the hand-written P1 `addBody`
  const addDoc = """
/-- Per-thread body of the add kernel: returns the output buffer after thread
`tid` of `n` has run.

`a` and `b` are the input buffers, `out` the output buffer before the write.
Precondition: `tid < n` (each thread is launched with its id inside the
buffer bound). Threads with `tid ≥ n` leave the buffer untouched, mirroring
the guard the crucible frontend must emit around every body.

Example: `@BODY@ 0 @N@ (fun _ => 10) (fun _ => 1) (fun _ => 0) 0 = 11`. -/
"""
  const genericDoc = """
/-- Per-thread body of the kernel: returns the output buffer after thread
`tid` of `n` has run.

Precondition: `tid < n` (each thread is launched with its id inside the
buffer bound). Threads with `tid ≥ n` leave the buffer untouched, mirroring
the guard the crucible frontend must emit around every body. -/
"""
  let doc = if model.isAddShape: addDoc else: genericDoc
  return doc.replace("@BODY@", model.bodyName).replace("@N@", $model.nValue) &
    "def " & model.bodyName & " (tid n : Nat) (a b out : Nat → UInt32) : Nat → UInt32 :=\n" &
    "  if tid < n then fun i => if i = tid then " & model.value & " else out i\n" &
    "  else out\n"

proc emitTheorems(model: KernelModel): string =
  ## Four spec theorems carry the `simp`-style proofs of the hand-written P1 module.
  ## Proofs close for any supported stored value: they depend on the def shape alone.
  const tail = """
/-- Postcondition, write half: thread `tid` wrote `@VALUE@` at its own index. -/
theorem @BODY@_writes_own_index {tid n : Nat} {a b out : Nat → UInt32} (h : tid < n) :
    @BODY@ tid n a b out tid = @VALUE@ := by
  simp [@BODY@, h]

/-- Postcondition, no-collateral-damage half: every other index keeps its old value. -/
theorem @BODY@_preserves_others {tid n : Nat} {a b out : Nat → UInt32} (h : tid < n)
    {i : Nat} (hi : i ≠ tid) :
    @BODY@ tid n a b out i = out i := by
  simp [@BODY@, h, hi]

/-- The per-thread postcondition as a single contract. -/
theorem @BODY@_post {tid n : Nat} {a b out : Nat → UInt32} (h : tid < n) :
    @BODY@ tid n a b out tid = @VALUE@ ∧
      ∀ i, i ≠ tid → @BODY@ tid n a b out i = out i :=
  ⟨@BODY@_writes_own_index h, fun _ hi => @BODY@_preserves_others h hi⟩

/-- Bounds theorem: the only index a thread can change is its own, which is
in bounds by the launch precondition. This is the property the codegen gate
must establish for every emitted kernel body. -/
theorem @BODY@_only_writes_tid {tid n : Nat} {a b out : Nat → UInt32} {i : Nat}
    (h : @BODY@ tid n a b out i ≠ out i) :
    i = tid ∧ tid < n := by
  by_cases hlt : tid < n
  · simp only [@BODY@, hlt] at h
    by_cases heq : i = tid
    · exact ⟨heq, hlt⟩
    · simp [heq] at h
  · simp [@BODY@, hlt] at h
"""
  return tail.replace("@BODY@", model.bodyName).replace("@VALUE@", model.value)

proc emitExamples(model: KernelModel): string =
  ## Worked `decide` examples over the host-side test vectors
  ## `a = [10, 20]`, `b = [1, 2]`. Add shape only
  if not model.isAddShape:
    return ""
  const tail = """
/-- Worked example matching the host-side test vectors `a = [10, 20]`,
`b = [1, 2]`: thread 0 writes 11. -/
example : @BODY@ 0 @N@ (fun _ => (10 : UInt32)) (fun _ => 1) (fun _ => 0) 0 = 11 := by
  decide

/-- `UInt32.add` wraps modulo 2^32 like the GPU `u32` add, so the spec needs
no overflow side conditions: thread 0 of a length-1 buffer holding
`0xFFFFFFFF` and `1` writes `0`. -/
example : @BODY@ 0 1 (fun _ => (0xFFFFFFFF : UInt32)) (fun _ => 1) (fun _ => 7) 0 = 0 := by
  decide
"""
  return tail.replace("@BODY@", model.bodyName).replace("@N@", $model.nValue)

# ── Verification pragma extraction (P3) ──────────────────────────────────────

type
  SpecKind* = enum
    skPrecondition
    skInvariant
    skPostcondition

  Scaffold = object
    kind: SpecKind
    payload: string   ## pragma payload verbatim, for the manifest
    leanName: string  ## emitted theorem name, empty for the assumed precondition
    statement: string ## Lean property text between `:` and `:= by`
    binders: string   ## buffer binders of the theorem, e.g. `a b out`
    attempt: string   ## mechanical proof attempt, downgraded to `sorry` on failure
    startLine: int    ## 1-based line of the scaffold's first line in the module

proc specKindByName(name: string): SpecKind =
  ## Maps a captured pragma name to its kind, refusing anything else
  case name
  of "precondition": skPrecondition
  of "invariant": skInvariant
  of "postcondition": skPostcondition
  else:
    raise newException(ValueError,
      "nim_to_lean: unsupported spec pragma '" & name & "'")

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

macro irVulkanLean*(body: typed): untyped =
  ## Like `irVulkan`, plus capture of the verification pragmas on the kernel proc.
  ## Evaluates to the tuple `(specs, gpu)`.
  ##
  ## - `specs` carries the raw pragma payloads
  ## - `gpu` is the post-pass GpuAst of the same proc
  ## - Crucible's frontend refuses verification pragmas, so they are stripped
  ##   before the conversion
  ##
  ## The call shape is in `t_p3_pragmas.nim`.
  ## The verification pragmas sit beside `global` on the kernel proc, each carrying a string payload.
  ## `emitLeanModuleWithSpecs` turns them into Lean scaffolds.
  ##
  ## Payloads that are not string literals, or the same pragma twice, are
  ## refused loudly at expansion time.
  if body.kind == nnkStmtList and body.len != 1:
    raise newException(ValueError, "nim_to_lean: expected exactly one kernel proc" &
      " in the guarded block, got " & $body.len & " statements")
  let procNode = if body.kind == nnkStmtList: body[0] else: body
  if procNode.kind != nnkProcDef:
    raise newException(ValueError,
      "nim_to_lean: expected a kernel proc, got node kind " & $procNode.kind)
  let misplaced = statementSpecPragma(procNode.body)
  if misplaced != "":
    raise newException(ValueError, "nim_to_lean: unsupported statement-level '" &
      misplaced & "' pragma inside the kernel body - verification pragmas" &
      " are captured at proc level only")
  var specs: seq[RawSpec]
  var kept = newNimNode(nnkPragma, procNode.pragma)
  for item in procNode.pragma:
    if item.kind in {nnkExprColonExpr, nnkCall} and
        item[0].kind in {nnkIdent, nnkSym} and
        item[0].strVal in ["precondition", "invariant", "postcondition"]:
      if item[1].kind != nnkStrLit:
        raise newException(ValueError, "nim_to_lean: pragma '" &
          item[0].strVal & "' payload must be a string literal")
      for existing in specs:
        if existing.name == item[0].strVal:
          raise newException(ValueError, "nim_to_lean: more than one '" &
            item[0].strVal & "' pragma on the kernel")
      specs.add RawSpec(name: item[0].strVal, payload: item[1].strVal)
    else:
      kept.add item
  let stripped = copyNimTree(procNode)
  # an empty nnkPragma trips crucible's `pragma.len > 0` assert at expansion.
  # The empty node also keeps `analyzeKernel`'s own `{.global.}` refusal loud
  stripped.pragma = if kept.len == 0: newEmptyNode() else: kept
  let astIn = if body.kind == nnkStmtList: newStmtList(stripped) else: stripped
  let gpuAst = toGpuAstVulkan(astIn)
  result = nnkPar.newTree(
    nnkExprColonExpr.newTree(ident"specs", newLit(specs)),
    nnkExprColonExpr.newTree(ident"gpu", newLit(gpuAst)))

proc prepareSpec(model: KernelModel, spec: RawSpec): Scaffold =
  ## Parses, validates and renders one captured pragma payload against the kernel model.
  ##
  ## - out-of-scope payloads are loud ValueErrors
  ## - a false property is not judged here. It fails the Lean proof attempt
  ##   and is downgraded (see `emitLeanModuleWithSpecs`)
  let kind = specKindByName(spec.name)
  let expr = parseSpecPayload(spec.payload)
  var nameMap: Table[string, string]
  for r in model.readBufs:
    nameMap[r] = r
  nameMap[model.writeBuf] = model.bodyName & " tid n " &
    (model.readBufs.join(" ") & " out")
  result = Scaffold(kind: kind, payload: spec.payload,
    statement: renderSpec(expr, nameMap))
  case kind
  of skPrecondition:
    # a precondition is assumed, never proved: it must match the model's
    # precondition `tid < n` exactly, anything else cannot be checked
    if result.statement != "tid < n":
      raise newException(ValueError,
        "nim_to_lean: unsupported precondition payload '" & spec.payload &
        "' - the tracer model's launch precondition is `tid < n`")
  of skInvariant:
    # a guard invariant is proved from the guard alone: over `tid` and `n` only
    let refs = specRefs(expr)
    for r in ["tid", "n"]:
      if r notin refs:
        raise newException(ValueError,
          "nim_to_lean: unsupported invariant payload '" & spec.payload &
          "' - the tracer invariant must be implied by the guard `tid < n`" &
          " and mention both `tid` and `n`")
    for r in refs:
      if r notin ["tid", "n"]:
        raise newException(ValueError,
          "nim_to_lean: unsupported invariant payload '" & spec.payload &
          "' - the tracer invariant may only mention `tid` and `n`")
    result.leanName = model.bodyName & "_invariant"
    result.attempt = "exact h"
  of skPostcondition:
    # the postcondition states the written buffer's post-state, rendered as
    # the per-thread body applied at `tid`
    if model.writeBuf notin specRefs(expr):
      raise newException(ValueError,
        "nim_to_lean: unsupported postcondition payload '" & spec.payload &
        "' - it must mention the written buffer '" & model.writeBuf & "'")
    result.leanName = model.bodyName & "_postcondition"
    result.binders = model.readBufs.join(" ") & " out"
    result.attempt = "simp [" & model.bodyName & ", h]"

proc scaffoldText(model: KernelModel, s: Scaffold, proof: string): string =
  ## One Lean spec scaffold: the precondition comment, or a theorem carrying
  ## the given proof text (the mechanical attempt, or `sorry` after downgrade)
  case s.kind
  of skPrecondition:
    const doc = """
/- Precondition from the `{.precondition.}` pragma on `@KERNEL@`: `@STMT@`.
Assumed at launch, carried as the hypothesis `h` by the scaffolds below. A
precondition is an assumption, so it gets no Lean declaration. -/
"""
    result = doc.replace("@KERNEL@", model.kernelName).replace("@STMT@", s.statement)
  of skInvariant:
    const doc = """
/-- Invariant from the `{.invariant.}` pragma on `@KERNEL@`, payload
`@PAYLOAD@`: a property implied by the guard `tid < n`, so it holds whenever
the guarded branch is entered. -/
theorem @NAME@ {tid n : Nat} (h : tid < n) :
    @STMT@ := by
  @PROOF@
"""
    result = doc.replace("@KERNEL@", model.kernelName).replace("@PAYLOAD@", s.payload)
      .replace("@NAME@", s.leanName).replace("@STMT@", s.statement)
      .replace("@PROOF@", proof)
  of skPostcondition:
    const doc = """
/-- Postcondition from the `{.postcondition.}` pragma on `@KERNEL@`, payload
`@PAYLOAD@`: the written buffer after thread `tid` of `n` ran. -/
theorem @NAME@ {tid n : Nat} {@BINDERS@ : Nat → UInt32} (h : tid < n) :
    @STMT@ := by
  @PROOF@
"""
    result = doc.replace("@KERNEL@", model.kernelName).replace("@PAYLOAD@", s.payload)
      .replace("@NAME@", s.leanName).replace("@BINDERS@", s.binders)
      .replace("@STMT@", s.statement).replace("@PROOF@", proof)

proc leanErrorExcerpts(leanOutput: string): seq[tuple[line: int, excerpt: string]] =
  ## Extracts the `error:` diagnostics of a Lean run as (line, message) pairs
  for ln in leanOutput.splitLines():
    let idx = ln.find(": error: ")
    if idx < 0:
      continue
    let head = ln[0 ..< idx].split(':')
    if head.len < 2:
      continue
    var excerpt = ln[idx + ": error: ".len .. ^1]
    if excerpt.len > 160:
      excerpt = excerpt[0 ..< 157] & "..."
    result.add (parseInt(head[^2]), excerpt)

proc findLakeDir(outPath: string): string =
  ## Walks up from `outPath` to the first directory containing a lakefile.lean
  var dir = parentDir(outPath)
  while dir != "" and not fileExists(dir / "lakefile.lean"):
    dir = parentDir(dir)
  if dir == "":
    raise newException(ValueError,
      "nim_to_lean: no lakefile.lean above the emitted module " & outPath)
  result = dir

proc emitLeanModuleWithSpecs*(kernel: GpuAst, specs: seq[RawSpec], outPath: string): string =
  ## Emits the Lean 4 verification module for one global kernel, with one spec scaffold per captured verification pragma.
  ##
  ## - writes the module to `outPath` and returns its text
  ## - with no pragmas the module is byte-identical to the P1/P2 statements
  ## - refuses loudly (ValueError, "nim_to_lean: ...") on any out-of-scope
  ##   construct or payload
  ## - settles each scaffold's proof attempt with `lake env lean`, downgrading
  ##   a failing proof to a `sorry`-backed scaffold recorded unproven in the JSON manifest beside the module
  let model = analyzeKernel(kernel)
  createDir(parentDir(outPath))
  result = leanHeader & "\n" & emitModuleDoc(model) & "\n" &
    emitBodyDef(model) & "\n" & emitTheorems(model)
  let examples = emitExamples(model)
  if examples != "":
    result &= "\n" & examples
  if specs.len == 0:
    writeFile(outPath, result)
    return

  # canonical pragma order, one of each kind
  var ordered: array[3, RawSpec]
  var seen: set[SpecKind]
  for spec in specs:
    let kind = specKindByName(spec.name)
    if kind in seen:
      raise newException(ValueError, "nim_to_lean: more than one '" &
        spec.name & "' pragma on the kernel")
    seen.incl kind
    ordered[ord(kind)] = spec

  var scaffolds: seq[Scaffold]
  for kind in [skPrecondition, skInvariant, skPostcondition]:
    if kind in seen:
      scaffolds.add prepareSpec(model, ordered[ord(kind)])

  # first pass: emit with the mechanical proof attempts
  var text = result
  for s in scaffolds:
    text &= "\n"
    var sc = s
    sc.startLine = text.count('\n') + 1
    text &= scaffoldText(model, sc, sc.attempt)
  writeFile(outPath, text)

  # settle the proof attempts by elaboration (a failed proof errors, a `sorry` only warns)
  let lakeDir = findLakeDir(outPath)
  let relFile = outPath.relativePath(lakeDir)
  let (outp, exitCode) = execCmdEx("lake env lean " & relFile, workingDir = lakeDir)
  var failed: Table[int, string] ## scaffold index -> error excerpt
  let lastLine = text.count('\n') + 1
  for (line, excerpt) in leanErrorExcerpts(outp):
    var owner = -1
    for i, s in scaffolds:
      let endLine = if i + 1 < scaffolds.len: scaffolds[i + 1].startLine - 1
                    else: lastLine
      if line >= s.startLine and line <= endLine:
        owner = i
    if owner < 0:
      raise newException(ValueError, "nim_to_lean: proof attempt failed " &
        "outside the pragma scaffolds (line " & $line & "): " & excerpt)
    failed[owner] = excerpt
  if exitCode != 0 and failed.len == 0:
    # no error line was attributable to a scaffold, so no proof was settled:
    # a `proven` verdict would be a guess, e.g. a missing toolchain, a Lean
    # crash or a diagnostic format change
    var excerpt = outp
    if excerpt.len > 800:
      excerpt = excerpt[0 ..< 797] & "..."
    raise newException(ValueError, "nim_to_lean: proof settling failed - " &
      "lean exited with " & $exitCode & " but no error was attributable" &
      " to a scaffold: " & excerpt)

  if failed.len > 0:
    # second pass: downgrade every failing proof to a sorry-backed scaffold
    text = result
    for i, s in scaffolds:
      text &= "\n"
      var sc = s
      sc.startLine = text.count('\n') + 1
      let proof = if i in failed: "sorry" else: s.attempt
      text &= scaffoldText(model, sc, proof)
    writeFile(outPath, text)

  # machine-readable extraction manifest
  var entries = newJArray()
  for i, s in scaffolds:
    var e = newJObject()
    e["kind"] = newJString(
      case s.kind
      of skPrecondition: "precondition"
      of skInvariant: "invariant"
      of skPostcondition: "postcondition")
    e["payload"] = newJString(s.payload)
    e["name"] = if s.leanName == "": newJNull() else: newJString(s.leanName)
    if s.kind == skPrecondition:
      e["proven"] = newJNull() # assumed, neither proved nor disproved
    else:
      e["proven"] = newJBool(i notin failed)
    e["error"] = if i in failed: newJString(failed[i]) else: newJNull()
    entries.add e
  var manifest = newJObject()
  manifest["kernel"] = newJString(model.kernelName)
  manifest["module"] = newJString(relFile)
  manifest["scaffolds"] = entries
  writeFile(outPath.changeFileExt("manifest.json"), manifest.pretty & "\n")
  result = text

proc emitLeanModule*(kernel: GpuAst, outPath: string): string =
  ## Emits the Lean 4 verification module for one global kernel: the per-thread
  ## body def plus derived spec theorems.
  ## - writes the module to `outPath` and returns its text
  ## - refuses loudly (ValueError, "nim_to_lean: unsupported ...") on any
  ##   construct outside the tracer's scope
  emitLeanModuleWithSpecs(kernel, @[], outPath)
