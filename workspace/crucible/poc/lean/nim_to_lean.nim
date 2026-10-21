# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim_to_lean: emit a Lean 4 verification module from crucible's post-pass GpuAst (P2)
##
## Walks the GpuAst of one `{.global.}` kernel and emits the per-thread body def plus
## derived spec theorems, in the exact shape of the hand-written `AddKernel.lean` (P1).
## Tracer POC scope: anything else raises a loud `nim_to_lean: unsupported ...` ValueError.
##
## Supported constructs (everything else is refused):
## - let-binding of `thread_position_in_grid.x` (emitted as the Lean `tid`)
## - one `if` without elif/else, guard `tidVar < literal` (the literal becomes `n`)
## - uint32 `+`, comparisons `<` and `==`, uint32 literals, indexed reads
## - indexed writes on `ptr UncheckedArray[uint32]` params, at the thread's own index
##
## Spec theorems come from a write-set analysis of the guarded branch:
## - one write at the thread's own index yields the writes-own-index,
##   preserves-others, composed-post and only-writes-tid theorems,
##   each carrying the `simp`-style proof the P1 module proved by hand
## - a worked `decide` example is emitted only for the add shape

import std/[macros, os, sets, strutils]
import workspace/crucible/src/crucible_internal_compiler

macro irVulkan*(body: typed): GpuAst =
  ## Evaluates to the post-pass GpuAst, the same common + Vulkan pass set
  ## as the production `codegenVulkan` macro
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerVulkanPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  result = newLit(gpuAst)

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
      return n.iArr.dOf.symbol.name & " " & emitExpr(n.iIndex, tidName, bufNames)
    raise unsupported(n.kind, "indexing into anything but a kernel buffer parameter")
  else:
    raise unsupported(n.kind, "expression")

proc emitGuard(cond: GpuAst, tidName: string): int =
  ## Recognizes the tracer guard `tidVar < literal` and returns the literal,
  ## which becomes the buffer length parameter `n` of the emitted def
  if cond.kind == gpuBinOp and cond.bOp.symbol.name == "<" and
      cond.bLeft.kind == gpuIdent and cond.bLeft.symbol.name == tidName and
      cond.bRight.kind == gpuLit and
      cond.bRight.lType.kind in {gtInt32, gtUint32, gtSize_t}:
    return parseInt(cond.bRight.lValue)
  raise unsupported(gpuIf, "guard is not of the shape `tidVar < literal`")

proc isUint32Buffer(p: GpuParam): bool =
  ## A kernel buffer: `ptr UncheckedArray[uint32]` (or the UA directly)
  (p.typ.kind == gtUA and p.typ.uaTo.kind == gtUint32) or
    (p.typ.kind == gtPtr and p.typ.to.kind == gtUA and p.typ.to.uaTo.kind == gtUint32)

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

proc leanBodyName(kernelName: string): string =
  ## `addKernel` -> `addBody`, the naming of the hand-written P1 module
  if kernelName.endsWith("Kernel"):
    kernelName[0 ..^ 7] & "Body"
  else:
    kernelName & "Body"

proc analyzeKernel(kernel: GpuAst): KernelModel =
  ## Walks one post-pass global kernel and derives the per-thread model:
  ## thread id, guard bound, write set, and the stored value
  if kernel.kind != gpuProc:
    raise unsupported(kernel.kind, "expected a gpuProc kernel")
  if attGlobal notin kernel.pAttributes:
    raise unsupported(gpuProc, "kernel is not `{.global.}`")

  var bufNames: HashSet[string]
  for p in kernel.pParams:
    if not isUint32Buffer(p):
      raise unsupported(gpuProc, "parameter '" & p.ident.symbol.name &
        "' is not a ptr UncheckedArray[uint32] buffer")
    bufNames.incl p.ident.symbol.name

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

proc emitLeanModule*(kernel: GpuAst, outPath: string): string =
  ## Emits the Lean 4 verification module for one global kernel: the per-thread
  ## body def plus derived spec theorems.
  ## - writes the module to `outPath` and returns its text
  ## - refuses loudly (ValueError, "nim_to_lean: unsupported ...") on any
  ##   construct outside the tracer's scope
  let model = analyzeKernel(kernel)
  createDir(parentDir(outPath))
  result = leanHeader & "\n" & emitModuleDoc(model) & "\n" &
    emitBodyDef(model) & "\n" & emitTheorems(model)
  let examples = emitExamples(model)
  if examples != "":
    result &= "\n" & examples
  writeFile(outPath, result)
