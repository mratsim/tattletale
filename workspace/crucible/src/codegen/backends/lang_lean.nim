# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## lang_lean: emit a Lean 4 verification module from a crucible CIRv2 kernel
##
##   post-pass CIRv2 (one `{.global.}` kernel) → this backend → Lean module
##
## Walks the body of one `{.global.}` kernel and emits the body def plus
## derived spec theorems. Anything outside the supported shape raises a loud
## `lean_lang: unsupported ...` ValueError:
##
## - guarded thread (the `AddKernel.lean` model): buffers `a`, `b` (read) plus `output` (written), all aptr-to-uint32 parameters
## - a let-bound `thread_position_in_grid.x` (the Lean `tid`), one `if` with no
##   elif/else branch, guard `tidVar < literal`, positive literal `n`
##
## A write is a functional update over buffers modelled as total functions
## `Nat → UInt32`. `UInt32.add` wraps modulo 2^32 like the kernel's GPU arithmetic.

import std/[os, sets, strutils]
import ../ir/cirv2_lang

proc leanRefusal(msg: string): ref ValueError =
  ## Loud refusal for every construct outside the supported kernel shapes
  result = newException(ValueError, "lean_lang: " & msg)

# ── Emission analysis ────────────────────────────────────────────────────────

type
  KernelModel = object
    kernelName: string   ## crucible kernel proc name, e.g. `addKernel`
    bodyName: string     ## emitted Lean def name, e.g. `addBody`
    readBufs: seq[string]  ## buffer params never written, in declaration order
    writeBuf: string     ## the buffer param written by the body
    value: string        ## emitted Lean text of the stored value, in terms of `tid`
    nValue: int          ## the guard literal, the buffer length of the emitted def

proc isUint32Buffer(t: CirType): bool =
  ## A kernel buffer: an aptr to uint32 elements
  t != nil and t.kind == ctAptr and t.pElem != nil and
    t.pElem.kind == ctUint and t.pElem.bits == 32

proc emitExpr(e: CirExpr, idxName, idxLean: string, paramNames: HashSet[string], mulAllowed: bool): string =
  ## Renders one CIRv2 expression as Lean text:
  ##
  ## - the index variable `idxName` renders as its Lean name `idxLean`
  ## - `paramNames` is the set of kernel parameters: buffers plus the axpy
  ##   scalar
  ## - an indexed buffer read renders as a function application
  ## - the multiply is emitted only when `mulAllowed` is true (the axpy shape)
  case e.kind
  of eSym:
    if e.name == idxName:
      return idxLean
    result = e.name
  of eIntLit:
    # Nim types every literal in a uint32 expression as uint32, so the value
    # renders as a Lean UInt32 literal
    result = "(" & $e.value & " : UInt32)"
  of eAdd, eMul:
    if e.kind == eMul and not mulAllowed:
      raise leanRefusal("unsupported expression eMul - the multiply is in scope" &
        " for the axpy shape only (a loop kernel with a scalar parameter)")
    let leanOp = if e.kind == eAdd: "+" else: "*"
    let l = emitExpr(e.left, idxName, idxLean, paramNames, mulAllowed)
    let r = emitExpr(e.right, idxName, idxLean, paramNames, mulAllowed)
    let lp = if e.left.kind in {eAdd, eSub, eMul}: "(" & l & ")" else: l
    let rp = if e.right.kind in {eAdd, eSub, eMul}: "(" & r & ")" else: r
    result = lp & " " & leanOp & " " & rp
  of ePat:
    if e.patIndex.kind != eSym or e.patIndex.name != idxName:
      raise leanRefusal("unsupported expression ePat - index expression is not" &
        " the index variable (the model covers reads at the thread's own index" &
        " or the loop variable's index only)")
    result = emitExpr(e.patBase, idxName, idxLean, paramNames, mulAllowed) &
      " " & idxLean
  else:
    raise leanRefusal("unsupported expression " & $e.kind)

proc emitGuard(cond: CirExpr, tidName: string): int =
  ## Recognizes the guard `tidVar < literal` and returns the literal.
  ## The literal becomes the buffer length parameter `n` of the emitted def
  ## and must be positive: the model runs thread 0 of a non-empty buffer.
  if cond.kind == eLt and cond.a.kind == eSym and cond.a.name == tidName and
      cond.b.kind == eIntLit:
    let n = parseInt($cond.b.value)
    if n <= 0:
      raise leanRefusal("unsupported guard - literal must be positive (got " &
        $cond.b.value & ")")
    return n
  raise leanRefusal("unsupported guard - not of the shape `tidVar < literal`")

proc checkWriteTarget(lhs: CirExpr): string =
  ## Validates an assignment target as a kernel buffer element write.
  ## Returns the written buffer's name.
  ##
  ## Shared by the loop path and the guarded-thread path, so both shapes
  ## refuse a malformed target with the same message.
  ## The index-variable check and the write-count policy stay at the call sites.
  if lhs.kind != ePat or lhs.patBase.kind != eSym:
    raise leanRefusal("unsupported assignment - target is not a buffer element")
  result = lhs.patBase.name

proc indexedReadBufs(e: CirExpr, bufNames: HashSet[string], found: var HashSet[string]) =
  ## Collects the kernel buffer parameters the expression reads by indexing
  case e.kind
  of ePat:
    if e.patBase.kind == eSym and e.patBase.name in bufNames:
      found.incl e.patBase.name
      indexedReadBufs(e.patIndex, bufNames, found)
  of eAdd, eSub, eMul:
    indexedReadBufs(e.left, bufNames, found)
    indexedReadBufs(e.right, bufNames, found)
  else:
    discard

proc leanBodyName(kernelName: string): string =
  ## `addKernel` -> `addBody`, the naming of the hand-written AddKernel.lean module
  if kernelName.endsWith("Kernel"):
    kernelName[0 ..^ 7] & "Body"
  else:
    kernelName & "Body"

proc analyzeKernel(kernel: CirProc): KernelModel =
  ## Walks one global kernel and derives the body model
  ##
  ## Fills:
  ## - the thread id binding and the guard literal
  ## - the write set and the stored value
  ##
  ## Enforces the demonstrated shape: buffer parameters exactly `a` and `b`
  ## (both read) plus `output` (written), the binders of the emitted Lean def
  var bufSeq: seq[string]
  var bufNames: HashSet[string]
  var scalarName = ""
  var scalarCount = 0
  for p in kernel.params:
    if isUint32Buffer(p.typ):
      bufSeq.add p.name
      bufNames.incl p.name
    elif p.typ.kind == ctUint and p.typ.bits == 32:
      inc scalarCount
      if scalarCount > 1:
        raise leanRefusal("unsupported kernel - more than one uint32 scalar" &
          " parameter, the model covers exactly one uniform scalar (the axpy shape)")
      scalarName = p.name
    else:
      raise leanRefusal("unsupported kernel - parameter '" & p.name &
        "' is neither a ptr UncheckedArray[uint32] buffer nor a uint32 scalar")
  if bufNames != ["a", "b", "output"].toHashSet():
    raise leanRefusal("unsupported kernel - buffer parameters must be exactly" &
      " `a` and `b` (read) plus `output` (written), the binders of the emitted" &
      " Lean def - got '" & bufSeq.join(", ") & "'")

  # walk the statements before the frontend's trailing void return
  var stmts = kernel.body
  if stmts.len > 0 and stmts[^1].kind == sRet:
    stmts = stmts[0 ..< stmts.len - 1]

  var tidName = ""
  var valueAst: CirExpr = nil
  for stmt in stmts:
    case stmt.kind
    of sVar:
      if tidName != "":
        raise leanRefusal("unsupported statement sVar - more than one local binding")
      let init = stmt.init
      if init != nil and init.kind == eDot and init.dotBase.kind == eSym and
          init.dotBase.builtin and
          init.dotBase.coordBuiltin == cbkThreadPositionInGrid and
          init.fieldName == "x":
        tidName = stmt.name
      else:
        raise leanRefusal("unsupported statement sVar - local binding whose" &
          " initializer is not `thread_position_in_grid.x`")
    of sIte:
      if tidName == "":
        raise leanRefusal("unsupported statement sIte - if before the thread-id binding")
      if stmt.elseBody.len > 0:
        raise leanRefusal("unsupported statement sIte - elif/else branches" &
          " (the model kernel is a bare if)")
      result.nValue = emitGuard(stmt.cond, tidName)
      for w in stmt.thenBody:
        if w.kind != sAsgn:
          raise leanRefusal("unsupported statement " & $w.kind &
            " - statement inside the guarded branch")
        let buf = checkWriteTarget(w.lhs)
        if w.lhs.patIndex.kind != eSym or w.lhs.patIndex.name != tidName:
          raise leanRefusal("unsupported assignment - write index is not the" &
            " thread id (write-set analysis covers the own-index shape only)")
        if result.writeBuf != "":
          raise leanRefusal("unsupported assignment - more than one write per thread")
        result.writeBuf = buf
        valueAst = w.rhs
        result.value = emitExpr(w.rhs, tidName, "tid", bufNames, false)
    else:
      raise leanRefusal("unsupported statement " & $stmt.kind &
        " - statement at kernel-body top level")

  if result.writeBuf == "":
    raise leanRefusal("unsupported assignment - no buffer write found in the kernel body")
  var readBufSet: HashSet[string]
  indexedReadBufs(valueAst, bufNames, readBufSet)
  if readBufSet != ["a", "b"].toHashSet():
    var readNames: seq[string]
    for r in readBufSet.items:
      readNames.add r
    raise leanRefusal("unsupported kernel - the kernel body must read both input" &
      " buffers `a` and `b`, the read binders of the emitted Lean def - got '" &
      (if readNames.len == 0: "none" else: readNames.join(", ")) & "'")

  result.kernelName = kernel.name
  result.bodyName = leanBodyName(result.kernelName)
  for p in kernel.params:
    if isUint32Buffer(p.typ) and p.name != result.writeBuf:
      result.readBufs.add p.name
  if scalarName != "":
    raise leanRefusal("unsupported kernel - scalar parameter '" & scalarName &
      "' on the guarded-thread shape, the guarded-thread shape takes no scalar" &
      " parameter (the scalar is the axpy loop shape)")

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
  const threadDoc = """
/-!
# Emitted @KERNEL@ verification module

Produced by `lang_lean.nim` from the post-pass CIRv2 of the crucible kernel
`@KERNEL@`. `tests/codegen/lean/t_lean_emit.nim` regenerates this file; edit the emitter,
not this file.

The model matches `AddKernel.lean`: buffers are total functions `Nat → UInt32`,
writes are functional updates, and `UInt32.add` wraps modulo 2^32 like the
GPU arithmetic the kernel emits.
-/
"""
  result = threadDoc.replace("@KERNEL@", model.kernelName)

proc emitBodyDef(model: KernelModel): string =
  ## Body def of the emitted module: the guarded-thread per-thread body
  const doc = """
/-- Per-thread body of the kernel: returns the output buffer after thread
`tid` of `n` has run.

`a` and `b` are the input buffers, `out` the output buffer before the write.
Precondition: `tid < n` (each thread is launched with its id inside the
buffer bound). Threads with `tid ≥ n` leave the buffer untouched, mirroring
the guard the crucible frontend must emit around every body. -/
"""
  result = doc &
    "def " & model.bodyName & " (tid n : Nat) (a b out : Nat → UInt32) : Nat → UInt32 :=\n" &
    "  if tid < n then fun i => if i = tid then " & model.value & " else out i\n" &
    "  else out\n"

proc emitTheorems(model: KernelModel): string =
  ## Emits the two postcondition theorems of the hand-written AddKernel.lean module.
  ## Every proof depends on the def shape alone, never on the stored value.
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
"""
  result = tail.replace("@BODY@", model.bodyName).replace("@VALUE@", model.value)

proc emitModule(model: KernelModel): string =
  ## Returns the full emitted module text: license header, module docstring,
  ## body def and spec theorems
  result = leanHeader & "\n" & emitModuleDoc(model) & "\n" &
    emitBodyDef(model) & "\n" & emitTheorems(model)

# ── Backend API ──────────────────────────────────────────────────────────────

proc emitText*(kernel: CirProc): string =
  ## Returns the Lean text of one kernel's verification module, no file output.
  ## Raises a loud `lean_lang: ...` ValueError on any construct outside the supported shape.
  result = emitModule(analyzeKernel(kernel))

proc autogenModule*(kernelName: string): string =
  ## Returns the default emitted-module file name of a kernel:
  ## `addKernel` emits `add.autogen.lean`. The `.autogen.lean` suffix marks
  ## machine-emitted modules and separates them from the hand-written ones
  let base = if kernelName.endsWith("Kernel"):
      kernelName[0 ..< ^6]
    else:
      kernelName
  result = base & ".autogen.lean"

proc codegen*(m: CirModule): string =
  ## Lean verification module of the module's single `{.global.}` kernel,
  ## the dispatch target of `codegen(m, cgLean)`
  var kernels: seq[CirProc]
  for p in m.procs:
    if p.isKernel:
      kernels.add p
  if kernels.len != 1:
    raise leanRefusal("unsupported module - expected exactly one `{.global.}`" &
      " kernel, got " & $kernels.len)
  result = emitText(kernels[0])
