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
## - loop (the axpy shape): one `for` over the compile-time constant `[0, n)`
##   range (start 0, step 1, exclusive), one indexed write, an optional uint32
##   scalar mapping to a trailing `UInt32` argument
##
## A write is a functional update over buffers modelled as total functions
## `Nat → UInt32`. `UInt32.add` wraps modulo 2^32 like the kernel's GPU arithmetic.

import std/[json, os, osproc, sets, strutils, tables]
import ../ir/cirv2_lang
import ./print_common

proc leanRefusal(msg: string): ref ValueError =
  ## Loud refusal for every construct outside the supported kernel shapes
  result = newException(ValueError, "lean_lang: " & msg)

# ── Emission analysis ────────────────────────────────────────────────────────

type
  KernelKind = enum
    kkGuardedThread ## let-bound thread id, one guarded write (the add shape)
    kkLoopRange     ## one `for` over a compile-time constant `[0, n)` range (the loop shape)

  KernelModel = object
    kind: KernelKind
    kernelName: string   ## crucible kernel proc name, e.g. `addKernel`
    bodyName: string     ## emitted Lean def name, e.g. `addBody`
    readBufs: seq[string]  ## buffer params never written, in declaration order
    writeBuf: string     ## the buffer param written by the body
    value: string        ## emitted Lean text of the stored value, in terms of `tid` or the update binder `j`
    nValue: int          ## the guard literal (guarded thread) or loop bound (loop shape)
    loopVar: string      ## loop shape: the Nim loop variable's name
    scalarName: string   ## the uint32 scalar parameter (axpy shape), "" when the kernel has none

proc isUint32Buffer(t: CirType): bool =
  ## A kernel buffer: an aptr to uint32 elements
  t != nil and t.kind == ctAptr and t.pElem != nil and
    t.pElem.kind == ctUint and t.pElem.bits == 32

proc emitExpr(e: CirExpr, idxName, idxLean: string, mulAllowed: bool): string =
  ## Renders one CIRv2 expression as Lean text:
  ##
  ## - the index variable `idxName` renders as its Lean name `idxLean`
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
    let l = emitExpr(e.left, idxName, idxLean, mulAllowed)
    let r = emitExpr(e.right, idxName, idxLean, mulAllowed)
    let lp = if e.left.kind in {eAdd, eSub, eMul}: "(" & l & ")" else: l
    let rp = if e.right.kind in {eAdd, eSub, eMul}: "(" & r & ")" else: r
    result = lp & " " & leanOp & " " & rp
  of ePat:
    if e.patIndex.kind != eSym or e.patIndex.name != idxName:
      raise leanRefusal("unsupported expression ePat - index expression is not" &
        " the index variable (the model covers reads at the thread's own index" &
        " or the loop variable's index only)")
    result = emitExpr(e.patBase, idxName, idxLean, mulAllowed) & " " & idxLean
  else:
    raise leanRefusal("unsupported expression " & $e.kind)

proc checkGuard(cond: CirExpr, tidName: string) =
  ## Recognizes the guard `tidVar < literal`. The literal becomes the buffer
  ## length parameter `n` of the emitted def and must be positive: the model runs
  ## thread 0 of a non-empty buffer.
  if cond.kind == eLt and cond.a.kind == eSym and cond.a.name == tidName and
      cond.b.kind == eIntLit and cond.b.value > 0:
    return
  if cond.kind == eLt and cond.a.kind == eSym and cond.a.name == tidName and
      cond.b.kind == eIntLit:
    raise leanRefusal("unsupported guard - literal must be positive (got " &
      $cond.b.value & ")")
  raise leanRefusal("unsupported guard - not of the shape `tidVar < literal`")

proc checkWriteTarget(lhs: CirExpr): string =
  ## Validates an assignment target as a write of the model's output buffer
  ## and returns the buffer's name.
  ##
  ## Shared by the loop path and the guarded-thread path, so both shapes
  ## refuse a malformed target with the same message.
  ## The index-variable check and the write-count policy stay at the call sites.
  if lhs.kind != ePat or lhs.patBase.kind != eSym:
    raise leanRefusal("unsupported assignment - target is not a buffer element")
  if lhs.patBase.name != "output":
    raise leanRefusal("unsupported assignment - the written buffer must be" &
      " `output` - got '" & lhs.patBase.name & "'")
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

proc stripKernelSuffix(kernelName: string): string =
  ## `addKernel` -> `add`, the base name of the emitted defs and modules
  if kernelName.endsWith("Kernel"):
    kernelName[0 ..< ^6]
  else:
    kernelName

proc leanBodyName(kernelName: string): string =
  ## `addKernel` -> `addBody`, the naming of the hand-written AddKernel.lean module
  stripKernelSuffix(kernelName) & "Body"

proc leanLoopName(kernelName: string): string =
  ## `loopAdd` -> `loopAddLoop`, the remaining-count recursion def of the loop shape
  stripKernelSuffix(kernelName) & "Loop"

proc analyzeLoopKernel(decl, start, loop: CirStmt, scalarName: string, model: var KernelModel): CirExpr =
  ## Walks the frontend's `for`-loop lowering: the loop-variable declaration,
  ## the `i = 0` start assignment, and the `sWhile` ending on the increment.
  ## The range must be the compile-time constant `[0, n)` with step 1, one indexed write over the loop variable in the body.
  ##
  ## Returns the value expression, input to the kernel-level read-set check,
  ## and fills the loop fields of the model
  let loopVar = decl.name
  # frontend renaming gives a shadowing loop variable a fresh name
  if loopVar == "j":
    raise leanRefusal("unsupported loop - loop variable 'j' collides with the" &
      " emitted update binder 'j'")
  if scalarName == "j":
    raise leanRefusal("unsupported kernel - scalar parameter 'j' collides with" &
      " the emitted update binder 'j' of the loop shape")
  model.loopVar = loopVar
  if start.rhs.kind != eIntLit or start.rhs.value != 0:
    let got = if start.rhs.kind == eIntLit: $start.rhs.value else: $start.rhs.kind
    raise leanRefusal("unsupported loop - start must be the literal 0 - got '" &
      got & "'")
  let cond = loop.whileCond
  if cond.kind != eLt or cond.a.kind != eSym or cond.a.name != loopVar:
    raise leanRefusal("unsupported loop - condition is not of the shape" &
      " `i < literal` (an inclusive `..` range, a `countup` step, or a symbolic bound)")
  if cond.b.kind != eIntLit:
    raise leanRefusal("unsupported loop - bound is not a compile-time literal")
  if cond.b.value <= 0:
    raise leanRefusal("unsupported loop - bound must be positive (got " &
      $cond.b.value & ")")
  model.nValue = int(cond.b.value)
  let body = loop.whileBody
  if body.len != 2:
    raise leanRefusal("unsupported loop - body is not a single write plus the" &
      " counter increment")
  let w = body[0]
  if w.kind != sAsgn:
    raise leanRefusal("unsupported statement " & $w.kind &
      " - statement inside the loop body")
  if w.lhs.patIndex.kind != eSym or w.lhs.patIndex.name != loopVar:
    raise leanRefusal("unsupported assignment - write index is not the loop" &
      " variable (write-set analysis covers the whole-range shape only)")
  model.writeBuf = checkWriteTarget(w.lhs)
  # increment: every supported range lowers with increment 1, any other condition shape refuses above
  result = w.rhs
  model.value = emitExpr(w.rhs, loopVar, "j", scalarName != "")

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

  var valueAst: CirExpr = nil
  if stmts.len == 3 and stmts[0].kind == sVar and stmts[0].init.isNil and
      stmts[1].kind == sAsgn and stmts[2].kind == sWhile:
    # loop shape: the frontend lowers `for` to a declaration, a start assignment and a while loop ending on the increment
    result.kind = kkLoopRange
    valueAst = analyzeLoopKernel(stmts[0], stmts[1], stmts[2], scalarName, result)
  else:
    result.kind = kkGuardedThread
    var tidName = ""
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
        checkGuard(stmt.cond, tidName)
        for w in stmt.thenBody:
          if w.kind != sAsgn:
            raise leanRefusal("unsupported statement " & $w.kind &
              " - statement inside the guarded branch")
          if w.lhs.patIndex.kind != eSym or w.lhs.patIndex.name != tidName:
            raise leanRefusal("unsupported assignment - write index is not the" &
              " thread id (write-set analysis covers the own-index shape only)")
          if result.writeBuf != "":
            raise leanRefusal("unsupported assignment - more than one write per thread")
          result.writeBuf = checkWriteTarget(w.lhs)
          valueAst = w.rhs
          result.value = emitExpr(w.rhs, tidName, "tid", false)
      else:
        raise leanRefusal("unsupported statement " & $stmt.kind &
          " - statement at kernel-body top level")
    if scalarName != "":
      raise leanRefusal("unsupported kernel - scalar parameter '" & scalarName &
        "' on the guarded-thread shape, the guarded-thread shape takes no scalar" &
        " parameter (the scalar is the axpy loop shape)")

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
  result.scalarName = scalarName
  for p in kernel.params:
    if isUint32Buffer(p.typ) and p.name != result.writeBuf:
      result.readBufs.add p.name

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
  ## Module docstring of the emitted Lean module, one variant per kernel shape
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
  const loopDoc = """
/-!
# Emitted @KERNEL@ verification module

Produced by `lang_lean.nim` from the post-pass CIRv2 of the crucible kernel
`@KERNEL@`. `tests/codegen/lean/t_lean_loop.nim` regenerates this file; edit the emitter,
not this file.

The kernel is one loop over the compile-time constant range `[0, @N@)`. The
emitted def runs the loop as a structural recursion on the remaining iteration
count `k`, threading the output buffer through functional updates, so the
kernel stays a loop in Lean and nothing is unrolled. Buffers are total
functions `Nat → UInt32`, and `UInt32.add` wraps modulo 2^32 like the GPU
arithmetic the kernel emits.
@SCALARDOC@
Divergence note: the Lean loop counter is a `Nat`, unbounded, while the GPU
counter is 32-bit and Nim wraps it at 2^32. The bound `@N@` is a compile-time
literal, so the counter provably stays in `[0, @N@)` and the two encodings
agree; a data-dependent bound would need an overflow side condition here.

Proof note: the loop-shape theorems may depend on the classical axioms
`[propext, Classical.choice, Quot.sound]` (the `by_cases` case splits and
`omega` fallbacks in the proofs), unlike the add-shape theorems, which stay
constructive with `[propext]` alone.
-/
"""
  let scalarDoc = if model.scalarName == "":
      ""
    else:
      "The scalar parameter `" & model.scalarName & "` is uniform across the\n" &
      "iterations: it maps to a `UInt32` argument appended after the buffer\n" &
      "binders of the emitted defs, and `UInt32.mul` wraps modulo 2^32 like\n" &
      "`UInt32.add`.\n\n"
  let doc = case model.kind
    of kkGuardedThread: threadDoc
    of kkLoopRange: loopDoc
  result = doc.replace("@KERNEL@", model.kernelName)
    .replace("@N@", $model.nValue).replace("@SCALARDOC@", scalarDoc)

proc scalarBinder(model: KernelModel): string =
  ## Lean binder text of the scalar parameter, "" when the kernel has none:
  ## a `UInt32` argument appended after the buffer binders of the emitted defs
  if model.scalarName == "":
    return ""
  result = " (" & model.scalarName & " : UInt32)"

proc scalarArg(model: KernelModel): string =
  ## Lean argument text of the scalar parameter, "" when the kernel has none
  if model.scalarName == "":
    return ""
  result = " " & model.scalarName

proc emitBodyDef(model: KernelModel): string =
  ## Emits the body def(s) of the module for the kernel shape.
  const doc = """
/-- Per-thread body of the kernel: returns the output buffer after thread
`tid` of `n` has run.

`a` and `b` are the input buffers, `out` the output buffer before the write.
Precondition: `tid < n` (each thread is launched with its id inside the
buffer bound). Threads with `tid ≥ n` leave the buffer untouched, mirroring
the guard the crucible frontend must emit around every body. -/
"""
  const loopDoc = """
/-- Remaining-count recursion of the loop: runs the iterations at remaining
count `k = @N@ .. 1`, each writing index `@N@ - k` of the running output
buffer, and returns the buffer after the last one.

`a` and `b` are the input buffers, `out` the output buffer before the loop.
The recursion is structural on the remaining count: it terminates by
construction, so no `termination_by` is emitted and the Lean kernel-reducer
can evaluate the def. -/
"""
  const loopEntryDoc = """
/-- Entry point: the output buffer after the whole `[0, @N@)` loop has run. -/
"""
  case model.kind
  of kkGuardedThread:
    result = doc &
      "def " & model.bodyName & " (tid n : Nat) (a b out : Nat → UInt32) : Nat → UInt32 :=\n" &
      "  if tid < n then fun i => if i = tid then " & model.value & " else out i\n" &
      "  else out\n"
  of kkLoopRange:
    let loopName = leanLoopName(model.kernelName)
    let n = $model.nValue
    result = loopDoc.replace("@N@", n) &
      "def " & loopName & " (k : Nat) (a b out : Nat → UInt32)" & model.scalarBinder &
      " : Nat → UInt32 :=\n" &
      "  match k with\n" &
      "  | 0 => out\n" &
      "  | k+1 =>\n" &
      "    " & loopName & " k a b " &
        "(fun j => if j = " & n & " - (k + 1) then " & model.value & " else out j)" &
        model.scalarArg & "\n" &
      "\n" & loopEntryDoc.replace("@N@", n) &
      "def " & model.bodyName & " (a b out : Nat → UInt32)" & model.scalarBinder &
      " : Nat → UInt32 :=\n" &
      "  " & loopName & " " & n & " a b out" & model.scalarArg & "\n"

proc emitLoopTheorems(model: KernelModel): string =
  ## Loop-shape spec theorems: the loop-state lemma, then the two postcondition
  ## halves derived from it at the entry count. Every proof depends on the def shape alone.
  let loopName = leanLoopName(model.kernelName)
  let n = $model.nValue
  let scalarBinder = model.scalarBinder
  let scalarArg = model.scalarArg
  let scalarImplicit = if model.scalarName == "":
      ""
    else:
      " {" & model.scalarName & " : UInt32}"
  result = "/-- Loop-state lemma: with the iterations at remaining count `k` still to\n" &
    "run, index `j` of the running buffer ends at the stored value iff it lies\n" &
    "in the not-yet-written range `[" & n & " - k, " & n & ")`, and keeps its old value\n" &
    "otherwise. Every spec theorem below derives from this induction\n" &
    "hypothesis at the entry count. -/\n" &
    "theorem " & loopName & "_eq (k : Nat) (a b out : Nat → UInt32)" & model.scalarBinder & " (j : Nat) :\n" &
    "    " & loopName & " k a b out" & model.scalarArg & " j =\n" &
    "      if " & n & " - k ≤ j ∧ j < " & n & " then " & model.value & " else out j := by\n" &
    "  induction k generalizing out j with\n" &
    "  | zero => rw [" & loopName & ", if_neg (by omega)]\n" &
    "  | succ k ih =>\n" &
    "    rw [" & loopName & ", ih]\n" &
    "    by_cases hc2 : " & n & " - k ≤ j ∧ j < " & n & "\n" &
    "    · by_cases hc1 : " & n & " - (k + 1) ≤ j ∧ j < " & n & "\n" &
    "      · rw [if_pos hc2, if_pos hc1]\n" &
    "      · exact absurd hc1 (by omega)\n" &
    "    · by_cases hc1 : " & n & " - (k + 1) ≤ j ∧ j < " & n & "\n" &
    "      · have heq : j = " & n & " - (k + 1) := by omega\n" &
    "        rw [if_neg hc2, if_pos hc1, if_pos heq]\n" &
    "      · have hne : ¬ (j = " & n & " - (k + 1)) := by omega\n" &
    "        rw [if_neg hc2, if_neg hc1, if_neg hne]\n" &
    "\n" &
    "/-- Postcondition, write half: every index of the `[0, " & n & ")` range\n" &
    "holds the stored value after the loop. -/\n" &
    "theorem " & model.bodyName & "_writes_own_index {a b out : Nat → UInt32}" & scalarImplicit & " {j : Nat} (h : j < " & n & ") :\n" &
    "    " & model.bodyName & " a b out" & scalarArg & " j = " & model.value & " := by\n" &
    "  rw [" & model.bodyName & ", " & loopName & "_eq, if_pos (by omega)]\n" &
    "\n" &
    "/-- Postcondition, no-collateral-damage half: every index outside the\n" &
    "range keeps its old value. -/\n" &
    "theorem " & model.bodyName & "_preserves_others {a b out : Nat → UInt32}" & scalarImplicit & " {j : Nat} (h : " & n & " ≤ j) :\n" &
    "    " & model.bodyName & " a b out" & scalarArg & " j = out j := by\n" &
    "  rw [" & model.bodyName & ", " & loopName & "_eq, if_neg (by omega)]\n"

proc emitTheorems(model: KernelModel): string =
  ## Emits the spec theorems of the kernel's shape.
  ## Every proof depends on the def shape alone, never on the stored value.
  if model.kind == kkLoopRange:
    return emitLoopTheorems(model)
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
  result = stripKernelSuffix(kernelName) & ".autogen.lean"

proc codegen*(m: CirModule): string =
  ## Lean verification module of the module's single `{.global.}` kernel,
  ## the dispatch target of `codegen(m, cgLean)`
  let kernels = getModuleKernels(m)
  if kernels.len != 1:
    raise leanRefusal("unsupported module - expected exactly one `{.global.}`" &
      " kernel, got " & $kernels.len)
  result = emitText(kernels[0])

# ── Verification scaffolds ───────────────────────────────────────────────────

## Lean properties cannot be written in Nim, so each pragma payload is a string holding
## a Lean property expression over the kernel's parameter names,
## plus the model binders `tid` (thread id) and `n` (buffer bound).
##
## - `{.precondition: "<payload>".}` - assumed at launch, never proved
## - `{.invariant: "<payload>".}` - true whenever the guarded branch is entered, implied by the guard `tid < n`
## - `{.postcondition: "<payload>".}` - true of the written buffer after the thread ran
##
## - `prop := cmp ('∧' cmp)*`
## - `cmp  := sum ('=' | '≠' | '<' | '≤') sum`
## - `sum  := term ('+' term)*`
## - `term := ident | ident 'tid' | uint literal`
##
## Example: `{.postcondition: "output tid = a tid + b tid".}` on the add kernel.
## A well-formed payload stating a false property fails its Lean proof attempt and is downgraded to a sorry-backed scaffold.
## Anything the grammar cannot parse is refused loudly.

type
  RawSpec* = object
    ## One pragma captured off the kernel proc, before the payload is parsed
    name*: string    ## one of precondition, invariant, postcondition
    payload*: string ## the string payload verbatim

  SpecKind* = enum
    skPrecondition
    skInvariant
    skPostcondition

  ScaffoldData* = object
    ## One prepared scaffold, a data record the compile-time capture hands
    ## to the runtime settling
    kind*: SpecKind
    kernelName*: string ## crucible kernel proc name, for the scaffold docs and the manifest
    payload*: string   ## pragma payload verbatim, for the manifest
    leanName*: string  ## emitted theorem name, empty for the assumed precondition
    statement*: string ## Lean property text between `:` and `:= by`
    binders*: string   ## buffer binders of the theorem, e.g. `a b out`
    attempt*: string   ## mechanical proof attempt, downgraded to `sorry` on failure
    nValue*: int       ## the buffer bound, rendered in the loop-shape scaffold doc
    loopShape*: bool   ## the loop-shape scaffold renders over the remaining count `k`

const specPragmaNames* = ["precondition", "invariant", "postcondition"]
  ## Verification pragma vocabulary: one source,
  ## consumed by the pragma capture (test harness) and `specKindByName`

proc specKindByName*(name: string): SpecKind =
  ## Maps a captured pragma name to its kind, refusing anything else
  for i, n in specPragmaNames:
    if n == name:
      return SpecKind(i)
  raise leanRefusal("unsupported spec pragma '" & name & "'")

static:
  # array order matches the SpecKind enum order
  doAssert specKindByName("precondition") == skPrecondition
  doAssert specKindByName("invariant") == skInvariant
  doAssert specKindByName("postcondition") == skPostcondition

template precondition*(expr: static string) {.pragma.}
  ## Payload: a Lean property assumed true at kernel launch, like `"tid < n"`.

template invariant*(expr: static string) {.pragma.}
  ## Payload: a Lean property implied by the guard of the guarded branch, `tid < n` in the model, like `"tid < n"`.

template postcondition*(expr: static string) {.pragma.}
  ## Payload: a Lean property over the kernel parameters, `tid` and `n`,
  ## stating the written buffer's state after the thread ran, like `"output tid = a tid + b tid"`.

# ── Payload grammar ──────────────────────────────────────────────────────────

type
  SpecTokenKind = enum
    tkIdent  ## `[A-Za-z_][A-Za-z0-9_]*`
    tkUint   ## `[0-9]+`
    tkPlus   ## `+`
    tkEq     ## `=`
    tkNeq    ## `≠`
    tkLt     ## `<`
    tkLe     ## `≤`
    tkAnd    ## `∧`
    tkEof

  SpecToken = object
    kind*: SpecTokenKind
    text*: string ## identifier name or literal digits

  SpecExprKind* = enum
    seIdent ## identifier binder: `tid` or `n`
    seApp   ## buffer read: `<buffer> tid`
    seUint  ## unsigned integer literal
    seBin   ## binary operator: `+`, `=`, `≠`, `<`, `≤`, `∧`

  SpecExpr* = ref object
    ## Parsed payload. The grammar fixes precedence (`∧` < cmp < `+`).
    ## `seBin` chains are left-associative, so rendering emits no parentheses.
    case kind*: SpecExprKind
    of seIdent:
      name*: string
    of seApp:
      fn*: string ## the buffer parameter name, applied to `tid`
    of seUint:
      value*: string
    of seBin:
      op*: string
      left*, right*: SpecExpr

proc tokenizeSpec(payload: string): seq[SpecToken] =
  ## Splits a payload into spec tokens, refusing characters outside the grammar with a `lean_lang: invalid spec payload` ValueError
  var i = 0
  while i < payload.len:
    let c = payload[i]
    if c in {' ', '\t'}:
      inc i
    elif c in IdentStartChars:
      let start = i
      while i < payload.len and payload[i] in IdentChars:
        inc i
      result.add SpecToken(kind: tkIdent, text: payload[start ..< i])
    elif c in {'0' .. '9'}:
      let start = i
      while i < payload.len and payload[i] in {'0' .. '9'}:
        inc i
      result.add SpecToken(kind: tkUint, text: payload[start ..< i])
    elif c == '+':
      result.add SpecToken(kind: tkPlus, text: "+")
      inc i
    elif c == '=':
      result.add SpecToken(kind: tkEq, text: "=")
      inc i
    elif c == '<':
      result.add SpecToken(kind: tkLt, text: "<")
      inc i
    elif payload[i ..< min(i + 3, payload.len)] == "≠":
      result.add SpecToken(kind: tkNeq, text: "≠")
      inc i, 3
    elif payload[i ..< min(i + 3, payload.len)] == "≤":
      result.add SpecToken(kind: tkLe, text: "≤")
      inc i, 3
    elif payload[i ..< min(i + 3, payload.len)] == "∧":
      result.add SpecToken(kind: tkAnd, text: "∧")
      inc i, 3
    else:
      raise leanRefusal("invalid spec payload '" & payload &
        "' - unexpected character '" & $c & "' at byte " & $i)

proc peek(toks: seq[SpecToken], i: int): SpecToken =
  ## Token at `i`, or `tkEof` at the end of input
  if i < toks.len:
    return toks[i]
  result = SpecToken(kind: tkEof)

proc parseSum(toks: seq[SpecToken], i: var int): SpecExpr
proc parseCmp(toks: seq[SpecToken], i: var int): SpecExpr
proc parseProp(toks: seq[SpecToken], i: var int): SpecExpr

proc parseTerm(toks: seq[SpecToken], i: var int): SpecExpr =
  ## term := ident | ident 'tid' | uint literal
  let tok = toks.peek(i)
  case tok.kind
  of tkIdent:
    inc i
    if toks.peek(i).kind == tkIdent and toks.peek(i).text == "tid":
      inc i
      result = SpecExpr(kind: seApp, fn: tok.text)
    else:
      result = SpecExpr(kind: seIdent, name: tok.text)
  of tkUint:
    inc i
    result = SpecExpr(kind: seUint, value: tok.text)
  else:
    raise leanRefusal("invalid spec payload - expected an identifier or a" &
      " uint literal, got '" & tok.text & "'")

proc parseSum(toks: seq[SpecToken], i: var int): SpecExpr =
  ## sum := term ('+' term)*
  result = parseTerm(toks, i)
  while toks.peek(i).kind == tkPlus:
    inc i
    let r = parseTerm(toks, i)
    result = SpecExpr(kind: seBin, op: "+", left: result, right: r)

proc parseCmp(toks: seq[SpecToken], i: var int): SpecExpr =
  ## cmp := sum ('=' | '≠' | '<' | '≤') sum
  result = parseSum(toks, i)
  let tok = toks.peek(i)
  if tok.kind in {tkEq, tkNeq, tkLt, tkLe}:
    inc i
    let r = parseSum(toks, i)
    result = SpecExpr(kind: seBin, op: tok.text, left: result, right: r)

proc parseProp(toks: seq[SpecToken], i: var int): SpecExpr =
  ## prop := cmp ('∧' cmp)*
  result = parseCmp(toks, i)
  while toks.peek(i).kind == tkAnd:
    inc i
    let r = parseCmp(toks, i)
    result = SpecExpr(kind: seBin, op: "∧", left: result, right: r)

proc parseSpecPayload*(payload: string): SpecExpr =
  ## Parses one payload into a `SpecExpr`, refusing every construct outside
  ## the grammar with a `lean_lang: invalid spec payload` ValueError
  let toks = tokenizeSpec(payload)
  var i = 0
  result = parseProp(toks, i)
  if toks.peek(i).kind != tkEof:
    raise leanRefusal("invalid spec payload '" & payload &
      "' - trailing input at token '" & toks.peek(i).text & "'")

proc specRefs*(e: SpecExpr): seq[string] =
  ## Names referenced by the payload: identifier binders (`tid`, `n`) and applied buffers, deduplicated in first-occurrence order
  case e.kind
  of seIdent:
    if e.name notin result:
      result.add e.name
  of seApp:
    if e.fn notin result:
      result.add e.fn
  of seUint:
    discard
  of seBin:
    result = specRefs(e.left)
    for r in specRefs(e.right):
      if r notin result:
        result.add r

proc renderSpec*(e: SpecExpr, nameMap: Table[string, string], bareMap: Table[string, string] = initTable[string, string]()): string =
  ## Renders the payload as Lean text over the model's names.
  ##
  ## - `nameMap` maps each kernel buffer parameter to its emitted Lean name,
  ##   the written buffer to the post-state application of the per-thread body
  ## - `bareMap` maps an identifier binder to its rendered Lean text: the loop
  ##   shape renders the loop counter over the recursion's remaining-count
  ##   parameter (`i` -> `4 - k`), any other identifier binder renders as itself
  ##   (`tid`, `n`)
  ## - a buffer read renders as `<mapped name> tid`
  ## - everything else is refused loudly: an unapplied buffer name or unknown
  ##   identifier is no Lean Prop over the model
  case e.kind
  of seIdent:
    if e.name in bareMap:
      return bareMap[e.name]
    if e.name in ["tid", "n"]:
      return e.name
    if e.name in nameMap:
      raise leanRefusal("invalid spec payload - '" & e.name &
        "' must be applied to `tid`")
    raise leanRefusal("invalid spec payload - '" & e.name &
      "' is neither `tid`, `n` nor a kernel buffer parameter")
  of seApp:
    if e.fn notin nameMap:
      raise leanRefusal("invalid spec payload - '" & e.fn &
        " tid' is not a read of a kernel buffer parameter")
    result = nameMap[e.fn] & " tid"
  of seUint:
    result = e.value
  of seBin:
    result = renderSpec(e.left, nameMap, bareMap) & " " & e.op & " " &
      renderSpec(e.right, nameMap, bareMap)

# ── Scaffold preparation (compile-time, against the kernel model) ───────────

proc prepareSpec(model: KernelModel, spec: RawSpec): ScaffoldData =
  ## Parses, validates and renders one captured pragma payload against the kernel model.
  ##
  ## - out-of-scope payloads are loud ValueErrors
  ## - a false property is not judged here. It fails the Lean proof attempt
  ##   and is downgraded (see `settleModule`)
  let kind = specKindByName(spec.name)
  if model.scalarName != "":
    raise leanRefusal("unsupported spec pragma '" & spec.name & "' on kernel '" &
      model.kernelName & "' - the scalar-parameter (axpy) shape is outside" &
      " the scaffold model")
  let expr = parseSpecPayload(spec.payload)
  var nameMap: Table[string, string]
  for r in model.readBufs:
    nameMap[r] = r
  var bareMap: Table[string, string]
  if model.kind == kkLoopRange:
    # loop-shape binders: remaining-count parameter `k` renders the loop
    # counter as `n - k`
    nameMap[model.writeBuf] = model.bodyName & " k a b out"
    bareMap[model.loopVar] = $model.nValue & " - k"
  else:
    nameMap[model.writeBuf] = model.bodyName & " tid n " &
      (model.readBufs.join(" ") & " out")
  result = ScaffoldData(kind: kind, kernelName: model.kernelName,
    payload: spec.payload, nValue: model.nValue,
    loopShape: model.kind == kkLoopRange,
    statement: renderSpec(expr, nameMap, bareMap))
  case kind
  of skPrecondition:
    # a precondition is assumed, never proved: it must match the model's
    # precondition `tid < n` exactly, anything else cannot be checked
    if model.kind == kkLoopRange:
      raise leanRefusal("unsupported precondition payload '" & spec.payload &
        "' - the loop shape has no thread-id launch precondition; the emitted" &
        " loop module proves the whole `[0, " & $model.nValue & ")` range")
    if result.statement != "tid < n":
      raise leanRefusal("unsupported precondition payload '" & spec.payload &
        "' - the model's launch precondition is `tid < n`")
  of skInvariant:
    let refs = specRefs(expr)
    if model.kind == kkLoopRange:
      # a loop-counter invariant, implied by the loop guard: over the loop
      # variable only. The running-buffer state property is outside the payload
      # grammar (the emitter derives it as the `<loop>_eq` lemma)
      if model.loopVar notin refs:
        raise leanRefusal("unsupported invariant payload '" & spec.payload &
          "' - the loop-shape invariant must mention the loop counter '" &
          model.loopVar & "'")
      for r in refs:
        if r != model.loopVar:
          raise leanRefusal("unsupported invariant payload '" & spec.payload &
            "' - the loop-shape invariant may only mention the loop counter '" &
            model.loopVar & "' (no `n`, no buffers: the loop bound is a" &
            " compile-time literal and the buffer state is the emitted" &
            " `" & leanLoopName(model.kernelName) & "_eq` lemma)")
      result.leanName = leanLoopName(model.kernelName) & "_invariant"
      result.attempt = "exact h"
    else:
      # a guard invariant is proved from the guard alone: over `tid` and `n` only
      for r in ["tid", "n"]:
        if r notin refs:
          raise leanRefusal("unsupported invariant payload '" & spec.payload &
            "' - the invariant must be implied by the guard `tid < n`" &
            " and mention both `tid` and `n`")
      for r in refs:
        if r notin ["tid", "n"]:
          raise leanRefusal("unsupported invariant payload '" & spec.payload &
            "' - the invariant may only mention `tid` and `n`")
      result.leanName = model.bodyName & "_invariant"
      result.attempt = "exact h"
  of skPostcondition:
    # postcondition: the written buffer's post-state, the per-thread body
    # applied at `tid`
    if model.kind == kkLoopRange:
      raise leanRefusal("unsupported postcondition payload '" & spec.payload &
        "' - the loop postcondition is the composed `∀ j` contract, outside" &
        " the payload grammar; the emitter derives it as `" &
        model.bodyName & "_post`")
    if model.writeBuf notin specRefs(expr):
      raise leanRefusal("unsupported postcondition payload '" & spec.payload &
        "' - it must mention the written buffer '" & model.writeBuf & "'")
    result.leanName = model.bodyName & "_postcondition"
    result.binders = model.readBufs.join(" ") & " out"
    result.attempt = "simp [" & model.bodyName & ", h]"

proc prepareScaffolds*(kernel: CirProc, specs: seq[RawSpec]): seq[ScaffoldData] =
  ## Analyzes the kernel and prepares one scaffold per captured verification pragma, in the canonical precondition/invariant/postcondition order.
  ## Raises a loud `lean_lang: ...` ValueError on any out-of-scope payload.
  let model = analyzeKernel(kernel)
  var ordered: array[3, RawSpec]
  var seen: set[SpecKind]
  for spec in specs:
    let kind = specKindByName(spec.name)
    if kind in seen:
      raise leanRefusal("more than one '" & spec.name & "' pragma on the kernel")
    seen.incl kind
    ordered[ord(kind)] = spec
  for i in 0 ..< 3:
    if SpecKind(i) in seen:
      result.add prepareSpec(model, ordered[i])

proc scaffoldText(s: ScaffoldData, proof: string): string =
  ## One Lean spec scaffold: the precondition comment, or a theorem carrying
  ## the given proof text (the mechanical attempt, or `sorry` after downgrade)
  case s.kind
  of skPrecondition:
    const doc = """
/- Precondition from the `{.precondition.}` pragma on `@KERNEL@`: `@STMT@`.
Assumed at launch, carried as the hypothesis `h` by the scaffolds below. A
precondition is an assumption, so it gets no Lean declaration. -/
"""
    result = doc.replace("@KERNEL@", s.kernelName).replace("@STMT@", s.statement)
  of skInvariant:
    if s.loopShape:
      const loopDoc = """
/-- Invariant from the `{.invariant.}` pragma on `@KERNEL@`, payload
`@PAYLOAD@`: a property of the loop counter, rendered over the recursion's
remaining-count parameter `k` (the iteration at remaining count `k` writes
index `@N@ - k`). Implied by the loop guard, so it holds whenever an
iteration body runs. -/
theorem @NAME@ {k : Nat} (h : @N@ - k < @N@) :
    @STMT@ := by
  @PROOF@
"""
      result = loopDoc.replace("@KERNEL@", s.kernelName)
        .replace("@PAYLOAD@", s.payload).replace("@NAME@", s.leanName)
        .replace("@N@", $s.nValue).replace("@STMT@", s.statement)
        .replace("@PROOF@", proof)
    else:
      const doc = """
/-- Invariant from the `{.invariant.}` pragma on `@KERNEL@`, payload
`@PAYLOAD@`: a property implied by the guard `tid < n`, so it holds whenever
the guarded branch is entered. -/
theorem @NAME@ {tid n : Nat} (h : tid < n) :
    @STMT@ := by
  @PROOF@
"""
      result = doc.replace("@KERNEL@", s.kernelName).replace("@PAYLOAD@", s.payload)
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
    result = doc.replace("@KERNEL@", s.kernelName).replace("@PAYLOAD@", s.payload)
      .replace("@NAME@", s.leanName).replace("@BINDERS@", s.binders)
      .replace("@STMT@", s.statement).replace("@PROOF@", proof)

# ── Settling (runtime: `lake env lean` cannot run inside a compile-time macro) ─

proc allDigits(s: string): bool =
  ## Whether every character of `s` is a decimal digit, false when empty
  if s.len == 0:
    return false
  for ch in s:
    if ch notin {'0' .. '9'}:
      return false
  result = true

proc parseLeanDiagnostics*(leanOutput: string): seq[tuple[line: int, excerpt: string]] =
  ## Extracts the `error:` diagnostics of a Lean run as (line, message) pairs.
  ##
  ## Returns:
  ## - one entry per error line: the `file:line:` position and the message text
  ## - `line = -1` for an error line the head cannot position, so the caller
  ##   treats it as unattributable and refuses on it
  for ln in leanOutput.splitLines():
    let idx = ln.find(": error: ")
    if idx < 0:
      continue
    let head = ln[0 ..< idx].split(':')
    var line = -1
    if head.len >= 2 and allDigits(head[^2]):
      line = parseInt(head[^2])
    var excerpt = ln[idx + ": error: ".len .. ^1]
    if excerpt.len > 160:
      excerpt = excerpt[0 ..< 157] & "..."
    result.add (line, excerpt)

proc findLakeDir(outPath: string): string =
  ## Walks up from `outPath` to the first directory containing a lakefile.lean
  var dir = parentDir(outPath)
  while dir != "" and not fileExists(dir / "lakefile.lean"):
    dir = parentDir(dir)
  if dir == "":
    raise leanRefusal("no lakefile.lean above the emitted module " & outPath)
  result = dir

proc settleModule*(baseText: string, scaffolds: seq[ScaffoldData], genDir: string, module: string): string =
  ## Settles the scaffold proof attempts of an emitted module and writes it:
  ##
  ## - stages the module with the mechanical proof attempts under
  ##   `<module>.settling.lean` and elaborates it through `lake env lean`
  ## - downgrades every failing proof to a `sorry`-backed scaffold recorded
  ##   unproven in the JSON manifest beside the module
  ## - writes the final module to `genDir / module`, returning the final text
  ##
  ## With no scaffolds the base module is written as-is, no manifest.
  ## Settling failures publish nothing: the staged file is dropped and no
  ## unverified module or manifest stays on disk
  let outPath = genDir / module
  createDir(parentDir(outPath))
  if scaffolds.len == 0:
    writeFile(outPath, baseText)
    return baseText

  # first pass: the mechanical proof attempts go to a staging file.
  # module output at `outPath` appears only after the settling succeeds.
  # a refusal drops the staged file, no unverified module stays on disk
  var text = baseText
  var startLines: seq[int]
  for s in scaffolds:
    text &= "\n"
    startLines.add text.count('\n') + 1
    text &= scaffoldText(s, s.attempt)
  let lakeDir = findLakeDir(outPath)
  let relFile = outPath.relativePath(lakeDir) ## the manifest's module field
  let stagingPath = outPath.changeFileExt("settling.lean")
  writeFile(stagingPath, text)

  proc abortSettling(msg: string): ref ValueError =
    ## Drops the staged module: nothing was settled, no artifact may stay
    removeFile(stagingPath)
    result = leanRefusal(msg)

  # settle the proof attempts by elaboration (a failed proof errors, a `sorry` only warns)
  let (outp, exitCode) = execCmdEx("lake env lean " & stagingPath.relativePath(lakeDir),
    workingDir = lakeDir)
  var failed: Table[int, string] ## scaffold index -> error excerpt
  var positionless: seq[string] ## error lines with no file:line position
  let lastLine = text.count('\n') + 1
  for (line, excerpt) in parseLeanDiagnostics(outp):
    if line < 0:
      positionless.add excerpt
      continue
    var owner = -1
    for i, startLine in startLines:
      let endLine = if i + 1 < startLines.len: startLines[i + 1] - 1
                    else: lastLine
      if line >= startLine and line <= endLine:
        owner = i
    if owner < 0:
      raise abortSettling("proof settling failed outside the pragma" &
        " scaffolds (line " & $line & "): " & excerpt)
    failed[owner] = excerpt
  if exitCode != 0 and positionless.len > 0:
    # an error line the extractor could not position: it may name the module
    # itself (a duplicate declaration, a stale olean), so the scaffold
    # verdicts below would be recorded over a module that does not elaborate
    raise abortSettling("proof settling failed - " & $positionless.len &
      " lean error line(s) carry no file:line position, the scaffold verdicts" &
      " would be a guess: " & positionless.join(" | "))
  if exitCode != 0 and failed.len == 0:
    # no error line was attributable to a scaffold, so no proof was settled:
    # a `proven` verdict would be a guess over a missing toolchain, a Lean crash or a diagnostic format change
    var excerpt = outp
    if excerpt.len > 800:
      excerpt = excerpt[0 ..< 797] & "..."
    raise abortSettling("proof settling failed - lean exited with " &
      $exitCode & " but no error was attributable to a scaffold: " & excerpt)

  if failed.len > 0:
    # second pass: downgrade every failing proof to a sorry-backed scaffold
    text = baseText
    for i, s in scaffolds:
      text &= "\n"
      let proof = if i in failed: "sorry" else: s.attempt
      text &= scaffoldText(s, proof)
  writeFile(outPath, text)
  removeFile(stagingPath)

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
  manifest["kernel"] = newJString(scaffolds[0].kernelName)
  manifest["module"] = newJString(relFile)
  manifest["scaffolds"] = entries
  writeFile(outPath.changeFileExt("manifest.json"), manifest.pretty & "\n")
  result = text
