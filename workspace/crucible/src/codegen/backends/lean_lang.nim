# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## lean_lang: emit a Lean 4 verification module from crucible's post-pass GpuAst
##
##   post-pass GpuAst (one `{.global.}` kernel) → this printer → Lean module + JSON manifest
##
## Walks the GpuAst of one `{.global.}` kernel, emitting the body def plus derived
## spec theorems. Anything outside the two supported kernel shapes raises a loud `lean_lang: unsupported ...` ValueError:
## - guarded thread (the `AddKernel.lean` model): buffers `a`, `b` (read) +
##   `output` (written), let-bound `thread_position_in_grid.x` (the Lean `tid`),
##   one bare `if` with guard `tidVar < literal`, positive literal `n` (the length)
## - loop: one `for` over the compile-time constant range `0 ..< n`, start 0,
##   step 1, exclusive, one indexed write over the loop variable and the buffers
##
## - `{.precondition.}` - an assumption, carried by the scaffolds as hypothesis `h` (guarded thread only)
## - `{.invariant.}` - implied by the guard, proved by `exact h` (loop shape: a loop-counter property over the remaining count `k`)
## - `{.postcondition.}` - stated against the per-thread body, attempted with the derived `simp` scripts (guarded thread only)
##
## A failed scaffold proof is downgraded to a `sorry`-backed scaffold, recorded unproven in a JSON manifest beside the module.
## The pragma payloads use the payload language of the frontend's `lean_pragmas.nim`.

import std/[json, os, osproc, sets, strutils, tables]
import workspace/crucible/src/crucible_internal_compiler

proc leanRefusal(msg: string): ref ValueError =
  result = newException(ValueError, msg)


# ── Payload grammar ──────────────────────────────────────────────────────────

type
  RawSpec* = object
    ## One pragma captured off the kernel proc, before the payload is parsed
    name*: string    ## one of precondition, invariant, postcondition
    payload*: string ## the string payload verbatim

  SpecTokenKind* = enum
    tkIdent  ## `[A-Za-z_][A-Za-z0-9_]*`
    tkUint   ## `[0-9]+`
    tkPlus   ## `+`
    tkEq     ## `=`
    tkNeq    ## `≠`
    tkLt     ## `<`
    tkLe     ## `≤`
    tkAnd    ## `∧`
    tkEof

  SpecToken* = object
    kind*: SpecTokenKind
    text*: string ## identifier name or literal digits

  SpecExprKind* = enum
    seIdent ## bare binder: `tid` or `n`
    seApp   ## buffer read: `<buffer> tid`
    seUint  ## unsigned integer literal
    seBin   ## binary operator: `+`, `=`, `≠`, `<`, `≤`, `∧`

  SpecExpr* = ref object
    ## Parsed payload. The grammar fixes precedence (`∧` < cmp < `+`).
    ## `seBin` chains are left-associative, so rendering needs no parentheses.
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
      raise leanRefusal( "lean_lang: invalid spec payload '" &
        payload & "' - unexpected character '" & $c & "' at byte " & $i)

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
    raise leanRefusal( "lean_lang: invalid spec payload - " &
      "expected an identifier or a uint literal, got '" & tok.text & "'")

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
    raise leanRefusal( "lean_lang: invalid spec payload '" &
      payload & "' - trailing input at token '" & toks.peek(i).text & "'")

proc specRefs*(e: SpecExpr): seq[string] =
  ## Names referenced by the payload: bare binders (`tid`, `n`) and applied buffers, deduplicated in first-occurrence order
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
  ##   the written buffer to the post-state application `addBody tid n a b out`
  ##   of the per-thread body
  ## - `bareMap` maps a bare binder to its rendered Lean text. The loop shape
  ##   renders the loop counter through `bareMap` over the recursion's
  ##   remaining-count parameter: `i` -> `4 - k`. Any other bare binder renders
  ##   as itself (`tid`, `n`)
  ## - a buffer read renders as `<mapped name> tid`
  ## - everything else is refused loudly: a bare buffer name or an unknown
  ##   identifier is not a Lean Prop over the model
  case e.kind
  of seIdent:
    if e.name in bareMap:
      return bareMap[e.name]
    if e.name in ["tid", "n"]:
      return e.name
    if e.name in nameMap:
      raise leanRefusal( "lean_lang: invalid spec payload - '" &
        e.name & "' must be applied to `tid`")
    raise leanRefusal( "lean_lang: invalid spec payload - '" &
      e.name & "' is neither `tid`, `n` nor a kernel buffer parameter")
  of seApp:
    if e.fn notin nameMap:
      raise leanRefusal( "lean_lang: invalid spec payload - '" &
        e.fn & " tid' is not a read of a kernel buffer parameter")
    result = nameMap[e.fn] & " tid"
  of seUint:
    result = e.value
  of seBin:
    result = renderSpec(e.left, nameMap, bareMap) & " " & e.op & " " &
      renderSpec(e.right, nameMap, bareMap)

# ── Emission analysis ────────────────────────────────────────────────────────

type
  KernelKind = enum
    kkGuardedThread ## let-bound thread id, one guarded write (the P2 add shape)
    kkLoopRange     ## one `for` over a compile-time constant `[0, n)` range (P4)

  KernelModel = object
    kind: KernelKind
    kernelName: string   ## crucible kernel proc name, e.g. `addKernel`
    bodyName: string     ## emitted Lean def name, e.g. `addBody`
    readBufs: seq[string]  ## buffer params never written, in declaration order
    writeBuf: string     ## the buffer param written by the body
    value: string        ## emitted Lean text of the stored value, in terms of `tid` or the update binder `j`
    nValue: int          ## the guard literal (guarded-thread shape) or loop bound (loop shape)
    loopVar: string      ## loop shape: the Nim loop variable's name
    isAddShape: bool     ## guarded-thread shape: stored value is `<buf1> tid + <buf2> tid`, enables examples

proc unsupported(kind: GpuNodeKind, detail: string): ref ValueError =
  ## Loud refusal for every construct outside the tracer's scope
  result = newException(ValueError,
    "lean_lang: unsupported GpuAst node " & $kind & " - " & detail)

proc emitExpr(n: GpuAst, idxName: string, idxLean: string, bufNames: HashSet[string]): string =
  ## Renders one GpuAst expression as Lean text. The index variable `idxName`
  ## renders as its Lean name `idxLean` (`tid` or the update binder `j`).
  ## An indexed buffer read renders as a function application
  case n.kind
  of gpuIdent:
    let name = n.symbol.name
    if name == idxName:
      return idxLean
    if name in bufNames:
      return name
    raise unsupported(n.kind, "identifier '" & name &
      "' is not a kernel parameter or the index variable")
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
      let l = emitExpr(n.bLeft, idxName, idxLean, bufNames)
      let r = emitExpr(n.bRight, idxName, idxLean, bufNames)
      let lp = if n.bLeft.kind == gpuBinOp: "(" & l & ")" else: l
      let rp = if n.bRight.kind == gpuBinOp: "(" & r & ")" else: r
      return lp & " " & leanOp & " " & rp
    else:
      raise unsupported(n.kind, "binary operator '" & op & "'")
  of gpuIndex:
    if n.iArr.kind == gpuDeref and n.iArr.dOf.kind == gpuIdent and
        n.iArr.dOf.symbol.name in bufNames:
      if n.iIndex.kind != gpuIdent or n.iIndex.symbol.name != idxName:
        # a nested index like `a[a[i]]` would render a bogus Lean application
        raise unsupported(n.kind,
          "index expression is not the index variable (the model covers reads" &
          " at the thread's own index or the loop variable's index only)")
      return n.iArr.dOf.symbol.name & " " & emitExpr(n.iIndex, idxName, idxLean, bufNames)
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
  p.typ.kind == gtPtr and p.typ.to.kind == gtUncheckedArray and p.typ.to.uaTo.kind == gtUint32

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

proc leanLoopName(kernelName: string): string =
  ## `loopAdd` -> `loopAddLoop`, the remaining-count recursion def of the loop shape
  let body = leanBodyName(kernelName)
  if body.endsWith("Body"):
    body[0 ..^ 5] & "Loop"
  else:
    body & "Loop"

proc checkWriteTarget(lhs: GpuAst, bufNames: HashSet[string]): string =
  ## Validates an assignment target as a kernel buffer element write.
  ##
  ## Returns the written buffer's name, refusing loudly when the target is
  ## not a buffer element or names a non-parameter buffer.
  ##
  ## Shared by the loop path and the guarded-thread path, so both shapes
  ## refuse a malformed target with the same message.
  ##
  ## The index-variable check and the write-count policy stay at the call sites.
  if lhs.kind != gpuIndex or lhs.iArr.kind != gpuDeref or
      lhs.iArr.dOf.kind != gpuIdent:
    raise unsupported(gpuAssign, "assignment target is not a buffer element")
  let buf = lhs.iArr.dOf.symbol.name
  if buf notin bufNames:
    raise unsupported(gpuAssign,
      "assignment to '" & buf & "' which is not a kernel parameter")
  result = buf

proc analyzeLoopKernel(loop: GpuAst, bufNames: HashSet[string], model: var KernelModel): GpuAst =
  ## Walks the single `gpuFor` of the loop shape: a compile-time constant
  ## exclusive range `[0, n)` with step 1, one indexed write of a supported
  ## expression over the loop variable.
  ##
  ## Fills the loop fields of the model and returns the value expression,
  ## input to the kernel-level read-set check
  if loop.fRangeKind != rkExclusive:
    raise unsupported(gpuFor, "loop range is not `[0 ..< n)` exclusive - got " &
      $loop.fRangeKind & " (e.g. Nim `0 .. 3` or `countup`)")
  if loop.fStart.kind != gpuLit:
    raise unsupported(gpuFor, "loop start is not a compile-time literal")
  if loop.fStart.lValue != "0":
    raise unsupported(gpuFor, "loop start must be the literal 0 - got '" &
      loop.fStart.lValue & "'")
  if loop.fEnd.kind != gpuLit:
    raise unsupported(gpuFor, "loop bound is not a compile-time literal")
  let n = parseInt(loop.fEnd.lValue)
  if n <= 0:
    raise unsupported(gpuFor, "loop bound must be positive (got " &
      loop.fEnd.lValue & ")")
  model.nValue = n
  if loop.fStep.kind != gpuLit or loop.fStep.lValue != "1":
    raise unsupported(gpuFor, "loop step must be the literal 1 - got '" &
      loop.fStep.lValue & "'")
  if loop.fVar.kind != gpuIdent:
    raise unsupported(gpuFor, "loop variable is not an identifier")
  let loopVar = loop.fVar.symbol.name
  if loopVar in bufNames or loopVar == "j":
    raise unsupported(gpuFor, "loop variable '" & loopVar &
      "' collides with a kernel buffer parameter or the emitted update binder 'j'")
  model.loopVar = loopVar
  if loop.fBody.kind != gpuBlock or loop.fBody.statements.len != 1:
    raise unsupported(gpuFor, "loop body is not a single-statement block")
  let w = loop.fBody.statements[0]
  if w.kind != gpuAssign:
    raise unsupported(w.kind, "statement inside the loop body")
  let lhs = w.aLeft
  let buf = checkWriteTarget(lhs, bufNames)
  if lhs.iIndex.kind != gpuIdent or lhs.iIndex.symbol.name != loopVar:
    raise unsupported(gpuAssign,
      "write index is not the loop variable (write-set analysis covers the" &
      " whole-range shape only)")
  if buf != "output":
    raise unsupported(gpuAssign, "the written buffer must be `output` - got '" &
      buf & "'")
  model.writeBuf = buf
  result = w.aRight
  model.value = emitExpr(w.aRight, loopVar, "j", bufNames)

proc analyzeKernel(kernel: GpuAst): KernelModel =
  ## Walks one post-pass global kernel and derives the body model
  ##
  ## Fills:
  ## - the shape (guarded thread or constant-bound loop)
  ## - the thread id or loop variable and the guard literal or loop bound
  ## - the write set and the stored value
  ##
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
  let stmts = kernel.pBody.statements
  if stmts.len == 1 and stmts[0].kind == gpuFor:
    # the loop shape: everything before/after the loop is refused, the tracer kernel
    # is exactly one constant-bound for loop
    result.kind = kkLoopRange
    valueAst = analyzeLoopKernel(stmts[0], bufNames, result)
  else:
    for stmt in stmts:
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
          let buf = checkWriteTarget(lhs, bufNames)
          if lhs.iIndex.kind != gpuIdent or lhs.iIndex.symbol.name != tidName:
            raise unsupported(gpuAssign,
              "write index is not the thread id (write-set analysis covers the own-index shape only)")
          if result.writeBuf != "":
            raise unsupported(gpuAssign, "more than one write per thread")
          result.writeBuf = buf
          valueAst = w.aRight
          result.value = emitExpr(w.aRight, tidName, "tid", bufNames)
      else:
        raise unsupported(stmt.kind, "statement at kernel-body top level")
    if tidName == "":
      raise unsupported(gpuVar, "no `thread_position_in_grid.x` binding found")
    result.kind = kkGuardedThread

  if result.writeBuf == "":
    raise unsupported(gpuAssign, "no buffer write found in the kernel body")
  var readBufSet: HashSet[string]
  indexedReadBufs(valueAst, bufNames, readBufSet)
  if readBufSet != ["a", "b"].toHashSet():
    var readNames: seq[string]
    for r in readBufSet.items:
      readNames.add r
    raise unsupported(gpuProc, "the kernel body must read both input buffers" &
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
  ## Module docstring of the emitted Lean module, one variant per kernel shape
  const threadDoc = """
/-!
# Emitted @KERNEL@ verification module (P2)

Produced by `lean_lang.nim` from the post-pass GpuAst of the crucible kernel
`@KERNEL@`. `tests/codegen/lean/t_lean_emit.nim` regenerates this file; edit the emitter,
not this file.

The model matches `AddKernel.lean`: buffers are total functions `Nat → UInt32`,
writes are functional updates, and `UInt32.add` wraps modulo 2^32 like the
GPU arithmetic the kernel emits.
-/
"""
  const loopDoc = """
/-!
# Emitted @KERNEL@ verification module (P4)

Produced by `lean_lang.nim` from the post-pass GpuAst of the crucible kernel
`@KERNEL@`. `tests/codegen/lean/t_lean_loop.nim` regenerates this file; edit the emitter,
not this file.

The kernel is one loop over the compile-time constant range `[0, @N@)`. The
emitted def runs the loop as a structural recursion on the remaining iteration
count `k`, threading the output buffer through functional updates, so the
kernel stays a loop in Lean and nothing is unrolled. Buffers are total
functions `Nat → UInt32`, and `UInt32.add` wraps modulo 2^32 like the GPU
arithmetic the kernel emits.

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
  let doc = case model.kind
    of kkGuardedThread: threadDoc
    of kkLoopRange: loopDoc
  result = doc.replace("@KERNEL@", model.kernelName).replace("@N@", $model.nValue)

proc emitBodyDef(model: KernelModel): string =
  ## Body def(s) of the emitted module, per kernel shape: the guarded-thread
  ## `addBody` def or the loop shape's remaining-count recursion plus entry point
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
  const loopDoc = """
/-- Remaining-count recursion of the loop: runs the iterations at remaining
count `k = @N@ .. 1`, each writing index `@N@ - k` of the running output
buffer, and returns the buffer after the last one.

`a` and `b` are the input buffers, `out` the output buffer before the loop.
The recursion is structural on the remaining count: it terminates by
construction, so no `termination_by` is emitted and the Lean kernel-reducer
can evaluate the def (worked `decide` examples stay available). -/
"""
  const loopEntryDoc = """
/-- Entry point: the output buffer after the whole `[0, @N@)` loop has run. -/
"""
  case model.kind
  of kkGuardedThread:
    let doc = if model.isAddShape: addDoc else: genericDoc
    return doc.replace("@BODY@", model.bodyName).replace("@N@", $model.nValue) &
      "def " & model.bodyName & " (tid n : Nat) (a b out : Nat → UInt32) : Nat → UInt32 :=\n" &
      "  if tid < n then fun i => if i = tid then " & model.value & " else out i\n" &
      "  else out\n"
  of kkLoopRange:
    return loopDoc.replace("@N@", $model.nValue) &
      "def " & leanLoopName(model.kernelName) &
      " (k : Nat) (a b out : Nat → UInt32) : Nat → UInt32 :=\n" &
      "  match k with\n" &
      "  | 0 => out\n" &
      "  | k+1 =>\n" &
      "    " & leanLoopName(model.kernelName) & " k a b " &
        "(fun j => if j = " & $model.nValue & " - (k + 1) then " & model.value & " else out j)\n" &
      "\n" & loopEntryDoc.replace("@N@", $model.nValue) &
      "def " & model.bodyName & " (a b out : Nat → UInt32) : Nat → UInt32 :=\n" &
      "  " & leanLoopName(model.kernelName) & " " & $model.nValue & " a b out\n"

proc emitLoopTheorems(model: KernelModel): string =
  ## Loop-shape spec theorems: the loop-state lemma, then the four spec
  ## theorems derived from it at the entry count.
  ## All proofs close for any supported stored value: they depend on the def shape alone.
  let loopName = leanLoopName(model.kernelName)
  let n = $model.nValue
  result = "/-- Loop-state lemma: with the iterations at remaining count `k` still to\n" &
    "run, index `j` of the running buffer ends at the stored value iff it lies\n" &
    "in the not-yet-written range `[" & n & " - k, " & n & ")`, and keeps its old value\n" &
    "otherwise. The loop's induction hypothesis, every spec theorem below\n" &
    "derives from it at the entry count. -/\n" &
    "theorem " & loopName & "_eq (k : Nat) (a b out : Nat → UInt32) (j : Nat) :\n" &
    "    " & loopName & " k a b out j =\n" &
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
    "theorem " & model.bodyName & "_writes_own_index {a b out : Nat → UInt32} {j : Nat} (h : j < " & n & ") :\n" &
    "    " & model.bodyName & " a b out j = " & model.value & " := by\n" &
    "  rw [" & model.bodyName & ", " & loopName & "_eq, if_pos (by omega)]\n" &
    "\n" &
    "/-- Postcondition, no-collateral-damage half: every index outside the\n" &
    "range keeps its old value. -/\n" &
    "theorem " & model.bodyName & "_preserves_others {a b out : Nat → UInt32} {j : Nat} (h : " & n & " ≤ j) :\n" &
    "    " & model.bodyName & " a b out j = out j := by\n" &
    "  rw [" & model.bodyName & ", " & loopName & "_eq, if_neg (by omega)]\n" &
    "\n" &
    "/-- The loop postcondition as a single contract. -/\n" &
    "theorem " & model.bodyName & "_post {a b out : Nat → UInt32} :\n" &
    "    (∀ j, j < " & n & " → " & model.bodyName & " a b out j = " & model.value & ") ∧\n" &
    "      (∀ j, " & n & " ≤ j → " & model.bodyName & " a b out j = out j) :=\n" &
    "  ⟨fun _ hj => " & model.bodyName & "_writes_own_index hj,\n" &
    "    fun _ hj => " & model.bodyName & "_preserves_others hj⟩\n" &
    "\n" &
    "/-- Bounds theorem: the loop only changes indices inside `[0, " & n & ")`. This\n" &
    "is the property the codegen gate must establish for every emitted kernel\n" &
    "body. -/\n" &
    "theorem " & model.bodyName & "_only_writes_in_range {a b out : Nat → UInt32} {j : Nat}\n" &
    "    (h : " & model.bodyName & " a b out j ≠ out j) : j < " & n & " := by\n" &
    "  rw [" & model.bodyName & ", " & loopName & "_eq] at h\n" &
    "  by_cases hc : " & n & " - " & n & " ≤ j ∧ j < " & n & "\n" &
    "  · exact hc.2\n" &
    "  · rw [if_neg hc] at h\n" &
    "    exact absurd rfl h\n"

proc emitTheorems(model: KernelModel): string =
  ## Spec theorems per kernel shape:
  ## - the guarded-thread shape carries the `simp`-style P1 module proofs
  ## - the loop shape derives its four theorems from the emitted loop-state lemma
  ##
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
  if model.kind == kkLoopRange:
    return emitLoopTheorems(model)
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

# ── Verification scaffold emission (P3) ──────────────────────────────────────

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
    raise leanRefusal(
      "lean_lang: unsupported spec pragma '" & name & "'")

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
  var bareMap: Table[string, string]
  if model.kind == kkLoopRange:
    # the loop shape's binders: the recursion's remaining-count parameter `k`.
    # The iteration at remaining count `k` writes index `n - k`.
    # The loop variable therefore renders as `n - k`
    nameMap[model.writeBuf] = model.bodyName & " k a b out"
    bareMap[model.loopVar] = $model.nValue & " - k"
  else:
    nameMap[model.writeBuf] = model.bodyName & " tid n " &
      (model.readBufs.join(" ") & " out")
  result = Scaffold(kind: kind, payload: spec.payload,
    statement: renderSpec(expr, nameMap, bareMap))
  case kind
  of skPrecondition:
    # a precondition is assumed, never proved: it must match the model's
    # precondition `tid < n` exactly, anything else cannot be checked
    if model.kind == kkLoopRange:
      raise leanRefusal(
        "lean_lang: unsupported precondition payload '" & spec.payload &
        "' - the loop shape has no thread-id launch precondition; the emitted" &
        " loop module proves the whole `[0, " & $model.nValue & ")` range")
    if result.statement != "tid < n":
      raise leanRefusal(
        "lean_lang: unsupported precondition payload '" & spec.payload &
        "' - the tracer model's launch precondition is `tid < n`")
  of skInvariant:
    let refs = specRefs(expr)
    if model.kind == kkLoopRange:
      # a loop-counter invariant, implied by the loop guard: over the loop
      # variable only. The running-buffer state property is outside the payload
      # grammar (the emitter derives it as the `<loop>_eq` lemma)
      if model.loopVar notin refs:
        raise leanRefusal(
          "lean_lang: unsupported invariant payload '" & spec.payload &
          "' - the loop-shape invariant must mention the loop counter '" &
          model.loopVar & "'")
      for r in refs:
        if r != model.loopVar:
          raise leanRefusal(
            "lean_lang: unsupported invariant payload '" & spec.payload &
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
          raise leanRefusal(
            "lean_lang: unsupported invariant payload '" & spec.payload &
            "' - the tracer invariant must be implied by the guard `tid < n`" &
            " and mention both `tid` and `n`")
      for r in refs:
        if r notin ["tid", "n"]:
          raise leanRefusal(
            "lean_lang: unsupported invariant payload '" & spec.payload &
            "' - the tracer invariant may only mention `tid` and `n`")
      result.leanName = model.bodyName & "_invariant"
      result.attempt = "exact h"
  of skPostcondition:
    # the postcondition states the written buffer's post-state:
    # the per-thread body applied at `tid`
    if model.kind == kkLoopRange:
      raise leanRefusal(
        "lean_lang: unsupported postcondition payload '" & spec.payload &
        "' - the loop postcondition is the composed `∀ j` contract, outside" &
        " the payload grammar; the emitter derives it as `" &
        model.bodyName & "_post`")
    if model.writeBuf notin specRefs(expr):
      raise leanRefusal(
        "lean_lang: unsupported postcondition payload '" & spec.payload &
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
    if model.kind == kkLoopRange:
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
      result = loopDoc.replace("@KERNEL@", model.kernelName)
        .replace("@PAYLOAD@", s.payload).replace("@NAME@", s.leanName)
        .replace("@N@", $model.nValue).replace("@STMT@", s.statement)
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
    raise leanRefusal(
      "lean_lang: no lakefile.lean above the emitted module " & outPath)
  result = dir

proc emitLeanModuleWithSpecs*(kernel: GpuAst, specs: seq[RawSpec], outPath: string): string =
  ## Emits the Lean 4 verification module for one global kernel, with one spec scaffold per captured verification pragma.
  ##
  ## - writes the module to `outPath` and returns its text
  ## - with no pragmas the module is byte-identical to the P1/P2 statements
  ## - refuses loudly (ValueError, "lean_lang: ...") on any out-of-scope
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
      raise leanRefusal( "lean_lang: more than one '" &
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
      raise leanRefusal( "lean_lang: proof attempt failed " &
        "outside the pragma scaffolds (line " & $line & "): " & excerpt)
    failed[owner] = excerpt
  if exitCode != 0 and failed.len == 0:
    # no error line was attributable to a scaffold, so no proof was settled:
    # a `proven` verdict would be a guess, e.g. a missing toolchain, a Lean crash
    # or a diagnostic format change
    var excerpt = outp
    if excerpt.len > 800:
      excerpt = excerpt[0 ..< 797] & "..."
    raise leanRefusal( "lean_lang: proof settling failed - " &
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
  ## Emits the Lean 4 verification module of one global kernel:
  ## the per-thread body def plus derived spec theorems.
  ## - writes the module to `outPath` and returns its text
  ## - refuses loudly (ValueError, "lean_lang: unsupported ...") on any
  ##   construct outside the tracer's scope
  emitLeanModuleWithSpecs(kernel, @[], outPath)
