# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CIRv2 → Nim emitter, the CPU backend's serial printer.
##
## Output is spliced Nim AST at compile time.
## Kernels compile with the host program, no runtime cc/dlopen/FFI boundary.
## The serial build (--threads:off) is the executable spec.
## The parallel launch must reproduce its output bitwise.
##
##   callees → emitCpuCallees → procdefs spliced at the launch site
##   kernel  → emitCpuKernelBody → statements inlined in the launch loop nest
##
## Kernel signature: `crucibleCtx: CpuLaunchContext` first, then the wrapper's formals verbatim, so launch sites stay typed end to end.
## Coordinate builtins print as launch-site expressions through the coordinate table.
##
## v0 refusals, each a compile-time error naming the construct and the kernel:
## - vector types and swizzles
## - threadgroup barriers, simd shuffles, subgroup reductions, atomics
## - inline asm, `{.emit.}` raw target text, string values, bfloat16 cells
##
## OpenArray spans and `var` parameters also refuse: the span split pass
## decomposes the spans and the CPU surface takes whole values.

import std/macros
import std/sequtils
import ../ir/cirv2_lang

export cirv2_lang

# ═══════════════════════════════════════════════════════════════════════
# Emission context and refusals
# ═══════════════════════════════════════════════════════════════════════

type
  CpuCoordTable* = object
    ## Coordinate-builtin rewrite expressions.
    ##
    ## - the launch loop provides them, the kernel body reads them instead
    ##   of the builtins
    ## - vec[k][c] is the expression for builtin k and component c, set only
    ##   for the builtins the kernel reads
    ## - flat is the expression for thread_index_in_threadgroup
    vec: array[CirCoordBuiltinKind, array[3, NimNode]]
    flat: NimNode

proc cpuCoordTable*(vec: array[CirCoordBuiltinKind, array[3, NimNode]], flat: NimNode): CpuCoordTable =
  ## Coordinate table constructor for the launch side.
  CpuCoordTable(vec: vec, flat: flat)

proc coordExpr(coords: CpuCoordTable, name: CirCoordBuiltinKind, comp: int): NimNode =
  ## Rewrite expression for one builtin read, nil when unset.
  if name == CirCoordBuiltinKind.cbkNone:
    return nil
  if name == CirCoordBuiltinKind.cbkThreadIndexInThreadgroup:
    return coords.flat
  if comp >= 0:
    return coords.vec[name][comp]
  # a whole-vector read reconstructs the three components
  let (e0, e1, e2) = (coords.vec[name][0], coords.vec[name][1], coords.vec[name][2])
  if e0.isNil or e1.isNil or e2.isNil:
    return nil
  nnkCall.newTree(ident"dim3", e0, e1, e2)

type
  EmitCtx = object
    ## Walk state for one proc.
    ##
    ## The module surface:
    ##
    ## - procName names the kernel in refusal messages
    ## - procs is the module proc table, the source of call-argument conversions
    ## - ret is the current proc's return type, the target of return conversions
    ## - suffix uniquifies callee references per launch site
    ##
    ## The launch shape:
    ##
    ## - coords is the coordinate-builtin rewrite table, empty outside the kernel body, coordinate reads refuse there
    ## - inlineKernel marks the kernel body walk, statements inline into the launch loop nest, returns become block breaks
    ## - subst maps identifiers to launch-site expressions for the baked static parameters
    procName: string
    procs: seq[CirProc]
    ret: CirType
    suffix: string
    coords: CpuCoordTable
    inlineKernel: bool
    subst: seq[(string, NimNode)]

proc cpuRefuse(ctx: EmitCtx, what: string) {.noreturn, compileTime.} =
  ## Compile-time refusal naming the construct and the kernel.
  error("cpu backend, kernel " & ctx.procName & ": " & what)

var cpuCoordUse {.global, compileTime.}: bool
  ## Coordinate-builtin use of the module under emission.
  ## The chevron-less launch schedule derives from it, reset per emission.

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitType(t: CirType, ctx: EmitCtx): NimNode =
  ## CIRv2 type → Nim type node, used for locals, nested types and struct declarations.
  ## Parameters come from the wrapper's original formals verbatim.
  case t.kind
  of ctVoid:
    ident"void"
  of ctBool:
    ident"bool"
  of ctInt, ctUint, ctFloat:
    if t.bits == -1:
      case t.kind
      of ctInt: ident"int"
      of ctUint: ident"uint"
      else: ident"float64"
    else:
      let base = case t.kind
                 of ctInt: "int"
                 of ctUint: "uint"
                 else: "float"
      ident(base & $t.bits)
  of ctVector:
    cpuRefuse(ctx,
      "vector types (" & $t.vN & " x " & $t.vElem.kind & ") are not supported in v0," &
      " the vector mapping is the optimization pass (P2)")
  of ctSpan:
    cpuRefuse(ctx,
      "openArray spans must split to a pointer and a length before printing" &
      " (the span split pass), a span survived to the cpu backend")
  of ctPtr:
    nnkPtrTy.newTree(emitType(t.pElem, ctx))
  of ctAptr:
    nnkPtrTy.newTree(nnkBracketExpr.newTree(ident"UncheckedArray", emitType(t.pElem, ctx)))
  of ctStruct:
    ident(t.name)
  of ctArray:
    nnkBracketExpr.newTree(ident"array", newLit(t.aN), emitType(t.aElem, ctx))
  of ctString:
    cpuRefuse(ctx, "string values are not supported in v0")
  of ctBf16:
    cpuRefuse(ctx, "bfloat16 values are not supported in v0")
  of ctProc:
    cpuRefuse(ctx, "function pointers are not supported in v0")

proc emitStructDefs(m: CirModule): NimNode =
  ## Nim type section over the module's nominal struct types, one object per
  ## struct in declaration order.
  ##
  ## CIRv2 dropped the frontend's source type nodes: the emitter prints
  ## its own declarations and references them by the CIRv2 struct name.
  ## Values crossing from the launch-site formals convert through `emitConv`.
  if m.types.len == 0:
    return nnkStmtList.newTree()
  var defs = nnkStmtList.newTree()
  var section = nnkTypeSection.newTree()
  for t in m.types:
    if t.kind != ctStruct:
      continue
    var recList = nnkRecList.newTree()
    for f in t.fields:
      recList.add(nnkIdentDefs.newTree(ident(f.name), emitType(f.typ, EmitCtx(procName: "module")), newEmptyNode()))
    let obj = nnkObjectTy.newTree(newEmptyNode(), newEmptyNode(), recList)
    section.add(nnkTypeDef.newTree(ident(t.name), newEmptyNode(), obj))
  if section.len == 0:
    return nnkStmtList.newTree()
  defs.add(section)
  defs

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc emitExpr(e: CirExpr, ctx: EmitCtx): NimNode
proc emitConv(target: CirType, e: CirExpr, ctx: EmitCtx): NimNode

proc isCoordRead(e: CirExpr): bool =
  ## Recognition of coordinate-builtin reads, possibly a component of one.
  ## The emitted Nim type is `int` whatever CIRv2 says, hence the conversion
  ## treatment at assignments and arguments.
  if e.isNil:
    return false
  case e.kind
  of eSym:
    e.builtin and e.coordBuiltin != CirCoordBuiltinKind.cbkNone
  of eDot:
    isCoordRead(e.dotBase)
  else:
    false

proc emitIntLit(e: CirExpr): NimNode =
  ## Typed literal `int32(5)` or `float32(1.5)`, a conversion call that keeps
  ## the CIRv2 literal width visible in Nim's inferred expressions.
  let t = e.litType
  if t != nil and t.kind in {ctInt, ctUint, ctFloat} and t.bits in {8, 16, 32, 64}:
    let base = case t.kind
               of ctInt: "int"
               of ctUint: "uint"
               else: "float"
    result = nnkCall.newTree(ident(base & $t.bits), newLit(e.value))
  else:
    result = newLit(e.value)

proc emitBuiltinRefusal(e: CirExpr, ctx: EmitCtx): NimNode =
  ## Refusal for every non-coordinate builtin category, always a compile-time error.
  ## The NimNode return keeps call sites type-correct.
  let what =
    if e.synchroBuiltin != CirSynchroBuiltinKind.cbkNone: "threadgroup barriers"
    elif e.shuffleBuiltin != CirShuffleBuiltinKind.cbkNone: "simd_shuffle/simd_shuffle_xor"
    elif e.reductionBuiltin != CirReductionBuiltinKind.cbkNone: "subgroup reductions"
    elif e.atomicsBuiltin != CirAtomicsBuiltinKind.cbkNone: "atomic builtins"
    else: "unknown builtins"
  cpuRefuse(ctx, what & " are not supported in v0")

proc emitExpr(e: CirExpr, ctx: EmitCtx): NimNode =
  case e.kind
  of eNil:
    result = ident"nil"
  of eTrue:
    result = ident"true"
  of eFalse:
    result = ident"false"
  of eStrLit:
    cpuRefuse(ctx, "string values are not supported in v0")
  of eSym:
    if e.builtin:
      if e.coordBuiltin != CirCoordBuiltinKind.cbkNone:
        if not ctx.inlineKernel:
          cpuRefuse(ctx,
            "coordinate builtins inside device callees are not supported in v0," &
            " pass the values as parameters or inline the helper")
        cpuCoordUse = true
        result = coordExpr(ctx.coords, e.coordBuiltin, -1)
        if result.isNil:
          cpuRefuse(ctx, "no launch expression for coordinate builtin " & $e.coordBuiltin)
      else:
        result = emitBuiltinRefusal(e, ctx)
    elif ctx.subst.len > 0:
      result = ident(e.name)
      for (name, expr) in ctx.subst:
        if name == e.name:
          result = expr
          break
    else:
      result = ident(e.name)
  of eIntLit:
    result = emitIntLit(e)
  of eFloatLit:
    let bits = if e.fLitType != nil and e.fLitType.bits in {32, 64}: e.fLitType.bits else: 64
    result = nnkCall.newTree(ident("float" & $bits), newLit(e.fValue))
  of eDeref:
    result = nnkBracketExpr.newTree(emitExpr(e.derefOperand, ctx))
  of eAt:
    result = nnkBracketExpr.newTree(emitExpr(e.atBase, ctx), emitExpr(e.atIndex, ctx))
  of ePat:
    # pointer-AT (aptr indexing), vector swizzles are not supported in v0
    let bt = getExprType(e.patBase)
    if bt != nil and bt.kind == ctVector:
      cpuRefuse(ctx, "vector swizzles are not supported in v0")
    result = nnkBracketExpr.newTree(emitExpr(e.patBase, ctx), emitExpr(e.patIndex, ctx))
  of eDot:
    let base = e.dotBase
    if base.kind == eSym and base.builtin and base.coordBuiltin != CirCoordBuiltinKind.cbkNone:
      if not ctx.inlineKernel:
        cpuRefuse(ctx,
          "coordinate builtins inside device callees are not supported in v0," &
          " pass the values as parameters or inline the helper")
      cpuCoordUse = true
      let comp = case e.fieldName
                 of "x": 0
                 of "y": 1
                 of "z": 2
                 else: -1
      result = coordExpr(ctx.coords, base.coordBuiltin, comp)
      if result.isNil:
        cpuRefuse(ctx, "no launch expression for coordinate builtin " & $base.coordBuiltin)
    else:
      result = nnkDotExpr.newTree(emitExpr(e.dotBase, ctx), ident(e.fieldName))
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    let op = case e.kind
             of eAdd: "+"
             of eSub: "-"
             of eMul: "*"
             of eDiv:
               if e.opType.kind == ctFloat: "/" else: "div"
             of eMod:
               if e.opType.kind == ctFloat: "%" else: "mod"
             of eShl: "shl"
             of eShr: "shr"
             of eBitAnd: "and"
             of eBitOr: "or"
             of eBitXor: "xor"
             else: ""
    result = nnkInfix.newTree(ident(op), emitExpr(e.left, ctx), emitExpr(e.right, ctx))
  of eNeg:
    result = nnkPrefix.newTree(ident"-", emitExpr(e.unOperand, ctx))
  of eEq, eNeq, eLt, eLe:
    let op = case e.kind
             of eEq: "=="
             of eNeq: "!="
             of eLt: "<"
             else: "<="
    result = nnkInfix.newTree(ident(op), emitExpr(e.a, ctx), emitExpr(e.b, ctx))
  of eNot:
    result = nnkPrefix.newTree(ident"not", nnkPar.newTree(emitExpr(e.notOperand, ctx)))
  of eAnd, eOr:
    let op = if e.kind == eAnd: "and" else: "or"
    result = nnkInfix.newTree(ident(op), emitExpr(e.andLeft, ctx), emitExpr(e.andRight, ctx))
  of eTernary:
    # a value-producing conditional, the if-expression keeps the eager
    # evaluation question away: CIRv2 ternary branches are pure
    result = nnkIfExpr.newTree(
      nnkElifExpr.newTree(emitExpr(e.cond, ctx), emitExpr(e.thenE, ctx)),
      nnkElseExpr.newTree(emitExpr(e.elseE, ctx)))
  of eCast:
    result = nnkCast.newTree(emitType(e.castType, ctx), emitExpr(e.castOperand, ctx))
  of eConv:
    result = nnkCall.newTree(emitType(e.castType, ctx), emitExpr(e.castOperand, ctx))
  of eAddr:
    result = nnkPrefix.newTree(ident"addr", emitExpr(e.addrOperand, ctx))
  of eSizeof:
    result = nnkCall.newTree(ident"sizeof", emitType(e.ofType, ctx))
  of eOconstr:
    let t = e.ocType
    if t.kind != ctStruct:
      cpuRefuse(ctx, "object constructor over a non-struct type")
    var ctor = nnkObjConstr.newTree(emitType(t, ctx))
    for kv in e.kvFields:
      ctor.add(nnkExprColonExpr.newTree(ident(kv.name), emitExpr(kv.value, ctx)))
    result = ctor
  of eAconstr:
    var bracket = nnkBracket.newTree()
    for el in e.elems:
      bracket.add(emitExpr(el, ctx))
    result = bracket
  of eCall:
    let callee = e.callee
    if callee.kind == eSym and callee.builtin:
      discard emitBuiltinRefusal(callee, ctx)
    var calleeRef = emitExpr(callee, ctx)
    if callee.kind == eSym and not callee.builtin:
      calleeRef = ident(callee.name & ctx.suffix)
    var call = nnkCall.newTree(calleeRef)
    for i, a in e.args:
      # call-argument conversions match the callee's parameter types,
      # matching the implicit conversions C performs at call sites
      var paramType: CirType = nil
      if callee.kind == eSym:
        for p in ctx.procs:
          if p.name == callee.name and i < p.params.len:
            paramType = p.params[i].typ
            break
      if paramType != nil:
        call.add(emitConv(paramType, a, ctx))
      else:
        call.add(emitExpr(a, ctx))
    result = call

proc emitConv(target: CirType, e: CirExpr, ctx: EmitCtx): NimNode =
  ## Conversion insertion where Nim needs what C does implicitly.
  ##
  ## - integer width changes at assignments and call arguments
  ## - coordinate-builtin reads (Nim `int`) into CIRv2 integer scalars
  ## - struct targets rebuild the value field by field under the target's emitted type, the launch-site formals carry the user's Nim type
  ##
  ## Floats convert only between widths, bools and composites pass through.
  let inner = emitExpr(e, ctx)
  if target.isNil or e.isNil:
    return inner
  case target.kind
  of ctInt, ctUint:
    let src = getExprType(e)
    let srcIsInt = src != nil and src.kind in {ctInt, ctUint}
    if isCoordRead(e) or (srcIsInt and src.bits != target.bits):
      let base = if target.kind == ctInt: "int" else: "uint"
      if target.bits == -1:
        return nnkCall.newTree(ident(base), inner)
      return nnkCall.newTree(ident(base & $target.bits), inner)
    return inner
  of ctFloat:
    let src = getExprType(e)
    if src != nil and src.kind == ctFloat and src.bits != target.bits:
      return nnkCall.newTree(ident("float" & $target.bits), inner)
    return inner
  of ctStruct:
    var ctor = nnkObjConstr.newTree(emitType(target, ctx))
    for f in target.fields:
      let fieldRead = CirExpr(kind: eDot, dotBase: e, fieldName: f.name)
      ctor.add(nnkExprColonExpr.newTree(ident(f.name),
        emitConv(f.typ, fieldRead, ctx)))
    return ctor
  else:
    return inner

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc emitStmts(body: seq[CirStmt], ctx: EmitCtx): NimNode =
  ## Statement-list emission.
  result = nnkStmtList.newTree()
  for s in body:
    case s.kind
    of sVar:
      case s.addressSpace
      of caThreadgroup:
        cpuRefuse(ctx, "threadgroup (shared) memory is not supported in v0")
      of caConstant, caDevice:
        cpuRefuse(ctx, "constant/device memory-class locals are not supported in v0")
      of caThread:
        discard
      if ctx.inlineKernel:
        # substitution works by name, a local shadowing a substituted
        # parameter would misbind
        for (name, _) in ctx.subst:
          if name == s.name:
            cpuRefuse(ctx, "kernel local '" & s.name &
              "' shadows a kernel parameter, rename one")
      let identDefs = nnkIdentDefs.newTree(ident(s.name), emitType(s.declType, ctx),
        if s.init != nil: emitConv(s.declType, s.init, ctx) else: newEmptyNode())
      if s.kind == sVar:
        result.add(nnkVarSection.newTree(identDefs))
      else:
        result.add(nnkLetSection.newTree(identDefs))
    of sAsgn:
      result.add(nnkAsgn.newTree(emitExpr(s.lhs, ctx),
        emitConv(getExprType(s.lhs), s.rhs, ctx)))
    of sIte:
      let cond = emitExpr(s.cond, ctx)
      let thenB = emitStmts(s.thenBody, ctx)
      let elseB = emitStmts(s.elseBody, ctx)
      # empty-branch shapes collapse: negation keeps an else-only branch
      # without an empty then branch
      if s.thenBody.len == 0 and s.elseBody.len == 0:
        discard # both branches empty, no code
      elif s.thenBody.len == 0:
        result.add(nnkIfStmt.newTree(nnkElifBranch.newTree(
          nnkPrefix.newTree(ident"not", nnkPar.newTree(cond)), elseB)))
      elif s.elseBody.len == 0:
        result.add(nnkIfStmt.newTree(nnkElifBranch.newTree(cond, thenB)))
      else:
        result.add(nnkIfStmt.newTree(nnkElifBranch.newTree(cond, thenB),
                                     nnkElse.newTree(elseB)))
    of sWhile:
      result.add(nnkWhileStmt.newTree(emitExpr(s.whileCond, ctx),
        emitStmts(s.whileBody, ctx)))
    of sRet:
      if ctx.inlineKernel:
        # inline shape: the body is part of the launch loop nest, a return exits
        # per-thread scope, each thread iteration sits in `block crucibleThread`
        if s.retVal != nil:
          cpuRefuse(ctx, "kernels are void, a return value has no meaning")
        result.add(nnkBreakStmt.newTree(ident"crucibleThread"))
      elif s.retVal != nil:
        result.add(nnkReturnStmt.newTree(emitConv(ctx.ret, s.retVal, ctx)))
      else:
        result.add(nnkReturnStmt.newTree(newEmptyNode()))
    of sDiscard:
      if s.discardVal != nil:
        result.add(nnkDiscardStmt.newTree(emitExpr(s.discardVal, ctx)))
      else:
        result.add(nnkDiscardStmt.newTree(newEmptyNode()))
    of sCall:
      var calleeRef = emitExpr(s.callCallee, ctx)
      if s.callCallee.kind == eSym and not s.callCallee.builtin:
        calleeRef = ident(s.callCallee.name & ctx.suffix)
      var call = nnkCall.newTree(calleeRef)
      for i, a in s.callArgs:
        var paramType: CirType = nil
        if s.callCallee.kind == eSym:
          for p in ctx.procs:
            if p.name == s.callCallee.name and i < p.params.len:
              paramType = p.params[i].typ
              break
        if paramType != nil:
          call.add(emitConv(paramType, a, ctx))
        else:
          call.add(emitExpr(a, ctx))
      result.add(call)
    of sAsm:
      cpuRefuse(ctx,
        "inline asm is target-specific and has no CPU meaning in v0")
    of sEmit:
      cpuRefuse(ctx,
        "`{.emit.}` raw target text has no Nim meaning in v0")

# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

proc findKernel(m: CirModule, ctx: EmitCtx): CirProc =
  ## Single-kernel lookup for the lowered module.
  for p in m.procs:
    if p.isKernel:
      if result != nil:
        cpuRefuse(ctx, "multiple kernels in one module")
      result = p
  if result.isNil:
    cpuRefuse(ctx, "no {.global.} kernel in the lowered module")

proc inlineSource(k: CirProc, m: CirModule, ctx: EmitCtx): tuple[bodyProc: CirProc, call: CirStmt] =
  ## Inlined-body lookup: the proc whose body is inlined into the launch loop nest,
  ## plus the wrapper call when the pipeline kept the kernel separate.
  ##
  ## Two shapes reach the emitter:
  ## - the wrapper body is one call to the user kernel proc, that body
  ##   inlines with the call arguments substituting the parameters
  ## - a pipeline transform already folded the user kernel into the wrapper,
  ##   the wrapper body is the kernel body itself, no call
  var call: CirStmt = nil
  for st in k.body:
    case st.kind
    of sCall:
      if call != nil:
        cpuRefuse(ctx, "the kernel wrapper body makes multiple calls")
      call = st
    else:
      discard
  if call.isNil:
    # folded shape, the wrapper body is the kernel body
    return (k, nil)
  let callee = call.callCallee
  if callee.kind != eSym or callee.builtin:
    cpuRefuse(ctx, "the kernel wrapper does not call a module proc")
  for p in m.procs:
    if p.name == callee.name:
      return (p, call)
  cpuRefuse(ctx, "the kernel wrapper calls an unknown proc '" & callee.name & "'")

proc emitProcParams(p: CirProc, ctx: EmitCtx): NimNode =
  ## Callee parameters from their CIRv2 types.
  ## Structs reference the emitter's own declarations, so the calls type-check.
  let params = nnkFormalParams.newTree(
    if p.ret.kind == ctVoid: newEmptyNode() else: emitType(p.ret, ctx))
  for param in p.params:
    if param.varParam:
      cpuRefuse(ctx,
        "`var` parameters are not supported in v0, the cpu backend takes whole values")
    params.add(nnkIdentDefs.newTree(ident(param.name), emitType(param.typ, ctx),
                                    newEmptyNode()))
  params

proc emitCpuCallees*(m: CirModule, suffix: string): NimNode =
  ## Callee proc emission for one lowered module.
  ##
  ## The module's nominal struct types emit as a Nim type section in front
  ## of the proc defs, the launch splices this output before the kernel
  ## body so the declarations precede every use.
  ##
  ## - callees emit under their CIRv2 names plus the launch-site suffix,
  ##   their parameter types reference the emitted declarations
  ## - `{.inline.}` callees emit the Nim `inline` pragma
  ## - the kernel wrapper and the inlined kernel proc emit nothing,
  ##   emitCpuKernelBody prints their bodies at the launch site
  ## - hoisted threadgroup/constant globals refuse in v0
  if m.globals.len > 0:
    cpuRefuse(EmitCtx(procName: "module"),
      "hoisted threadgroup/constant globals are not supported in v0")
  var kctx = EmitCtx(procName: "module", procs: m.procs)
  let inlinedName = if m.procs.anyIt(it.isKernel):
    inlineSource(findKernel(m, kctx), m, kctx).bodyProc.name
  else:
    ""
  result = nnkStmtList.newTree()
  result.add(emitStructDefs(m))
  for p in m.procs:
    if p.isKernel or p.name == inlinedName:
      continue
    var ctx = EmitCtx(procName: p.name, procs: m.procs, ret: p.ret, suffix: suffix)
    let pragmas = if p.inline: nnkPragma.newTree(ident"inline")
                  else: nnkPragma.newTree()
    result.add(nnkProcDef.newTree(
      ident(p.name & suffix),
      newEmptyNode(), newEmptyNode(),
      emitProcParams(p, ctx),
      pragmas,
      newEmptyNode(),
      emitStmts(p.body, ctx)))

proc emitCpuKernelBody*(m: CirModule, suffix: string, coords: CpuCoordTable): tuple[stmts: NimNode, coordUse: bool] =
  ## Kernel body emission, inlined form.
  ##
  ## Returns the kernel body statements plus the module's coordinate-builtin use.
  ## The launch loop nest wraps the statements and a kernel return breaks
  ## `block crucibleThread`, the per-thread block around each iteration.
  ##
  ## - the inlined proc is the user kernel, its parameters are the wrapper's
  ##   formals verbatim plus baked statics
  ## - static parameters substitute to the literals the wrapper passes
  ## - coordinate builtins rewrite through `coords`
  ## - `suffix` uniquifies callee references per launch site
  cpuCoordUse = false
  var kctx = EmitCtx(procName: "kernel", procs: m.procs, suffix: suffix)
  let k = findKernel(m, kctx)
  let (bodyProc, call) = inlineSource(k, m, kctx)

  # static parameters: the wrapper call passes their values positionally
  # and the body references substitute to those literals
  # a folded body has the baked values, the substitution stays empty
  var subst: seq[(string, NimNode)] = @[]
  var bctx = EmitCtx(procName: bodyProc.name, procs: m.procs, ret: bodyProc.ret,
                     suffix: suffix, coords: coords, inlineKernel: true)
  if not call.isNil:
    for i, param in bodyProc.params:
      if i < call.callArgs.len:
        # runtime params substitute to their identical kernel formal names,
        # statics to the baked literals, args convert at the substitution
        subst.add((param.name, emitExpr(call.callArgs[i], bctx)))
    bctx.subst = subst
  result.stmts = emitStmts(bodyProc.body, bctx)
  result.coordUse = cpuCoordUse
