# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            C printer
#
# ############################################################

import std/[macros, strutils]
import ../ir/cirv2_lang
import ../../builtins/builtins_gpu_types
import ../../builtins/builtins_decl_helpers
import ./targets_registry
import ./print_common

export cirv2_lang

# ═══════════════════════════════════════════════════════════════════════
# Emission context
# ═══════════════════════════════════════════════════════════════════════

type
  CpuCoordTableC* = object
    ## Coordinate-builtin rewrite expressions as C text, provided by the launch.
    ## Kernel-body reads go through them for every builtin access.
    ##
    ## - vec[k][c]: the expression for builtin k and component c, present
    ##   only for the builtins the kernel reads
    ## - flat: the expression for thread_index_in_threadgroup
    vec*: array[CirCoordBuiltinKind, array[3, string]]
    flat*: string

proc cpuCoordTableC*(vec: array[CirCoordBuiltinKind, array[3, string]], flat: string): CpuCoordTableC =
  CpuCoordTableC(vec: vec, flat: flat)

proc coordExprC(coords: CpuCoordTableC, name: CirCoordBuiltinKind, comp: int): string =
  ## Rewrite expression for one builtin read, empty when unset.
  if name == CirCoordBuiltinKind.cbkNone:
    return ""
  if name == CirCoordBuiltinKind.cbkThreadIndexInThreadgroup:
    return coords.flat
  if comp >= 0:
    return coords.vec[name][comp]
  let (e0, e1, e2) = (coords.vec[name][0], coords.vec[name][1], coords.vec[name][2])
  if e0.len == 0 or e1.len == 0 or e2.len == 0:
    return ""
  "(dim3){" & e0 & ", " & e1 & ", " & e2 & "}"

const printerTarget = "cpu"

const cpuEmit = TargetEmit(
  vocab: BuiltinVocab(),
  functionalCtor: false,
  ctorOpen: "{",
  ctorClose: "}",
  ctorCast: true,
  ctorZero: "",
  aconstrCasts: false,
  dotDerefParens: true,
  vectorDots: false,
  noPointerValues: (spaces: {}, text: ""),
  noPointerRebind: "",
  prefixKernel: "",
  prefixInline: "",
  prefixDevice: "")

type
  PrintCtx* = object of WalkCtx
    ## Per-target walk context. The shared walk procs instantiate against
    ## the derived type and the overlay procs below resolve per target.
    ##
    ## - suffix: uniquifies callee references per launch site
    ## - coords: the coordinate-builtin rewrite table, coordinate reads
    ##   refuse outside the kernel body
    ## - inlineKernel: marks the kernel body walk, returns become gotos
    ##   to the per-thread exit label, subst maps identifiers to C text
    ##   for baked static parameters
    suffix: string
    coords: CpuCoordTableC
    inlineKernel: bool
    subst: seq[(string, string)]

type
  EmitCtxC = object
    ## Walk state for one proc.
    ##
    ## The module surface:
    ##
    ## - procName: names the kernel in refusal messages
    ## - suffix: uniquifies callee references per launch site
    ##
    ## The launch shape:
    ##
    ## - coords: the coordinate-builtin rewrite table, coordinate reads
    ##   refuse outside the kernel body
    ## - inlineKernel: marks the kernel body walk, returns become gotos
    ##   to the per-thread exit label, subst maps identifiers to C text
    ##   for baked static parameters
    procName: string
    suffix: string
    coords: CpuCoordTableC
    inlineKernel: bool
    subst: seq[(string, string)]

proc cRefuse(ctx: EmitCtxC, what: string) {.noreturn, compileTime.} =
  error("cpu workgroup launch, kernel " & ctx.procName & ": " & what)

var cpuCoordUseC {.global, compileTime.}: bool
  ## Coordinate-builtin use of the module under emission, reset per emission, read by the launch after the body prints.

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitTypeC(t: CirType, ctx: EmitCtxC): string =
  ## CIRv2 type as C type text, used for locals, params, data members.
  case t.kind
  of ctVoid:
    result = "void"
  of ctBool:
    result = "bool"
  of ctInt, ctUint, ctFloat:
    let stem = case t.kind
               of ctInt: "int"
               of ctUint: "uint"
               else: ""
    if t.bits == -1:
      # machine word: Nim int/uint map to int64_t/uint64_t on Nim's targets
      result = if t.kind == ctFloat: "double" else: stem & "64_t"
    else:
      result = if t.kind == ctFloat:
        (if t.bits == 32: "float" else: "double")
      else:
        stem & $t.bits & "_t"
  of ctVector:
    cRefuse(ctx,
      "vector types (" & $t.vN & " x " & $t.vElem.kind & ") are not supported in v0," &
      " the vector mapping is the optimization pass (P2)")
  of ctSpan:
    cRefuse(ctx,
      "openArray spans must split to a pointer and a length before printing" &
      " (the span split pass), a span survived to the cpu backend")
  of ctPtr:
    result = emitTypeC(t.pElem, ctx) & "*"
  of ctAptr:
    # UncheckedArray: a raw pointer with array indexing
    result = emitTypeC(t.pElem, ctx) & "*"
  of ctStruct:
    result = "struct " & t.name & ctx.suffix
  of ctArray:
    # element type only, the declaration sites append every dimension
    # (declaratorC, struct fields, data-struct members)
    result = emitTypeC(t.aElem, ctx)
  of ctString:
    cRefuse(ctx, "string values are not supported in v0")
  of ctBf16:
    cRefuse(ctx, "bfloat16 values are not supported in v0")
  of ctProc:
    cRefuse(ctx, "function pointers are not supported in v0")

proc splitArrayDimsC(t: CirType): tuple[elem: CirType, dims: string] =
  ## Array type split for a declaration site: element type plus stacked
  ## C declarator dimensions.
  ##
  ##   array[3, array[4, float32]]  ->  elem `float32`, dims `[3][4]`
  ##
  ## A non-array type returns itself with empty dims.
  result.elem = t
  while result.elem.kind == ctArray:
    result.dims &= "[" & $result.elem.aN & "]"
    result.elem = result.elem.aElem

proc declaratorC(t: CirType, name: string, ctx: EmitCtxC): string =
  ## Declaration text binding `t` to `name`.
  ## Array dimensions move behind the identifier and a pointer to an array
  ## prints the parenthesized declarator:
  ##
  ##   ptr array[4, int32] p  ->  int32_t (*p)[4]
  ##
  ## With `T* p` the kernel's `(*p)[i]` element reads reject at C compile time.
  ## An empty name prints the abstract form casts use, `int32_t (*)[4]`.
  let (elem, dims) = splitArrayDimsC(t)
  if elem.kind == ctPtr and elem.pElem.kind == ctArray:
    var inner = elem
    var ptrs = ""
    var innerDims = ""
    while inner.kind == ctPtr:
      ptrs &= "*"
      inner = inner.pElem
    while inner.kind == ctArray:
      innerDims &= "[" & $inner.aN & "]"
      inner = inner.aElem
    result = emitTypeC(inner, ctx) & " (" & ptrs & name & dims & ")" & innerDims
  elif name.len == 0:
    result = emitTypeC(elem, ctx) & dims
  else:
    result = emitTypeC(elem, ctx) & " " & name & dims

proc collectStructs(t: CirType, seen: var seq[CirType]) =
  ## Post-order collection of every nominal struct type reachable
  ## from t. Fields collect before the struct, the printed definitions
  ## come out in an order C accepts.
  if t.isNil:
    return
  case t.kind
  of ctStruct:
    for s in seen:
      if s == t:
        return
    for f in t.fields:
      collectStructs(f.typ, seen)
    seen.add(t)
  of ctPtr, ctAptr:
    collectStructs(t.pElem, seen)
  of ctArray:
    collectStructs(t.aElem, seen)
  else:
    discard

proc moduleStructs(m: CirModule): seq[CirType] =
  ## Every printable nominal struct: declared types plus the proc
  ## parameter and return types.
  var seen: seq[CirType]
  for t in m.types:
    collectStructs(t, seen)
  for p in m.procs:
    for param in p.params:
      collectStructs(param.typ, seen)
    collectStructs(p.ret, seen)
  seen

proc emitStructDefsC(m: CirModule, ctx: EmitCtxC): string =
  ## One definition per nominal struct type, dependency order.
  var lines: seq[string]
  for t in moduleStructs(m):
    var def = "struct " & t.name & ctx.suffix & " {\n"
    for f in t.fields:
      let (elem, dims) = splitArrayDimsC(f.typ)
      def.add("    " & emitTypeC(elem, ctx) & " " & f.name & dims & ";\n")
    def.add("};")
    lines.add(def)
  if lines.len > 0:
    lines.join("\n") & "\n"
  else:
    ""

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printExpr(e: CirExpr, ctx: EmitCtxC): string

proc isCoordReadC(e: CirExpr): bool =
  ## Recognition of coordinate-builtin reads, possibly a component of one.
  if e.isNil:
    return false
  case e.kind
  of eSym:
    e.builtin and e.coordBuiltin != CirCoordBuiltinKind.cbkNone
  of eDot:
    isCoordReadC(e.dotBase)
  else:
    false

proc printIntLit(e: CirExpr): string =
  ## Integer literal, C suffix when the width exceeds int.
  let bits = if e.litType != nil and e.litType.kind in {ctInt, ctUint}:
               e.litType.bits else: -1
  if e.litType != nil and e.litType.kind == ctUint:
    result = $e.value & "u"
    if bits == 64:
      result &= "ll"
  elif bits == 64:
    result = $e.value & "ll"
  else:
    result = $e.value

proc printFloatLit(e: CirExpr): string =
  ## Float literal, f suffix for 32-bit.
  let bits = if e.fLitType != nil and e.fLitType.kind == ctFloat:
               e.fLitType.bits else: 64
  if bits == 32:
    # repr keeps the decimal form exact through the Nim float64 value
    result = repr(e.fValue) & "f"
  else:
    result = repr(e.fValue)

proc printBuiltinRefusalC(e: CirExpr, ctx: EmitCtxC): string =
  let what =
    if e.synchroBuiltin != CirSynchroBuiltinKind.cbkNone: "threadgroup barriers"
    elif e.shuffleBuiltin != CirShuffleBuiltinKind.cbkNone: "simd_shuffle/simd_shuffle_xor"
    elif e.reductionBuiltin != CirReductionBuiltinKind.cbkNone: "subgroup reductions"
    elif e.atomicsBuiltin != CirAtomicsBuiltinKind.cbkNone: "atomic builtins"
    else: "unknown builtins"
  cRefuse(ctx, what & " are not supported in v0")

proc calleeSymbolC(name: string, ctx: EmitCtxC): string =
  ## C symbol for a callee: kernel-printed callees append the launch-site suffix,
  ## `__builtin_` compiler builtins pass verbatim, the decomposeArrayAssign pass emits them.
  if name.startsWith("__builtin_"):
    name
  else:
    name & ctx.suffix

proc printExpr(e: CirExpr, ctx: EmitCtxC): string =
  case e.kind
  of eNil:
    result = "NULL"
  of eTrue:
    result = "true"
  of eFalse:
    result = "false"
  of eStrLit:
    cRefuse(ctx, "string values are not supported in v0")
  of eSym:
    if e.builtin:
      if e.coordBuiltin != CirCoordBuiltinKind.cbkNone:
        if not ctx.inlineKernel:
          cRefuse(ctx,
            "coordinate builtins inside device callees are not supported in v0," &
            " pass the values as parameters or inline the helper")
        cpuCoordUseC = true
        result = coordExprC(ctx.coords, e.coordBuiltin, -1)
        if result.len == 0:
          cRefuse(ctx, "no launch expression for coordinate builtin " & $e.coordBuiltin)
      else:
        result = printBuiltinRefusalC(e, ctx)
    elif ctx.subst.len > 0:
      result = e.name
      for (name, txt) in ctx.subst:
        if name == e.name:
          result = txt
          break
    else:
      result = e.name
  of eIntLit:
    result = printIntLit(e)
  of eFloatLit:
    result = printFloatLit(e)
  of eDeref:
    result = "*" & printExpr(e.derefOperand, ctx)
  of eAt:
    let base = printExpr(e.atBase, ctx)
    # C parses a prefix dereference tighter than the index,
    # `*p[i]` reads `*(p[i])`
    if e.atBase.kind == eDeref:
      result = "(" & base & ")[" & printExpr(e.atIndex, ctx) & "]"
    else:
      result = base & "[" & printExpr(e.atIndex, ctx) & "]"
  of ePat:
    # pointer-AT: aptr indexing, vector swizzles are not supported in v0
    let bt = getExprType(e.patBase)
    if bt != nil and bt.kind == ctVector:
      cRefuse(ctx, "vector swizzles are not supported in v0")
    result = printExpr(e.patBase, ctx) & "[" & printExpr(e.patIndex, ctx) & "]"
  of eDot:
    let base = e.dotBase
    if base.kind == eSym and base.builtin and base.coordBuiltin != CirCoordBuiltinKind.cbkNone:
      if not ctx.inlineKernel:
        cRefuse(ctx,
          "coordinate builtins inside device callees are not supported in v0," &
          " pass the values as parameters or inline the helper")
      cpuCoordUseC = true
      let comp = case e.fieldName
                 of "x": 0
                 of "y": 1
                 of "z": 2
                 else: -1
      result = coordExprC(ctx.coords, base.coordBuiltin, comp)
      if result.len == 0:
        cRefuse(ctx, "no launch expression for coordinate builtin " & $base.coordBuiltin)
    else:
      let baseTxt = printExpr(e.dotBase, ctx)
      # C parses a prefix dereference tighter than the field access,
      # `*p.field` reads `*(p.field)`
      if e.dotBase.kind == eDeref:
        result = "(" & baseTxt & ")." & e.fieldName
      else:
        result = baseTxt & "." & e.fieldName
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    let op = case e.kind
             of eAdd: "+"
             of eSub: "-"
             of eMul: "*"
             of eDiv: "/"
             of eMod:
               if e.opType.kind == ctFloat:
                 cRefuse(ctx, "float mod has no C operator, CIRv2 lowers it for integer selectors only")
               else: "%"
             of eShl: "<<"
             of eShr: ">>"
             of eBitAnd: "&"
             of eBitOr: "|"
             of eBitXor: "^"
             else: ""
    result = "(" & printExpr(e.left, ctx) & " " & op & " " & printExpr(e.right, ctx) & ")"
  of eNeg:
    result = "(-" & printExpr(e.unOperand, ctx) & ")"
  of eEq, eNeq, eLt, eLe:
    let op = case e.kind
             of eEq: "=="
             of eNeq: "!="
             of eLt: "<"
             else: "<="
    result = "(" & printExpr(e.a, ctx) & " " & op & " " & printExpr(e.b, ctx) & ")"
  of eNot:
    result = "(!" & printExpr(e.notOperand, ctx) & ")"
  of eAnd, eOr:
    let op = if e.kind == eAnd: "&&" else: "||"
    result = "(" & printExpr(e.andLeft, ctx) & " " & op & " " & printExpr(e.andRight, ctx) & ")"
  of eTernary:
    result = "(" & printExpr(e.cond, ctx) & " ? " & printExpr(e.thenE, ctx) &
             " : " & printExpr(e.elseE, ctx) & ")"
  of eCast:
    # bit-level reinterpret through a compound-literal union: header-free, aliasing-safe, the compiler folds it to a register move
    let srcT = getExprType(e.castOperand)
    if srcT == nil or e.castType.kind notin
        {ctBool, ctInt, ctUint, ctFloat, ctPtr, ctAptr} or
        srcT.kind notin {ctBool, ctInt, ctUint, ctFloat, ctPtr, ctAptr}:
      cRefuse(ctx, "cast between non-scalar types is not supported in v0")
    result = "((union { " & emitTypeC(srcT, ctx) & " _from; " &
             emitTypeC(e.castType, ctx) & " _to; }){ ._from = " &
             printExpr(e.castOperand, ctx) & " })._to"
  of eConv:
    result = "(" & emitTypeC(e.castType, ctx) & ")" & printExpr(e.castOperand, ctx)
  of eAddr:
    result = "&" & printExpr(e.addrOperand, ctx)
  of eSizeof:
    let (elem, dims) = splitArrayDimsC(e.ofType)
    result = "sizeof(" & emitTypeC(elem, ctx) & dims & ")"
  of eOconstr:
    let t = e.ocType
    if t.kind != ctStruct:
      cRefuse(ctx, "object constructor over a non-struct type")
    var parts: seq[string]
    for kv in e.kvFields:
      parts.add("." & kv.name & " = " & printExpr(kv.value, ctx))
    result = "(" & emitTypeC(t, ctx) & "){" & parts.join(", ") & "}"
  of eAconstr:
    var parts: seq[string]
    for el in e.elems:
      parts.add(printExpr(el, ctx))
    result = "{" & parts.join(", ") & "}"
  of eCall:
    let callee = e.callee
    if callee.kind == eSym and callee.builtin:
      discard printBuiltinRefusalC(callee, ctx)
    var calleeRef = printExpr(callee, ctx)
    if callee.kind == eSym and not callee.builtin:
      calleeRef = calleeSymbolC(callee.name, ctx)
    var args: seq[string]
    for a in e.args:
      # C converts implicitly at call sites, CIRv2 conversions print
      # explicitly as casts where they exist
      args.add(printExpr(a, ctx))
    result = calleeRef & "(" & args.join(", ") & ")"

const LaunchCounterNamesC = [
  # counters the iteration function generates around the inlined kernel
  # body (see emitCpuWorkgroupLaunchC), the body shares their scope
  "crucibleWg", "crucibleWgTotal", "crucibleWgEnd",
  "crucibleGx", "crucibleGy", "crucibleGz",
  "crucibleBx", "crucibleBy", "crucibleBz",
]

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc printStmts(body: seq[CirStmt], ctx: EmitCtxC, indent: int, lines: var seq[string]) =
  var i = 0
  while i < body.len:
    let s = body[i]
    let pad = spaces(indent * 2)
    case s.kind
    of sVar:
      case s.addressSpace
      of caThreadgroup:
        cRefuse(ctx, "threadgroup (shared) memory is not supported in v0")
      of caConstant, caDevice:
        cRefuse(ctx, "constant/device memory-class locals are not supported in v0")
      of caThread:
        discard
      if ctx.inlineKernel:
        # substitution works by name, a local shadowing a substituted
        # parameter would misbind
        for (name, _) in ctx.subst:
          if name == s.name:
            cRefuse(ctx, "kernel local '" & s.name &
              "' shadows a kernel parameter, rename one")
        if s.name in LaunchCounterNamesC:
          # kernel locals must avoid the launch's generated counter names:
          # a shadowing local silently corrupts the coordinate reads
          cRefuse(ctx, "kernel local '" & s.name &
            "' shadows a counter the launch generates, rename it")
      let decl =
        declaratorC(s.declType, s.name, ctx) &
        (if s.init != nil: " = " & printExpr(s.init, ctx) else: "") & ";"
      lines.add(pad & decl)
      i += 1
    of sAsgn:
      lines.add(pad & printExpr(s.lhs, ctx) & " = " & printExpr(s.rhs, ctx) & ";")
      i += 1
    of sIte:
      let cond = printExpr(s.cond, ctx)
      var thenLines: seq[string]
      var elseLines: seq[string]
      printStmts(s.thenBody, ctx, indent + 1, thenLines)
      printStmts(s.elseBody, ctx, indent + 1, elseLines)
      # empty-branch shapes collapse: negation keeps an else-only
      # branch without an empty then branch
      if s.thenBody.len == 0 and s.elseBody.len == 0:
        # both branches lower to zero statements, the printed condition
        # keeps its side effects
        lines.add(pad & "(void)(" & cond & ");")
      elif s.thenBody.len == 0:
        lines.add(pad & "if (!" & cond & ") {")
        lines.add(elseLines)
        lines.add(pad & "}")
      elif s.elseBody.len == 0:
        lines.add(pad & "if (" & cond & ") {")
        lines.add(thenLines)
        lines.add(pad & "}")
      else:
        lines.add(pad & "if (" & cond & ") {")
        lines.add(thenLines)
        lines.add(pad & "} else {")
        lines.add(elseLines)
        lines.add(pad & "}")
      i += 1
    of sWhile:
      lines.add(pad & "while (" & printExpr(s.whileCond, ctx) & ") {")
      printStmts(s.whileBody, ctx, indent + 1, lines)
      lines.add(pad & "}")
      i += 1
    of sRet:
      if ctx.inlineKernel:
        # inline shape: the body sits in the per-thread loop nest, a return exits
        # per-thread scope through the launch's done label
        if s.retVal != nil:
          cRefuse(ctx, "kernels are void, a return value has no meaning")
        lines.add(pad & "goto crucibleThreadDone;")
      elif s.retVal != nil:
        lines.add(pad & "return " & printExpr(s.retVal, ctx) & ";")
      else:
        lines.add(pad & "return;")
      i += 1
    of sDiscard:
      if s.discardVal != nil:
        lines.add(pad & "(void)(" & printExpr(s.discardVal, ctx) & ");")
      else:
        lines.add(pad & ";")
      i += 1
    of sCall:
      var calleeRef = printExpr(s.callCallee, ctx)
      if s.callCallee.kind == eSym and not s.callCallee.builtin:
        calleeRef = calleeSymbolC(s.callCallee.name, ctx)
      var args: seq[string]
      for a in s.callArgs:
        args.add(printExpr(a, ctx))
      lines.add(pad & calleeRef & "(" & args.join(", ") & ");")
      i += 1
    of sAsm:
      cRefuse(ctx,
        "inline asm is target-specific and has no CPU meaning in v0")
    of sEmit:
      cRefuse(ctx,
        "`{.emit.}` raw target text has no C meaning in v0")

proc printProcBodyC(p: CirProc, ctx: EmitCtxC): string =
  var lines: seq[string]
  printStmts(p.body, ctx, 1, lines)
  lines.join("\n")

# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

proc findKernelC(m: CirModule, ctx: EmitCtxC): CirProc =
  for p in m.procs:
    if p.isKernel:
      if result != nil:
        cRefuse(ctx, "multiple kernels in one module")
      result = p
  if result.isNil:
    cRefuse(ctx, "no {.global.} kernel in the lowered module")

proc inlineSourceC(k: CirProc, m: CirModule, ctx: EmitCtxC): tuple[bodyProc: CirProc, call: CirStmt] =
  ## Inlined-body lookup: the proc whose body inlines into the launch
  ## thread loop nest, plus the wrapper call when the pipeline kept them separate.
  ## - the wrapper body is one call to the user kernel proc, that body
  ##   inlines with the call arguments substituting the parameters
  ## - a pipeline transform already folded the user kernel into the wrapper,
  ##   the wrapper body is the kernel body itself, no call
  var call: CirStmt = nil
  for st in k.body:
    case st.kind
    of sCall:
      if call != nil:
        cRefuse(ctx, "the kernel wrapper body makes multiple calls")
      call = st
    else:
      discard
  if call.isNil:
    return (k, nil)
  let callee = call.callCallee
  if callee.kind != eSym or callee.builtin:
    cRefuse(ctx, "the kernel wrapper does not call a module proc")
  for p in m.procs:
    if p.name == callee.name:
      return (p, call)
  cRefuse(ctx, "the kernel wrapper calls an unknown proc '" & callee.name & "'")

proc kernelRuntimeParams*(m: CirModule): seq[tuple[name: string, typ: CirType]] =
  ## Kernel parameters only: the launch macro bakes static parameters into the call before CIRv2 exists.
  for p in m.procs:
    if p.isKernel:
      for param in p.params:
        result.add((param.name, param.typ))
      return

proc emitCpuCalleesC*(m: CirModule, suffix: string): string =
  ## Callee proc emission as C function definitions.
  ##
  ## - callees emit as file-scope definitions, prototypes first:
  ##   the frontend harvests a callee after its caller, so a mutual
  ##   recursion cycle needs a forward declaration
  ## - `{.inline.}` callees print the `static inline` qualifier
  ## - the kernel wrapper and the inlined kernel proc print nothing,
  ##   their bodies print at the launch site through emitCpuWorkgroupLaunchC
  ## - hoisted threadgroup/constant globals refuse in v0
  if m.globals.len > 0:
    cRefuse(EmitCtxC(procName: "module"),
      "hoisted threadgroup/constant globals are not supported in v0")
  var kctx = EmitCtxC(procName: "module")
  var inlinedName = ""
  for p in m.procs:
    if p.isKernel:
      inlinedName = inlineSourceC(findKernelC(m, kctx), m, kctx).bodyProc.name
      break
  var protos: seq[string]
  var blocks: seq[string]
  for p in m.procs:
    if p.isKernel or p.name == inlinedName:
      continue
    var ctx = EmitCtxC(procName: p.name, suffix: suffix)
    var params: seq[string]
    for param in p.params:
      if param.varParam:
        cRefuse(ctx,
          "`var` parameters are not supported in v0, the cpu backend takes whole values")
      params.add(declaratorC(param.typ, param.name, ctx))
    let retT = if p.ret.kind == ctVoid: "void" else: emitTypeC(p.ret, ctx)
    # C99 `inline` alone emits no external definition, a call that the optimizer does not inline fails to link,
    # `static` gives an inline definition file-local linkage
    let qual = if p.inline: "static inline " else: ""
    let signature = qual & retT & " " & p.name & suffix & "(" & params.join(", ") & ")"
    protos.add(signature & ";")
    blocks.add(signature & " {\n" & printProcBodyC(p, ctx) & "\n}")
  if protos.len > 0:
    result = protos.join("\n") & "\n\n" & blocks.join("\n\n")
  else:
    result = blocks.join("\n\n")

proc kernelBodyBoundC(m: CirModule, suffix: string, coords: CpuCoordTableC,
                      dataPrefix: string,
                      memberNames: seq[string]): tuple[body: string, coordUse: bool] =
  ## Kernel body emission with the runtime parameters bound to the data
  ## struct members: parameter i reads `dataPrefix & memberNames[i]`,
  ## baked statics substitute to their C literals.
  cpuCoordUseC = false
  var kctx = EmitCtxC(procName: "kernel", suffix: suffix)
  let k = findKernelC(m, kctx)
  let (bodyProc, call) = inlineSourceC(k, m, kctx)

  var subst: seq[(string, string)] = @[]
  var bctx = EmitCtxC(procName: bodyProc.name,
                      suffix: suffix, coords: coords, inlineKernel: true)
  if call.isNil:
    # folded shape: the wrapper body is the kernel body, every parameter reads its data struct member
    for i, param in bodyProc.params:
      if i < memberNames.len:
        subst.add((param.name, dataPrefix & memberNames[i]))
  else:
    for i, param in bodyProc.params:
      if i < call.callArgs.len:
        let arg = call.callArgs[i]
        if arg.kind == eSym and not arg.builtin and arg.name == param.name:
          # runtime param: the wrapper passes the formal verbatim, the body reads the data struct member
          subst.add((param.name, dataPrefix & memberNames[i]))
        else:
          # baked static: the wrapper passes the literal
          subst.add((param.name, printExpr(arg, bctx)))
  bctx.subst = subst
  var lines: seq[string]
  printStmts(bodyProc.body, bctx, 0, lines)
  result.body = lines.join("\n")
  result.coordUse = cpuCoordUseC

proc emitCpuWorkgroupLaunchC*(m: CirModule, suffix: string, coords: CpuCoordTableC, runtimeParams: seq[tuple[name: string, typ: CirType]], serial: bool): tuple[typesAndData: string, funcs: string, coordUse: bool] =
  ## Workgroup task machinery as C text, returned in two parts:
  ##
  ## - typesAndData: the nominal struct definitions and the per-launch
  ##   data struct
  ## - funcs: the iteration function with decomposition prologue, thread
  ##   loop nest and kernel body, plus the task function except in serial mode
  ##
  ## Launch assembly: typesAndData, then the callee definitions, then funcs.
  ## Callees reference the structs, the kernel body references callees.
  ##
  ## - the iteration function has the tpc_range_fn shape (int32, void*),
  ##   it casts the env to the data struct, one call covers the task's
  ##   contiguous workgroup range, crucibleWgsPerTask workgroups at a time
  ## - the task function reads the env self-pointer and hands the task
  ##   to the parallelForRunner, TPC_DEFINE_PARFOR shape
  ## - serial mode omits the task function, the launch tail
  ##   (emitCpuSerialTailC) calls the iteration function directly
  ##
  ## Parameter names must avoid `tpc_` prefixes and the launch-internal crucibleWg/crucibleWgsPerTask/crucibleGrid/crucibleBlk names.
  if m.globals.len > 0:
    cRefuse(EmitCtxC(procName: "module"),
      "hoisted threadgroup/constant globals are not supported in v0")
  for p in runtimeParams:
    if p.name.startsWith("tpc_") or p.name in
        ["crucibleWg", "crucibleWgsPerTask", "crucibleGrid", "crucibleBlk"]:
      error("cpu backend: kernel parameter '" & p.name & "' collides with the launch's internal names")

  let structDefs = emitStructDefsC(m,
      EmitCtxC(procName: "module", suffix: suffix))

  var members: seq[string]
  members.add("int32_t crucibleWg;")
  members.add("int32_t crucibleGrid[3];")
  members.add("int32_t crucibleBlk[3];")
  members.add("int32_t crucibleWgsPerTask;")
  var memberNames: seq[string]
  let mctx = EmitCtxC(procName: "module", suffix: suffix)
  for p in runtimeParams:
    members.add(declaratorC(p.typ, p.name, mctx) & ";")
    memberNames.add(p.name)
  let dataName = "crucibleData" & suffix
  var defs = structDefs
  defs.add("struct " & dataName & " {\n")
  for mem in members:
    defs.add("    " & mem & "\n")
  defs.add("};\n")

  let (bodyC, coordUse) = kernelBodyBoundC(m, suffix, coords, "tpc_d_->", memberNames)

  # iteration function: one call per pool task, tpc_i is the task index.
  # Each task covers crucibleWgsPerTask adjacent workgroups in x-fastest flat order.
  # Every workgroup runs the decomposition prologue, the thread loop nest and the kernel body. The last task clamps at the grid total.
  # grid dims and the workgroup total are bounded to int32 by the launch guard
  # an open range end can exceed the total by up to workgroupsPerTask - 1 workgroups,
  # int64 arithmetic precedes the clamp
  var iter = "void crucibleIter" & suffix & "(int32_t tpc_i, void* tpc_env_) {\n"
  iter.add("    struct " & dataName & "* tpc_d_ = (struct " & dataName & "*)tpc_env_;\n")
  iter.add("    int32_t crucibleWgTotal = tpc_d_->crucibleGrid[0] * tpc_d_->crucibleGrid[1] * tpc_d_->crucibleGrid[2];\n")
  iter.add("    int64_t crucibleWgEnd = (int64_t)tpc_i * tpc_d_->crucibleWgsPerTask + tpc_d_->crucibleWgsPerTask;\n")
  iter.add("    if (crucibleWgEnd > crucibleWgTotal) crucibleWgEnd = crucibleWgTotal;\n")
  iter.add("    for (int32_t crucibleWg = tpc_i * tpc_d_->crucibleWgsPerTask;\n")
  iter.add("         crucibleWg < crucibleWgEnd; ++crucibleWg) {\n")
  iter.add("        int32_t crucibleGx = crucibleWg % tpc_d_->crucibleGrid[0];\n")
  iter.add("        int32_t crucibleGy = (crucibleWg / tpc_d_->crucibleGrid[0]) % tpc_d_->crucibleGrid[1];\n")
  iter.add("        int32_t crucibleGz = crucibleWg / (tpc_d_->crucibleGrid[0] * tpc_d_->crucibleGrid[1]);\n")
  # per-thread-iteration do-block: a kernel return jumps to the label
  # inside the loops and the remaining thread iterations still run
  iter.add("        for (int32_t crucibleBz = 0; crucibleBz < tpc_d_->crucibleBlk[2]; ++crucibleBz)\n")
  iter.add("            for (int32_t crucibleBy = 0; crucibleBy < tpc_d_->crucibleBlk[1]; ++crucibleBy)\n")
  iter.add("                for (int32_t crucibleBx = 0; crucibleBx < tpc_d_->crucibleBlk[0]; ++crucibleBx) {\n")
  iter.add("                    do {\n")
  for line in bodyC.split('\n'):
    if line.len > 0:
      iter.add("                        " & line & "\n")
    else:
      iter.add("\n")
  iter.add("                    } while (0);\n")
  iter.add("                    crucibleThreadDone: ;\n")
  iter.add("                }\n")
  iter.add("    }\n")
  iter.add("}\n")

  result.typesAndData = defs
  result.funcs = iter
  if not serial:
    # task function: the iteration function already has the tpc_range_fn
    # shape (int32, void*), the task reads the env self-pointer and hands
    # itself to the runner
    var task = "void crucibleTask" & suffix & "(void* tpc_env_) {\n"
    task.add("    tpc_task* tpc_this = *(tpc_task**)tpc_env_;\n")
    task.add("    tpc_worker_context* tpc_ctx = tpc_workerContext();\n")
    task.add("    tpc_parallelForRunner(tpc_ctx, tpc_this, crucibleIter" & suffix & ");\n")
    task.add("}\n")
    result.funcs.add(task)
  result.coordUse = coordUse

proc emitCpuDataFillC*(suffix: string, runtimeParams: seq[tuple[name: string, typ: CirType]]): string =
  ## Data-struct fill for a launch tail, both launch modes.
  ##
  ## Grid and block dims copy from the launch's `crucibleGrid`/`crucibleBlk`
  ## dim3 locals, runtime parameters copy from their Nim formals.
  ##
  ## The caller appends the mode's workgroup-range and launch-call lines.
  let mctx = EmitCtxC(procName: "module", suffix: suffix)
  var fill = ""
  for (field, axis) in [("x", 0), ("y", 1), ("z", 2)]:
    fill.add("    crucibleData.crucibleGrid[" & $axis & "] = (int32_t)`crucibleGrid`." & field & ";\n")
    fill.add("    crucibleData.crucibleBlk[" & $axis & "] = (int32_t)`crucibleBlk`." & field & ";\n")
  for p in runtimeParams:
    let sym = "`" & p.name & "`"
    case p.typ.kind
    of ctStruct:
      fill.add("    crucibleData." & p.name & " = *(" & declaratorC(p.typ, "", mctx) & "*)&" & sym & ";\n")
    of ctArray:
      # C assigns arrays only element-wise, the compiler builtin copies
      # without a string.h include
      fill.add("    __builtin_memcpy(crucibleData." & p.name & ", " & sym &
               ", sizeof(crucibleData." & p.name & "));\n")
    else:
      fill.add("    crucibleData." & p.name & " = (" & declaratorC(p.typ, "", mctx) & ")(" & sym & ");\n")
  fill

proc emitCpuSerialTailC*(suffix: string, runtimeParams: seq[tuple[name: string, typ: CirType]]): string =
  ## Serial launch tail: the data struct on the caller stack, one direct call of the iteration function over the whole grid.
  ##
  ## - crucibleWgsPerTask is the grid workgroup total. One call covers every workgroup in x-fastest flat order
  ## - the iteration function clamps the range at the grid total
  ## - the product computes in int64 and the launch's dim guard keeps the total within int32
  var tail = "    extern void crucibleIter" & suffix & "(int32_t tpc_i, void* tpc_env_);\n"
  tail.add("    struct crucibleData" & suffix & " crucibleData;\n")
  tail.add(emitCpuDataFillC(suffix, runtimeParams))
  tail.add("    crucibleData.crucibleWgsPerTask = (int32_t)((int64_t)`crucibleGrid`.x * `crucibleGrid`.y * `crucibleGrid`.z);\n")
  tail.add("    crucibleIter" & suffix & "(0, &crucibleData);\n")
  tail
