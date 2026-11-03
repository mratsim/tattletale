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
import ../../crucible_codegen
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

const cpuModuleShape = ModuleShape(
  spaces: {caThread},
  structTypedef: false,
  structPadding: "char _",
  structFieldTerm: ';',
  structClose: "};",
  fwdDecls: false,
  moduleOrderProcs: false,
  kernelSep: "",
  customProcHead: false,
  printsRefBlocks: false,
  printsParamBindings: false,
  printsReduceTemplates: false)

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

var cpuCoordUseC {.global, compileTime.}: bool
  ## Coordinate-builtin use of the module under emission, reset per emission, read by the launch after the body prints.

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitTypeName(ctx: var PrintCtx, t: CirType): string =
  ## C type name of `t` with no identifier, as casts and sizeof print it.
  ## Refused kinds name the v0 restriction, the shared walk reads the recorded refusal.
  case t.kind
  of ctVector:
    ctx.fail("vector types (" & $t.vN & " x " & $t.vElem.kind & ") are not supported in v0, the vector mapping is the optimization pass (P2)")
    ""
  of ctSpan:
    ctx.fail("openArray spans must split to a pointer and a length before printing (the span split pass)")
    ""
  of ctString:
    ctx.fail("string types are not supported in v0")
    ""
  of ctBf16:
    ctx.fail("bfloat16 values are not supported in v0")
    ""
  of ctProc:
    ctx.fail("function pointers are not supported in v0")
    ""
  else:
    getFullTypeName(TargetVocabs[cgCpu].scalarNames, cpuVectorRule, t, "struct ", ctx.suffix)

proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  ## C declaration text of one struct member via the shared declarator.
  printCDecl(TargetVocabs[cgCpu].scalarNames, cpuVectorRule, printerTarget,
    f.typ, f.name, false, "struct ", ctx.suffix)

proc structDeclName(ctx: var PrintCtx, t: CirType): string =
  ## Header name of one struct declaration with the launch-site suffix.
  t.name & ctx.suffix

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printExpr(ctx: var PrintCtx, e: CirExpr): string

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

proc printBuiltinRefusalC(e: CirExpr, ctx: var PrintCtx): string =
  let what =
    if e.synchroBuiltin != CirSynchroBuiltinKind.cbkNone: "threadgroup barriers"
    elif e.shuffleBuiltin != CirShuffleBuiltinKind.cbkNone: "simd_shuffle/simd_shuffle_xor"
    elif e.reductionBuiltin != CirReductionBuiltinKind.cbkNone: "subgroup reductions"
    elif e.atomicsBuiltin != CirAtomicsBuiltinKind.cbkNone: "atomic builtins"
    else: "unknown builtins"
  ctx.fail(what & " are not supported in v0")
  e.name

proc calleeSymbolC(name: string, ctx: PrintCtx): string =
  ## C symbol for a callee: kernel-printed callees append the launch-site suffix,
  ## `__builtin_` compiler builtins pass verbatim, the decomposeArrayAssign pass emits them.
  if name.startsWith("__builtin_"):
    name
  else:
    name & ctx.suffix

proc printExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## C text of one structured expression over the cpu launch's rewrite state.
  case e.kind
  of eNil:
    result = "NULL"
  of eTrue:
    result = "true"
  of eFalse:
    result = "false"
  of eStrLit:
    ctx.fail("string values are not supported in v0")
  of eSym:
    if e.builtin:
      if e.coordBuiltin != CirCoordBuiltinKind.cbkNone:
        if not ctx.inlineKernel:
          ctx.fail("coordinate builtins in device callees are not supported in v0, pass them as parameters")
        cpuCoordUseC = true
        result = coordExprC(ctx.coords, e.coordBuiltin, -1)
        if result.len == 0:
          ctx.fail("no launch expression for coordinate builtin " & $e.coordBuiltin)
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
    result = "*" & ctx.printExpr(e.derefOperand)
  of eAt:
    let base = ctx.printExpr(e.atBase)
    # C parses a prefix dereference tighter than the index,
    # `*p[i]` reads `*(p[i])`
    if e.atBase.kind == eDeref:
      result = "(" & base & ")[" & ctx.printExpr(e.atIndex) & "]"
    else:
      result = base & "[" & ctx.printExpr(e.atIndex) & "]"
  of ePat:
    # pointer-AT: aptr indexing, vector swizzles are not supported in v0
    let bt = getExprType(e.patBase)
    if bt != nil and bt.kind == ctVector:
      ctx.fail("vector swizzles are not supported in v0")
    result = ctx.printExpr(e.patBase) & "[" & ctx.printExpr(e.patIndex) & "]"
  of eDot:
    let base = e.dotBase
    if base.kind == eSym and base.builtin and base.coordBuiltin != CirCoordBuiltinKind.cbkNone:
      if not ctx.inlineKernel:
        ctx.fail("coordinate builtin components in device callees are not supported in v0, pass them as parameters")
      cpuCoordUseC = true
      let comp = case e.fieldName
                 of "x": 0
                 of "y": 1
                 of "z": 2
                 else: -1
      result = coordExprC(ctx.coords, base.coordBuiltin, comp)
      if result.len == 0:
        ctx.fail("no launch expression for " & $base.coordBuiltin & "." & e.fieldName)
    else:
      let baseTxt = ctx.printExpr(e.dotBase)
      # C parses a prefix dereference tighter than the field access,
      # `*p.field` reads `*(p.field)`
      if e.dotBase.kind == eDeref:
        result = "(" & baseTxt & ")." & e.fieldName
      else:
        result = baseTxt & "." & e.fieldName
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    if e.kind == eMod and e.opType.kind == ctFloat:
      ctx.fail("float mod has no C operator, CIRv2 lowers it for integer selectors only")
    let op = case e.kind
             of eAdd: "+"
             of eSub: "-"
             of eMul: "*"
             of eDiv: "/"
             of eMod: "%"
             of eShl: "<<"
             of eShr: ">>"
             of eBitAnd: "&"
             of eBitOr: "|"
             of eBitXor: "^"
             else: ""
    result = "(" & ctx.printExpr(e.left) & " " & op & " " & ctx.printExpr(e.right) & ")"
  of eNeg:
    result = "(-" & ctx.printExpr(e.unOperand) & ")"
  of eEq, eNeq, eLt, eLe:
    let op = case e.kind
             of eEq: "=="
             of eNeq: "!="
             of eLt: "<"
             else: "<="
    result = "(" & ctx.printExpr(e.a) & " " & op & " " & ctx.printExpr(e.b) & ")"
  of eNot:
    result = "(!" & ctx.printExpr(e.notOperand) & ")"
  of eAnd, eOr:
    let op = if e.kind == eAnd: "&&" else: "||"
    result = "(" & ctx.printExpr(e.andLeft) & " " & op & " " & ctx.printExpr(e.andRight) & ")"
  of eTernary:
    result = "(" & ctx.printExpr(e.cond) & " ? " & ctx.printExpr(e.thenE) &
             " : " & ctx.printExpr(e.elseE) & ")"
  of eCast:
    # bit-level reinterpret through a compound-literal union: header-free, aliasing-safe, the compiler folds it to a register move
    let srcT = getExprType(e.castOperand)
    if srcT == nil or e.castType.kind notin
        {ctBool, ctInt, ctUint, ctFloat, ctPtr, ctAptr} or
        srcT.kind notin {ctBool, ctInt, ctUint, ctFloat, ctPtr, ctAptr}:
      ctx.fail("cast between non-scalar types is not supported in v0")
    result = "((union { " & ctx.emitTypeName(srcT) & " _from; " &
             ctx.emitTypeName(e.castType) & " _to; }){ ._from = " &
             ctx.printExpr(e.castOperand) & " })._to"
  of eConv:
    result = "(" & ctx.emitTypeName(e.castType) & ")" & ctx.printExpr(e.castOperand)
  of eAddr:
    result = "&" & ctx.printExpr(e.addrOperand)
  of eSizeof:
    # array dimensions move behind the type name, the C sizeof form
    var t = e.ofType
    var dims = ""
    while t.kind == ctArray:
      dims &= "[" & $t.aN & "]"
      t = t.aElem
    result = "sizeof(" & ctx.emitTypeName(t) & dims & ")"
  of eOconstr:
    let t = e.ocType
    if t.kind != ctStruct:
      ctx.fail("object constructor over a non-struct type")
    var parts: seq[string]
    for kv in e.kvFields:
      parts.add("." & kv.name & " = " & ctx.printExpr(kv.value))
    result = "(" & ctx.emitTypeName(t) & "){" & parts.join(", ") & "}"
  of eAconstr:
    var parts: seq[string]
    for el in e.elems:
      parts.add(ctx.printExpr(el))
    result = "{" & parts.join(", ") & "}"
  of eCall:
    let callee = e.callee
    if callee.kind == eSym and callee.builtin:
      discard printBuiltinRefusalC(callee, ctx)
    var calleeRef = ctx.printExpr(callee)
    if callee.kind == eSym and not callee.builtin:
      calleeRef = calleeSymbolC(callee.name, ctx)
    var args: seq[string]
    for a in e.args:
      # C converts implicitly at call sites, CIRv2 conversions print
      # explicitly as casts where they exist
      args.add(ctx.printExpr(a))
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

proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool =
  ## Local declaration policy of the cpu launch: refused address spaces
  ## and the inline kernel's shadow refusals.
  case s.addressSpace
  of caThreadgroup:
    ctx.fail("threadgroup (shared) memory is not supported in v0")
    return false
  of caConstant, caDevice:
    ctx.fail("constant/device memory-class locals are not supported in v0")
    return false
  of caThread:
    discard
  if ctx.inlineKernel:
    # substitution works by name, a local shadowing a substituted
    # parameter would misbind
    for (name, _) in ctx.subst:
      if name == s.name:
        ctx.fail("kernel local '" & s.name & "' shadows a kernel parameter, rename one")
        return false
    if s.name in LaunchCounterNamesC:
      # kernel locals must avoid the launch's generated counter names:
      # a shadowing local silently corrupts the coordinate reads
      ctx.fail("kernel local '" & s.name & "' shadows a counter the launch generates, rename it")
      return false
  true

proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  ## C declaration text of one local via the shared declarator.
  printCDecl(TargetVocabs[cgCpu].scalarNames, cpuVectorRule, printerTarget,
    s.declType, s.name, false, "struct ", ctx.suffix)

proc printRet(ctx: var PrintCtx, s: CirStmt, pad: string): string =
  ## Return statement of the cpu launch: the inline kernel exits per-thread
  ## scope through the launch's done label, callees return values.
  if ctx.inlineKernel:
    # a return inside the inlined body exits at the launch's done label
    if s.retVal != nil:
      ctx.fail("kernels are void, a return value has no meaning")
    return pad & "goto crucibleThreadDone;"
  if s.retVal != nil:
    return pad & "return " & ctx.printExpr(s.retVal) & ";"
  pad & "return;"

proc printDiscard(ctx: var PrintCtx, s: CirStmt, pad: string): string =
  ## Discard statement of the cpu launch: the value expression keeps its side effects, cast to void.
  if s.discardVal != nil:
    pad & "(void)(" & ctx.printExpr(s.discardVal) & ");"
  else:
    pad & ";"

# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

proc findKernelC(m: CirModule, ctx: var PrintCtx): CirProc =
  for p in m.procs:
    if p.isKernel:
      if result != nil:
        ctx.fail("multiple kernels in one module")
      result = p
  if result.isNil:
    ctx.fail("no {.global.} kernel in the lowered module")

proc inlineSourceC(k: CirProc, m: CirModule, ctx: var PrintCtx): tuple[bodyProc: CirProc, call: CirStmt] =
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
        ctx.fail("the kernel wrapper body makes multiple calls")
      call = st
    else:
      discard
  if call.isNil:
    return (k, nil)
  let callee = call.callCallee
  if callee.kind != eSym or callee.builtin:
    ctx.fail("the kernel wrapper does not call a module proc")
  for p in m.procs:
    if p.name == callee.name:
      return (p, call)
  ctx.fail("the kernel wrapper calls an unknown proc '" & callee.name & "'")

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
  var kctx = initPrintCtx(cpuWalkPolicy, cpuVectorRule, cpuEmit, cgCpu)
  var inlinedName = ""
  for p in m.procs:
    if p.isKernel:
      inlinedName = inlineSourceC(findKernelC(m, kctx), m, kctx).bodyProc.name
      break
  if kctx.failing():
    error(kctx.err)
  var protos: seq[string]
  var blocks: seq[string]
  for p in m.procs:
    if p.isKernel or p.name == inlinedName:
      continue
    var ctx = initPrintCtx(cpuWalkPolicy, cpuVectorRule, cpuEmit, cgCpu)
    ctx.fnName = p.name
    ctx.suffix = suffix
    var params: seq[string]
    for param in p.params:
      if param.varParam:
        ctx.fail("`var` parameters are not supported in v0, the cpu backend takes whole values")
      params.add(printCDecl(TargetVocabs[cgCpu].scalarNames, cpuVectorRule, printerTarget,
        param.typ, param.name, false, "struct ", ctx.suffix))
    let retT = ctx.emitTypeName(p.ret)
    # C99 `inline` alone emits no external definition, a call that the optimizer does not inline fails to link,
    # `static` gives an inline definition file-local linkage
    let qual = if p.inline: "static inline " else: ""
    let signature = qual & retT & " " & p.name & suffix & "(" & params.join(", ") & ")"
    protos.add(signature & ";")
    ctx.indent = 1
    blocks.add(signature & " {\n" & ctx.printStmts(p.body) & "\n}")
    if ctx.failing():
      error(ctx.err)
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
  var kctx = initPrintCtx(cpuWalkPolicy, cpuVectorRule, cpuEmit, cgCpu)
  kctx.fnName = "kernel"
  kctx.suffix = suffix
  let k = findKernelC(m, kctx)
  if kctx.failing():
    error(kctx.err)
  let (bodyProc, call) = inlineSourceC(k, m, kctx)
  if kctx.failing():
    error(kctx.err)

  var subst: seq[(string, string)] = @[]
  var bctx = initPrintCtx(cpuWalkPolicy, cpuVectorRule, cpuEmit, cgCpu)
  bctx.fnName = bodyProc.name
  bctx.suffix = suffix
  bctx.coords = coords
  bctx.inlineKernel = true
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
          subst.add((param.name, bctx.printExpr(arg)))
  bctx.subst = subst
  result.body = bctx.printStmts(bodyProc.body)
  result.coordUse = cpuCoordUseC
  if bctx.failing():
    error(bctx.err)

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
  var mctx = initPrintCtx(cpuWalkPolicy, cpuVectorRule, cpuEmit, cgCpu)
  mctx.suffix = suffix
  if m.globals.len > 0:
    mctx.fail("hoisted threadgroup/constant globals are not supported in v0")
    error(mctx.err)
  for p in runtimeParams:
    if p.name.startsWith("tpc_") or p.name in
        ["crucibleWg", "crucibleWgsPerTask", "crucibleGrid", "crucibleBlk"]:
      error("cpu backend: kernel parameter '" & p.name & "' collides with the launch's internal names")

  let structDefs = mctx.printStructs(m, cpuModuleShape)
  if mctx.failing():
    error(mctx.err)

  var members: seq[string]
  members.add("int32_t crucibleWg;")
  members.add("int32_t crucibleGrid[3];")
  members.add("int32_t crucibleBlk[3];")
  members.add("int32_t crucibleWgsPerTask;")
  var memberNames: seq[string]
  for p in runtimeParams:
    members.add(printCDecl(TargetVocabs[cgCpu].scalarNames, cpuVectorRule, printerTarget,
      p.typ, p.name, false, "struct ", suffix) & ";")
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
  var fill = ""
  for (field, axis) in [("x", 0), ("y", 1), ("z", 2)]:
    fill.add("    crucibleData.crucibleGrid[" & $axis & "] = (int32_t)`crucibleGrid`." & field & ";\n")
    fill.add("    crucibleData.crucibleBlk[" & $axis & "] = (int32_t)`crucibleBlk`." & field & ";\n")
  for p in runtimeParams:
    let sym = "`" & p.name & "`"
    case p.typ.kind
    of ctStruct:
      fill.add("    crucibleData." & p.name & " = *(" & printCDecl(
        TargetVocabs[cgCpu].scalarNames, cpuVectorRule, printerTarget, p.typ, "", true,
        "struct ", suffix) & "*)&" & sym & ";\n")
    of ctArray:
      # C assigns arrays only element-wise, the compiler builtin copies
      # without a string.h include
      fill.add("    __builtin_memcpy(crucibleData." & p.name & ", " & sym &
               ", sizeof(crucibleData." & p.name & "));\n")
    else:
      fill.add("    crucibleData." & p.name & " = (" & printCDecl(
        TargetVocabs[cgCpu].scalarNames, cpuVectorRule, printerTarget, p.typ, "", true,
        "struct ", suffix) & ")(" & sym & ");\n")
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
