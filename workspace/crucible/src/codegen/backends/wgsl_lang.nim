# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## WebGPU WGSL printer: a structured CIRv2 module rendered as WGSL source.
## The universal walk lives in `print_common`, this overlay carries the WGSL delta.
##
## Kernel parameter bindings, one `@group(0) @binding(i)` per parameter in host `store` packing order, output first:
## - device buffers declare `var<storage, read_write> name: array<T>;`, bound writable by the engine
## - scalars and value structs declare `var<storage, read> name: T;`, packed into the 256-byte-aligned scratch buffer
## - a bool scalar or bool struct field binds as the `NIM_BOOL` alias of `u32`
##   (bool is not host-shareable in WGSL storage)
##
## Bool storage and the kernel signature:
## - reads of `NIM_BOOL` storage coerce to bool, writes coerce to NIM_BOOL
## - the kernel signature takes only `@builtin(...)` coordinate parameters, first-use order
##
## Addressing, memory classes, jumps:
## - the WGSL lowering pass rewrites pointer locals into their pointees,
##   pointer-holding struct locals into their leaf symbols
## - the WGSL lowering pass relocates a smem/const_mem kernel local into `CirModule.globals`
## - struct declarations emit for every struct of `CirModule.types` in declaration order,
##   a dissolved one included, an unused declaration is legal WGSL,
##   a pointer field drops from the printed declaration (WGSL has no pointer struct members)
## - a loop prints `while (cond)`, an unrolled loop notes the unroll directive
##   has no WGSL form

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import ../passes/pass_helpers
import std/[macros, math, sets, strutils, tables]
import ../ir/ast_datatypes
import ./print_common
import ./targets_registry

const printerTarget = "wgsl"

type PrintCtx = object of WalkCtx
  ## Per-target walk context. The shared walk procs instantiate against
  ## the derived type and the overlay procs below resolve per target.
  nimBoolNames: HashSet[string]
  ## Names whose storage binds as `NIM_BOOL`: kernel bool parameters,
  ## bool workgroup variables.
  ## Reads coerce to `bool`, writes coerce to `NIM_BOOL`.
  storageBoolSpelling: bool
  ## When set, a bool leaf in `emitWgslScalar` becomes `NIM_BOOL`.
  ## Set only around interface-storage declarations.

proc coordRef(ctx: var PrintCtx, e: CirExpr): string
proc coordDot(ctx: var PrintCtx, e: CirExpr): string
proc printConv(ctx: var PrintCtx, e: CirExpr): string
proc emitTypeName(ctx: var PrintCtx, t: CirType): string
proc printSizeof(ctx: var PrintCtx, e: CirExpr): string
proc printExtraExpr(ctx: var PrintCtx, e: CirExpr): string
proc getSymbolText(ctx: var PrintCtx, e: CirExpr): string
proc getDotText(ctx: var PrintCtx, e: CirExpr, baseTxt: string): string
proc getCtorValue(ctx: var PrintCtx, t: CirType, kvv: CirKv): string
proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string
proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool
proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string
proc constDecl(ctx: var PrintCtx, s: CirStmt): string

# ═══════════════════════════════════════════════════════════════════════
# Target tables
# ═══════════════════════════════════════════════════════════════════════

const wgslVocab = BuiltinVocab(
  reductionText: makeTexts([
    (cbkReduceAdd, "subgroupAdd($1)"),
    (cbkReduceMax, "subgroupMax($1)")]))

const wgslEmit = TargetEmit(
  vocab: wgslVocab,
  functionalCtor: true,
  ctorOpen: "(",
  ctorClose: ")",
  ctorCast: false,
  ctorZero: "$1(0u)",
  aconstrCasts: false,
  dotDerefParens: false,
  vectorDots: true,
  noPointerValues: (spaces: {}, text: ""),
  noPointerRebind: "",
  prefixKernel: "",
  prefixInline: "",
  prefixPlain: "")

const wgslModuleShape = ModuleShape(
  spaces: {caThreadgroup, caConstant},
  structTypedef: false,
  structPadding: "_padding: u32",
  structFieldTerm: ',',
  structClose: "}",
  fwdDecls: false,
  moduleOrderProcs: false,
  kernelSep: "\n",
  customProcHead: true,
  printsRefBlocks: false,
  printsParamBindings: true,
  printsReduceTemplates: false)

# WGSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
# `bitcast` from the unsigned integer of the same width. Literals without a suffix
# are abstract and work in any typed context, u32 gains the `u` suffix, f16 the `h`.

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitWgslScalar(ctx: var PrintCtx, t: CirType): string =
  ## Returns the WGSL name of a non-array leaf type, "" when the caller
  ## composes the name.
  ##
  ## 64-bit leaves refuse: their device features exist but the vendored
  ## wgpu-native exposes no WGPUFeatureName to request them.
  case t.kind
  of ctBool:
    if ctx.storageBoolSpelling:
      result = "NIM_BOOL"
    else:
      result = getScalarName(wgslScalarNames, t)
  of ctVoid:
    result = getScalarName(wgslScalarNames, t)
  of ctInt, ctUint:
    result = getScalarName(wgslScalarNames, t)
    if result.len == 0:
      if t.bits == 64:
        ctx.fail("i64/u64 need the shader-int64 device feature, the vendored wgpu-native cannot request it")
      else:
        ctx.fail($t.bits & "-bit integers have no WGSL type, only 32")
  of ctFloat:
    result = getScalarName(wgslScalarNames, t)
    if result.len == 0:
      if t.bits == 64:
        ctx.fail("f64 needs the shader-f64 device feature, the vendored wgpu-native cannot request it")
      else:
        ctx.fail($t.bits & "-bit floats have no WGSL type")
  of ctBf16:
    ctx.fail("bfloat16 has no WGSL type")
  of ctVector:
    let elem = ctx.emitWgslScalar(t.vElem)
    result = getVectorName(wgslVectorRule, wgslScalarNames, t)
    if result.len == 0 and elem.len > 0:
      ctx.fail("vector of " & elem & "x" & $t.vN & " has no WGSL type name, only 2..4-wide vectors have a name")
  of ctStruct:
    result = t.name
  of ctString:
    ctx.fail("string has no WGSL type")
  of ctPtr, ctAptr, ctArray, ctProc, ctSpan:
    result = ""

proc emitTypeName(ctx: var PrintCtx, t: CirType): string =
  ## Returns the WGSL name of `t` with no identifier, the form that casts
  ## and constructors use.
  case t.kind
  of ctArray:
    let elem = ctx.emitTypeName(t.aElem)
    if t.aN == 0:
      result = "array<" & elem & ">"
    else:
      result = "array<" & elem & ", " & $t.aN & ">"
  else:
    result = ctx.emitWgslScalar(t)

proc structFieldIsBool(t: CirType, field: string): bool =
  ## True when `field` names a bool member of struct `t`, a NIM_BOOL storage word.
  if t.isNil or t.kind != ctStruct:
    return false
  for f in t.fields:
    if f.name == field:
      return f.typ.kind == ctBool
  return false

# ═══════════════════════════════════════════════════════════════════════
# Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

type WgslBuiltinParam = tuple[attribute, param, wgslType: string]
  ## Canonical coordinate builtin name -> WGSL `@builtin(...)` kernel parameter.
  ## - `attribute`: the WGSL builtin attribute.
  ## - `param`: the identifier the kernel body reads.
  ## - `wgslType`: the WGSL type (scalar for the flat thread index).

proc getWgslBuiltinParam(k: CirCoordBuiltinKind): WgslBuiltinParam =
  ## Returns the WGSL parameter of a coordinate builtin kind, an empty tuple when none exists.
  ## `threads_per_threadgroup` is absent, WGSL exposes the threadgroup size
  ## only as the `@workgroup_size` entry-point attribute, the printer refuses a read.
  case k
  of cbkThreadPositionInGrid:
    ("global_invocation_id", "global_id", "vec3<u32>")
  of cbkThreadgroupPositionInGrid:
    ("workgroup_id", "workgroup_id", "vec3<u32>")
  of cbkThreadPositionInThreadgroup:
    ("local_invocation_id", "local_invocation_id", "vec3<u32>")
  of cbkThreadgroupsPerGrid:
    ("num_workgroups", "num_workgroups", "vec3<u32>")
  of cbkThreadIndexInThreadgroup:
    ("local_invocation_index", "local_invocation_index", "u32")
  else:
    ("", "", "")

proc getWgslCoordComponent(k: CirCoordBuiltinKind, field: string): string =
  case field
  of "x", "y", "z":
    let p = getWgslBuiltinParam(k)
    if p.attribute.len > 0:
      result = "i32(" & p.param & "." & field & ")"
  else:
    result = ""

proc getWgslWholeVector(k: CirCoordBuiltinKind): string =
  let p = getWgslBuiltinParam(k)
  result = "vec3<i32>(i32(" & p.param & ".x), i32(" & p.param & ".y), i32(" &
    p.param & ".z))"

const wgslEntryInputs*: EntryInputs = block:
  ## WGSL entry-input text per coordinate builtin kind: the `@builtin(...)` parameter that the entry signature declares.
  ##
  ## - built from the `getWgslBuiltinParam` table, one text per kind that has a parameter
  ## - a kind without a WGSL builtin has no text: the printer refuses a `threads_per_threadgroup` read,
  ##   WGSL has only the `@workgroup_size` attribute
  var inputs: EntryInputs
  for k in cbkThreadPositionInGrid..cbkThreadIndexInThreadgroup:
    let p = getWgslBuiltinParam(k)
    if p.attribute.len > 0:
      inputs[k] = "@builtin(" & p.attribute & ") " & p.param & ": " & p.wgslType
  inputs

proc coordRef(ctx: var PrintCtx, e: CirExpr): string =
  ## One coordinate builtin read in the kernel body. WGSL builtins are
  ## entry-point parameters, a device function read refuses.
  if not ctx.inKernel:
    ctx.fail("coordinate builtin " & CoordBuiltinNames[e.coordBuiltin] & " in a device function, WGSL builtins are entry-point parameters")
    return ""
  if e.coordBuiltin == cbkThreadsPerThreadgroup:
    ctx.fail("threads_per_threadgroup has no WGSL builtin, WGSL has only the @workgroup_size attribute")
    return ""
  ctx.builtins.incl(e.coordBuiltin)
  if e.coordBuiltin == cbkThreadIndexInThreadgroup:
    return "i32(" & getWgslBuiltinParam(e.coordBuiltin).param & ")"
  getWgslWholeVector(e.coordBuiltin)

proc coordDot(ctx: var PrintCtx, e: CirExpr): string =
  ## One coordinate builtin component read in the kernel body.
  if not ctx.inKernel:
    ctx.fail("a device function reads " & CoordBuiltinNames[e.dotBase.coordBuiltin] & ", WGSL coordinate builtins are entry-point parameters")
    return ""
  let mapped = getWgslCoordComponent(e.dotBase.coordBuiltin, e.fieldName)
  if mapped.len == 0:
    if e.dotBase.coordBuiltin == cbkThreadsPerThreadgroup:
      ctx.fail("threads_per_threadgroup reads refuse on WGSL, only the @workgroup_size attribute exists")
    else:
      ctx.fail("coordinate builtin has no component " & e.fieldName)
    return ""
  ctx.builtins.incl(e.dotBase.coordBuiltin)
  mapped

# ═══════════════════════════════════════════════════════════════════════
# Expressions: the WGSL hooks over the universal variants
# ═══════════════════════════════════════════════════════════════════════

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## WGSL has no bool constructor and no bool-to-numeric conversion.
  if e.castType.kind == ctBool:
    ctx.fail("conversion to bool, WGSL has no bool constructor")
    return ""
  let bt = getExprType(e.castOperand)
  if bt != nil and bt.kind == ctBool:
    ctx.fail("conversion from bool, WGSL has no bool-to-numeric conversion")
    return ""
  ctx.emitTypeName(e.castType) & "(" & ctx.printExpr(e.castOperand) & ")"

proc printSizeof(ctx: var PrintCtx, e: CirExpr): string =
  ctx.fail(getUnhandledExprRejection(e.kind))
  ""

proc getSymbolText(ctx: var PrintCtx, e: CirExpr): string =
  ## A NIM_BOOL storage word coerces to bool on read.
  if e.typ != nil and e.typ.kind == ctBool and e.name in ctx.nimBoolNames:
    return "bool(" & e.name & ")"
  e.name

proc getDotText(ctx: var PrintCtx, e: CirExpr, baseTxt: string): string =
  ## A NIM_BOOL storage word coerces to bool on read.
  if structFieldIsBool(getExprType(e.dotBase), e.fieldName):
    return "bool(" & baseTxt & "." & e.fieldName & ")"
  baseTxt & '.' & e.fieldName

proc getCtorValue(ctx: var PrintCtx, t: CirType, kvv: CirKv): string =
  ## A NIM_BOOL storage word coerces the stored value.
  if structFieldIsBool(t, kvv.name):
    return "NIM_BOOL(" & ctx.printExpr(kvv.value) & ")"
  ctx.printExpr(kvv.value)

proc printExtraExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## WGSL routes the literal, ternary and addressing kinds itself.
  case e.kind
  of eFloatLit:
    # every bitcast<f64> operand form fails, a u32-suffixed 64-bit literal is a parse error
    if e.fValue.classify in {fcInf, fcNegInf, fcNan} and e.fLitType.bits == 64:
      ctx.fail("a 64-bit non-finite float has no WGSL token, bitcast<f64> needs the unrequestable shader-f64 feature")
      return ""
    emitLiteral(e, ctx.lit, ctx.scalarNames)
  of eTernary:
    # select(f, t, cond) puts the false value first, both branches evaluate
    # eagerly and a faulting read in either branch runs whatever the condition
    "select(" & ctx.printExpr(e.elseE) & ", " & ctx.printExpr(e.thenE) &
      ", " & ctx.printExpr(e.cond) & ")"
  of eAt, ePat:
    let base = if e.kind == eAt: e.atBase else: e.patBase
    var (baseTxt, idxTxt) = ctx.printSubscriptParts(e)
    let bt = getExprType(base)
    if bt != nil and bt.kind in {ctPtr, ctAptr} and
        not (bt.kind == ctAptr and bt.space == caDevice):
      baseTxt = "(*" & baseTxt & ")[" & idxTxt & "]"
    else:
      baseTxt = baseTxt & "[" & idxTxt & "]"
    if base.kind == eSym and base.name in ctx.nimBoolNames:
      # storage is a u32 word, element reads expect bool
      return "bool(" & baseTxt & ")"
    baseTxt
  else:
    ctx.fail(getUnhandledExprRejection(e.kind))
    ""


proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string =
  ## Returns the WGSL barrier statement texts. WGSL has no combined barrier:
  ## the workgroup fence runs first and the storage fence comes second.
  case s.callCallee.synchroBuiltin
  of cbkThreadgroupBarrier:
    pad & "workgroupBarrier();"
  of cbkThreadgroupBarrierDevice:
    pad & "workgroupBarrier();\n" & pad & "storageBarrier();"
  else:
    ""

# ═══════════════════════════════════════════════════════════════════════
# Statements: the WGSL space policy and declaration forms
# ═══════════════════════════════════════════════════════════════════════

proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool =
  ## WGSL locals are function or workgroup storage, the WGSL lowering pass
  ## hoists the storage spaces to program scope.
  if s.space == caThreadgroup:
    ctx.fail("local " & s.name & " in threadgroup space, the WGSL lowering" &
      " pass hoists threadgroup locals to program scope")
    return false
  elif s.space == caConstant:
    ctx.fail("local " & s.name & " in constant space, the WGSL lowering" &
      " pass hoists const_mem locals to program scope")
    return false
  elif s.space == caDevice:
    ctx.fail("local " & s.name & " in device space, WGSL locals are" &
      " function or workgroup")
    return false
  true

proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  "var " & s.name & ": " & ctx.emitTypeName(s.declType)

proc constDecl(ctx: var PrintCtx, s: CirStmt): string =
  "let " & s.name & ": " & ctx.emitTypeName(s.declType)

proc printAsgn(ctx: var PrintCtx, s: CirStmt): string =
  ## One assignment. A NIM_BOOL storage target coerces the stored value:
  ## a bool parameter, a bool struct member, a bool workgroup variable.
  let pad = spaces(ctx.indent * 2)
  var boolTarget = ""
  if s.lhs.kind == eSym and s.lhs.name in ctx.nimBoolNames:
    boolTarget = s.lhs.name
  elif s.lhs.kind == eDot and
      structFieldIsBool(getExprType(s.lhs.dotBase), s.lhs.fieldName):
    boolTarget = ctx.printExpr(s.lhs.dotBase) & "." & s.lhs.fieldName
  elif s.lhs.kind in {eAt, ePat}:
    let lhsBase = if s.lhs.kind == eAt: s.lhs.atBase else: s.lhs.patBase
    if lhsBase.kind == eSym and lhsBase.name in ctx.nimBoolNames:
      let idx = if s.lhs.kind == eAt: s.lhs.atIndex else: s.lhs.patIndex
      boolTarget = lhsBase.name & "[" & ctx.printExpr(idx) & "]"
  if boolTarget.len > 0:
    return pad & boolTarget & " = NIM_BOOL(" & ctx.printExpr(s.rhs) & ");"
  result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules: the shader hooks over the shared engine
# ═══════════════════════════════════════════════════════════════════════

proc printProcHead(ctx: var PrintCtx, fn: CirProc): string =
  ## Head of one proc, signature and open brace:
  ##
  ## - a device function: the bare `fn name(params) -> ret` signature, a void return emits no arrow (WGSL has no `void` type name)
  ## - a pointer parameter takes the `ptr<function, T>` form
  ## - the entry kernel: `@compute @workgroup_size(...)` attribute lines, then `fn <entry>(<entry inputs>)` over the body-walk inputs
  ##   exportc renames the entry for the host runtime lookup.
  if fn.isKernel:
    let wg = getWorkgroup(fn, [64, 1, 1])
    let entryName = if fn.exportName.len > 0: fn.exportName else: fn.name
    result = "@compute @workgroup_size(" & $wg[0] & ", " & $wg[1] & ", " &
      $wg[2] & ")\nfn " & entryName & "(" &
      emitEntryInputs(ctx.builtins, wgslEntryInputs) & ") {\n"
    return
  var params: seq[string]
  for p in fn.params:
    case p.typ.kind
    of ctPtr:
      # WGSL signatures take only this pointer form, the call inlining removes the rest
      params.add p.name & ": ptr<function, " & ctx.emitTypeName(p.typ.pElem) & ">"
    else:
      params.add p.name & ": " & ctx.emitTypeName(p.typ)
  result = "fn " & fn.name & "(" & params.join(", ") & ")"
  if fn.ret.kind != ctVoid:
    result &= " -> " & ctx.emitTypeName(fn.ret)

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope declaration text of one relocated storage local.
  ## No initializer survives relocation, the rewrite ran before printing.
  if s.space == caThreadgroup:
    ctx.storageBoolSpelling = true
    let ty = ctx.emitTypeName(s.declType)
    ctx.storageBoolSpelling = false
    return "var<workgroup> " & s.name & ": " & ty & ";"
  if s.init == nil:
    ctx.fail("const_mem local " & s.name & " without an initializer")
    return ""
  printDeclStmt("", "const " & s.name & ": " & ctx.emitTypeName(s.declType), ctx.printExpr(s.init))

proc emitBinding(ctx: var PrintCtx, p: CirParam, i: int): string =
  ## `@group(0) @binding(i)` declaration of one kernel parameter in host `store` packing order:
  ## - device buffers bind writable storage, `var<storage, read_write>`
  ## - scalars and value structs bind read-only storage, packed in the engine scratch buffer
  let attr = "@group(0) @binding(" & $i & ") "
  case p.typ.kind
  of ctAptr:
    result = attr & "var<storage, read_write> " & p.name & ": array<" &
      ctx.emitTypeName(p.typ.pElem) & ">;"
  of ctPtr:
    ctx.fail("kernel parameter " & p.name & " of pointer type has no WGSL binding")
    return ""
  of ctArray:
    ctx.fail("kernel parameter " & p.name & " of array type has no WGSL binding")
    return ""
  of ctStruct:
    for f in p.typ.fields:
      if f.typ.kind == ctVector:
        ctx.fail("kernel parameter " & p.name & " field " & f.name & " is a vector with no host-packed WGSL storage")
        return ""
      if f.typ.kind == ctArray and f.typ.aElem.kind == ctBool:
        ctx.fail("kernel parameter " & p.name & " field " & f.name & " is a bool array, only a scalar bool field maps to a NIM_BOOL word")
        return ""
    result = attr & "var<storage, read> " & p.name & ": " &
      ctx.emitTypeName(p.typ) & ";"
  of ctBool:
    result = attr & "var<storage, read> " & p.name & ": NIM_BOOL;"
  else:
    result = attr & "var<storage, read> " & p.name & ": " &
      ctx.emitTypeName(p.typ) & ";"

proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  ## WGSL declaration text of one struct member.
  ## A bool member stores as NIM_BOOL. A pointer-holding member
  ## returns "": the dissolution passes rewrote its uses.
  if f.typ.containsPtrDeep:
    return ""
  ctx.storageBoolSpelling = f.typ.kind == ctBool
  result = f.name & ": " & ctx.emitTypeName(f.typ)
  ctx.storageBoolSpelling = false

proc printParamBindings(ctx: var PrintCtx, kernels: seq[CirProc]): string =
  for i, p in kernels[0].params:
    result &= ctx.emitBinding(p, i) & "\n"
  if result.len > 0:
    result &= "\n"

proc printPrelude(ctx: var PrintCtx, m: CirModule): string =
  ## `enable f16` from the stored module feature scan `CirModule.features`,
  ## filled by the feature-scan pass, then the `NIM_BOOL` alias when interface
  ## storage or a declared struct field is bool.
  var usesNimBool = ctx.nimBoolNames.len > 0
  if not usesNimBool:
    for t in m.types:
      if t.kind == ctStruct:
        for f in t.fields:
          if f.typ.kind == ctBool:
            usesNimBool = true
  if m.features.fp16:
    result = declareExtensions(m.features, [(true, "enable f16;\n")])
  if usesNimBool:
    result &= "alias NIM_BOOL = u32;\n\n"

proc printWgslModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full WGSL module through the shared module skeleton, returning "" as the error when clean, else the first refusal.
  var ctx = initPrintCtx(wgslWalkPolicy, wgslLitTokens, wgslScalarNames,
    wgslVectorRule, wgslCastRules, wgslEmit, bkWebGPU)
  ctx.mathNames = wgslMathBuiltinNames
  let kernel = getEntryKernel(ctx, m)
  if ctx.failing():
    return ("", ctx.err)
  for p in kernel.params:
    if p.typ.kind == ctBool:
      ctx.nimBoolNames.incl p.name
  for s in m.globals:
    if s.kind == sVar and s.space == caThreadgroup and s.declType.kind == ctBool:
      ctx.nimBoolNames.incl s.name
  ctx.fnName = kernel.name
  printModule(ctx, m, @[kernel], wgslModuleShape)

proc codegen*(ctx: var GpuContext): string =
  ## `backends_api` entry point: prints the module in `ctx` as WGSL source.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printWgslModule(ctx.module))
