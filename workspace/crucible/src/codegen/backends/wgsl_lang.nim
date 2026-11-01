# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## WebGPU WGSL printer: a structured CIRv2 module rendered as WGSL source,
## table-driven over the closed set of type kinds.
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
##   a dissolved one included, an unused declaration is legal WGSL, a pointer-holding struct refuses
## - a loop prints `while (cond)` (WGSL's `while` takes a condition),
##   an unrolled loop notes the unroll directive has no WGSL form

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import ../passes/pass_helpers
import std/[macros, math, sets, strutils, tables]
import ../ir/ast_datatypes
import ./print_common
import ./print_compute
import ./print_shaders
import ./targets_rows
import ../passes/layout_std430

type
  PrintCtx = object
    fnName: string
    err: string
    ## First refusal of the walk, "" while clean.
    indent: int
    policy: KindRow
    ## Kind row of the target. The shared walk reads its policy fields.
    lit: LitRow
    ## Literal token forms of the target. The shared literal head reads them.
    row: ScalarRow
    ## Scalar row of the target. The narrow integer cast names its type from it.
    inKernel: bool
    ## False inside a device function, a device function reading a coordinate builtin refuses.
    paramNames: HashSet[string]
    ## Parameter names of the proc in flight, a pointer-typed parameter prints
    ## as-is (WGSL device functions take real pointer parameters).
    ## Only pointer locals refuse as values.
    builtins: OrderedSet[CoordBuiltinKind]
    ## Canonical coordinate builtin kinds the kernel body references, first-use order.
    layouts: Table[string, StructLayout]
    ## Stored struct layouts of the module, which the layout pass filled.
    nimBoolNames: HashSet[string]
    ## Names whose storage binds as `NIM_BOOL`: kernel bool parameters,
    ## bool workgroup variables.
    ## Reads coerce to `bool`, writes coerce to `NIM_BOOL`.
    storageBoolSpelling: bool
    ## When set, a bool leaf in `emitWgslTypeBare` becomes `NIM_BOOL`.
    ## Set only around interface-storage declarations.

const printerTarget = "wgsl"

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitWgslScalar(ctx: var PrintCtx, t: CirType): string =
  ## WGSL name of a non-array leaf type, "" when the caller composes the name.
  ## A struct name registers the struct for reference-driven emission.
  ##
  ## 64-bit leaves refuse: their device features exist but the vendored
  ## wgpu-native exposes no WGPUFeatureName to request them.
  case t.kind
  of ctBool:
    if ctx.storageBoolSpelling:
      result = "NIM_BOOL"
    else:
      result = scalarName(wgslScalarRow, t)
  of ctVoid:
    result = scalarName(wgslScalarRow, t)
  of ctInt, ctUint:
    result = scalarName(wgslScalarRow, t)
    if result.len == 0:
      if t.bits == 64:
        ctx.fail("i64/u64 need the shader-int64 device feature, the vendored wgpu-native cannot request it")
      else:
        ctx.fail($t.bits & "-bit integers have no WGSL type, only 32")
  of ctFloat:
    result = scalarName(wgslScalarRow, t)
    if result.len == 0:
      if t.bits == 64:
        ctx.fail("f64 needs the shader-f64 device feature, the vendored wgpu-native cannot request it")
      else:
        ctx.fail($t.bits & "-bit floats have no WGSL type")
  of ctVector:
    let elem = ctx.emitWgslScalar(t.vElem)
    result = vectorName(wgslVectorRule, wgslScalarRow, t)
    if result.len == 0 and elem.len > 0:
      ctx.fail("vector of " & elem & "x" & $t.vN & " has no WGSL type name, only 2..4-wide vectors have a name")
  of ctStruct:
    result = t.name
  of ctPtr, ctAptr, ctArray, ctProc, ctSpan:
    result = ""

proc emitWgslTypeBare(ctx: var PrintCtx, t: CirType): string =
  ## WGSL name of `t` with no identifier, in the form that casts and constructors use.
  case t.kind
  of ctArray:
    let elem = ctx.emitWgslTypeBare(t.aElem)
    if t.aN == 0:
      result = "array<" & elem & ">"
    else:
      result = "array<" & elem & ", " & $t.aN & ">"
  else:
    result = ctx.emitWgslScalar(t)

proc emitWgslPtrType(ctx: var PrintCtx, t: CirType): string =
  ## WGSL pointer type of a device-function `var T` parameter: `ptr<function, T>`,
  ## the only pointer form a signature takes. The call inlining removes every
  ## device function whose pointers cannot cross a WGSL function signature.
  let elem = ctx.emitWgslTypeBare(t.pElem)
  result = "ptr<function, " & elem & ">"

# ═══════════════════════════════════════════════════════════════════════
# Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

type WgslBuiltinParam = tuple[attribute, param, wgslType: string]
  ## Coordinate builtin kind -> WGSL `@builtin(...)` kernel parameter.
  ## - `attribute`: the WGSL builtin attribute.
  ## - `param`: the identifier the kernel body reads.
  ## - `wgslType`: the WGSL type (scalar for the flat thread index).

proc wgslBuiltinParam(kind: CoordBuiltinKind): WgslBuiltinParam =
  ## Returns the WGSL parameter of a canonical coordinate builtin kind, an empty tuple when none exists.
  ## `threads_per_threadgroup` is absent, WGSL exposes the threadgroup size
  ## only as the `@workgroup_size` entry-point attribute, a read refuses.
  case kind
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
  of cbkNone, cbkThreadsPerThreadgroup:
    ("", "", "")

proc wgslCoordComponent(kind: CoordBuiltinKind, field: string): string =
  case field
  of "x", "y", "z":
    let p = wgslBuiltinParam(kind)
    if p.attribute.len > 0:
      result = "i32(" & p.param & "." & field & ")"
  else:
    result = ""

proc wgslWholeVector(kind: CoordBuiltinKind): string =
  let p = wgslBuiltinParam(kind)
  result = "vec3<i32>(i32(" & p.param & ".x), i32(" & p.param & ".y), i32(" &
    p.param & ".z))"

const wgslEntryInputs*: EntryInputRow = block:
  ## WGSL entry-input text per canonical coordinate builtin kind: the `@builtin(...)` parameter the entry signature declares.
  ##
  ## - built from the `wgslBuiltinParam` table, one text per kind that has a parameter
  ## - "" for kinds without a WGSL builtin: `threads_per_threadgroup` reads refuse, WGSL has only the `@workgroup_size` attribute
  var row: EntryInputRow
  for kind in CoordBuiltinKind:
    let p = wgslBuiltinParam(kind)
    if p.attribute.len > 0:
      row[kind] = "@builtin(" & p.attribute & ") " & p.param & ": " & p.wgslType
  row

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

# WGSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
# `bitcast` from the unsigned integer of the same width. Bare literals are
# abstract and work in any typed context, u32 gains the `u` suffix, f16 the `h`.

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## One numeric conversion in the constructor-call form: WGSL has no bool
  ## constructor and no bool-to-numeric conversion.
  if e.castType.kind == ctBool:
    ctx.fail("conversion to bool, WGSL has no bool constructor")
    return ""
  let bt = getExprType(e.castOperand)
  if bt != nil and bt.kind == ctBool:
    ctx.fail("conversion from bool, WGSL has no bool-to-numeric conversion")
    return ""
  return ctx.emitWgslTypeBare(e.castType) & "(" & ctx.printExpr(e.castOperand) & ")"

proc printBuiltinCall(ctx: var PrintCtx, call: CirExpr, args: seq[string]): string =
  ## Returns the WGSL text of an expression-position builtin call,
  ## "" when `call.callee` is not a builtin.
  if call.callee.kind == eSym:
    # builtin vocabulary dispatches on the typed kinds, never on names,
    # `rsqrt` stays a name case, a math intrinsic outside the four categories
    case call.callee.reductionBuiltin
    of rbkReduceAdd:
      return "subgroupAdd(" & args.join(", ") & ")"
    of rbkReduceMax:
      return "subgroupMax(" & args.join(", ") & ")"
    of rbkNone: discard
    if call.callee.atomicsBuiltin == abkAtomicFetchAdd:
      # WGSL atomics cover i32/u32 on atomic<T> storage types only,
      # f32 atomics do not exist, and the atomic<T> retype of the binding
      # is a declaration-site change the printer does not make
      ctx.fail("atomicFetchAdd has no WGSL form, WGSL atomics cover i32/u32 pointees through atomic<T> storage types only")
      return ""
    if call.callee.synchroBuiltin != sbkNone:
      ctx.fail("barrier is a statement, a value-position barrier is outside the WGSL subset")
      return ""
  # a math intrinsic reroutes only when the callee IS the builtin
  # a user proc that shares the name stays an ordinary call
  if call.callee.kind == eSym and call.callee.builtin:
    case call.callee.name
    of "rsqrt":
      return "inverseSqrt(" & args.join(", ") & ")"
    else: discard
  if call.callee.kind == eSym and call.callee.builtin and
      wgslConvBuiltinNames.hasKey(call.callee.name):
    return wgslConvBuiltinNames.getOrDefault(call.callee.name) % args.join(", ")
  return ""
proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  ## Declaration text of one local, whose init the caller emits.
  ## A pointer-typed local refuses earlier: the WGSL lowering pass rewrites it.
  result = s.name & ": " & ctx.emitWgslTypeBare(s.declType)

# ═══════════════════════════════════════════════════════════════════════
# Statements: per-target policy arms, the shared dispatch lives in print_common
# ═══════════════════════════════════════════════════════════════════════

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## WGSL text of the kinds in `targetExprKinds`.
  case e.kind
  of eNil:
    ctx.fail("nil literal is outside the WGSL subset")
    return ""
  of eFloatLit:
    # a 64-bit non-finite float has no token in the lit row:
    # - `bitcast<f64>` requires the shader-f64 device feature the runtime cannot request
    # - every bitcast operand form fails, a u32-suffixed 64-bit literal is a parse error
    if e.fValue.classify in {fcInf, fcNegInf, fcNan} and e.fLitType.bits == 64:
      ctx.fail("a 64-bit non-finite float has no WGSL token, bitcast<f64> needs the unrequestable shader-f64 feature")
      return ""
    return emitLiteral(e, ctx.lit, ctx.row)
  of eSym:
    # coordinate builtins are entry-point parameters, names pass through,
    # pointer-value and whole-pointer-struct reads refuse in the lowering pass
    if e.typ != nil and e.typ.kind == ctBool and e.name in ctx.nimBoolNames:
      # storage is a u32 word, body reads expect bool
      return "bool(" & e.name & ")"
    if e.coordBuiltin != cbkNone:
      if not ctx.inKernel:
        ctx.fail("coordinate builtin " & e.name & " in a device function, WGSL builtins are entry-point parameters")
        return ""
      if e.coordBuiltin == cbkThreadsPerThreadgroup:
        ctx.fail("threads_per_threadgroup has no WGSL builtin, WGSL has only the @workgroup_size attribute")
        return ""
      ctx.builtins.incl(e.coordBuiltin)
      if e.coordBuiltin == cbkThreadIndexInThreadgroup:
        return "i32(" & wgslBuiltinParam(e.coordBuiltin).param & ")"
      return wgslWholeVector(e.coordBuiltin)
    return e.name
  of eAt, ePat:
    # a pointer local indexes through an explicit dereference
    let base = if e.kind == eAt: e.atBase else: e.patBase
    let idx = if e.kind == eAt: e.atIndex else: e.patIndex
    let idxTxt = ctx.printExpr(idx)
    let bt = getExprType(base)
    var baseTxt = ctx.printExpr(base)
    if base.kind in {eCast, eConv}:
      # a cast binds looser than the subscript: parenthesize the base
      baseTxt = "(" & baseTxt & ")"
    if bt != nil and bt.kind in {ctPtr, ctAptr} and
        not (bt.kind == ctAptr and bt.space == caDevice):
      baseTxt = "(*" & baseTxt & ")[" & idxTxt & "]"
    else:
      baseTxt = baseTxt & "[" & idxTxt & "]"
    if base.kind == eSym and base.name in ctx.nimBoolNames:
      # storage is a u32 word, element reads expect bool
      return "bool(" & baseTxt & ")"
    return baseTxt
  of eDot:
    # a coordinate component takes its canonical token, a vector component
    # takes its letter, names pass through
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      if not ctx.inKernel:
        ctx.fail("a device function reads " & e.dotBase.name & ", WGSL coordinate builtins are entry-point parameters")
        return ""
      let mapped = wgslCoordComponent(e.dotBase.coordBuiltin, e.fieldName)
      if mapped.len == 0:
        if e.dotBase.coordBuiltin == cbkThreadsPerThreadgroup:
          ctx.fail("threads_per_threadgroup reads refuse on WGSL, only the @workgroup_size attribute exists")
        else:
          ctx.fail("no coordinate component " & e.fieldName)
        return ""
      ctx.builtins.incl(e.dotBase.coordBuiltin)
      return mapped
    let bt = getExprType(e.dotBase)
    if bt != nil and bt.kind == ctVector:
      let comp = emitVectorComponent(e.fieldName)
      if comp.len == 0:
        ctx.fail("no vector component " & e.fieldName)
        return ""
      return ctx.printExpr(e.dotBase) & "." & comp
    if bt != nil and bt.kind == ctStruct:
      for f in bt.fields:
        if f.name == e.fieldName and f.typ.kind == ctBool:
          # field stores a NIM_BOOL word, body reads expect bool
          return "bool(" & ctx.printExpr(e.dotBase) & "." & e.fieldName & ")"
    return ctx.printExpr(e.dotBase) & "." & e.fieldName
  of eTernary:
    # Ternary via WGSL's select:
    # - `select(f, t, cond)` yields t when cond is true, f when cond is false
    # - both branches evaluate eagerly, a faulting read in either branch runs whatever the condition
    return "select(" & ctx.printExpr(e.elseE) & ", " & ctx.printExpr(e.thenE) &
      ", " & ctx.printExpr(e.cond) & ")"
  of eCast:
    return ctx.printCast(e, wgslCastRules, ctx.emitWgslTypeBare(e.castType))
  of eConv:
    return ctx.printConv(e)
  of eSizeof:
    return $typeByteSize(ctx.layouts, e.ofType, launchAbiBoolWidth)
  of eOconstr:
    # constructor-call form, a zero-field struct's padding member zero-initializes
    if e.kvFields.len == 0:
      return ctx.emitWgslTypeBare(e.ocType) & "(0u)"
    result = ctx.emitWgslTypeBare(e.ocType) & "("
    for i, kvv in e.kvFields:
      if i > 0:
        result.add ", "
      var fieldIsBool = false
      for f in e.ocType.fields:
        if f.name == kvv.name:
          fieldIsBool = f.typ.kind == ctBool
      if fieldIsBool:
        result.add "NIM_BOOL(" & ctx.printExpr(kvv.value) & ")"
      else:
        result.add ctx.printExpr(kvv.value)
    result.add ")"
  of eAconstr:
    # constructor-call form
    result = ctx.emitWgslTypeBare(e.acType) & "("
    for i, el in e.elems:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(el)
    result.add ")"
  of eCall:
    let args = printCallArgs(ctx, e)
    return printCallText(ctx, e, args, printBuiltinCall(ctx, e, args))
  else:
    # dispatcher sends only the kinds above, anything else is a routing bug
    ctx.fail("expression kind " & $e.kind & " has no WGSL emission")


proc printVar(ctx: var PrintCtx, s: CirStmt): string =
  ## One local declaration: storage-class refusals, then the declaration text.
  ## Pointer locals and pointer-holding struct locals never reach here: the WGSL lowering pass
  ## rewrites the shapes into pointees and leaf symbols.
  let pad = spaces(ctx.indent * 2)
  if s.space == caThreadgroup:
    # WGSL lowering pass hoists threadgroup locals to `CirModule.globals`,
    # a survivor here was never relocated
    ctx.fail("local " & s.name & " in threadgroup space, the WGSL lowering" &
      " pass hoists threadgroup locals to program scope")
    return ""
  elif s.space == caConstant:
    ctx.fail("local " & s.name & " in constant space, the WGSL lowering" &
      " pass hoists const_mem locals to program scope")
    return ""
  elif s.space == caDevice:
    ctx.fail("local " & s.name & " in device space, WGSL locals are" &
      " function or workgroup")
    return ""
  else:
    result = pad & "var " & ctx.emitLocalDecl(s)
    if s.init != nil:
      result &= " = " & ctx.printExpr(s.init)
    result &= ";"

proc printConst(ctx: var PrintCtx, s: CirStmt): string =
  ## One local constant declaration in the `let` form.
  let pad = spaces(ctx.indent * 2)
  result = pad & "let " & s.name & ": " &
    ctx.emitWgslTypeBare(s.declType) & " = " & ctx.printExpr(s.init) & ";"

proc printAsgn(ctx: var PrintCtx, s: CirStmt): string =
  ## One assignment: the NIM_BOOL storage mappings, then the assignment text.
  let pad = spaces(ctx.indent * 2)
  if s.lhs.kind == eSym and s.lhs.name in ctx.nimBoolNames:
    # storage is a u32 word, assigned values are bool
    return pad & s.lhs.name & " = NIM_BOOL(" & ctx.printExpr(s.rhs) & ");"
  if s.lhs.kind == eDot:
    let bt = getExprType(s.lhs.dotBase)
    if bt != nil and bt.kind == ctStruct:
      for f in bt.fields:
        if f.name == s.lhs.fieldName and f.typ.kind == ctBool:
          return pad & ctx.printExpr(s.lhs.dotBase) & "." & s.lhs.fieldName &
            " = NIM_BOOL(" & ctx.printExpr(s.rhs) & ");"
  var lhsBase = CirExpr(nil)
  if s.lhs.kind == eAt:
    lhsBase = s.lhs.atBase
  elif s.lhs.kind == ePat:
    lhsBase = s.lhs.patBase
  if lhsBase != nil and lhsBase.kind == eSym and
      lhsBase.name in ctx.nimBoolNames:
    let idx = if s.lhs.kind == eAt: s.lhs.atIndex else: s.lhs.patIndex
    return pad & lhsBase.name & "[" & ctx.printExpr(idx) &
      "] = NIM_BOOL(" & ctx.printExpr(s.rhs) & ");"
  result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"

proc printCall(ctx: var PrintCtx, s: CirStmt): string =
  ## One call statement: the barrier vocabulary, then the plain call form.
  let pad = spaces(ctx.indent * 2)
  var parts: seq[string]
  parts.add ctx.printExpr(s.callCallee)
  for a in s.callArgs:
    parts.add ctx.printExpr(a)
  if s.callCallee.kind == eSym and
      s.callCallee.synchroBuiltin == sbkThreadgroupBarrier:
    result = pad & "workgroupBarrier();"
  elif s.callCallee.kind == eSym and
      s.callCallee.synchroBuiltin == sbkThreadgroupBarrierDevice:
    # WGSL has no combined barrier: the workgroup fence, then the storage fence
    result = pad & "workgroupBarrier();\n" & pad & "storageBarrier();"
  else:
    result = pad & parts[0] & "(" & parts[1 ..^ 1].join(", ") & ");"

proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  ## One inline assembly statement: refused.
  ctx.fail("inline assembly has no WGSL form")
  result = ""

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules: shader-tier hooks over print_shaders
# ═══════════════════════════════════════════════════════════════════════

proc prepShaderProc(ctx: var PrintCtx, fn: CirProc) =
  ## Per-proc state of the WGSL walk:
  ##
  ## - the parameter names, whose pointer reads print as-is: WGSL device functions take real pointer parameters, only pointer locals refuse
  ## - the kernel flag, a coordinate-builtin read outside the entry refuses
  ctx.paramNames = HashSet[string]()
  for p in fn.params:
    ctx.paramNames.incl p.name
  ctx.inKernel = fn.isKernel

proc emitShaderSig(ctx: var PrintCtx, fn: CirProc): string =
  ## Head of one proc, signature and open brace:
  ##
  ## - a device function: the bare `fn name(params) -> ret` signature, a void return emits no arrow (WGSL has no `void` type name)
  ## - a pointer parameter takes the `ptr<function, T>` form
  ## - the entry kernel: `@compute @workgroup_size(...)` attribute lines, then `fn <entry>(<entry inputs>)` over the body-walk inputs
  ##   exportc renames the entry for the host runtime lookup.
  if fn.isKernel:
    let wg = (if fn.workgroup == [0, 0, 0]: [64, 1, 1] else: fn.workgroup)
    let entryName = if fn.exportName.len > 0: fn.exportName else: fn.name
    result = "@compute @workgroup_size(" & $wg[0] & ", " & $wg[1] & ", " &
      $wg[2] & ")\nfn " & entryName & "(" &
      emitEntryInputs(ctx.builtins, wgslEntryInputs) & ") {\n"
    return
  var params: seq[string]
  for p in fn.params:
    case p.typ.kind
    of ctPtr:
      params.add p.name & ": " & ctx.emitWgslPtrType(p.typ)
    else:
      params.add p.name & ": " & ctx.emitWgslTypeBare(p.typ)
  result = "fn " & fn.name & "(" & params.join(", ") & ")"
  if fn.ret.kind != ctVoid:
    result &= " -> " & ctx.emitWgslTypeBare(fn.ret)

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope declaration text of one relocated storage local:
  ##
  ## - a threadgroup local declares `var<workgroup>` storage, no initializer survives relocation (the rewrite ran before printing)
  ## - a bool workgroup variable stores as the NIM_BOOL alias of u32
  ## - a constant local declares `const` with its initializer
  if s.space == caThreadgroup:
    ctx.storageBoolSpelling = true
    let ty = ctx.emitWgslTypeBare(s.declType)
    ctx.storageBoolSpelling = false
    return "var<workgroup> " & s.name & ": " & ty & ";"
  if s.init == nil:
    ctx.fail("const_mem local " & s.name & " without an initializer")
    return ""
  return "const " & s.name & ": " & ctx.emitWgslTypeBare(s.declType) &
    " = " & ctx.printExpr(s.init) & ";"

proc emitBinding(ctx: var PrintCtx, p: CirParam, i: int): string =
  ## `@group(0) @binding(i)` declaration of one kernel parameter in host `store` packing order:
  ## - device buffers bind writable storage, `var<storage, read_write>`
  ## - scalars and value structs bind read-only storage, packed in the engine scratch buffer
  let attr = "@group(0) @binding(" & $i & ") "
  case p.typ.kind
  of ctAptr:
    result = attr & "var<storage, read_write> " & p.name & ": array<" &
      ctx.emitWgslTypeBare(p.typ.pElem) & ">;"
  of ctPtr:
    ctx.fail("kernel parameter " & p.name &
      " of pointer type, the engine binds device buffers and values")
    return ""
  of ctArray:
    ctx.fail("kernel parameter " & p.name &
      " of array type, the engine packs the value, the parameter is a struct or a scalar")
    return ""
  of ctStruct:
    for f in p.typ.fields:
      if f.typ.kind == ctVector:
        ctx.fail("kernel parameter " & p.name & " field " & f.name &
          " is a vector, the WGSL storage layout of vectors (16-byte alignment" &
          " for 3-component vectors) does not match the host packing")
        return ""
      if f.typ.kind == ctArray and f.typ.aElem.kind == ctBool:
        ctx.fail("kernel parameter " & p.name & " field " & f.name &
          " is a bool array, only a scalar bool field maps to a NIM_BOOL word")
        return ""
    result = attr & "var<storage, read> " & p.name & ": " &
      ctx.emitWgslTypeBare(p.typ) & ";"
  of ctBool:
    result = attr & "var<storage, read> " & p.name & ": NIM_BOOL;"
  else:
    result = attr & "var<storage, read> " & p.name & ": " &
      ctx.emitWgslTypeBare(p.typ) & ";"

proc emitWgslStruct(ctx: var PrintCtx, t: CirType): string =
  ## One struct declaration, WGSL requires at least one member.
  if t.containsPtrDeep:
    ctx.fail("struct " & t.name & " holds a pointer, WGSL has no pointer struct members")
    return ""
  result = "struct " & t.name & " {\n"
  if t.fields.len == 0:
    result &= "  _padding: u32,\n"
  else:
    for f in t.fields:
      ctx.storageBoolSpelling = f.typ.kind == ctBool
      let field = f.name & ": " & ctx.emitWgslTypeBare(f.typ)
      ctx.storageBoolSpelling = false
      result &= "  " & field & ",\n"
  result &= '}'

proc emitShaderStructs(ctx: var PrintCtx, m: CirModule): string =
  ## Struct declarations for every struct in `CirModule.types`, in declaration order.
  ## WGSL accepts an unused struct declaration, so a struct the lowering
  ## dissolved emits as dead text. A pointer-holding struct refuses in `emitWgslStruct`.
  for t in m.types:
    if t.kind == ctStruct:
      result &= ctx.emitWgslStruct(t) & "\n\n"

proc emitShaderBindings(ctx: var PrintCtx, kernel: CirProc): string =
  ## The per-parameter policy is `emitBinding`.
  for i, p in kernel.params:
    result &= ctx.emitBinding(p, i) & "\n"
  if result.len > 0:
    result &= "\n"

proc emitShaderFns(ctx: var PrintCtx, m: CirModule): string =
  for fn in moduleDeviceFns(m):
    result &= ctx.printShaderProc(fn) & "\n\n"

proc printWgslModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full WGSL module, one `@group(0) @binding(i)` per kernel parameter:
  ##
  ## - `enable f16` when any type in the module uses f16, and the `NIM_BOOL` alias when interface storage or a declared struct field is bool
  ## - struct declarations for the referenced value structs, then bindings, then hoisted module-scope declarations, then device functions
  ## - the `@compute` kernel last, its builtin parameters over the body-walk input set
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>", policy: wgslKindRow,
    lit: wgslLitRow, row: wgslScalarRow)
  ctx.layouts = m.layouts
  let kernel = selectShaderKernel(ctx, m)
  if ctx.failing():
    return ("", ctx.err)
  # names whose storage binds as NIM_BOOL: kernel bool parameters and bool workgroup variables
  # reads coerce to bool, writes coerce to NIM_BOOL (see the module header)
  for p in kernel.params:
    if p.typ.kind == ctBool:
      ctx.nimBoolNames.incl p.name
  for s in m.globals:
    if s.kind == sVar and s.space == caThreadgroup and s.declType.kind == ctBool:
      ctx.nimBoolNames.incl s.name

  # `enable f16` from the stored module feature scan `CirModule.features`, filled by the feature-scan pass
  # NIM_BOOL appears in interface storage names and in bool fields of declared structs
  var usesNimBool = ctx.nimBoolNames.len > 0
  for t in m.types:
    if t.kind == ctStruct:
      for f in t.fields:
        if f.typ.kind == ctBool:
          usesNimBool = true
  var prelude = ""
  if m.features.fp16:
    prelude = "enable f16;\n\n"
  if usesNimBool:
    prelude &= "alias NIM_BOOL = u32;\n\n"

  printShaderModule(ctx, m, kernel, prelude, {caThreadgroup, caConstant})


proc codegen*(ctx: var GpuContext): string =
  ## `backends_api` entry point: prints the module in `ctx` as WGSL source.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printWgslModule(ctx.module))
