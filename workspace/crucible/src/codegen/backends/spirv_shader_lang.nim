# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ═══════════════════════════════════════════════════════════════════════
#  SPIR-V emission for the Vulkan environment
# ═══════════════════════════════════════════════════════════════════════

import std/algorithm
import std/macros
import std/sets
import std/strutils
import std/tables
import ../ir/ast_datatypes
import ./spirv_encoder {.all.}

const spvScalarKinds = {gtVoid, gtBool, gtUint8, gtUint16, gtInt16, gtUint32,
                         gtInt32, gtUint64, gtInt64, gtFloat32, gtFloat64,
                         gtFloat16, gtSize_t}

func uint32Type(): GpuType =
  GpuType(kind: gtUint32)

func roundUp(x, step: int): int =
  ((x + step - 1) div step) * step

func getScalarByteSize(t: GpuType): int =
  case t.kind
  of gtVoid: 0
  of gtUint8: 1
  of gtUint16, gtInt16, gtFloat16: 2
  of gtBool, gtUint32, gtInt32, gtFloat32, gtSize_t: 4
  of gtUint64, gtInt64, gtFloat64: 8
  else: error("not a scalar GpuType: " & $t.kind)

template spvRefuse(what: string) =
  error("SPIR-V backend does not support: " & what)

# ═══════════════════════════════════════════════════════════════════════
#  Type descriptors
# ═══════════════════════════════════════════════════════════════════════

type
  SpvTyKind = enum
    stkScalar
    stkArray
    stkRuntimeArray
    stkStruct
    stkPointer

  SpvTy = ref object
    case kind*: SpvTyKind
    of stkScalar:
      scalar*: GpuType
    of stkArray:
      elem*: SpvTy
      len*: int
    of stkRuntimeArray:
      relem*: SpvTy
    of stkStruct:
      members*: seq[SpvTy]
    of stkPointer:
      storage*: uint32
      pointee*: SpvTy

func spvScalar(t: GpuType): SpvTy =
  doAssert t.kind in spvScalarKinds, "not a scalar GpuType: " & $t.kind
  SpvTy(kind: stkScalar, scalar: t)

func spvArray(elem: SpvTy, len: int): SpvTy =
  doAssert len >= 1, "length 0 means runtime-sized, use spvRuntimeArray"
  SpvTy(kind: stkArray, elem: elem, len: len)

func spvRuntimeArray(elem: SpvTy): SpvTy =
  SpvTy(kind: stkRuntimeArray, relem: elem)

func spvStruct(members: seq[SpvTy]): SpvTy =
  SpvTy(kind: stkStruct, members: members)

func spvPointer(storage: uint32, pointee: SpvTy): SpvTy =
  SpvTy(kind: stkPointer, storage: storage, pointee: pointee)

func spvTy(t: GpuType): SpvTy =
  case t.kind
  of spvScalarKinds: spvScalar(t)
  of gtArray:
    if t.aLen > 0:
      spvArray(spvTy(t.aTyp), t.aLen)
    else:
      spvRuntimeArray(spvTy(t.aTyp))
  of gtUncheckedArray: spvRuntimeArray(spvTy(t.uaTo))
  of gtObject:
    var ms: seq[SpvTy]
    for f in t.fields:
      ms.add spvTy(f.typ)
    spvStruct(ms)
  of gtPtr: spvPointer(scPhysicalStorageBuffer, spvTy(t.to))
  of gtBf16, gtF8E4M3, gtF8E5M2:
    error("bf16/f8 need OpTypeFloat FPEncoding operands and extension machinery")
  else:
    error("GpuType outside the SPIR-V subset: " & $t.kind)

func isStaticInt(t: GpuType): bool =
  t.kind == gtStatic or
      (t.kind == gtGenericInst and t.gName == "Int" and t.gArgs.len == 1 and
       t.gArgs[0].kind == gtStatic)

func staticIntValue(t: GpuType): int =
  if t.kind == gtStatic:
    t.sValue
  else:
    t.gArgs[0].sValue

func valueType(t: GpuType): GpuType =
  if isStaticInt(t):
    GpuType(kind: gtInt32)
  else:
    t

func valueTy(t: GpuType): SpvTy =
  case t.kind
  of spvScalarKinds: spvScalar(t)
  of gtStatic: spvScalar(GpuType(kind: gtInt32))
  of gtPtr, gtUncheckedArray: spvScalar(GpuType(kind: gtUint64))
  of gtArray:
    doAssert t.aTyp.kind in spvScalarKinds or isStaticInt(t.aTyp),
      "array elements outside the SPIR-V subset: " & $t.aTyp.kind
    spvArray(spvScalar(valueType(t.aTyp)), t.aLen)
  of gtObject, gtGenericInst:
    if isStaticInt(t):
      return spvScalar(GpuType(kind: gtInt32))
    var ms: seq[SpvTy]
    for f in t.fields:
      ms.add valueTy(f.typ)
    spvStruct(ms)
  else:
    error("GpuType outside the SPIR-V subset: " & $t.kind)

# ═══════════════════════════════════════════════════════════════════════
#  std430 layout
# ═══════════════════════════════════════════════════════════════════════

type SpvLayout = object
  size*: int
  align*: int

func std430Layout(t: SpvTy): SpvLayout =
  case t.kind
  of stkScalar:
    let s = getScalarByteSize(t.scalar)
    SpvLayout(size: s, align: s)
  of stkArray:
    let e = std430Layout(t.elem)
    let stride = roundUp(e.size, e.align)
    SpvLayout(size: stride * t.len, align: e.align)
  of stkRuntimeArray:
    let e = std430Layout(t.relem)
    SpvLayout(size: 0, align: roundUp(e.size, e.align))
  of stkStruct:
    var off = 0
    var align = 1
    for m in t.members:
      let ml = std430Layout(m)
      off = roundUp(off, ml.align) + ml.size
      align = max(align, ml.align)
    SpvLayout(size: roundUp(off, align), align: align)
  of stkPointer:
    SpvLayout(size: 8, align: 8)

func std430Offsets(members: seq[SpvTy]): seq[int] =
  var off = 0
  for m in members:
    let ml = std430Layout(m)
    off = roundUp(off, ml.align)
    result.add off
    off += ml.size

# ═══════════════════════════════════════════════════════════════════════
#  Emission context
# ═══════════════════════════════════════════════════════════════════════

type
  SpvLangCtx = object
    builder*: SpvBuilder
    caps*: HashSet[uint32]              ## pending capabilities, flushCapabilities emits them
    structMemo*: Table[seq[uint32], uint32]
    builtins*: Table[uint32, tuple[varId, vecTy: uint32]]
    interfaceVars*: seq[uint32]
    scalarBuiltins*: Table[uint32, uint32]
    fns*: Table[string, GpuAst]
    emittedFns*: Table[string, uint32]
    emitting*: HashSet[string]
    exts*: seq[string]

proc initSpvLangCtx(): SpvLangCtx =
  result.builder = initSpvBuilder()
  result.caps.incl capShader

proc scalarId(ctx: var SpvLangCtx, t: GpuType): uint32 =
  case t.kind
  of gtVoid: ctx.builder.tyVoid()
  of gtBool: ctx.builder.tyBool()
  of gtUint8:
    ctx.caps.incl capInt8
    ctx.builder.tyInt(8, false)
  of gtUint16, gtInt16:
    ctx.caps.incl capInt16
    ctx.builder.tyInt(16, t.kind == gtInt16)
  of gtUint32: ctx.builder.tyInt(32, false)
  of gtInt32: ctx.builder.tyInt(32, true)
  of gtUint64, gtInt64:
    ctx.caps.incl capInt64
    ctx.builder.tyInt(64, t.kind == gtInt64)
  of gtFloat16:
    ctx.caps.incl capFloat16
    ctx.builder.tyFloat(16)
  of gtFloat32: ctx.builder.tyFloat(32)
  of gtFloat64:
    ctx.caps.incl capFloat64
    ctx.builder.tyFloat(64)
  of gtSize_t: ctx.builder.tyInt(32, false)
  else: error("not a scalar GpuType: " & $t.kind)

proc tyId(ctx: var SpvLangCtx, t: SpvTy): uint32

proc arrayTy(ctx: var SpvLangCtx, elem: SpvTy, len: int): uint32 =
  doAssert len >= 1
  let elemId = ctx.tyId(elem)
  let lenId = ctx.builder.constUint(ctx.scalarId(uint32Type()), uint64(len))
  let key: SpvMemoKey = (OpTypeArray, @[elemId, lenId])
  if key in ctx.builder.memo:
    return ctx.builder.memo[key]
  result = ctx.builder.memoInst(OpTypeArray, @[elemId, lenId])
  let e = std430Layout(elem)
  ctx.builder.emitDecorate(result, decArrayStride, @[uint32(roundUp(e.size, e.align))])

proc runtimeArrayTy(ctx: var SpvLangCtx, elem: SpvTy): uint32 =
  let elemId = ctx.tyId(elem)
  let key: SpvMemoKey = (OpTypeRuntimeArray, @[elemId])
  if key in ctx.builder.memo:
    return ctx.builder.memo[key]
  result = ctx.builder.memoInst(OpTypeRuntimeArray, @[elemId])
  let e = std430Layout(elem)
  ctx.builder.emitDecorate(result, decArrayStride, @[uint32(roundUp(e.size, e.align))])

proc structTy(ctx: var SpvLangCtx, members: seq[SpvTy], isBlock = false): uint32 =
  let offsets = std430Offsets(members)
  var memberIds: seq[uint32]
  for m in members:
    memberIds.add ctx.tyId(m)
  var key = memberIds
  for off in offsets:
    key.add uint32(off)
  key.add uint32(isBlock)
  if key in ctx.structMemo:
    return ctx.structMemo[key]
  result = ctx.builder.module.allocId()
  # result-carrying instructions lead with the result id, emitInstSeq does not insert it
  var words = @[result]
  for mid in memberIds:
    words.add mid
  ctx.builder.module.emitInstSeq(ssTypesConstantsGlobals,
                                 OpTypeStruct, words)
  for i, off in offsets:
    ctx.builder.emitMemberDecorate(result, uint32(i), decOffset, @[uint32(off)])
  if isBlock:
    ctx.builder.emitDecorate(result, decBlock)
  ctx.structMemo[key] = result

proc tyId(ctx: var SpvLangCtx, t: SpvTy): uint32 =
  case t.kind
  of stkScalar: ctx.scalarId(t.scalar)
  of stkArray: ctx.arrayTy(t.elem, t.len)
  of stkRuntimeArray: ctx.runtimeArrayTy(t.relem)
  of stkStruct: ctx.structTy(t.members)
  of stkPointer:
    if t.storage == scPhysicalStorageBuffer:
      ctx.caps.incl capPhysicalStorageBufferAddresses
    ctx.builder.tyPointer(t.storage, ctx.tyId(t.pointee))

type
  SpvPushBlock = object
    structTy*: uint32     ## Block-decorated struct type id
    ptrTy*: uint32        ## pointer to the struct in the PushConstant class
    varId*: uint32        ## the block variable
    members*: seq[SpvTy]  ## lowered member types, one per kernel parameter
    offsets*: seq[int]    ## std430 offset per member
    byteSize*: int        ## packed block size, the runtime push-blob length

func pushMemberTy(p: GpuType): SpvTy =
  case p.kind
  of gtPtr, gtVoidPtr: spvScalar(GpuType(kind: gtUint64))
  else: spvTy(p)

proc buildPushBlockTys(ctx: var SpvLangCtx, members: seq[SpvTy]): SpvPushBlock =
  # bool members never reach here: the patchBoolToI32 pass retypes bool
  # kernel params to i32 before emission
  for m in members:
    if m.kind == stkScalar and getScalarByteSize(m.scalar) < 4:
      spvRefuse("push-constant member " & $m.scalar.kind &
        " narrower than 32 bits, 8/16-bit push members need SPIR-V 1.5 capabilities")
  result.members = members
  result.offsets = std430Offsets(result.members)
  result.byteSize = std430Layout(spvStruct(result.members)).size
  result.structTy = ctx.structTy(result.members, isBlock = true)
  result.ptrTy = ctx.builder.tyPointer(scPushConstant, result.structTy)
  result.varId = ctx.builder.opVariable(result.ptrTy, scPushConstant)

proc builtinInput(ctx: var SpvLangCtx, builtin: uint32): uint32 =
  if builtin in ctx.builtins:
    return ctx.builtins[builtin].varId
  let vecTy = ctx.builder.tyVector(ctx.scalarId(uint32Type()), 3)
  let ptrTy = ctx.builder.tyPointer(scInput, vecTy)
  let varId = ctx.builder.opVariable(ptrTy, scInput)
  ctx.builder.emitDecorate(varId, decBuiltIn, @[builtin])
  ctx.builtins[builtin] = (varId, vecTy)
  ctx.interfaceVars.add varId
  result = varId

proc loadBuiltinComponent(ctx: var SpvLangCtx, builtin: uint32, component: int): uint32 =
  let varId = ctx.builtinInput(builtin)
  let v = ctx.builder.opLoad(ctx.builtins[builtin].vecTy, varId)
  ctx.builder.opCompositeExtract(ctx.scalarId(uint32Type()), v, @[component])

func coordBuiltinInput(k: GpuCoordBuiltinKind): uint32 =
  case k
  of gbkThreadPositionInGrid: biGlobalInvocationId
  of gbkThreadgroupPositionInGrid: biWorkgroupId
  of gbkThreadPositionInThreadgroup: biLocalInvocationId
  of gbkThreadsPerThreadgroup: biWorkgroupSize
  of gbkThreadgroupsPerGrid: biNumWorkgroups
  of gbkThreadIndexInThreadgroup: biLocalInvocationIndex
  of gbkNone:
    error("not a coordinate builtin")

proc scalarBuiltinInput(ctx: var SpvLangCtx, builtin: uint32): uint32 =
  if builtin in ctx.scalarBuiltins:
    return ctx.scalarBuiltins[builtin]
  let u32Ty = ctx.scalarId(uint32Type())
  let ptrTy = ctx.builder.tyPointer(scInput, u32Ty)
  let varId = ctx.builder.opVariable(ptrTy, scInput)
  ctx.builder.emitDecorate(varId, decBuiltIn, @[builtin])
  ctx.scalarBuiltins[builtin] = varId
  ctx.interfaceVars.add varId
  result = varId

proc loadScalarBuiltin(ctx: var SpvLangCtx, builtin: uint32): uint32 =
  let varId = ctx.scalarBuiltinInput(builtin)
  ctx.builder.opLoad(ctx.scalarId(uint32Type()), varId)

proc flushCapabilities(ctx: var SpvLangCtx) =
  var caps: seq[uint32]
  for c in ctx.caps:
    caps.add c
  caps.sort()
  for c in caps:
    ctx.builder.emitCapability(c)

# ═══════════════════════════════════════════════════════════════════════
#  Kernel body printer
# ═══════════════════════════════════════════════════════════════════════

type
  SpvFnCtx = object
    locals*: Table[string, uint32]     ## symbol iSym to Function-storage variable id
    paramVals*: Table[string, uint32]  ## scalar param iSym to hoisted value id
    paramPtrs*: Table[string, uint32]  ## pointer param iSym to hoisted BDA pointer id
    addrs*: Table[string, uint32]
    addrNames*: Table[string, uint32]
    addrNamePointees*: Table[string, GpuType]
    localTys*: Table[string, GpuType]
    localsBySym*: Table[string, uint32]
    localTysBySym*: Table[string, GpuType]
    addrPointee*: Table[string, GpuType]
    terminated*: bool                   ## a terminator already ended this fn
    fns*: Table[string, GpuAst]         ## device fn bodies keyed by callee iSym

const spvSignedInts = {gtInt16, gtInt32, gtInt64}

proc emitStmt(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst)

func pointeeElem(t: GpuType): GpuType =
  case t.kind
  of gtPtr:
    if t.to.kind == gtUncheckedArray:
      t.to.uaTo
    else:
      t.to
  else:
    spvRefuse("element type of non-pointer type " & $t.kind)

func indexBaseSym(n: GpuAst): Symbol =
  let base = if n.iArr.kind == gpuDeref: n.iArr.dOf else: n.iArr
  doAssert base.kind == gpuIdent, "index base is not a symbol"
  base.symbol

proc fieldEntry(t: GpuType, name: string): tuple[idx: int, typ: GpuType] =
  doAssert t.kind in {gtObject, gtGenericInst},
    "field access on non-object type " & $t.kind
  for i, f in t.fields:
    if f.name == name:
      return (i, f.typ)
  error("no field " & name & " on type " & t.pretty())

func derefTy(t: GpuType): GpuType =
  case t.kind
  of gtPtr: t.to
  of gtUncheckedArray: t.uaTo
  else:
    error("deref of non-pointer type " & $t.kind)

proc valueTypeOfField(t: GpuType, name: string): GpuType =
  valueType(fieldEntry(t, name).typ)

type DotChain = object
  root*: GpuAst
  fields*: seq[GpuAst]

func dotChain(n: GpuAst): DotChain =
  doAssert n.kind == gpuDot, "not a member access"
  var cur = n
  while cur.kind == gpuDot:
    doAssert cur.dField.kind == gpuIdent, "dot field is not an identifier"
    result.fields.insert(cur.dField)
    cur = cur.dParent
  result.root = cur

func chainRootType(root: GpuAst): GpuType =
  case root.kind
  of gpuIdent:
    let t = root.symbol.typ
    if t.kind in {gtPtr, gtUncheckedArray}:
      t.to
    else:
      t
  of gpuDeref: chainRootType(root.dOf)
  else:
    spvRefuse("member chain root node " & $root.kind)

proc exprGpuType(n: GpuAst): GpuType =
  case n.kind
  of gpuLit: n.lType
  of gpuIdent:
    if n.symbol.coordBuiltin == gbkThreadIndexInThreadgroup:
      # the scalar builtin reads as u32, conversions handle the cast
      GpuType(kind: gtUint32)
    else:
      n.symbol.typ
  of gpuBinOp: n.bType
  of gpuVar: n.vType
  of gpuConv: n.convTo
  of gpuCast: n.cTo
  of gpuObjConstr: valueType(n.ocType)
  of gpuDot:
    let ch = dotChain(n)
    if ch.root.kind == gpuIdent and ch.root.symbol.coordBuiltin != gbkNone:
      GpuType(kind: gtUint32)   ## coordinate builtin components are u32
    else:
      var t = chainRootType(ch.root)
      for f in ch.fields:
        t = valueTypeOfField(t, f.symbol.name)
      t
  of gpuIndex:
    let inner = if n.iArr.kind == gpuDeref: n.iArr.dOf else: n.iArr
    if inner.kind == gpuIdent:
      let st = inner.symbol.typ
      if st.kind == gtArray:
        valueType(st.aTyp)
      else:
        pointeeElem(st)
    else:
      let ch = dotChain(inner)
      var t = chainRootType(ch.root)
      for f in ch.fields:
        t = valueTypeOfField(t, f.symbol.name)
      case t.kind
      of gtArray: valueType(t.aTyp)
      of gtUncheckedArray: valueType(t.uaTo)
      of gtPtr: valueType(pointeeElem(t))
      else:
        spvRefuse("index of non-array member type " & $t.kind)
  of gpuDeref: valueType(derefTy(exprGpuType(n.dOf)))
  of gpuPrefix: exprGpuType(n.pVal)
  of gpuArrayLit:
    GpuType(kind: gtArray, aTyp: n.aLitType, aLen: n.aValues.len)
  else:
    spvRefuse("result type of expression node " & $n.kind)

proc spvConstLit(ctx: var SpvLangCtx, t: GpuType, lValue: string): uint32 =
  let tyId = ctx.tyId(spvScalar(t))
  case t.kind
  of gtUint32, gtSize_t: ctx.builder.constUint(tyId, uint64(uint32(parseUInt(lValue))))
  of gtInt32: ctx.builder.constUint(tyId, uint64(uint32(parseBiggestInt(lValue))))
  of gtUint64: ctx.builder.constUint(tyId, parseUInt(lValue))
  of gtInt64: ctx.builder.constUint(tyId, uint64(parseBiggestInt(lValue)))
  of gtFloat32: ctx.builder.constUint(tyId, cast[uint32](parseFloat(lValue).float32))
  of gtFloat64: ctx.builder.constUint(tyId, cast[uint64](parseFloat(lValue)))
  else:
    spvRefuse("literal of type " & $t.kind)

proc convertValue(ctx: var SpvLangCtx, v: uint32, source, to: GpuType): uint32 =
  if source.kind == to.kind:
    return v
  let fFloatTy = source.kind in {gtFloat16, gtFloat32, gtFloat64}
  if to.kind == gtBool and not fFloatTy:
    # Bool reads as truthiness: any nonzero integer means true.
    # This is the patchBoolToI32 pass rule, the WGSL bool(i32) form.
    let zero = ctx.builder.constUint(ctx.scalarId(source), 0)
    return ctx.builder.freshInst(OpINotEqual, @[ctx.tyId(spvScalar(to)), v, zero])
  let tFloatTy = to.kind in {gtFloat16, gtFloat32, gtFloat64}
  let fInt = source.kind in spvScalarKinds and source.kind != gtBool and not fFloatTy
  let tInt = to.kind in spvScalarKinds and to.kind != gtBool and not tFloatTy
  let fFloat = source.kind in {gtFloat32, gtFloat64}
  let tFloat = to.kind in {gtFloat32, gtFloat64}
  let inst = if fInt and tInt:
      if getScalarByteSize(source) == getScalarByteSize(to):
        OpBitcast
      elif source.kind in spvSignedInts:
        OpSConvert
      else:
        OpUConvert
    elif fInt and tFloat:
      if source.kind in spvSignedInts:
        OpConvertSToF
      else:
        OpConvertUToF
    elif fFloat and tInt:
      if to.kind in spvSignedInts:
        OpConvertFToS
      else:
        OpConvertFToU
    elif fFloat and tFloat:
      OpFConvert
    else:
      error("no SPIR-V scalar conversion " & $source.kind & " -> " & $to.kind)
  ctx.builder.freshInst(inst, @[ctx.tyId(spvScalar(to)), v])

proc bdaPointerTy(ctx: var SpvLangCtx, elem: GpuType): uint32 =
  doAssert elem.kind in spvScalarKinds,
    "SPIR-V buffer references address scalar elements only: " & $elem.kind
  ctx.caps.incl capPhysicalStorageBufferAddresses
  let refStruct = ctx.structTy(@[spvRuntimeArray(spvScalar(elem))], isBlock = true)
  ctx.builder.tyPointer(scPhysicalStorageBuffer, refStruct)

proc getLocalVarId(fn: SpvFnCtx, sym: Symbol): uint32 =
  if sym.iSym in fn.localsBySym:
    return fn.localsBySym[sym.iSym]
  if sym.name in fn.locals:
    return fn.locals[sym.name]
  spvRefuse("unknown local symbol " & sym.name)

proc getLocalType(fn: SpvFnCtx, sym: Symbol): GpuType =
  if sym.iSym in fn.localTysBySym:
    return fn.localTysBySym[sym.iSym]
  if sym.name in fn.localTys:
    return fn.localTys[sym.name]
  spvRefuse("unknown local symbol " & sym.name)

proc emitIdentRef(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): uint32 =
  let sym = n.symbol
  case sym.coordBuiltin
  of gbkThreadIndexInThreadgroup:
    return ctx.loadScalarBuiltin(biLocalInvocationIndex)
  of gbkNone: discard
  else:
    spvRefuse("whole-vector use of coordinate builtin " & sym.name &
              ", index a component instead")
  if sym.name == "true" or sym.name == "false":
    let boolTy = ctx.tyId(spvScalar(GpuType(kind: gtBool)))
    return ctx.builder.constBool(boolTy, sym.name == "true")
  if sym.iSym in fn.localsBySym or sym.name in fn.locals:
    let t = if sym.typ.kind == gtVoid: getLocalType(fn, sym) else: sym.typ
    return ctx.builder.opLoad(ctx.tyId(valueTy(t)), getLocalVarId(fn, sym))
  if sym.iSym in fn.paramVals:
    return fn.paramVals[sym.iSym]
  if sym.iSym in fn.addrs:
    return ctx.builder.opLoad(ctx.tyId(valueTy(sym.typ)), fn.addrs[sym.iSym])
  if sym.iSym in fn.paramPtrs:
    spvRefuse("bare pointer value use of parameter " & sym.name)
  spvRefuse("unknown symbol " & sym.name)

func coordComponent(field: GpuAst): int =
  doAssert field.kind == gpuIdent, "coordinate component is not an identifier"
  case field.symbol.name
  of "x": 0
  of "y": 1
  of "z": 2
  else:
    spvRefuse("coordinate component " & field.symbol.name)

func binopInst(op: GpuAst, resultTy, operandTy: GpuType): SpvOp =
  doAssert op.kind == gpuIdent, "binary operator is not an identifier"
  let name = op.symbol.name
  let isFloat = operandTy.kind in {gtFloat32, gtFloat64}
  let signed = operandTy.kind in spvSignedInts
  if isFloat and name in ["==", "!=", "<", "<=", ">", ">="]:
    spvRefuse("float comparison " & name)
  case name
  of "+": (if isFloat: OpFAdd else: OpIAdd)
  of "-": (if isFloat: OpFSub else: OpISub)
  of "*": (if isFloat: OpFMul else: OpIMul)
  of "/":
    if isFloat:
      OpFDiv
    elif signed:
      OpSDiv
    else:
      OpUDiv
  of "==":
    if operandTy.kind == gtBool:
      OpLogicalEqual
    else:
      OpIEqual
  of "!=":
    if operandTy.kind == gtBool:
      OpLogicalNotEqual
    else:
      OpINotEqual
  of "<": (if signed: OpSLessThan else: OpULessThan)
  of "<=": (if signed: OpSLessThanEqual else: OpULessThanEqual)
  of ">": (if signed: OpSGreaterThan else: OpUGreaterThan)
  of ">=": (if signed: OpSGreaterThanEqual else: OpUGreaterThanEqual)
  of "%":
    # a float `%` refuses: no float-mod opcode exists in the table
    # an OpUMod over float operands is invalid
    # the frontend refuses the same construct as an undeclared identifier
    if isFloat:
      spvRefuse("float modulo %")
    elif signed:
      OpSMod
    else:
      OpUMod
  of "&&": OpLogicalAnd
  of "||": OpLogicalOr
  else:
    spvRefuse("binary operator " & name)

proc emitExpr(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): uint32

proc exprType(fn: SpvFnCtx, n: GpuAst): GpuType

proc resolvePtrRoot(ctx: var SpvLangCtx, fn: var SpvFnCtx, root: GpuAst): tuple[ptrId: uint32, pointee: GpuType, isBda: bool] =
  case root.kind
  of gpuIdent:
    let sym = root.symbol
    if sym.iSym in fn.localsBySym or sym.name in fn.locals:
      let t = if sym.typ.kind == gtVoid: getLocalType(fn, sym) else: sym.typ
      return (getLocalVarId(fn, sym), t, false)
    if sym.iSym in fn.addrs:
      # pointer params address their pointee, a chain root through them
      # starts at the pointee type, an implicit-ptr parameter's declared type
      # names only the element
      let t = fn.addrPointee.getOrDefault(sym.iSym, sym.typ)
      let pt = if t.kind in {gtPtr, gtUncheckedArray}: derefTy(t) else: t
      return (fn.addrs[sym.iSym], pt, false)
    if sym.iSym in fn.paramPtrs:
      return (fn.paramPtrs[sym.iSym], pointeeElem(sym.typ), true)
    if sym.name in fn.addrNames:
      # re-symbol'd assign target over a parameter: the name survives,
      # the type comes from the registration site
      let t = fn.addrNamePointees.getOrDefault(sym.name, sym.typ)
      let pt = if t.kind in {gtPtr, gtUncheckedArray}: derefTy(t) else: t
      return (fn.addrNames[sym.name], pt, false)
    spvRefuse("member chain root is not a local or parameter: " & sym.name)
  of gpuAddr: result = resolvePtrRoot(ctx, fn, root.aOf)
  of gpuDeref: result = resolvePtrRoot(ctx, fn, root.dOf)
  else:
    spvRefuse("member chain root node " & $root.kind)

proc emitChainPtr(ctx: var SpvLangCtx, fn: var SpvFnCtx, root: GpuAst, fields: seq[GpuAst]): tuple[ptrId: uint32, pointee: GpuType] =
  let (base, baseTy, isBda) = resolvePtrRoot(ctx, fn, root)
  if isBda:
    spvRefuse("member access through a buffer device address parameter")
  var t = baseTy
  var idxs: seq[uint32]
  for f in fields:
    let (i, fty) = fieldEntry(t, f.symbol.name)
    idxs.add ctx.builder.constUint(ctx.scalarId(uint32Type()), uint64(i))
    t = fty
  let ptrTy = ctx.builder.tyPointer(scFunction, ctx.tyId(valueTy(t)))
  result.ptrId = ctx.builder.opAccessChain(ptrTy, base, idxs)
  result.pointee = t

proc emitDotValue(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): uint32 =
  let ch = dotChain(n)
  if ch.root.kind == gpuIdent and ch.root.symbol.coordBuiltin != gbkNone:
    doAssert ch.fields.len == 1,
      "coordinate builtin chains are single component accesses"
    return ctx.loadBuiltinComponent(
      coordBuiltinInput(ch.root.symbol.coordBuiltin),
      coordComponent(ch.fields[0]))
  if ch.root.kind in {gpuIdent, gpuAddr, gpuDeref}:
    let (ptrId, pointee) = emitChainPtr(ctx, fn, ch.root, ch.fields)
    return ctx.builder.opLoad(ctx.tyId(valueTy(pointee)), ptrId)
  var v = emitExpr(ctx, fn, ch.root)
  var idxs: seq[int]
  var t = exprType(fn, ch.root)
  for f in ch.fields:
    let (i, fty) = fieldEntry(t, f.symbol.name)
    doAssert fty.kind notin {gtPtr, gtUncheckedArray},
      "device address member on a value-rooted chain"
    idxs.add i
    t = valueType(fty)
  return ctx.builder.opCompositeExtract(ctx.tyId(valueTy(t)), v, idxs)

proc emitAccessPtr(ctx: var SpvLangCtx, fn: var SpvFnCtx, base: GpuAst, index: GpuAst): tuple[ptrId: uint32, elem: GpuType, isBda: bool] =
  let inner = if base.kind == gpuDeref: base.dOf else: base
  let idxVal = emitExpr(ctx, fn, index)
  if inner.kind == gpuIdent:
    let sym = inner.symbol
    if sym.iSym in fn.paramPtrs:
      result.elem = pointeeElem(sym.typ)
      doAssert result.elem.kind in spvScalarKinds,
        "SPIR-V buffer references address scalar elements only: " & $result.elem.kind
      ctx.caps.incl capPhysicalStorageBufferAddresses
      let elemPtrTy = ctx.builder.tyPointer(scPhysicalStorageBuffer,
                                            ctx.tyId(spvScalar(result.elem)))
      let member0 = ctx.builder.constUint(ctx.scalarId(uint32Type()), 0)
      result.ptrId = ctx.builder.opAccessChain(elemPtrTy, fn.paramPtrs[sym.iSym],
                                               @[member0, idxVal])
      result.isBda = true
      return
    if sym.iSym in fn.addrs:
      # Function-storage pointer parameter (`var T`): the recovered pointee
      # is the array the call sites pass, one access chain indexes it
      let pointee = fn.addrPointee.getOrDefault(sym.iSym, derefTy(sym.typ))
      doAssert pointee.kind == gtArray,
        "indexing a Function-storage pointer to a non-array: " & sym.name
      result.elem = valueType(pointee.aTyp)
      doAssert result.elem.kind in spvScalarKinds,
        "composite element stores address scalar elements only"
      let elemPtrTy = ctx.builder.tyPointer(scFunction,
                                            ctx.tyId(spvScalar(result.elem)))
      result.ptrId = ctx.builder.opAccessChain(elemPtrTy, fn.addrs[sym.iSym],
                                               @[idxVal])
      result.isBda = false
      return
    # local array: the Function-storage variable already is a pointer to it
    let arrTy = getLocalType(fn, sym)
    doAssert arrTy.kind == gtArray, "indexing a non-array local: " & sym.name
    result.elem = valueType(arrTy.aTyp)
    doAssert result.elem.kind in spvScalarKinds,
      "composite element stores address scalar elements only"
    let elemPtrTy = ctx.builder.tyPointer(scFunction,
                                          ctx.tyId(spvScalar(result.elem)))
    result.ptrId = ctx.builder.opAccessChain(elemPtrTy, getLocalVarId(fn, sym),
                                             @[idxVal])
    result.isBda = false
    return
  doAssert inner.kind == gpuDot, "index base outside the SPIR-V subset: " & $inner.kind
  let ch = dotChain(inner)
  var t = chainRootType(ch.root)
  var memberIdxs: seq[uint32]
  for f in ch.fields:
    let (i, fty) = fieldEntry(t, f.symbol.name)
    memberIdxs.add ctx.builder.constUint(ctx.scalarId(uint32Type()), uint64(i))
    t = fty
  if t.kind in {gtPtr, gtUncheckedArray}:
    # device memory through the u64 address member
    result.elem = if t.kind == gtUncheckedArray: t.uaTo else: pointeeElem(t)
    doAssert result.elem.kind in spvScalarKinds,
      "SPIR-V buffer references address scalar elements only: " & $result.elem.kind
    let (addrPtr, _) = emitChainPtr(ctx, fn, ch.root, ch.fields)
    let u64Ty = ctx.tyId(spvScalar(GpuType(kind: gtUint64)))
    let addrVal = ctx.builder.opLoad(u64Ty, addrPtr)
    ctx.caps.incl capPhysicalStorageBufferAddresses
    let bdaPtr = ctx.builder.opConvertUToPtr(ctx.bdaPointerTy(result.elem), addrVal)
    let elemPtrTy = ctx.builder.tyPointer(scPhysicalStorageBuffer,
                                          ctx.tyId(spvScalar(result.elem)))
    let member0 = ctx.builder.constUint(ctx.scalarId(uint32Type()), 0)
    result.ptrId = ctx.builder.opAccessChain(elemPtrTy, bdaPtr,
                                             @[member0, idxVal])
    result.isBda = true
  else:
    # Function-storage composite member: the member chain already exists
    # (emitChainPtr), the element chain adds the array index alone
    doAssert t.kind == gtArray, "indexed member is not an array: " & $t.kind
    result.elem = valueType(t.aTyp)
    doAssert result.elem.kind in spvScalarKinds,
      "composite element stores address scalar elements only"
    let (basePtr, _) = emitChainPtr(ctx, fn, ch.root, ch.fields)
    let elemPtrTy = ctx.builder.tyPointer(scFunction,
                                          ctx.tyId(spvScalar(result.elem)))
    result.ptrId = ctx.builder.opAccessChain(elemPtrTy, basePtr, @[idxVal])
    result.isBda = false

proc emitAddr(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): tuple[ptrId: uint32, pointee: GpuType, isBda: bool] =
  case n.kind
  of gpuIdent, gpuAddr, gpuDeref:
    result = resolvePtrRoot(ctx, fn, n)
  of gpuDot:
    let ch = dotChain(n)
    let (ptrId, pointee) = emitChainPtr(ctx, fn, ch.root, ch.fields)
    result = (ptrId, pointee, false)
  else:
    spvRefuse("address of node " & $n.kind)

proc exprType(fn: SpvFnCtx, n: GpuAst): GpuType =
  if n.kind == gpuIdent and (n.symbol.name == "true" or n.symbol.name == "false"):
    # the passes type bool literals as Void idents, emitExpr constBools them
    return GpuType(kind: gtBool)
  if n.kind == gpuCall:
    let procNode = fn.fns.getOrDefault(n.cName.symbol.iSym)
    if not procNode.isNil:
      doAssert not procNode.pRetType.isNil,
        "fn-table entry with nil return type: " & n.cName.symbol.name
      return valueType(procNode.pRetType)
    let argTy = exprGpuType(n.cArgs[0])
    if argTy.isNil:
      spvRefuse("result type of a call to " & n.cName.symbol.name &
        " whose first argument carries no static type")
    return valueType(argTy)
  return exprGpuType(n)

proc fnRetTyId(ctx: var SpvLangCtx, retTy: GpuType): uint32 =
  if retTy.kind == gtVoid:
    ctx.builder.tyVoid()
  elif retTy.kind in spvScalarKinds:
    ctx.tyId(spvScalar(retTy))
  else:
    ctx.tyId(valueTy(valueType(retTy)))

proc emitObjConstr(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): uint32 =
  if isStaticInt(n.ocType):
    return ctx.builder.constUint(ctx.tyId(spvScalar(GpuType(kind: gtInt32))),
                                 uint64(uint32(staticIntValue(n.ocType))))
  let structId = ctx.tyId(valueTy(n.ocType))
  var members: seq[uint32]
  for f in n.ocFields:
    let v = emitExpr(ctx, fn, f.value)
    if f.typ.kind in spvScalarKinds or isStaticInt(f.typ):
      members.add ctx.convertValue(v, exprType(fn, f.value), valueType(f.typ))
    else:
      members.add v
  ctx.builder.freshInst(OpCompositeConstruct, @[structId] & members)

proc emitScalarBuiltinCall(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): uint32 =
  let name = n.cName.symbol.name
  if name notin ["max", "min"]:
    spvRefuse("call to unknown function " & name)
  doAssert n.cArgs.len == 2, "max/min take two arguments"
  let opTy = valueType(exprType(fn, n.cArgs[0]))
  let a = ctx.convertValue(emitExpr(ctx, fn, n.cArgs[0]),
                           exprType(fn, n.cArgs[0]), opTy)
  let b = ctx.convertValue(emitExpr(ctx, fn, n.cArgs[1]),
                           exprType(fn, n.cArgs[1]), opTy)
  let signed = opTy.kind in spvSignedInts
  let isFloat = opTy.kind in {gtFloat32, gtFloat64}
  let cmpOp = if isFloat:
                # ordered float comparison, the counterpart of the integer compare opcodes
                (if name == "max": OpFOrdGreaterThan else: OpFOrdLessThan)
              elif name == "max":
                (if signed: OpSGreaterThan else: OpUGreaterThan)
              else:
                (if signed: OpSLessThan else: OpULessThan)
  let boolTy = ctx.tyId(spvScalar(GpuType(kind: gtBool)))
  let cond = ctx.builder.freshInst(cmpOp, @[boolTy, a, b])
  ctx.builder.opSelect(ctx.tyId(spvScalar(opTy)), cond, a, b)

proc emitReduceAdd(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): uint32 =
  doAssert n.cArgs.len == 1, "reduceAdd takes one argument"
  let v = emitExpr(ctx, fn, n.cArgs[0])
  let resTy = valueType(exprType(fn, n.cArgs[0]))
  doAssert resTy.kind == gtFloat32, "reduceAdd lowers f32 only: " & $resTy.kind
  ctx.caps.incl capGroupNonUniformArithmetic
  let scopeSub = ctx.builder.constUint(ctx.scalarId(uint32Type()), uint64(scopeSubgroup))
  ctx.builder.freshInst(OpGroupNonUniformFAdd, @[ctx.tyId(spvScalar(resTy)), scopeSub, groupOpReduce, v])

proc emitAtomicFetchAdd(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): uint32 =
  doAssert n.cArgs.len == 2, "atomicFetchAdd takes a pointer and a value"
  let addrNode = n.cArgs[0]
  doAssert addrNode.kind == gpuAddr and addrNode.aOf.kind == gpuIndex,
    "atomicFetchAdd address is not addr(index(...))"
  let (ptrId, elem, isBda) = emitAccessPtr(ctx, fn, addrNode.aOf.iArr,
                                           addrNode.aOf.iIndex)
  doAssert isBda, "atomicFetchAdd lowers through a buffer device address only"
  doAssert elem.kind == gtFloat32, "atomicFetchAdd lowers f32 only: " & $elem.kind
  let v = emitExpr(ctx, fn, n.cArgs[1])
  let value = ctx.convertValue(v, exprType(fn, n.cArgs[1]), valueType(elem))
  ctx.caps.incl capAtomicFloat32AddEXT
  if "SPV_EXT_shader_atomic_float_add" notin ctx.exts:
    ctx.exts.add "SPV_EXT_shader_atomic_float_add"
  let scopeDev = ctx.builder.constUint(ctx.scalarId(uint32Type()), uint64(scopeDevice))
  let sem = ctx.builder.constUint(ctx.scalarId(uint32Type()), uint64(semUniformMemory or semAcquireRelease))
  ctx.builder.freshInst(OpAtomicFAddEXT, @[ctx.tyId(spvScalar(elem)), ptrId, scopeDev, sem, value])

proc emitCall(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): uint32 =
  let isym = n.cName.symbol.iSym
  case n.cName.symbol.reductionBuiltin
  of gbkReduceAdd:
    return emitReduceAdd(ctx, fn, n)
  of gbkNone: discard
  else:
    spvRefuse("reduction builtin " & $n.cName.symbol.reductionBuiltin)
  case n.cName.symbol.atomicsBuiltin
  of gbkAtomicFetchAdd:
    return emitAtomicFetchAdd(ctx, fn, n)
  of gbkNone: discard
  else:
    spvRefuse("atomics builtin " & $n.cName.symbol.atomicsBuiltin)
  if isym notin fn.fns:
    return emitScalarBuiltinCall(ctx, fn, n)
  let procNode = fn.fns[isym]
  let fnId = ctx.emittedFns.getOrDefault(isym, 0)
  doAssert fnId != 0, "callee " & n.cName.symbol.name & " was not pre-emitted"
  doAssert n.cArgs.len == procNode.pParams.len, "call argument count mismatch"
  var args: seq[uint32]
  for i, a in n.cArgs:
    let p = procNode.pParams[i]
    if p.passByRef or p.typ.kind in {gtPtr, gtVoidPtr}:
      let (ptrId, _, isBda) = emitAddr(ctx, fn, a)
      doAssert not isBda,
        "buffer device address pointers are not call arguments in the SPIR-V subset"
      args.add ptrId
    else:
      let v = emitExpr(ctx, fn, a)
      args.add ctx.convertValue(v, exprType(fn, a), valueType(p.typ))
  let retTyId = fnRetTyId(ctx, procNode.pRetType)
  ctx.builder.freshInst(OpFunctionCall, @[retTyId, fnId] & args)

proc emitExpr(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst): uint32 =
  case n.kind
  of gpuLit:
    if n.lValue == "DEFAULT":
      spvRefuse("DEFAULT literal")
    spvConstLit(ctx, n.lType, n.lValue)
  of gpuIdent:
    emitIdentRef(ctx, fn, n)
  of gpuBinOp:
    # comparisons yield gtBool and operate on the operand type, arithmetic on the self-carried result type
    let resTy = n.bType
    let opTy = if resTy.kind == gtBool: exprType(fn, n.bLeft) else: resTy
    let l = convertValue(ctx, emitExpr(ctx, fn, n.bLeft),
                         exprType(fn, n.bLeft), opTy)
    let r = convertValue(ctx, emitExpr(ctx, fn, n.bRight),
                         exprType(fn, n.bRight), opTy)
    ctx.builder.freshInst(binopInst(n.bOp, resTy, opTy),
                          @[ctx.tyId(spvScalar(resTy)), l, r])
  of gpuDot:
    emitDotValue(ctx, fn, n)
  of gpuIndex:
    let (ptrId, elem, isBda) = emitAccessPtr(ctx, fn, n.iArr, n.iIndex)
    if isBda:
      let elemAlign = uint32(std430Layout(spvScalar(elem)).align)
      ctx.builder.opLoadAligned(ctx.tyId(spvScalar(elem)), ptrId, elemAlign)
    else:
      ctx.builder.opLoad(ctx.tyId(spvScalar(valueType(elem))), ptrId)
  of gpuObjConstr:
    emitObjConstr(ctx, fn, n)
  of gpuCall:
    emitCall(ctx, fn, n)
  of gpuConv, gpuCast:
    let target = if n.kind == gpuConv: n.convTo else: n.cTo
    let src = if n.kind == gpuConv: n.convExpr else: n.cExpr
    convertValue(ctx, emitExpr(ctx, fn, src), exprType(fn, src), target)
  of gpuDeref:
    let (ptrId, pointee, isBda) = resolvePtrRoot(ctx, fn, n.dOf)
    doAssert not isBda,
      "deref load through a buffer device address is outside the SPIR-V subset"
    ctx.builder.opLoad(ctx.tyId(valueTy(pointee)), ptrId)
  of gpuPrefix:
    let v = emitExpr(ctx, fn, n.pVal)
    let t = exprGpuType(n.pVal)
    case n.pOp
    of "!":
      doAssert t.kind == gtBool, "logical not of a non-bool operand"
      ctx.builder.freshInst(OpLogicalNot, @[ctx.tyId(spvScalar(t)), v])
    of "-":
      let inst = if t.kind in {gtFloat32, gtFloat64}: OpFNegate else: OpSNegate
      ctx.builder.freshInst(inst, @[ctx.tyId(spvScalar(t)), v])
    else:
      spvRefuse("prefix operator " & n.pOp)
  of gpuArrayLit:
    let elemTy = valueType(n.aLitType)
    let arrTy = ctx.tyId(spvArray(spvScalar(elemTy), n.aValues.len))
    var vals: seq[uint32]
    for v in n.aValues:
      let e = emitExpr(ctx, fn, v)
      vals.add ctx.convertValue(e, exprType(fn, v), elemTy)
    ctx.builder.freshInst(OpCompositeConstruct, @[arrTy] & vals)
  else:
    spvRefuse("expression node " & $n.kind)

proc emitAssign(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst) =
  doAssert not n.aRequiresMemcpy, "memcpy assignment outside the SPIR-V subset"
  # compound assignments never reach the backend: the SPIR-V pipeline
  # rewrites them into plain assignments of the binop
  let rhs = emitExpr(ctx, fn, n.aRight)
  case n.aLeft.kind
  of gpuDot:
    let ch = dotChain(n.aLeft)
    let (ptrId, pointee) = emitChainPtr(ctx, fn, ch.root, ch.fields)
    let value = ctx.convertValue(rhs, exprType(fn, n.aRight), valueType(pointee))
    ctx.builder.opStore(ptrId, value)
  of gpuDeref:
    let (ptrId, pointee, isBda) = resolvePtrRoot(ctx, fn, n.aLeft)
    doAssert not isBda,
      "deref store through a buffer device address is outside the SPIR-V subset"
    let value = ctx.convertValue(rhs, exprType(fn, n.aRight), valueType(pointee))
    ctx.builder.opStore(ptrId, value)
  of gpuIndex:
    let (ptrId, elem, isBda) = emitAccessPtr(ctx, fn, n.aLeft.iArr, n.aLeft.iIndex)
    let value = ctx.convertValue(rhs, exprType(fn, n.aRight), valueType(elem))
    if isBda:
      let elemAlign = uint32(std430Layout(spvScalar(elem)).align)
      ctx.builder.opStoreAligned(ptrId, value, elemAlign)
    else:
      ctx.builder.opStore(ptrId, value)
  of gpuIdent:
    let sym = n.aLeft.symbol
    # iSym first: same-name sibling locals (two `total`, two `accs`) carry
    # distinct iSyms, name-keyed lookup would hit the wrong one.
    # The name fallback covers assign targets whose symbols the passes
    # re-symbol, their iSym keys change.
    let varId = if sym.iSym in fn.localsBySym:
                  fn.localsBySym[sym.iSym]
                elif sym.name in fn.locals:
                  fn.locals[sym.name]
                else:
                  spvRefuse("assignment to non-local " & sym.name)
    let t = if sym.iSym in fn.localTysBySym: fn.localTysBySym[sym.iSym]
            elif sym.typ.kind == gtVoid and sym.name in fn.localTys:
              fn.localTys[sym.name]
            else: sym.typ
    if t.kind in spvScalarKinds or isStaticInt(t):
      let value = ctx.convertValue(rhs, exprType(fn, n.aRight), valueType(t))
      ctx.builder.opStore(varId, value)
    else:
      ctx.builder.opStore(varId, rhs)
  else:
    spvRefuse("assignment target " & $n.aLeft.kind)

proc emitIfChain(ctx: var SpvLangCtx, fn: var SpvFnCtx, branches: seq[tuple[cond, body: GpuAst]], elseBody: GpuAst) =
  let mergeLbl = ctx.builder.freshLabel()
  let thenLbl = ctx.builder.freshLabel()
  let cond = emitExpr(ctx, fn, branches[0].cond)
  var mergeReachable = false
  if branches.len == 1 and elseBody.kind == gpuDiscard:
    ctx.builder.opSelectionMerge(mergeLbl, selNone)
    ctx.builder.opBranchConditional(cond, thenLbl, mergeLbl)
    mergeReachable = true
    ctx.builder.opLabelAt(thenLbl)
    fn.terminated = false
    emitStmt(ctx, fn, branches[0].body)
    if not fn.terminated:
      ctx.builder.opBranch(mergeLbl)
  else:
    let falseLbl = ctx.builder.freshLabel()
    ctx.builder.opSelectionMerge(mergeLbl, selNone)
    ctx.builder.opBranchConditional(cond, thenLbl, falseLbl)
    ctx.builder.opLabelAt(thenLbl)
    fn.terminated = false
    emitStmt(ctx, fn, branches[0].body)
    if not fn.terminated:
      ctx.builder.opBranch(mergeLbl)
      mergeReachable = true
    ctx.builder.opLabelAt(falseLbl)
    fn.terminated = false
    if branches.len == 1:
      emitStmt(ctx, fn, elseBody)
      if not fn.terminated:
        ctx.builder.opBranch(mergeLbl)
        mergeReachable = true
    else:
      emitIfChain(ctx, fn, branches[1 ..^ 1], elseBody)
      # the nested chain ends on its own merge block as the current block
      if not fn.terminated:
        ctx.builder.opBranch(mergeLbl)
        mergeReachable = true
  if not mergeReachable:
    spvRefuse("if-statement whose every branch returns, the merge block would be unreachable")
  ctx.builder.opLabelAt(mergeLbl)
  fn.terminated = false

proc emitIf(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst) =
  doAssert not n.ifIsExpr, "expression if outside the SPIR-V subset"
  var branches: seq[tuple[cond, body: GpuAst]] = @[(n.ifCond, n.ifThen)]
  for el in n.ifElifs:
    branches.add (el.cond, el.body)
  emitIfChain(ctx, fn, branches, n.ifElse)

proc emitFor(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst) =
  let indSym = n.fVar.symbol
  let indVar = getLocalVarId(fn, indSym)
  doAssert (indSym.iSym in fn.localTysBySym) or (indSym.name in fn.localTys),
    "induction variable was not hoisted"
  let indTy = ctx.tyId(spvScalar(indSym.typ))
  let boolTy = ctx.tyId(spvScalar(GpuType(kind: gtBool)))
  let startV = convertValue(ctx, emitExpr(ctx, fn, n.fStart),
                            exprType(fn, n.fStart), indSym.typ)
  ctx.builder.opStore(indVar, startV)
  # the bound evaluates once, in the pre-header block
  let endV = convertValue(ctx, emitExpr(ctx, fn, n.fEnd),
                          exprType(fn, n.fEnd), indSym.typ)
  let headerLbl = ctx.builder.freshLabel()
  ctx.builder.opBranch(headerLbl)
  let bodyLbl = ctx.builder.freshLabel()
  let contLbl = ctx.builder.freshLabel()
  let mergeLbl = ctx.builder.freshLabel()
  ctx.builder.opLabelAt(headerLbl)
  let cur = ctx.builder.opLoad(indTy, indVar)
  let cmpOp = case n.fRangeKind
              of rkExclusive:
                if indSym.typ.kind in spvSignedInts:
                  OpSLessThan
                else:
                  OpULessThan
              of rkInclusive:
                if indSym.typ.kind in spvSignedInts:
                  OpSLessThanEqual
                else:
                  OpULessThanEqual
  let cond = ctx.builder.freshInst(cmpOp, @[boolTy, cur, endV])
  # OpLoopMerge must be the second-to-last instruction of its block,
  # immediately before the branch terminator
  ctx.builder.opLoopMerge(mergeLbl, contLbl, lcNone)
  ctx.builder.opBranchConditional(cond, bodyLbl, mergeLbl)
  ctx.builder.opLabelAt(bodyLbl)
  fn.terminated = false
  emitStmt(ctx, fn, n.fBody)
  if fn.terminated:
    spvRefuse("return inside a loop body, the loop continue target would be unreachable")
  ctx.builder.opBranch(contLbl)
  ctx.builder.opLabelAt(contLbl)
  fn.terminated = false
  let stepV = convertValue(ctx, emitExpr(ctx, fn, n.fStep),
                           exprType(fn, n.fStep), indSym.typ)
  let iNow = ctx.builder.opLoad(indTy, indVar)
  let next = ctx.builder.opIAdd(indTy, iNow, stepV)
  ctx.builder.opStore(indVar, next)
  ctx.builder.opBranch(headerLbl)
  ctx.builder.opLabelAt(mergeLbl)
  fn.terminated = false

proc hoistLocals(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst) =
  case n.kind
  of gpuBlock:
    for s in n.statements:
      hoistLocals(ctx, fn, s)
  of gpuVar:
    doAssert n.addressSpace == asRMEM,
      "shared/constant locals are outside the SPIR-V subset"
    doAssert not n.vRequiresMemcpy,
      "memcpy locals are outside the SPIR-V subset"
    let sym = n.vName.symbol
    let varId = ctx.builder.opVariableFn(
      ctx.builder.tyPointer(scFunction, ctx.tyId(valueTy(n.vType))))
    fn.localsBySym[sym.iSym] = varId
    fn.localTysBySym[sym.iSym] = n.vType
    if sym.name notin fn.locals:
      fn.locals[sym.name] = varId
      fn.localTys[sym.name] = n.vType
  of gpuFor:
    let indSym = n.fVar.symbol
    let varId = ctx.builder.opVariableFn(
      ctx.builder.tyPointer(scFunction, ctx.tyId(spvScalar(indSym.typ))))
    fn.localsBySym[indSym.iSym] = varId
    fn.localTysBySym[indSym.iSym] = indSym.typ
    if indSym.name notin fn.locals:
      fn.locals[indSym.name] = varId
      fn.localTys[indSym.name] = indSym.typ
    hoistLocals(ctx, fn, n.fBody)
  of gpuIf:
    hoistLocals(ctx, fn, n.ifThen)
    for el in n.ifElifs:
      hoistLocals(ctx, fn, el.body)
    if n.ifElse.kind != gpuDiscard:
      hoistLocals(ctx, fn, n.ifElse)
  else:
    discard

proc emitStmt(ctx: var SpvLangCtx, fn: var SpvFnCtx, n: GpuAst) =
  case n.kind
  of gpuDiscard:
    discard
  of gpuBlock:
    doAssert not n.isExpr, "block expressions are outside the SPIR-V subset"
    for s in n.statements:
      emitStmt(ctx, fn, s)
  of gpuAssign:
    emitAssign(ctx, fn, n)
  of gpuVar:
    # storage hoisted to the setup block, the initializer stores here
    let sym = n.vName.symbol
    let varId = getLocalVarId(fn, sym)
    if n.vInit.kind != gpuDiscard:
      let v = emitExpr(ctx, fn, n.vInit)
      if n.vType.kind in spvScalarKinds or isStaticInt(n.vType):
        let value = ctx.convertValue(v, exprType(fn, n.vInit), valueType(n.vType))
        ctx.builder.opStore(varId, value)
      else:
        ctx.builder.opStore(varId, v)
  of gpuIf:
    emitIf(ctx, fn, n)
  of gpuFor:
    emitFor(ctx, fn, n)
  of gpuCall:
    discard emitCall(ctx, fn, n)
  of gpuConstexpr:
    # compile-time constant: the passes folded the value into literals
    discard
  of gpuReturn:
    if fn.terminated:
      spvRefuse("a second return in one basic block")
    if n.rValue.kind != gpuDiscard:
      let v = emitExpr(ctx, fn, n.rValue)
      ctx.builder.opReturnValue(v)
    else:
      ctx.builder.opReturn()
    fn.terminated = true
  else:
    spvRefuse("statement node " & $n.kind)

proc scanTree(n: GpuAst, calls: var seq[string]) =
  case n.kind
  of gpuCall:
    calls.add n.cName.symbol.iSym
    for a in n.cArgs:
      scanTree(a, calls)
  of gpuProc:
    scanTree(n.pBody, calls)
  of gpuBlock:
    for s in n.statements:
      scanTree(s, calls)
  of gpuVar:
    scanTree(n.vInit, calls)
  of gpuAssign:
    scanTree(n.aLeft, calls)
    scanTree(n.aRight, calls)
  of gpuIf:
    scanTree(n.ifCond, calls)
    scanTree(n.ifThen, calls)
    for el in n.ifElifs:
      scanTree(el.cond, calls)
      scanTree(el.body, calls)
    if n.ifElse.kind != gpuDiscard:
      scanTree(n.ifElse, calls)
  of gpuFor:
    scanTree(n.fStart, calls)
    scanTree(n.fEnd, calls)
    scanTree(n.fStep, calls)
    scanTree(n.fBody, calls)
  of gpuWhile:
    scanTree(n.wCond, calls)
    scanTree(n.wBody, calls)
  of gpuReturn:
    scanTree(n.rValue, calls)
  of gpuBinOp:
    scanTree(n.bLeft, calls)
    scanTree(n.bRight, calls)
  of gpuDot:
    scanTree(n.dParent, calls)
    scanTree(n.dField, calls)
  of gpuIndex:
    scanTree(n.iArr, calls)
    scanTree(n.iIndex, calls)
  of gpuPrefix:
    scanTree(n.pVal, calls)
  of gpuAddr:
    scanTree(n.aOf, calls)
  of gpuDeref:
    scanTree(n.dOf, calls)
  of gpuConv:
    scanTree(n.convExpr, calls)
  of gpuCast:
    scanTree(n.cExpr, calls)
  of gpuObjConstr:
    for f in n.ocFields:
      scanTree(f.value, calls)
  of gpuArrayLit:
    for v in n.aValues:
      scanTree(v, calls)
  of gpuMaterialize:
    scanTree(n.mExpr, calls)
  of gpuConstexpr:
    scanTree(n.cValue, calls)
  else:
    discard

proc emitDeviceFn(ctx: var SpvLangCtx, procNode: GpuAst): uint32 =
  doAssert procNode.kind == gpuProc, "not a function"
  doAssert not procNode.forwardDeclare, "forward declarations do not emit"
  let retTyId = fnRetTyId(ctx, procNode.pRetType)
  var paramTyIds: seq[uint32]
  var objParams: seq[tuple[sym: Symbol, pid: uint32, t: GpuType]]
  for i, p in procNode.pParams:
    if p.typ.kind == gtPtr:
      doAssert p.typ.to.kind notin {gtUncheckedArray},
        "unchecked-array pointer params are kernel BDA parameters"
      # `var T` parameters: call sites pass whole arrays or addresses
      # of locals, the declared pointee types the Function-storage pointer
      paramTyIds.add ctx.builder.tyPointer(scFunction,
        ctx.tyId(valueTy(p.typ.to)))
    elif p.passByRef:
      doAssert p.typ.kind in {gtObject, gtGenericInst},
        "byRef scalar params are outside the SPIR-V subset"
      paramTyIds.add ctx.builder.tyPointer(scFunction, ctx.tyId(valueTy(p.typ)))
    else:
      paramTyIds.add ctx.tyId(valueTy(valueType(p.typ)))
  let fnTy = ctx.builder.tyFunction(retTyId, paramTyIds)
  result = ctx.builder.opFunction(retTyId, fnTy)
  var fn = SpvFnCtx(fns: ctx.fns)
  # parameters come first, the entry label opens the body block
  for i, p in procNode.pParams:
    let pid = ctx.builder.opFunctionParameter(paramTyIds[i])
    let sym = p.ident.symbol
    if p.typ.kind == gtPtr or p.passByRef:
      fn.addrs[sym.iSym] = pid
      fn.addrNames[sym.name] = pid
      if p.typ.kind == gtPtr:
        if p.typ.implicit:
          fn.addrPointee[sym.iSym] = p.typ.to
        else:
          fn.addrPointee[sym.iSym] = pointeeElem(p.typ)
      else:
        fn.addrPointee[sym.iSym] = p.typ
        fn.addrNamePointees[sym.name] = p.typ
    elif p.typ.kind in {gtObject, gtGenericInst}:
      # by-value object parameter: remembered here, then copied into one Function-storage
      # variable right after the entry label, which gives member chains an address.
      # Stores still go through its pointer fields
      objParams.add (sym, pid, p.typ)
    else:
      fn.paramVals[sym.iSym] = pid
  discard ctx.builder.opLabel()
  hoistLocals(ctx, fn, procNode.pBody)
  # by-value object parameter copies: every OpVariable in a function must
  # precede any load, so they declare here before the copies store
  var copies: seq[tuple[varId: uint32, pid: uint32]]
  for (sym, pid, t) in objParams:
    let varId = ctx.builder.opVariableFn(
      ctx.builder.tyPointer(scFunction, ctx.tyId(valueTy(t))))
    copies.add (varId, pid)
    fn.addrs[sym.iSym] = varId
    fn.addrPointee[sym.iSym] = t
    fn.addrNames[sym.name] = varId
    fn.addrNamePointees[sym.name] = t
  # every OpVariable precedes every store in the entry block
  for (varId, pid) in copies:
    ctx.builder.opStore(varId, pid)
  emitStmt(ctx, fn, procNode.pBody)
  if not fn.terminated:
    ctx.builder.opReturn()
  ctx.builder.opFunctionEnd()

proc emitDeviceFnRec(ctx: var SpvLangCtx, procNode: GpuAst): uint32 =
  let isym = procNode.pName.symbol.iSym
  if isym in ctx.emittedFns:
    return ctx.emittedFns[isym]
  doAssert isym notin ctx.emitting,
    "recursive device function calls are outside the SPIR-V subset"
  ctx.emitting.incl isym
  var calls: seq[string]
  scanTree(procNode.pBody, calls)
  var seen: HashSet[string]
  for c in calls:
    if c in ctx.fns and c notin seen:
      seen.incl c
      discard ctx.emitDeviceFnRec(ctx.fns[c])
  result = ctx.emitDeviceFn(procNode)
  ctx.emittedFns[isym] = result
  ctx.emitting.excl isym

proc emitKernel(ctx: var SpvLangCtx, kernel: GpuAst, fns: Table[string, GpuAst]): uint32 =
  doAssert kernel.kind == gpuProc, "not a function"
  doAssert attGlobal in kernel.pAttributes, "only global kernels emit"
  doAssert kernel.pRetType.kind == gtVoid, "kernels are void"
  doAssert not kernel.forwardDeclare, "forward declarations do not emit"

  ctx.fns = fns
  # every kernel parameter is flat by now: the flattenObjectParams pass
  # decomposed object params into one scalar/pointer param per leaf.
  # The push block holds one member per parameter in parameter order
  var memberTys: seq[SpvTy]
  var hasBda = false
  for p in kernel.pParams:
    doAssert p.addressSpace != asSMEM,
      "shared-memory params are outside the SPIR-V subset"
    if p.typ.kind == gtVoidPtr:
      spvRefuse("void ptr params")
    if p.typ.kind in {gtPtr, gtUncheckedArray}:
      hasBda = true
      memberTys.add spvScalar(GpuType(kind: gtUint64))
    else:
      doAssert not p.passByRef,
        "by-reference kernel params are flattened to leaves before emission"
      memberTys.add pushMemberTy(p.typ)

  let push = ctx.buildPushBlockTys(memberTys)

  # device fns emit first, an OpFunctionCall result id must exist at the call site
  var calls: seq[string]
  scanTree(kernel.pBody, calls)
  for c in calls:
    if c in fns and c notin ctx.emittedFns:
      discard ctx.emitDeviceFnRec(fns[c])

  let voidTy = ctx.builder.tyVoid()
  let fnTy = ctx.builder.tyFunction(voidTy, @[])
  result = ctx.builder.opFunction(voidTy, fnTy)
  var fn = SpvFnCtx(fns: ctx.fns)
  discard ctx.builder.opLabel()

  # setup block: Function-storage variables first, then hoisted parameter loads,
  # all later blocks are dominated by the entry block
  hoistLocals(ctx, fn, kernel.pBody)
  for i, p in kernel.pParams:
    let sym = p.ident.symbol
    if p.typ.kind in {gtPtr, gtUncheckedArray}:
      # u64 device address in the push block, converted to a buffer pointer
      let u64Ty = ctx.tyId(spvScalar(GpuType(kind: gtUint64)))
      let slotPtrTy = ctx.builder.tyPointer(scPushConstant, u64Ty)
      let memberIdx = ctx.builder.constUint(ctx.scalarId(uint32Type()), uint64(i))
      let slot = ctx.builder.opAccessChain(slotPtrTy, push.varId, @[memberIdx])
      let devAddr = ctx.builder.opLoad(u64Ty, slot)
      fn.paramPtrs[sym.iSym] =
        ctx.builder.opConvertUToPtr(ctx.bdaPointerTy(pointeeElem(p.typ)), devAddr)
    else:
      let memberTy = memberTys[i]
      let slotPtrTy = ctx.builder.tyPointer(scPushConstant, ctx.tyId(memberTy))
      let memberIdx = ctx.builder.constUint(ctx.scalarId(uint32Type()), uint64(i))
      let slot = ctx.builder.opAccessChain(slotPtrTy, push.varId, @[memberIdx])
      fn.paramVals[sym.iSym] = ctx.builder.opLoad(ctx.tyId(memberTy), slot)

  emitStmt(ctx, fn, kernel.pBody)
  if not fn.terminated:
    ctx.builder.opReturn()
  ctx.builder.opFunctionEnd()

  # BDA loads through u64 address members pull
  # the PhysicalStorageBuffer capability too: memory model and extension
  # must pair with it
  let usePhys = hasBda or (capPhysicalStorageBufferAddresses in ctx.caps)
  if usePhys:
    # PhysicalStorageBufferAddresses is extension-gated on SPIR-V 1.3
    ctx.builder.emitExtension("SPV_KHR_physical_storage_buffer")
  var pendingExts: seq[string]
  for e in ctx.exts:
    pendingExts.add e
  pendingExts.sort()
  for e in pendingExts:
    ctx.builder.emitExtension(e)
  let lws = if kernel.pWorkgroupSize == (0, 0, 0): (256, 1, 1)
            else: kernel.pWorkgroupSize
  ctx.builder.emitEntryPoint(emGLCompute, result, kernel.pName.ident(),
                             ctx.interfaceVars)
  ctx.builder.emitExecutionMode(result, modeLocalSize,
                                @[uint32(lws[0]), uint32(lws[1]), uint32(lws[2])])
  ctx.builder.emitMemoryModel(
    (if usePhys: amPhysicalStorageBuffer64 else: amLogical), mmGLSL450)
  ctx.flushCapabilities()

# ═══════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════

proc codegen*(ctx: var GpuContext): seq[uint32] =
  ## Returns the SPIR-V words of one `spirv:` block for the Vulkan environment
  ## (SPIR-V 1.3, target environment vulkan1.4):
  ## - exactly one `{.global.}` kernel per block, refusal otherwise
  ## - device functions used by the kernel emit alongside it
  ## - pointer parameters carry one u64 device address push member each
  ## - the words ingest directly into the Vulkan engine
  var lang = initSpvLangCtx()
  var kernels: seq[GpuAst]
  for _, fn in ctx.fnTab:
    if fn.isGlobal():
      kernels.add fn
  if kernels.len != 1:
    error "SPIR-V codegen requires exactly one `{.global.}` kernel per `spirv:` block, found " & $kernels.len
  var fns: Table[string, GpuAst] = initTable[string, GpuAst]()
  for _, fn in ctx.allFnTab:
    if fn.kind == gpuProc and not fn.isGlobal() and not fn.forwardDeclare:
      fns[fn.pName.symbol.iSym] = fn
  discard lang.emitKernel(kernels[0], fns)
  result = lang.builder.module.finalize()
