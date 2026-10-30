# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SPIR-V word constants shared by the printer (writer) and the runtime
## module scanner (reader), one source of truth.
## The opcode, capability, and execution-mode numbers are the SPIR-V core
## grammar values (SPIRV.html, sections 3.3-3.6).

const
  opEntryPoint* = 15'u32
  opExecutionMode* = 16'u32
  opCapability* = 17'u32

  capPhysicalStorageBufferAddresses* = 5347'u32
  modeLocalSize* = 17'u32

# ═══════════════════════════════════════════════════════════════════════
#  Value types
# ═══════════════════════════════════════════════════════════════════════

## SPIR-V emitter over CIRv2, Vulkan 1.4 environment on the encoder's
## SPIR-V 1.3 baseline:
##
## - one `{.global.}` kernel per module, the device functions it calls emitted alongside
## - one push-constant block binds the kernel parameters in parameter order, the host `store` packing order
## - buffer parameters contribute one u64 device address each, converted to physical storage buffer pointers in the kernel prologue

import std/[algorithm, sets, strutils, tables, macros]
import ../ir/cirv2_lang
import ./spirv_encoder {.all.}
import ../ir/ast_datatypes

template spvRefuse(what: string) =
  error("SPIR-V backend does not support: " & what)

proc cirUint32(): CirType =
  CirType(kind: ctUint, bits: 32)

proc cirInt32(): CirType =
  CirType(kind: ctInt, bits: 32)

proc cirUint64(): CirType =
  CirType(kind: ctUint, bits: 64)

type
  SpvTyKind* = enum
    stkScalar
    stkVector
    stkArray
    stkRuntimeArray
    stkStruct
    stkPointer

  SpvTy* = ref object
    ## SPIR-V value-type descriptor lowered from a CirType.
    ## `blockDecorated` marks push-constant and buffer-reference struct
    ## wrappers with the Block decoration Vulkan requires on them.
    blockDecorated*: bool
    case kind*: SpvTyKind
    of stkScalar:
      scalar*: CirType          ## ctInt/ctUint/ctFloat/ctBool/ctVoid
    of stkVector:
      vElem*: SpvTy
      vN*: int
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

proc spvScalar(t: CirType): SpvTy =
  doAssert t.kind in {ctVoid, ctBool, ctInt, ctUint, ctFloat},
    "not a scalar CirType: " & $t.kind
  SpvTy(kind: stkScalar, scalar: t)

proc spvVector(elem: SpvTy, n: int): SpvTy =
  SpvTy(kind: stkVector, vElem: elem, vN: n)

proc spvArray(elem: SpvTy, len: int): SpvTy =
  doAssert len >= 1, "length 0 means runtime-sized, use spvRuntimeArray"
  SpvTy(kind: stkArray, elem: elem, len: len)

proc spvStruct(members: seq[SpvTy]): SpvTy =
  SpvTy(kind: stkStruct, members: members)

proc spvPointer(storage: uint32, pointee: SpvTy): SpvTy =
  SpvTy(kind: stkPointer, storage: storage, pointee: pointee)

proc isSpvScalarKind(t: CirType): bool =
  t.kind in {ctVoid, ctBool, ctInt, ctUint, ctFloat}

proc spvTyKey(t: SpvTy): string =
  ## Structural key, the type-identity memo for OpType* dedup.
  case t.kind
  of stkScalar:
    result = "s:" & $t.scalar.kind
    if t.scalar.kind in {ctInt, ctUint, ctFloat}:
      result &= ":" & $t.scalar.bits
  of stkVector: result = "v:" & spvTyKey(t.vElem) & ":" & $t.vN
  of stkArray: result = "a:" & spvTyKey(t.elem) & ":" & $t.len
  of stkRuntimeArray: result = "r:" & spvTyKey(t.relem)
  of stkStruct:
    result = "t:"
    for m in t.members:
      result &= spvTyKey(m) & ","
    if t.blockDecorated:
      result &= "B"
  of stkPointer: result = "p:" & $t.storage & ":" & spvTyKey(t.pointee)

# ═══════════════════════════════════════════════════════════════════════
#  std430 layout
# ═══════════════════════════════════════════════════════════════════════

proc roundUpMultiple(v, multiple: int): int =
  doAssert multiple > 0
  (v + multiple - 1) div multiple * multiple

type SpvLayout = object
  size*: int
  align*: int

proc scalarByteSize(t: CirType): int =
  doAssert t.kind in {ctInt, ctUint, ctFloat, ctBool}
  if t.kind == ctBool:
    1
  else:
    t.bits div 8

proc std430Layout(t: SpvTy): SpvLayout =
  case t.kind
  of stkScalar:
    let s = scalarByteSize(t.scalar)
    SpvLayout(size: s, align: s)
  of stkVector:
    let e = std430Layout(t.vElem)
    # std430 base alignment: 1x/2x the component's for vec1/vec2, 4x for vec3/vec4
    let alignMul = if t.vN <= 2: t.vN else: 4
    SpvLayout(size: e.size * t.vN, align: alignMul * e.align)
  of stkArray:
    let e = std430Layout(t.elem)
    let stride = roundUpMultiple(e.size, e.align)
    SpvLayout(size: stride * t.len, align: e.align)
  of stkRuntimeArray:
    let e = std430Layout(t.relem)
    SpvLayout(size: 0, align: roundUpMultiple(e.size, e.align))
  of stkStruct:
    var off = 0
    var align = 1
    for m in t.members:
      let ml = std430Layout(m)
      off = roundUpMultiple(off, ml.align) + ml.size
      align = max(align, ml.align)
    SpvLayout(size: roundUpMultiple(off, align), align: align)
  of stkPointer:
    SpvLayout(size: 8, align: 8)

proc std430Offsets(members: seq[SpvTy]): seq[int] =
  var off = 0
  for m in members:
    let ml = std430Layout(m)
    off = roundUpMultiple(off, ml.align)
    result.add off
    off += ml.size

# ═══════════════════════════════════════════════════════════════════════
#  Emission context
# ═══════════════════════════════════════════════════════════════════════

type
  SpvLangCtx* = object
    builder*: SpvBuilder
    caps*: HashSet[uint32]
    tyMemo*: Table[string, uint32]
    structMemo*: Table[string, uint32]
    builtins*: Table[uint32, tuple[varId, vecTy: uint32]]
    scalarBuiltins*: Table[uint32, uint32]
    interfaceVars*: seq[uint32]
    exts*: seq[string]
    emittedFns*: Table[string, uint32]
    fnProcs*: Table[string, CirProc]
    emitting*: HashSet[string]
    fnTypes*: Table[string, CirType]
      ## proc name -> value type of its CirProc, the fn-value-type lookup
    err*: string

proc initSpvLangCtx(): SpvLangCtx =
  result.builder = initSpvBuilder()
  result.caps.incl capShader

proc failSpv(ctx: var SpvLangCtx, what: string) =
  if ctx.err.len == 0:
    ctx.err = "SPIR-V backend does not support: " & what

proc scalarId(ctx: var SpvLangCtx, t: CirType): uint32 =
  case t.kind
  of ctVoid: ctx.builder.tyVoid()
  of ctBool: ctx.builder.tyBool()
  of ctUint:
    case t.bits
    of 8: ctx.caps.incl capInt8; ctx.builder.tyInt(8, false)
    of 16: ctx.caps.incl capInt16; ctx.builder.tyInt(16, false)
    of 32: ctx.builder.tyInt(32, false)
    of 64: ctx.caps.incl capInt64; ctx.builder.tyInt(64, false)
    else: ctx.failSpv("uint width " & $t.bits); return 0
  of ctInt:
    case t.bits
    of 16: ctx.caps.incl capInt16; ctx.builder.tyInt(16, true)
    of 32: ctx.builder.tyInt(32, true)
    of 64: ctx.caps.incl capInt64; ctx.builder.tyInt(64, true)
    else: ctx.failSpv("int width " & $t.bits); return 0
  of ctFloat:
    case t.bits
    of 16: ctx.caps.incl capFloat16; ctx.builder.tyFloat(16)
    of 32: ctx.builder.tyFloat(32)
    of 64: ctx.caps.incl capFloat64; ctx.builder.tyFloat(64)
    else: ctx.failSpv("float width " & $t.bits); return 0
  else:
    ctx.failSpv("scalar id of " & $t.kind)
    return 0

proc valueTy(ctx: var SpvLangCtx, t: CirType): SpvTy

proc valueTy(ctx: var SpvLangCtx, t: CirType): SpvTy =
  ## SPIR-V value type of a CirType.
  ##
  ## Pointer-typed members keep their pointer type, storing pointer values
  ## needs the variable-pointers capability and such a member declares it.
  case t.kind
  of ctVoid, ctBool, ctInt, ctUint, ctFloat: spvScalar(t)
  of ctVector: spvVector(ctx.valueTy(t.vElem), t.vN)
  of ctArray: spvArray(ctx.valueTy(t.aElem), t.aN)
  of ctStruct:
    var ms: seq[SpvTy]
    for f in t.fields:
      ms.add ctx.valueTy(f.typ)
    spvStruct(ms)
  of ctPtr:
    # an explicit pointer, the Function-storage pointer of var params/locals
    spvPointer(scFunction, ctx.valueTy(t.pElem))
  of ctAptr:
    if t.space != caDevice:
      ctx.failSpv("array pointer in " & $t.space & " space")
      return nil
    if t.pElem == nil or not isSpvScalarKind(t.pElem):
      ctx.failSpv("array pointer to a non-scalar element")
      return nil
    ctx.caps.incl capPhysicalStorageBufferAddresses
    # buffer reference shape: a Block struct over a runtime array
    # element access chains through member 0 and the index
    let arr = SpvTy(kind: stkRuntimeArray, relem: ctx.valueTy(t.pElem))
    var wrapper = spvStruct(@[arr])
    wrapper.blockDecorated = true
    spvPointer(scPhysicalStorageBuffer, wrapper)
  else:
    ctx.failSpv("CirType outside the SPIR-V subset: " & $t.kind)
    return nil

proc tyId(ctx: var SpvLangCtx, t: SpvTy): uint32 =
  ## SPIR-V type id of a value type, memoized on the structural key.
  let key = spvTyKey(t)
  if ctx.tyMemo.hasKey(key):
    return ctx.tyMemo[key]
  case t.kind
  of stkScalar: result = ctx.scalarId(t.scalar)
  of stkVector:
    result = ctx.builder.tyVector(ctx.tyId(t.vElem), t.vN)
  of stkArray:
    doAssert t.len >= 1
    let elemId = ctx.tyId(t.elem)
    let lenId = ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(t.len))
    let memoKey: SpvMemoKey = (OpTypeArray, @[elemId, lenId])
    if memoKey in ctx.builder.memo:
      return ctx.builder.memo[memoKey]
    result = ctx.builder.memoInst(OpTypeArray, @[elemId, lenId])
    let e = std430Layout(t.elem)
    ctx.builder.emitDecorate(result, decArrayStride,
      @[uint32(roundUpMultiple(e.size, e.align))])
  of stkRuntimeArray:
    let elemId = ctx.tyId(t.relem)
    let memoKey: SpvMemoKey = (OpTypeRuntimeArray, @[elemId])
    if memoKey in ctx.builder.memo:
      return ctx.builder.memo[memoKey]
    result = ctx.builder.memoInst(OpTypeRuntimeArray, @[elemId])
    let e = std430Layout(t.relem)
    ctx.builder.emitDecorate(result, decArrayStride,
      @[uint32(roundUpMultiple(e.size, e.align))])
  of stkStruct:
    var memberIds: seq[uint32]
    for m in t.members:
      memberIds.add ctx.tyId(m)
    let offsets = std430Offsets(t.members)
    var memoKey = "t:"
    for mid in memberIds:
      memoKey &= $mid & ","
    for off in offsets:
      memoKey &= $off & ","
    if t.blockDecorated:
      memoKey &= "B"
    if ctx.structMemo.hasKey(memoKey):
      return ctx.structMemo[memoKey]
    result = ctx.builder.module.allocId()
    var words = @[result]
    for mid in memberIds:
      words.add mid
    ctx.builder.module.emitInstSeq(ssTypesConstantsGlobals, OpTypeStruct, words)
    for i, off in offsets:
      ctx.builder.emitMemberDecorate(result, uint32(i), decOffset, @[uint32(off)])
    if t.blockDecorated:
      ctx.builder.emitDecorate(result, decBlock)
    ctx.structMemo[memoKey] = result
  of stkPointer:
    if t.storage == scPhysicalStorageBuffer:
      ctx.caps.incl capPhysicalStorageBufferAddresses
    result = ctx.builder.tyPointer(t.storage, ctx.tyId(t.pointee))
  ctx.tyMemo[key] = result

type
  SpvPushBlock = object
    structTy*: uint32
    ptrTy*: uint32
    varId*: uint32
    members*: seq[SpvTy]
    offsets*: seq[int]
    byteSize*: int

proc pushMemberTy(ctx: var SpvLangCtx, p: CirParam): SpvTy =
  ## One push-block member per kernel parameter, in parameter order.
  ## A device array pointer contributes its u64 device address
  ## and everything else its value type.
  if p.typ.kind == ctAptr:
    spvScalar(cirUint64())
  else:
    ctx.valueTy(p.typ)

proc bdaPointerTy(ctx: var SpvLangCtx, elemTy: CirType): uint32 =
  ## Physical storage buffer pointer type of a buffer reference
  ## the `valueTy` shape of the matching `aptr` type.
  ctx.tyId(ctx.valueTy(CirType(kind: ctAptr, space: caDevice, pElem: elemTy)))

proc buildPushBlock(ctx: var SpvLangCtx, params: seq[CirParam]): SpvPushBlock =
  ## One OpTypeStruct push block in parameter order, offsets are the host
  ## `store` packing numbers.
  ##
  ## Buffer args bind their address in place.
  var members: seq[SpvTy]
  for p in params:
    if p.typ.kind == ctAptr:
      if p.typ.space != caDevice:
        ctx.failSpv("kernel parameter " & p.name & " in " & $p.typ.space &
          " space, only device buffers carry a push address")
      members.add spvScalar(cirUint64())
    elif p.typ.kind in {ctPtr, ctAptr}:
      ctx.failSpv("kernel parameter " & p.name & " of pointer type")
    else:
      if p.typ.kind == ctBool:
        ctx.failSpv("bool kernel parameter " & p.name &
          ", the 4-byte push convention has no bool member")
      let m = ctx.valueTy(p.typ)
      let l = std430Layout(m)
      if l.align < 4:
        ctx.failSpv("push member " & p.name & " is narrower than 32 bits (" &
          $l.size & " bytes), narrow members need SPIR-V 1.5 storage capabilities")
      members.add m
  result.members = members
  result.offsets = std430Offsets(members)
  var blockTy: SpvTy = spvStruct(members)
  result.byteSize = std430Layout(blockTy).size
  result.structTy = ctx.tyId(blockTy)
  ctx.builder.emitDecorate(result.structTy, decBlock)
  result.ptrTy = ctx.builder.tyPointer(scPushConstant, result.structTy)
  result.varId = ctx.builder.opVariable(result.ptrTy, scPushConstant)

proc builtinInput(ctx: var SpvLangCtx, builtin: uint32): uint32 =
  ## Input-storage builtin variable for a vector coordinate builtin,
  ## the OpEntryPoint interface lists it.
  if ctx.builtins.hasKey(builtin):
    return ctx.builtins[builtin].varId
  let vecTy = ctx.builder.tyVector(ctx.scalarId(cirUint32()), 3)
  let ptrTy = ctx.builder.tyPointer(scInput, vecTy)
  let varId = ctx.builder.opVariable(ptrTy, scInput)
  ctx.builder.emitDecorate(varId, decBuiltIn, @[builtin])
  ctx.builtins[builtin] = (varId, vecTy)
  ctx.interfaceVars.add varId
  result = varId

proc convertValue(ctx: var SpvLangCtx, v: uint32, source, to: CirType): uint32

proc loadBuiltinComponent(ctx: var SpvLangCtx, builtin: uint32, component: int, target: CirType): uint32 =
  ## One coordinate-builtin component, cast to the canonical int type
  ## the CIRv2 vocabulary reads.
  let varId = ctx.builtinInput(builtin)
  let v = ctx.builder.opLoad(ctx.builtins[builtin].vecTy, varId)
  let raw = ctx.builder.opCompositeExtract(ctx.scalarId(cirUint32()), v, @[component])
  ctx.convertValue(raw, cirUint32(), target)

proc coordBuiltinInput(ctx: var SpvLangCtx, kind: CoordBuiltinKind): uint32 =
  ## Canonical coordinate vocabulary mapped onto the BuiltIn enumerants.
  case kind
  of cbkThreadPositionInGrid: result = biGlobalInvocationId
  of cbkThreadgroupPositionInGrid: result = biWorkgroupId
  of cbkThreadPositionInThreadgroup: result = biLocalInvocationId
  of cbkThreadsPerThreadgroup: result = biWorkgroupSize
  of cbkThreadgroupsPerGrid: result = biNumWorkgroups
  of cbkThreadIndexInThreadgroup: result = biLocalInvocationIndex
  of cbkNone:
    ctx.failSpv("not a coordinate builtin kind")

proc scalarBuiltinInput(ctx: var SpvLangCtx, builtin: uint32): uint32 =
  if ctx.scalarBuiltins.hasKey(builtin):
    return ctx.scalarBuiltins[builtin]
  let u32Ty = ctx.scalarId(cirUint32())
  let ptrTy = ctx.builder.tyPointer(scInput, u32Ty)
  let varId = ctx.builder.opVariable(ptrTy, scInput)
  ctx.builder.emitDecorate(varId, decBuiltIn, @[builtin])
  ctx.scalarBuiltins[builtin] = varId
  ctx.interfaceVars.add varId
  result = varId

proc loadScalarBuiltin(ctx: var SpvLangCtx, builtin: uint32, target: CirType): uint32 =
  ## Flat thread index, cast to the canonical int type the CIRv2 reads.
  let varId = ctx.scalarBuiltinInput(builtin)
  let raw = ctx.builder.opLoad(ctx.scalarId(cirUint32()), varId)
  ctx.convertValue(raw, cirUint32(), target)

proc flushCapabilities(ctx: var SpvLangCtx) =
  var caps: seq[uint32]
  for c in ctx.caps:
    caps.add c
  caps.sort()
  for c in caps:
    ctx.builder.emitCapability(c)

proc flushExtensions(ctx: var SpvLangCtx) =
  var exts = ctx.exts
  exts.sort()
  for e in exts:
    ctx.builder.emitExtension(e)

# ═══════════════════════════════════════════════════════════════════════
#  Function body emission
# ═══════════════════════════════════════════════════════════════════════

type
  SpvFnCtx* = object
    name*: string
    locals*: Table[string, uint32]
      ## local name -> Function-storage variable id
    localTys*: Table[string, CirType]
    consts*: Table[string, uint32]
      ## const name -> its constant id
    params*: Table[string, uint32]
      ## scalar/struct param name -> hoisted value id
    bdaParams*: Table[string, uint32]
      ## device buffer param name -> the physical storage buffer pointer value
    addrParams*: Table[string, uint32]
      ## Function-storage pointer param name -> the parameter's pointer id
    addrPointee*: Table[string, CirType]
      ## pointer param name -> the pointee value type
    labels*: Table[string, uint32]
      ## label name -> block id, filled forward for jmp targets
    loopMerge*: uint32
      ## the merge block of the loop this body belongs to, 0 outside a loop
    loopCont*: uint32
      ## the continue target of the enclosing loop
    loopExit*: string
      ## the break label name of the enclosing loop
    ptrTargets*: Table[string, uint32]
      ## pointer-typed local name -> the Function-storage variable it names,
      ## a pointer to a Function object cannot live in a Function variable
    terminated*: bool

proc emitExpr(ctx: var SpvLangCtx, fn: var SpvFnCtx, e: CirExpr): uint32
proc resolveExprType(ctx: var SpvLangCtx, fn: SpvFnCtx, e: CirExpr): CirType

proc failFn(ctx: var SpvFnCtx, what: string): string =
  what

proc hoistLocals(ctx: var SpvLangCtx, fn: var SpvFnCtx, ss: seq[CirStmt]) =
  ## Function-storage variables must all precede the entry block's first load.
  ## Every sVar declares here, the declaration site stores its value.
  for s in ss:
    case s.kind
    of sVar:
      if s.space != caThread:
        ctx.failSpv("local " & s.name & " in " & $s.space &
          " space, shared/constant locals are outside the SPIR-V subset")
      if s.declType.kind == ctPtr:
        # a pointer-typed local resolves to its pointee variable, no storage
        discard
      else:
        let vt = ctx.valueTy(s.declType)
        let varId = ctx.builder.opVariableFn(
          ctx.builder.tyPointer(scFunction, ctx.tyId(vt)))
        fn.locals[s.name] = varId
        fn.localTys[s.name] = s.declType
    of sConst:
      discard
    of sIte:
      ctx.hoistLocals(fn, s.thenBody)
      ctx.hoistLocals(fn, s.elseBody)
    of sLoop:
      ctx.hoistLocals(fn, s.loopBody)
    of sScope:
      ctx.hoistLocals(fn, s.scopeBody)
    else:
      discard

proc constId(ctx: var SpvLangCtx, fn: SpvFnCtx, e: CirExpr): uint32 =
  ## Constant id of a literal, or the registered const name.
  case e.kind
  of eIntLit:
    let tyId = ctx.tyId(spvScalar(e.litType))
    let width = ctx.builder.getScalarWidth(tyId)
    let mask =
      if width >= 64: not 0'u64
      else: (1'u64 shl uint64(width)) - 1
    ctx.builder.constUint(tyId, uint64(e.value) and mask)
  of eFloatLit:
    let tyId = ctx.tyId(spvScalar(e.fLitType))
    let width = ctx.builder.getScalarWidth(tyId)
    if width == 32:
      ctx.builder.constUint(tyId, uint64(cast[uint32](e.fValue.float32)))
    else:
      ctx.builder.constUint(tyId, cast[uint64](e.fValue))
  of eTrue:
    let boolTy = ctx.tyId(spvScalar(CirType(kind: ctBool)))
    ctx.builder.constBool(boolTy, true)
  of eFalse:
    let boolTy = ctx.tyId(spvScalar(CirType(kind: ctBool)))
    ctx.builder.constBool(boolTy, false)
  of eSym:
    if fn.consts.hasKey(e.name):
      return fn.consts[e.name]
    ctx.failSpv("literal expected, got symbol " & e.name)
    0
  else:
    ctx.failSpv("literal expected, got expression kind " & $e.kind)
    0

proc scalarBits(t: CirType): int =
  ## Bit width of a scalar type, 0 for widthless kinds.
  if t == nil:
    return 0
  if t.kind in {ctInt, ctUint, ctFloat}:
    t.bits
  else:
    0

proc convertValue(ctx: var SpvLangCtx, v: uint32, source, to: CirType): uint32 =
  ## Scalar conversion opcode between two value types.
  if source == nil or to == nil:
    ctx.failSpv("scalar conversion with a missing operand type")
    return 0
  if source.kind == to.kind and scalarBits(source) == scalarBits(to):
    return v
  let fFloat = source.kind == ctFloat
  let tFloat = to.kind == ctFloat
  let fInt = source.kind in {ctInt, ctUint}
  let tInt = to.kind in {ctInt, ctUint}
  if to.kind == ctBool and not fFloat:
    # bool reads as truthiness, any nonzero integer means true
    let zero = ctx.builder.constUint(ctx.scalarId(source), 0)
    return ctx.builder.freshInst(OpINotEqual,
      @[ctx.tyId(spvScalar(CirType(kind: ctBool))), v, zero])
  if source.kind == ctBool and tInt:
    let boolTy = ctx.tyId(spvScalar(CirType(kind: ctBool)))
    let one = ctx.builder.constUint(ctx.scalarId(to), 1)
    let zero = ctx.builder.constUint(ctx.scalarId(to), 0)
    return ctx.builder.opSelect(ctx.scalarId(to),
      ctx.builder.freshInst(OpLogicalEqual, @[boolTy, v,
        ctx.builder.constBool(boolTy, true)]), one, zero)
  if fInt and tInt:
    if scalarBits(source) == scalarBits(to):
      return ctx.builder.freshInst(OpBitcast, @[ctx.scalarId(to), v])
    if to.kind == ctInt:
      return ctx.builder.freshInst(OpSConvert, @[ctx.scalarId(to), v])
    return ctx.builder.freshInst(OpUConvert, @[ctx.scalarId(to), v])
  if fInt and tFloat:
    if source.kind == ctInt:
      return ctx.builder.freshInst(OpConvertSToF, @[ctx.scalarId(to), v])
    return ctx.builder.freshInst(OpConvertUToF, @[ctx.scalarId(to), v])
  if fFloat and tInt:
    if to.kind == ctInt:
      return ctx.builder.freshInst(OpConvertFToS, @[ctx.scalarId(to), v])
    return ctx.builder.freshInst(OpConvertFToU, @[ctx.scalarId(to), v])
  if fFloat and tFloat:
    return ctx.builder.freshInst(OpFConvert, @[ctx.scalarId(to), v])
  ctx.failSpv("no scalar conversion " & $source.kind & $scalarBits(source) &
    " -> " & $to.kind & $scalarBits(to))
  0

proc memberIndex(t: CirType, name: string): int =
  ## Member index of a named struct field or a vector component
  ## (`F0`/`F1`/...), -1 when absent.
  if t.kind == ctStruct:
    for i, f in t.fields:
      if f.name == name:
        return i
    return -1
  if t.kind == ctVector and name.len >= 2 and name[0] == 'F':
    let i = parseInt(name[1 ..^ 1])
    if i >= 0 and i < t.vN:
      return i
    return -1
  return -1

proc resolveExprType(ctx: var SpvLangCtx, fn: SpvFnCtx, e: CirExpr): CirType =
  ## CirType of an expression, the read type for conversions.
  case e.kind
  of eNil: result = e.nilType
  of eSym:
    # temp syms carry no type, the hoisted declaration's type is the truth
    result = fn.localTys.getOrDefault(e.name, e.typ)
  of eIntLit: result = e.litType
  of eFloatLit: result = e.fLitType
  of eTrue, eFalse: result = CirType(kind: ctBool)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    result = e.opType
  of eBitNot, eNeg: result = e.unType
  of eEq, eNeq, eLt, eLe:
    # a comparison is a value of bool, cmpType names the operand type
    result = CirType(kind: ctBool)
  of eNot: result = CirType(kind: ctBool)
  of eAnd, eOr: result = CirType(kind: ctBool)
  of eCast, eConv: result = e.castType
  of eAddr: result = e.addrType
  of eSizeof: result = cirUint32()
  of eOconstr: result = e.ocType
  of eAconstr: result = e.acType
  of eDeref:
    let t = ctx.resolveExprType(fn, e.derefOperand)
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.pElem
    else:
      ctx.failSpv("deref of a non-pointer operand")
  of eAt:
    let t = ctx.resolveExprType(fn, e.atBase)
    if t != nil and t.kind == ctArray:
      result = t.aElem
    elif t != nil and t.kind in {ctPtr, ctAptr} and t.pElem != nil:
      result = t.pElem
    else:
      ctx.failSpv("index of a non-array operand")
  of ePat:
    let t = ctx.resolveExprType(fn, e.patBase)
    if t != nil and t.kind in {ctPtr, ctAptr} and t.pElem != nil:
      result = t.pElem
    else:
      ctx.failSpv("pattern access on a non-pointer operand")
  of eDot:
    let t = ctx.resolveExprType(fn, e.dotBase)
    if t != nil and t.kind == ctStruct:
      let i = memberIndex(t, e.fieldName)
      if i >= 0:
        result = t.fields[i].typ
      else:
        ctx.failSpv("no field " & e.fieldName & " on " & t.name)
    elif t != nil and t.kind == ctVector:
      let i = memberIndex(t, e.fieldName)
      if i >= 0:
        result = t.vElem
      else:
        ctx.failSpv("no component " & e.fieldName & " on the vector")
    else:
      ctx.failSpv("field access on a non-struct operand")
  of eCall:
    if e.callee.kind == eSym and ctx.fnTypes.hasKey(e.callee.name):
      result = ctx.fnTypes[e.callee.name]
    elif e.callee.kind == eSym and e.callee.name == "atomicFetchAdd" and
        e.args.len == 2 and e.args[0].kind == eAddr:
      # an atomic add yields the addressed element, its first argument
      # names the address
      let pat = e.args[0].addrOperand
      if pat.kind == ePat:
        let t = ctx.resolveExprType(fn, pat.patBase)
        if t != nil and t.kind in {ctPtr, ctAptr} and t.pElem != nil:
          result = t.pElem
          return
      ctx.failSpv("atomicFetchAdd address is not the address of an element")
    elif e.callee.kind == eSym and e.args.len > 0:
      # a builtin call, the result type is the first argument's type
      result = ctx.resolveExprType(fn, e.args[0])
    else:
      ctx.failSpv("call result type of " & (if e.callee.kind == eSym: e.callee.name else: "non-symbol callee"))

proc emitLhsPtr(ctx: var SpvLangCtx, fn: var SpvFnCtx, e: CirExpr): tuple[ptrId: uint32, pointee: CirType, isBda: bool, isPush: bool]

proc emitCallValue(ctx: var SpvLangCtx, fn: var SpvFnCtx, e: CirExpr): uint32

proc binopInst(ctx: var SpvLangCtx, name: string, operandTy: CirType): SpvOp =
  ## SPIR-V opcode of an arithmetic or comparison operator.
  ## OpNop marks an operator outside the supported set, the caller records it.
  let isFloat = operandTy.kind == ctFloat
  let signed = operandTy.kind == ctInt
  case name
  of "+": result = (if isFloat: OpFAdd else: OpIAdd)
  of "-": result = (if isFloat: OpFSub else: OpISub)
  of "*": result = (if isFloat: OpFMul else: OpIMul)
  of "/":
    result = (if isFloat: OpFDiv elif signed: OpSDiv else: OpUDiv)
  of "%":
    if isFloat:
      result = OpNop
      return
    result = (if signed: OpSRem else: OpUMod)
  of "<<": result = OpShiftLeftLogical
  of ">>":
    result = (if signed: OpShiftRightArithmetic else: OpShiftRightLogical)
  of "&": result = OpBitwiseAnd
  of "|": result = OpBitwiseOr
  of "xor": result = OpBitwiseXor
  of "==": result = (if isFloat: OpFOrdEqual
                    elif operandTy.kind == ctBool: OpLogicalEqual
                    else: OpIEqual)
  of "!=": result = (if isFloat: OpFOrdNotEqual
                    elif operandTy.kind == ctBool: OpLogicalNotEqual
                    else: OpINotEqual)
  of "<": result = (if isFloat: OpFOrdLessThan
                   elif signed: OpSLessThan else: OpULessThan)
  of "<=": result = (if isFloat: OpFOrdLessThanEqual
                    elif signed: OpSLessThanEqual else: OpULessThanEqual)
  else: result = OpNop

proc cmpInst(ctx: var SpvLangCtx, name: string, operandTy: CirType): SpvOp =
  ## Comparison opcode, the CIRv2 comparison vocabulary is eq/neq/lt/le,
  ## OpNop when the operator is outside the supported set.
  case name
  of "==": result = (if operandTy.kind == ctFloat: OpFOrdEqual
                    elif operandTy.kind == ctBool: OpLogicalEqual
                    else: OpIEqual)
  of "!=": result = (if operandTy.kind == ctFloat: OpFOrdNotEqual
                    elif operandTy.kind == ctBool: OpLogicalNotEqual
                    else: OpINotEqual)
  of "<": result = (if operandTy.kind == ctFloat: OpFOrdLessThan
                   elif operandTy.kind == ctInt: OpSLessThan else: OpULessThan)
  of "<=": result = (if operandTy.kind == ctFloat: OpFOrdLessThanEqual
                    elif operandTy.kind == ctInt: OpSLessThanEqual
                    else: OpULessThanEqual)
  else: result = OpNop

proc chainValueRoot(ctx: var SpvLangCtx, fn: var SpvFnCtx, root: CirExpr): tuple[ptrId: uint32, isPush: bool] =
  ## Pointer of an addressable dot-chain root.
  ##
  ## - a local or a Function-storage pointer param is addressable
  ## - a hoisted param value is not, push constants are read-only and bound
  ##   as whole values
  case root.kind
  of eSym:
    if fn.locals.hasKey(root.name):
      return (fn.locals[root.name], false)
    if fn.addrParams.hasKey(root.name):
      return (fn.addrParams[root.name], false)
    ctx.failSpv("member chain root is not an addressable local: " & root.name)
    (0, false)
  else:
    ctx.failSpv("member chain root node " & $root.kind)
    (0, false)

proc emitExpr(ctx: var SpvLangCtx, fn: var SpvFnCtx, e: CirExpr): uint32 =
  ## One expression, the SPIR-V id of its value.
  if ctx.err.len > 0:
    return 0
  case e.kind
  of eNil:
    let t = ctx.tyId(ctx.valueTy(e.nilType))
    ctx.builder.freshInst(OpUndef, @[t])
  of eIntLit, eFloatLit, eTrue, eFalse:
    ctx.constId(fn, e)
  of eSym:
    if e.coordBuiltin != cbkNone:
      if e.coordBuiltin == cbkThreadIndexInThreadgroup:
        return ctx.loadScalarBuiltin(biLocalInvocationIndex, e.typ)
      ctx.failSpv("whole-vector use of coordinate builtin " & e.name &
        ", index a component instead")
      return 0
    if fn.consts.hasKey(e.name):
      return fn.consts[e.name]
    if fn.params.hasKey(e.name):
      return fn.params[e.name]
    if fn.bdaParams.hasKey(e.name):
      return fn.bdaParams[e.name]
    if fn.locals.hasKey(e.name):
      let t = fn.localTys.getOrDefault(e.name, ctx.resolveExprType(fn, e))
      return ctx.builder.opLoad(ctx.tyId(ctx.valueTy(t)), fn.locals[e.name])
    if fn.ptrTargets.hasKey(e.name):
      # a pointer-typed local, its value is the pointee variable's address
      return fn.ptrTargets[e.name]
    if fn.addrParams.hasKey(e.name):
      let t = ctx.resolveExprType(fn, e)
      return ctx.builder.opLoad(ctx.tyId(ctx.valueTy(t)), fn.addrParams[e.name])
    ctx.failSpv("unknown symbol " & e.name & " in " & fn.name)
    0
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    let opTy = e.opType
    let l = ctx.emitExpr(fn, e.left)
    let r = ctx.emitExpr(fn, e.right)
    let li = ctx.convertValue(l, ctx.resolveExprType(fn, e.left), opTy)
    let ri = ctx.convertValue(r, ctx.resolveExprType(fn, e.right), opTy)
    let opName = case e.kind
      of eAdd: "+"
      of eSub: "-"
      of eMul: "*"
      of eDiv: "/"
      of eMod: "%"
      of eShl: "<<"
      of eShr: ">>"
      of eBitAnd: "&"
      of eBitOr: "|"
      else: "xor"
    let inst = ctx.binopInst(opName, opTy)
    if inst == OpNop:
      ctx.failSpv("binary operator " & opName)
      return 0
    ctx.builder.freshInst(inst, @[ctx.tyId(ctx.valueTy(opTy)), li, ri])
  of eEq, eNeq, eLt, eLe:
    let opTy = e.cmpType
    let l = ctx.emitExpr(fn, e.a)
    let r = ctx.emitExpr(fn, e.b)
    let li = ctx.convertValue(l, ctx.resolveExprType(fn, e.a), opTy)
    let ri = ctx.convertValue(r, ctx.resolveExprType(fn, e.b), opTy)
    let name = case e.kind
      of eEq: "=="
      of eNeq: "!="
      of eLt: "<"
      else: "<="
    let inst = ctx.cmpInst(name, opTy)
    if inst == OpNop:
      ctx.failSpv("comparison operator " & name)
      return 0
    ctx.builder.freshInst(inst,
      @[ctx.tyId(spvScalar(CirType(kind: ctBool))), li, ri])
  of eNot:
    let v = ctx.emitExpr(fn, e.notOperand)
    ctx.builder.freshInst(OpLogicalNot,
      @[ctx.tyId(spvScalar(CirType(kind: ctBool))), v])
  of eNeg:
    let t = ctx.resolveExprType(fn, e.unOperand)
    let v = ctx.emitExpr(fn, e.unOperand)
    let inst = (if t.kind == ctFloat: OpFNegate else: OpSNegate)
    ctx.builder.freshInst(inst, @[ctx.tyId(ctx.valueTy(t)), v])
  of eBitNot:
    let t = ctx.resolveExprType(fn, e.unOperand)
    let v = ctx.emitExpr(fn, e.unOperand)
    ctx.builder.freshInst(OpNot, @[ctx.tyId(ctx.valueTy(t)), v])
  of eAnd:
    let l = ctx.emitExpr(fn, e.andLeft)
    let r = ctx.emitExpr(fn, e.andRight)
    ctx.builder.freshInst(OpLogicalAnd,
      @[ctx.tyId(spvScalar(CirType(kind: ctBool))), l, r])
  of eOr:
    let l = ctx.emitExpr(fn, e.andLeft)
    let r = ctx.emitExpr(fn, e.andRight)
    ctx.builder.freshInst(OpLogicalOr,
      @[ctx.tyId(spvScalar(CirType(kind: ctBool))), l, r])
  of eCast:
    let srcTy = ctx.resolveExprType(fn, e.castOperand)
    doAssert scalarBits(srcTy) == scalarBits(e.castType), "cast of differing bit widths"
    let v = ctx.emitExpr(fn, e.castOperand)
    ctx.builder.freshInst(OpBitcast, @[ctx.tyId(ctx.valueTy(e.castType)), v])
  of eConv:
    let srcTy = ctx.resolveExprType(fn, e.castOperand)
    let v = ctx.emitExpr(fn, e.castOperand)
    ctx.convertValue(v, srcTy, e.castType)
  of eAddr:
    # an addressable expression's address is its pointer, no instruction
    let (ptrId, _, _, _) = ctx.emitLhsPtr(fn, e.addrOperand)
    ptrId
  of eSizeof:
    let l = std430Layout(ctx.valueTy(e.ofType))
    ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(l.size))
  of eDeref:
    # load through a pointer value: the operand evaluates to a pointer
    let ptrVal = ctx.emitExpr(fn, e.derefOperand)
    let pt = ctx.resolveExprType(fn, e)
    ctx.builder.opLoad(ctx.tyId(ctx.valueTy(pt)), ptrVal)
  of eAt, ePat:
    let (ptrId, pointee, isBda, _) = ctx.emitLhsPtr(fn, e)
    if isBda:
      let elemAlign = uint32(std430Layout(ctx.valueTy(pointee)).align)
      ctx.builder.opLoadAligned(ctx.tyId(ctx.valueTy(pointee)), ptrId, elemAlign)
    else:
      ctx.builder.opLoad(ctx.tyId(ctx.valueTy(pointee)), ptrId)
  of eDot:
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone and
        e.dotBase.coordBuiltin != cbkThreadIndexInThreadgroup:
      # a coordinate-builtin component, read from the builtin input variable
      let comp =
        case e.fieldName
        of "x": 0
        of "y": 1
        of "z": 2
        else:
          ctx.failSpv("coordinate component " & e.fieldName)
          -1
      if comp < 0:
        return 0
      let compTy = ctx.resolveExprType(fn, e)
      return ctx.loadBuiltinComponent(ctx.coordBuiltinInput(e.dotBase.coordBuiltin),
        comp, compTy)
    let baseTy = ctx.resolveExprType(fn, e.dotBase)
    if baseTy != nil and baseTy.kind == ctVector:
      let i = memberIndex(baseTy, e.fieldName)
      if i < 0:
        ctx.failSpv("no component " & e.fieldName & " on the vector")
        return 0
      let v = ctx.emitExpr(fn, e.dotBase)
      ctx.builder.opCompositeExtract(ctx.tyId(ctx.valueTy(baseTy.vElem)), v, @[i])
    elif baseTy != nil and baseTy.kind == ctStruct:
      let i = memberIndex(baseTy, e.fieldName)
      if i < 0:
        ctx.failSpv("no field " & e.fieldName & " on " & baseTy.name)
        return 0
      # dot base is a value: composite extract, no storage
      let v = ctx.emitExpr(fn, e.dotBase)
      ctx.builder.opCompositeExtract(ctx.tyId(ctx.valueTy(baseTy.fields[i].typ)),
        v, @[i])
    else:
      # dot base is a pointer-valued symbol: access chain through it
      if e.dotBase.kind != eSym:
        ctx.failSpv("field access base outside the supported set: " & $e.dotBase.kind)
        return 0
      let rootTy = ctx.resolveExprType(fn, e.dotBase)
      if rootTy == nil or rootTy.kind != ctPtr:
        ctx.failSpv("field access through a non-pointer: " & e.dotBase.name)
        return 0
      let i = memberIndex(rootTy.pElem, e.fieldName)
      if i < 0:
        ctx.failSpv("no field " & e.fieldName & " on " & rootTy.pElem.name)
        return 0
      let ptrVal = ctx.emitExpr(fn, e.dotBase)
      let memberPtrTy = ctx.builder.tyPointer(scFunction,
        ctx.tyId(ctx.valueTy(rootTy.pElem.fields[i].typ)))
      let idx0 = ctx.builder.constUint(ctx.scalarId(cirUint32()), 0)
      let memberPtr = ctx.builder.opAccessChain(memberPtrTy, ptrVal, @[idx0])
      ctx.builder.opLoad(ctx.tyId(ctx.valueTy(rootTy.pElem.fields[i].typ)),
        memberPtr)
  of eOconstr:
    let structTy = ctx.valueTy(e.ocType)
    let structId = ctx.tyId(structTy)
    var members: seq[uint32]
    for f in e.kvFields:
      let fi = memberIndex(e.ocType, f.name)
      doAssert fi >= 0, "constructor field " & f.name & " not on " & e.ocType.name
      let v = ctx.emitExpr(fn, f.value)
      let ft = ctx.resolveExprType(fn, f.value)
      let target =
        if e.ocType.kind == ctVector: e.ocType.vElem
        else: e.ocType.fields[fi].typ
      if isSpvScalarKind(target):
        members.add ctx.convertValue(v, ft, target)
      else:
        members.add v
    ctx.builder.freshInst(OpCompositeConstruct, @[structId] & members)
  of eAconstr:
    let arrTy = ctx.tyId(ctx.valueTy(e.acType))
    var vals: seq[uint32]
    for el in e.elems:
      let v = ctx.emitExpr(fn, el)
      vals.add v
    ctx.builder.freshInst(OpCompositeConstruct, @[arrTy] & vals)
  of eCall:
    # a call binds to its destination as an asgn right-hand side
    # value emission is the call itself
    ctx.emitCallValue(fn, e)

proc emitLhsPtr(ctx: var SpvLangCtx, fn: var SpvFnCtx, e: CirExpr):
    tuple[ptrId: uint32, pointee: CirType, isBda: bool, isPush: bool] =
  ## Pointer of an addressable expression, the OpStore/OpAccessChain target.
  if ctx.err.len > 0:
    return
  case e.kind
  of eSym:
    if fn.locals.hasKey(e.name):
      # stored value's type is the variable's member type
      # a pointer-typed local stores pointer values
      let t = fn.localTys.getOrDefault(e.name, ctx.resolveExprType(fn, e))
      return (fn.locals[e.name], t, false, false)
    if fn.addrParams.hasKey(e.name):
      return (fn.addrParams[e.name], fn.addrPointee[e.name], false, false)
    ctx.failSpv("assignment target is not a local: " & e.name)
  of eDeref:
    # store through a pointer value held in an expression
    let ptrVal = ctx.emitExpr(fn, e.derefOperand)
    let pt = ctx.resolveExprType(fn, e)
    return (ptrVal, pt, false, false)
  of eDot:
    # field write on a local struct: access chain, member write
    if e.dotBase.kind != eSym or not fn.locals.hasKey(e.dotBase.name):
      ctx.failSpv("field write outside a local struct")
      return
    let rootTy = ctx.resolveExprType(fn, e.dotBase)
    if rootTy == nil or rootTy.kind != ctStruct:
      ctx.failSpv("field write on a non-struct local")
      return
    let i = memberIndex(rootTy, e.fieldName)
    if i < 0:
      ctx.failSpv("no field " & e.fieldName & " on " & rootTy.name)
      return
    let memberPtrTy = ctx.builder.tyPointer(scFunction,
      ctx.tyId(ctx.valueTy(rootTy.fields[i].typ)))
    let idx0 = ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(i))
    let ptrId = ctx.builder.opAccessChain(memberPtrTy, fn.locals[e.dotBase.name],
      @[idx0])
    result = (ptrId, rootTy.fields[i].typ, false, false)
  of ePat:
    # element write through a pointer value: the base evaluates to a pointer
    let baseTy = ctx.resolveExprType(fn, e.patBase)
    if baseTy == nil or baseTy.kind notin {ctPtr, ctAptr}:
      ctx.failSpv("pattern write base is not a pointer")
      return
    if baseTy.kind == ctAptr:
      if baseTy.space != caDevice:
        ctx.failSpv("pattern write through a " & $baseTy.space & " array pointer")
        return
      let idxVal = ctx.emitExpr(fn, e.patIndex)
      let idxInt = ctx.convertValue(idxVal, ctx.resolveExprType(fn, e.patIndex), cirInt32())
      let elemTy = baseTy.pElem
      let elemPtrTy = ctx.builder.tyPointer(scPhysicalStorageBuffer,
        ctx.tyId(ctx.valueTy(elemTy)))
      let idx0 = ctx.builder.constUint(ctx.scalarId(cirUint32()), 0)
      let ptrId = ctx.builder.opAccessChain(elemPtrTy,
        ctx.emitExpr(fn, e.patBase), @[idx0, idxInt])
      return (ptrId, elemTy, true, false)
    else:
      # a pointer to array in Function storage: the parameter names the array
      let idxVal = ctx.emitExpr(fn, e.patIndex)
      let idxInt = ctx.convertValue(idxVal, ctx.resolveExprType(fn, e.patIndex), cirInt32())
      let elemTy = baseTy.pElem
      let elemPtrTy = ctx.builder.tyPointer(scFunction,
        ctx.tyId(ctx.valueTy(elemTy)))
      let ptrId = ctx.builder.opAccessChain(elemPtrTy,
        ctx.emitExpr(fn, e.patBase), @[idxInt])
      return (ptrId, elemTy, false, false)
  of eAt:
    # base chain rooted at a local: a plain local or the struct behind
    # a pointer-typed local: the access chain runs on the variable directly
    var rootVar = 0'u32
    var memberIdxs: seq[int]
    var chainOk = true
    var walk = e.atBase
    while chainOk:
      case walk.kind
      of eSym:
        if fn.locals.hasKey(walk.name):
          rootVar = fn.locals[walk.name]
        else:
          chainOk = false
        break
      of eDeref:
        if walk.derefOperand.kind == eSym and fn.locals.hasKey(walk.derefOperand.name):
          rootVar = fn.locals[walk.derefOperand.name]
          break
        if walk.derefOperand.kind == eSym and fn.ptrTargets.hasKey(walk.derefOperand.name):
          rootVar = fn.ptrTargets[walk.derefOperand.name]
          break
        chainOk = false
      of eDot:
        let innerTy = ctx.resolveExprType(fn, walk.dotBase)
        if innerTy == nil or innerTy.kind != ctStruct:
          chainOk = false
          break
        let mi = memberIndex(innerTy, walk.fieldName)
        if mi < 0:
          chainOk = false
          break
        memberIdxs.insert(mi, 0)
        walk = walk.dotBase
      else:
        chainOk = false
    let arrTy = ctx.resolveExprType(fn, e.atBase)
    if chainOk and arrTy != nil and arrTy.kind == ctArray:
      let idxVal = ctx.emitExpr(fn, e.atIndex)
      let idxInt = ctx.convertValue(idxVal, ctx.resolveExprType(fn, e.atIndex), cirInt32())
      let elemPtrTy = ctx.builder.tyPointer(scFunction,
        ctx.tyId(ctx.valueTy(arrTy.aElem)))
      var idxs: seq[uint32]
      for mi in memberIdxs:
        idxs.add ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(mi))
      idxs.add idxInt
      let ptrId = ctx.builder.opAccessChain(elemPtrTy, rootVar, idxs)
      result = (ptrId, arrTy.aElem, false, false)
      return
    let baseTy = ctx.resolveExprType(fn, e.atBase)
    if baseTy == nil or baseTy.kind notin {ctArray, ctAptr, ctPtr}:
      ctx.failSpv("array element access on a non-array base")
      return
    let idxVal = ctx.emitExpr(fn, e.atIndex)
    let idxInt = ctx.convertValue(idxVal, ctx.resolveExprType(fn, e.atIndex), cirInt32())
    let elemTy =
      if baseTy.kind == ctArray: baseTy.aElem
      else: baseTy.pElem
    let elemPtrTy = ctx.builder.tyPointer(scFunction,
      ctx.tyId(ctx.valueTy(elemTy)))
    if baseTy.kind == ctArray:
      if e.atBase.kind != eSym or not fn.locals.hasKey(e.atBase.name):
        ctx.failSpv("array element write outside a local array")
        return
      let ptrId = ctx.builder.opAccessChain(elemPtrTy, fn.locals[e.atBase.name],
        @[idxInt])
      result = (ptrId, elemTy, false, false)
    elif baseTy.kind == ctAptr:
      if baseTy.space != caDevice:
        ctx.failSpv("array element access through a " & $baseTy.space & " pointer")
        return
      let devElemPtrTy = ctx.builder.tyPointer(scPhysicalStorageBuffer,
        ctx.tyId(ctx.valueTy(elemTy)))
      let idx0 = ctx.builder.constUint(ctx.scalarId(cirUint32()), 0)
      let basePtr = ctx.emitExpr(fn, e.atBase)
      let ptrId = ctx.builder.opAccessChain(devElemPtrTy, basePtr, @[idx0, idxInt])
      result = (ptrId, elemTy, true, false)
    else:
      # a pointer to the element storage in Function scope, index directly
      let basePtr = ctx.emitExpr(fn, e.atBase)
      let ptrId = ctx.builder.opAccessChain(elemPtrTy, basePtr, @[idxInt])
      result = (ptrId, elemTy, false, false)
  else:
    ctx.failSpv("assignment target " & $e.kind)

proc emitCallValue(ctx: var SpvLangCtx, fn: var SpvFnCtx, e: CirExpr): uint32 =
  ## Value of a call: builtin vocabulary maps to native instructions
  ## and a device function emits an OpFunctionCall.
  if ctx.err.len > 0:
    return 0
  doAssert e.callee.kind == eSym, "call callee is not a symbol"
  let name = e.callee.name
  # builtin vocabulary dispatches on the typed kinds, never on names,
  # `max`/`min` stay a name case, harvested magics outside the four categories
  if e.callee.reductionBuiltin == rbkReduceAdd:
    doAssert e.args.len == 1, "reduceAdd takes one argument"
    let v = ctx.emitExpr(fn, e.args[0])
    let resTy = ctx.resolveExprType(fn, e.args[0])
    if resTy.kind != ctFloat or resTy.bits != 32:
      ctx.failSpv("reduceAdd lowers f32 only")
      return 0
    ctx.caps.incl capGroupNonUniformArithmetic
    let scopeSub = ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(scopeSubgroup))
    return ctx.builder.freshInst(OpGroupNonUniformFAdd,
      @[ctx.tyId(spvScalar(resTy)), scopeSub, groupOpReduce, v])
  if e.callee.kind == eSym and e.callee.builtin and
      e.callee.name in ["simd_shuffle", "simd_shuffle_xor"]:
    doAssert e.args.len == 2, "simd_shuffle takes a value and a lane index"
    let v = ctx.emitExpr(fn, e.args[0])
    let lane = ctx.emitExpr(fn, e.args[1])
    let resTy = ctx.resolveExprType(fn, e.args[0])
    if resTy == nil or resTy.kind notin {ctInt, ctUint, ctFloat, ctBool}:
      ctx.failSpv("simd_shuffle lowers scalars only")
      return 0
    # Invocation Id and Mask take the unsigned width the grammar demands
    let laneU = ctx.convertValue(lane, ctx.resolveExprType(fn, e.args[1]), cirUint32())
    ctx.caps.incl capGroupNonUniformShuffle
    let scopeSub = ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(scopeSubgroup))
    let op = if e.callee.name == "simd_shuffle": OpGroupNonUniformShuffle
             else: OpGroupNonUniformShuffleXor
    return ctx.builder.freshInst(op, @[ctx.tyId(spvScalar(resTy)), scopeSub, v, laneU])
  if e.callee.atomicsBuiltin == abkAtomicFetchAdd:
    doAssert e.args.len == 2, "atomicFetchAdd takes a pointer and a value"
    if e.args[0].kind != eAddr or e.args[0].addrOperand.kind != ePat:
      ctx.failSpv("atomicFetchAdd address is not the address of an element")
      return 0
    let pat = e.args[0].addrOperand
    let baseTy = ctx.resolveExprType(fn, pat.patBase)
    if baseTy == nil or baseTy.kind != ctAptr or baseTy.space != caDevice:
      ctx.failSpv("atomicFetchAdd lowers through a device buffer element only")
      return 0
    let elemTy = baseTy.pElem
    if elemTy.kind != ctFloat or elemTy.bits != 32:
      ctx.failSpv("atomicFetchAdd lowers f32 only")
      return 0
    let (ptrId, _, _, _) = ctx.emitLhsPtr(fn, pat)
    let v = ctx.emitExpr(fn, e.args[1])
    let value = ctx.convertValue(v, ctx.resolveExprType(fn, e.args[1]), elemTy)
    ctx.caps.incl capAtomicFloat32AddEXT
    if "SPV_EXT_shader_atomic_float_add" notin ctx.exts:
      ctx.exts.add "SPV_EXT_shader_atomic_float_add"
    let scopeDev = ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(scopeDevice))
    let sem = ctx.builder.constUint(ctx.scalarId(cirUint32()),
      uint64(semUniformMemory or semAcquireRelease))
    return ctx.builder.freshInst(OpAtomicFAddEXT,
      @[ctx.tyId(spvScalar(elemTy)), ptrId, scopeDev, sem, value])
  case name
  of "max", "min":
    doAssert e.args.len == 2, name & " takes two arguments"
    let opTy = ctx.resolveExprType(fn, e.args[0])
    let a = ctx.convertValue(ctx.emitExpr(fn, e.args[0]),
      ctx.resolveExprType(fn, e.args[0]), opTy)
    let b = ctx.convertValue(ctx.emitExpr(fn, e.args[1]),
      ctx.resolveExprType(fn, e.args[1]), opTy)
    let signed = opTy.kind == ctInt
    let isFloat = opTy.kind == ctFloat
    let cmpOp = if isFloat:
                  (if name == "max": OpFOrdGreaterThan else: OpFOrdLessThan)
                elif name == "max":
                  (if signed: OpSGreaterThan else: OpUGreaterThan)
                else:
                  (if signed: OpSLessThan else: OpULessThan)
    let boolTy = ctx.tyId(spvScalar(CirType(kind: ctBool)))
    let cond = ctx.builder.freshInst(cmpOp, @[boolTy, a, b])
    return ctx.builder.opSelect(ctx.tyId(spvScalar(opTy)), cond, a, b)
  if e.callee.synchroBuiltin == sbkThreadgroupBarrier:
    let scopeDev = ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(scopeDevice))
    let sem = ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(semAcquireRelease))
    discard ctx.builder.freshInst(OpControlBarrier, @[scopeDev, scopeDev, sem])
    return 0
  # a device function call
  if not ctx.emittedFns.hasKey(name):
    ctx.failSpv("callee " & name & " was not pre-emitted")
    return 0
  let fnId = ctx.emittedFns[name]
  let retTy = ctx.fnTypes[name]
  var args: seq[uint32]
  for a in e.args:
    args.add ctx.emitExpr(fn, a)
  return ctx.builder.freshInst(OpFunctionCall,
    @[ctx.tyId(ctx.valueTy(retTy)), fnId] & args)

proc emitStmts(ctx: var SpvLangCtx, fn: var SpvFnCtx, ss: seq[CirStmt])

proc freshBlock(ctx: var SpvLangCtx, fn: var SpvFnCtx, name: string): uint32 =
  ## A new basic block for a label, branching from the current block when open.
  let lbl = ctx.builder.module.allocId()
  if not fn.terminated:
    ctx.builder.opBranch(lbl)
  ctx.builder.opLabelAt(lbl)
  fn.terminated = false
  fn.labels[name] = lbl
  lbl

proc emitStmts(ctx: var SpvLangCtx, fn: var SpvFnCtx, ss: seq[CirStmt]) =
  ## One statement list into the current basic block.
  for s in ss:
    if ctx.err.len > 0:
      return
    case s.kind
    of sVar:
      # storage hoisted to the entry block, the initializer stores here
      if s.init != nil and s.init.kind != eNil:
        let v = ctx.emitExpr(fn, s.init)
        let t = ctx.resolveExprType(fn, s.init)
        let target = fn.localTys[s.name]
        if t != nil:
          let value = ctx.convertValue(v, t, target)
          ctx.builder.opStore(fn.locals[s.name], value)
        else:
          ctx.builder.opStore(fn.locals[s.name], v)
    of sConst:
      # a compile-time literal binding, reads use the constant id
      if s.init != nil and s.init.kind in {eIntLit, eFloatLit, eTrue, eFalse}:
        fn.consts[s.name] = ctx.constId(fn, s.init)
      else:
        ctx.failSpv("const " & s.name & " initializer outside the literal set")
    of sAsgn:
      if s.lhs.kind == eSym:
        let lhsTy = fn.localTys.getOrDefault(s.lhs.name, s.lhs.typ)
        if lhsTy != nil and lhsTy.kind == ctPtr and
            not fn.ptrTargets.hasKey(s.lhs.name):
          # a pointer-typed local binds the address of its pointee
          # pointee is a hoisted local, binding is symbolic
          if s.rhs.kind != eAddr or s.rhs.addrOperand.kind != eSym or
              not fn.locals.hasKey(s.rhs.addrOperand.name):
            ctx.failSpv("pointer local " & s.lhs.name &
              " binds an address outside a local")
            return
          fn.ptrTargets[s.lhs.name] = fn.locals[s.rhs.addrOperand.name]
          continue
      let rhs = ctx.emitExpr(fn, s.rhs)
      let (ptrId, pointee, isBda, isPush) = ctx.emitLhsPtr(fn, s.lhs)
      if isPush:
        ctx.failSpv("assignment through push-constant storage, push constants are read-only")
        return
      let rhsTy = ctx.resolveExprType(fn, s.rhs)
      let value = ctx.convertValue(rhs, rhsTy, pointee)
      if isBda:
        let elemAlign = uint32(std430Layout(ctx.valueTy(pointee)).align)
        ctx.builder.opStoreAligned(ptrId, value, elemAlign)
      else:
        ctx.builder.opStore(ptrId, value)
    of sIte:
      let cond = ctx.emitExpr(fn, s.cond)
      let mergeLbl = ctx.builder.freshLabel()
      let thenLbl = ctx.builder.freshLabel()
      let elseLbl = ctx.builder.freshLabel()
      ctx.builder.opSelectionMerge(mergeLbl, selNone)
      ctx.builder.opBranchConditional(cond, thenLbl, elseLbl)
      ctx.builder.opLabelAt(thenLbl)
      fn.terminated = false
      ctx.emitStmts(fn, s.thenBody)
      if not fn.terminated:
        ctx.builder.opBranch(mergeLbl)
      ctx.builder.opLabelAt(elseLbl)
      fn.terminated = false
      ctx.emitStmts(fn, s.elseBody)
      if not fn.terminated:
        ctx.builder.opBranch(mergeLbl)
      ctx.builder.opLabelAt(mergeLbl)
      fn.terminated = false
    of sLoop:
      # loop body break targets the merge block, the exit label after the loop
      # aliases it and the continue target ends the body
      var localLabs: HashSet[string]
      proc scanLabs(ss: seq[CirStmt]) =
        for s in ss:
          case s.kind
          of sLab: localLabs.incl s.labName
          of sIte: scanLabs(s.thenBody); scanLabs(s.elseBody)
          of sLoop: scanLabs(s.loopBody)
          of sScope: scanLabs(s.scopeBody)
          else: discard
      scanLabs(s.loopBody)
      var exitName = ""
      proc scanJmps(ss: seq[CirStmt]) =
        for s in ss:
          case s.kind
          of sJmp:
            if s.jmpTarget notin localLabs and exitName == "":
              exitName = s.jmpTarget
          of sIte: scanJmps(s.thenBody); scanJmps(s.elseBody)
          of sLoop: scanJmps(s.loopBody)
          of sScope: scanJmps(s.scopeBody)
          else: discard
      scanJmps(s.loopBody)
      let headerLbl = ctx.builder.freshLabel()
      let bodyLbl = ctx.builder.freshLabel()
      let contLbl = ctx.builder.freshLabel()
      let mergeLbl = ctx.builder.freshLabel()
      let savedMerge = fn.loopMerge
      let savedCont = fn.loopCont
      let savedExit = fn.loopExit
      ctx.builder.opBranch(headerLbl)
      ctx.builder.opLabelAt(headerLbl)
      fn.terminated = false
      ctx.builder.opLoopMerge(mergeLbl, contLbl, if s.unroll: lcUnroll else: lcNone)
      ctx.builder.opBranch(bodyLbl)
      ctx.builder.opLabelAt(bodyLbl)
      fn.terminated = false
      fn.loopMerge = mergeLbl
      fn.loopCont = contLbl
      fn.loopExit = exitName
      ctx.emitStmts(fn, s.loopBody)
      fn.loopMerge = savedMerge
      fn.loopCont = savedCont
      fn.loopExit = savedExit
      if fn.terminated:
        ctx.failSpv("return inside a loop body, the continue target would be unreachable")
        return
      ctx.builder.opBranch(contLbl)
      ctx.builder.opLabelAt(contLbl)
      fn.terminated = false
      ctx.builder.opBranch(headerLbl)
      ctx.builder.opLabelAt(mergeLbl)
      fn.terminated = false
    of sLab:
      # a label the control flow reaches by falling into it or by a jmp
      if fn.labels.hasKey(s.labName):
        # forward jmp target: the block was created at the jmp site
        let lbl = fn.labels[s.labName]
        if not fn.terminated:
          ctx.builder.opBranch(lbl)
        ctx.builder.opLabelAt(lbl)
        fn.terminated = false
      elif s.labName == fn.loopExit:
        # an exit label after a loop: the merge block is this block
        discard
      else:
        discard ctx.freshBlock(fn, s.labName)
    of sJmp:
      if s.jmpTarget == fn.loopExit:
        ctx.builder.opBranch(fn.loopMerge)
        fn.terminated = true
      elif fn.labels.hasKey(s.jmpTarget):
        ctx.builder.opBranch(fn.labels[s.jmpTarget])
        fn.terminated = true
      else:
        # forward jump, the label block is created now and filled at the lab
        let lbl = ctx.builder.module.allocId()
        fn.labels[s.jmpTarget] = lbl
        ctx.builder.opBranch(lbl)
        fn.terminated = true
    of sScope:
      ctx.emitStmts(fn, s.scopeBody)
    of sAssume:
      # an optimizer hint, no instruction
      discard
    of sRet:
      if fn.terminated:
        ctx.failSpv("a second return in one basic block")
        return
      if s.retVal != nil and s.retVal.kind != eNil:
        let v = ctx.emitExpr(fn, s.retVal)
        ctx.builder.opReturnValue(v)
      else:
        ctx.builder.opReturn()
      fn.terminated = true
    of sDiscard:
      discard
    of sCall:
      # statement-position call: void calls emit, value calls compute
      # into a fresh temp the caller stores (the frontend lowers both shapes)
      if s.callCallee.kind == eSym and ctx.fnTypes.hasKey(s.callCallee.name) and
          ctx.fnTypes[s.callCallee.name].kind == ctVoid:
        var callE = CirExpr(kind: eCall)
        callE.callee = s.callCallee
        callE.args = s.callArgs
        discard ctx.emitCallValue(fn, callE)
      else:
        ctx.failSpv("statement call " & s.callCallee.name &
          " without a destination binding")
    of sCase, sAsm, sEmit:
      ctx.failSpv("statement construct " & $s.kind &
        " outside the SPIR-V subset, it must lower earlier")

# ═══════════════════════════════════════════════════════════════════════
#  Function and kernel emission
# ═══════════════════════════════════════════════════════════════════════

proc scanCallExpr(e: CirExpr, calls: var seq[string]) =
  ## Callee names of one expression tree, nested calls included.
  if e.isNil:
    return
  case e.kind
  of eCall:
    if e.callee.kind == eSym:
      calls.add e.callee.name
    for a in e.args:
      scanCallExpr(a, calls)
  of eNil, eTrue, eFalse, eIntLit, eFloatLit, eSizeof:
    discard
  of eSym:
    discard
  of eDeref: scanCallExpr(e.derefOperand, calls)
  of eAt: scanCallExpr(e.atBase, calls); scanCallExpr(e.atIndex, calls)
  of ePat: scanCallExpr(e.patBase, calls); scanCallExpr(e.patIndex, calls)
  of eDot: scanCallExpr(e.dotBase, calls)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    scanCallExpr(e.left, calls); scanCallExpr(e.right, calls)
  of eBitNot, eNeg: scanCallExpr(e.unOperand, calls)
  of eEq, eNeq, eLt, eLe: scanCallExpr(e.a, calls); scanCallExpr(e.b, calls)
  of eNot: scanCallExpr(e.notOperand, calls)
  of eAnd, eOr: scanCallExpr(e.andLeft, calls); scanCallExpr(e.andRight, calls)
  of eCast, eConv: scanCallExpr(e.castOperand, calls)
  of eAddr: scanCallExpr(e.addrOperand, calls)
  of eOconstr:
    for f in e.kvFields:
      scanCallExpr(f.value, calls)
  of eAconstr:
    for el in e.elems:
      scanCallExpr(el, calls)

proc scanCalls(ss: seq[CirStmt], calls: var seq[string]) =
  ## Callee names of one statement list, nested bodies and expressions included.
  for s in ss:
    case s.kind
    of sCall:
      if s.callCallee.kind == eSym:
        calls.add s.callCallee.name
      for a in s.callArgs:
        scanCallExpr(a, calls)
    of sAsgn:
      scanCallExpr(s.lhs, calls)
      scanCallExpr(s.rhs, calls)
    of sVar, sConst:
      if s.init != nil:
        scanCallExpr(s.init, calls)
    of sIte:
      scanCallExpr(s.cond, calls)
      scanCalls(s.thenBody, calls)
      scanCalls(s.elseBody, calls)
    of sLoop:
      scanCalls(s.loopBody, calls)
    of sScope:
      scanCalls(s.scopeBody, calls)
    of sRet:
      if s.retVal != nil:
        scanCallExpr(s.retVal, calls)
    of sAssume:
      scanCallExpr(s.assumeCond, calls)
    else:
      discard


proc emitDeviceFn(ctx: var SpvLangCtx, procNode: CirProc): uint32 =
  ## One device function.
  ##
  ## Pointer parameters are Function-storage pointers and value parameters
  ## pass by value.
  let retTyId =
    if procNode.ret.kind == ctVoid: ctx.builder.tyVoid()
    else: ctx.tyId(ctx.valueTy(procNode.ret))
  var paramTyIds: seq[uint32]
  for p in procNode.params:
    if p.typ.kind == ctPtr:
      # a var parameter: the caller passes the address of a local
      paramTyIds.add ctx.builder.tyPointer(scFunction, ctx.tyId(ctx.valueTy(p.typ.pElem)))
    else:
      paramTyIds.add ctx.tyId(ctx.valueTy(p.typ))
  let fnTy = ctx.builder.tyFunction(retTyId, paramTyIds)
  result = ctx.builder.opFunction(retTyId, fnTy)
  var fn: SpvFnCtx = SpvFnCtx(name: "device fn " & procNode.name)
  for i, p in procNode.params:
    let pid = ctx.builder.opFunctionParameter(paramTyIds[i])
    if p.typ.kind == ctPtr:
      fn.addrParams[p.name] = pid
      fn.addrPointee[p.name] = p.typ.pElem
    else:
      fn.params[p.name] = pid
  discard ctx.builder.opLabel()
  ctx.hoistLocals(fn, procNode.body)
  ctx.emitStmts(fn, procNode.body)
  if not fn.terminated:
    ctx.builder.opReturn()
  ctx.builder.opFunctionEnd()

proc emitDeviceFnRec(ctx: var SpvLangCtx, procNode: CirProc): uint32 =
  ## One device function with its transitive callees emitted first.
  if ctx.emittedFns.hasKey(procNode.name):
    return ctx.emittedFns[procNode.name]
  doAssert procNode.name notin ctx.emitting,
    "recursive device function calls are outside the SPIR-V subset"
  ctx.emitting.incl procNode.name
  ctx.fnTypes[procNode.name] = procNode.ret
  var calls: seq[string]
  scanCalls(procNode.body, calls)
  var seen: HashSet[string]
  for c in calls:
    if ctx.fnProcs.hasKey(c) and c notin seen:
      seen.incl c
      discard ctx.emitDeviceFnRec(ctx.fnProcs[c])
  result = ctx.emitDeviceFn(procNode)
  ctx.emittedFns[procNode.name] = result
  ctx.emitting.excl procNode.name

proc emitKernel(ctx: var SpvLangCtx, kernel: CirProc): uint32 =
  ## Compute entry point.
  ##
  ## One push block in parameter order, the builtin inputs on the interface
  ## and structured control flow from the CIRv2 body.
  doAssert kernel.ret.kind == ctVoid, "kernels are void"
  let push = ctx.buildPushBlock(kernel.params)
  # device fns emit first, an OpFunctionCall result id must exist at the call site
  var calls: seq[string]
  scanCalls(kernel.body, calls)
  var seen: HashSet[string]
  for c in calls:
    if ctx.fnProcs.hasKey(c) and c notin seen:
      seen.incl c
      discard ctx.emitDeviceFnRec(ctx.fnProcs[c])
  let voidTy = ctx.builder.tyVoid()
  let fnTy = ctx.builder.tyFunction(voidTy, @[])
  result = ctx.builder.opFunction(voidTy, fnTy)
  var fn: SpvFnCtx = SpvFnCtx(name: "kernel " & kernel.name)
  discard ctx.builder.opLabel()
  # prologue: hoisted locals, then the parameter bindings
  ctx.hoistLocals(fn, kernel.body)
  for i, p in kernel.params:
    if p.typ.kind == ctAptr:
      # u64 device address in the push block, converted to a buffer pointer
      let u64Ty = ctx.tyId(spvScalar(cirUint64()))
      let slotPtrTy = ctx.builder.tyPointer(scPushConstant, u64Ty)
      let memberIdx = ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(i))
      let slot = ctx.builder.opAccessChain(slotPtrTy, push.varId, @[memberIdx])
      let devAddr = ctx.builder.opLoad(u64Ty, slot)
      let pointeeTyId = ctx.bdaPointerTy(p.typ.pElem)
      fn.bdaParams[p.name] = ctx.builder.opConvertUToPtr(pointeeTyId, devAddr)
    else:
      let memberTy = ctx.pushMemberTy(p)
      let slotPtrTy = ctx.builder.tyPointer(scPushConstant, ctx.tyId(memberTy))
      let memberIdx = ctx.builder.constUint(ctx.scalarId(cirUint32()), uint64(i))
      let slot = ctx.builder.opAccessChain(slotPtrTy, push.varId, @[memberIdx])
      fn.params[p.name] = ctx.builder.opLoad(ctx.tyId(memberTy), slot)
  ctx.emitStmts(fn, kernel.body)
  if not fn.terminated:
    ctx.builder.opReturn()
  ctx.builder.opFunctionEnd()
  let usePhys = capPhysicalStorageBufferAddresses in ctx.caps
  if usePhys:
    # PhysicalStorageBufferAddresses is extension-gated on SPIR-V 1.3
    ctx.builder.emitExtension("SPV_KHR_physical_storage_buffer")
  ctx.flushExtensions()
  let lws = if kernel.workgroup == [0, 0, 0]: [256, 1, 1]
            else: kernel.workgroup
  ctx.builder.emitEntryPoint(emGLCompute, result, kernel.name, ctx.interfaceVars)
  ctx.builder.emitExecutionMode(result, modeLocalSize,
    @[uint32(lws[0]), uint32(lws[1]), uint32(lws[2])])
  ctx.builder.emitMemoryModel(
    (if usePhys: amPhysicalStorageBuffer64 else: amLogical), mmGLSL450)
  ctx.flushCapabilities()

# ═══════════════════════════════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════════════════════════════

proc printSpirvModule*(m: CirModule): tuple[words: seq[uint32], err: string] =
  ## Returns the SPIR-V words of one CIRv2 module for the Vulkan environment.
  ##
  ## Exactly one `{.global.}` kernel per module, the device functions it
  ## calls emitted alongside.
  ## A refusal names the construct otherwise.
  var lang = initSpvLangCtx()
  var kernels: seq[CirProc]
  for fn in m.procs:
    if fn.isKernel:
      kernels.add fn
  if kernels.len != 1:
    lang.failSpv("exactly one kernel per module, found " & $kernels.len)
    return (@[], lang.err)
  lang.fnProcs = initTable[string, CirProc]()
  for fn in m.procs:
    if not fn.isKernel:
      lang.fnProcs[fn.name] = fn
  discard lang.emitKernel(kernels[0])
  if lang.err.len > 0:
    return (@[], lang.err)
  (lang.builder.module.finalize(), "")


proc codegen*(ctx: var GpuContext): seq[uint32] =
  ## Encodes the module in `ctx` as SPIR-V words for the Vulkan environment,
  ## the `backends_api` entry point.
  ## An encoder refusal raises, the macro layer turns it into a compile error
  ## naming the construct.
  let (words, err) = printSpirvModule(ctx.module)
  if err.len > 0:
    error(err)
  result = words
