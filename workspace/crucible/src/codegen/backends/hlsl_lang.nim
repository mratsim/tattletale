# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## HLSL emission: one compute kernel per source, device functions emitted natively.
##
## DXC compiles multi-function HLSL, so device functions emit as ordinary
## functions with forward declarations and the kernel emits as `main`.
##
## Emission pipeline (codegen): recursion check → parameter classification → write-scans → declarations → forward declarations → fn definitions → kernel.
##
## Resource mapping, kernels:
## - pointer parameters become `[[vk::binding(N, 0)]]` StructuredBuffer / RWStructuredBuffer resources,
##   numbered in declaration order, read-only vs read-write decided by the write-scan
## - one wrapper struct (`struct WrapX { uint v; }`) per buffer parameter, MoltenVK merges identically typed storage blocks into one MSL struct
## - scalar parameters become `[[vk::push_constant]] ConstantBuffer<Push> push` members, reads render `push.<name>`
## - coordinate builtins join `main`'s signature only when read, mapped to SV_DispatchThreadID / SV_GroupID / SV_GroupIndex
##
## Resource mapping, device functions:
## - pointer parameters become StructuredBuffer / RWStructuredBuffer resource params
##   without binding attributes, the resource identity flows from the call site
## - read-only vs read-write decided by the function's own write-scan, one wrapper struct per element type
## - scalars pass by value, large-struct params bind `inout`
##
## glslang quirks honored here:
## - the entry point is always `main`, matching the `-e main` glslang invocation
## - `[numthreads]` comes from pWorkgroupSize, (0, 0, 0) means 256x1x1
## - brace-initialization is declaration-only: an object constructor assigned to an already-declared local lowers to member-wise assignment
## - only int/uint/float/double/bool have HLSL type names, int64/uint64 are rejected by the printer

import std / [macros, strutils, tables, sets, hashes]

import ../../builtins/builtins_functions
import ../../builtins/builtins_gpu_types
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./lang_utils

# ═══════════════════════════════════════════════════════════════════════
# Type mapping
# ═══════════════════════════════════════════════════════════════════════

proc gpuTypeToString*(t: GpuTypeKind): string =
  case t
  of gtBool: "bool"
  of gtInt32: "int"
  of gtUint32, gtSize_t: "uint" # size_t launches as a 32-bit uint, matching the Vulkan launch ABI
  of gtFloat32: "float"
  of gtFloat64: "double"
  of gtVoid: "void"
  of gtStatic: "int"
  of gtObject, gtGenericInst, gtArray: "struct"
  of gtInt64, gtUint64:
    error "HLSL via glslang: int64_t is a parse error in the HLSL frontend and uint64_t pulls the SPIR-V Int64 capability, 64-bit integers are rejected"
  else:
    error "HLSL has no type name for " & $t

proc gpuTypeToString*(t: GpuType, ident: string = "", allowEmptyIdent = false): string =
  ## HLSL type text, struct types render their own name
  var skipIdent = false
  case t.kind
  of gtArray:
    if t.aLen == 0:
      error "HLSL: runtime-sized arrays exist only as buffer resources"
    if ident.len == 0 and not allowEmptyIdent:
      error "Invalid call, got an array type but don't have an identifier: " & $t
    result = gpuTypeToString(t.aTyp, allowEmptyIdent = true) & ' ' & ident & '[' & $t.aLen & ']'
    skipIdent = true
  of gtObject: result = t.name
  of gtGenericInst:
    result = t.gName
    if t.gArgs.len == 1:
      result.add gpuTypeToShortString(t.gArgs[0])
    elif t.gArgs.len > 1:
      # 2+ args collapse to a short hash
      result.add mangledHashName("", $hash(t))
  of gtStatic: result = "int"
  else:        result = gpuTypeToString(t.kind)

  if ident.len > 0 and not skipIdent:
    result.add ' ' & ident

proc genLit*(ast: GpuAst): string =
  ## HLSL literal text, suffixed per its type: without the suffix the glslang
  ## HLSL frontend changes width or signedness
  if ast.lType.kind == gtString:
    result = '"' & ast.lValue & '"'
  elif ast.lValue == "DEFAULT":
    error "HLSL: DEFAULT literal has no emission"
  else:
    case ast.lType.kind
    of gtBool:
      doAssert ast.lValue in ["true", "false"], "bool literal is " & ast.lValue
      result = ast.lValue
    of gtInt32: result = ast.lValue
    of gtUint32, gtSize_t: result = ast.lValue & "u"
    of gtFloat32:
      result = if '.' in ast.lValue or 'e' in ast.lValue or 'E' in ast.lValue: ast.lValue & "f"
               else: ast.lValue & ".0f"
    of gtFloat64:
      result = if '.' in ast.lValue or 'e' in ast.lValue or 'E' in ast.lValue: ast.lValue
               else: ast.lValue & ".0"
    else:
      error "HLSL: literal of type " & $ast.lType.kind & " has no emission"

# ═══════════════════════════════════════════════════════════════════════
# Per-function emission state
# ═══════════════════════════════════════════════════════════════════════

type HlslBuffer = object
  name: string        ## display name of the buffer parameter, the resource variable name
  elem: GpuType       ## element type addressed through the buffer
  wrap: string        ## wrapper struct name, distinct per kernel binding (MoltenVK block-merge trap), per element type on device functions
  binding: int        ## vk::binding ordinal among kernel buffer parameters, unused on device function params
  readOnly: bool      ## true when the write-scan finds no store through the buffer

type HlslKernel = ref object
  ## Per-function emission state, codegen assembles the collected pieces after the body scan.
  ## A ref so the gpuEmit renderer closure can capture and mutate it: Nim closures cannot capture `var` parameters.
  buffers: Table[string, HlslBuffer] ## buffer param iSym -> resource info
  pushParams: Table[string, GpuType] ## scalar param iSym -> push-block member type, kernels only
  isKernel: bool                     ## false for device functions: scalars are plain params, returns carry values
  gridRead: bool                     ## a thread_position_in_grid component was read
  groupIdName: string                ## threadgroup_position_in_grid input name, "" when unread
  groupIndexName: string             ## thread_index_in_threadgroup input name, "" when unread

func fnBufferWrap(elem: GpuType): string =
  ## Wrapper struct name of a device function buffer parameter, keyed by element type
  "Wrap" & gpuTypeToShortString(elem)

func indexBase(n: GpuAst): GpuAst =
  ## Array expression an index node addresses, transparent through gpuDeref
  ## nodes around pointer bases
  if n.iArr.kind == gpuDeref:
    n.iArr.dOf
  else:
    n.iArr

proc classifyParams(st: HlslKernel, fn: GpuAst) =
  ## Splits function parameters into buffer resources and scalars,
  ## one wrapper struct name and one binding ordinal per buffer parameter:
  ## - kernels: scalars join the push-constant block, buffer wrapper structs are per parameter
  ## - device functions: scalars pass by value, buffer wrapper structs are per element type
  ##
  ## Rejects the parameter shapes the HLSL resource model cannot bind,
  ## every message names the rejected construct and the enclosing function.
  let fnName = fn.pName.ident()
  var ordinal = 0
  for param in fn.pParams:
    let sym = param.ident.symbol
    if sym.coordBuiltin != gbkNone:
      # coordinate builtins are inputs joined on read, never parameters
      continue
    if param.flavor == kVar:
      error "HLSL: `var` (pass-by-reference) parameter " & sym.name & " in `" & fnName &
        "` has no binding, buffers pass as `ptr UncheckedArray[T]`"
    if param.passByRef and st.isKernel:
      error "HLSL: by-reference parameter " & sym.name & " in kernel `" & fnName & "` has no binding"
    if param.addressSpace == asSMEM:
      error "HLSL: {.smem.} parameter " & sym.name & " in `" & fnName & "` has no binding"
    case param.typ.kind
    of gtVoidPtr:
      error "HLSL: void pointer parameter " & sym.name & " in `" & fnName & "` has no resource binding"
    of gtPtr:
      if param.typ.to.kind != gtUncheckedArray:
        error "HLSL: buffer parameter " & sym.name & " in `" & fnName &
          "` must be `ptr UncheckedArray[T]`, got ptr " & $param.typ.to.kind
      st.buffers[sym.iSym] = HlslBuffer(
        name: sym.name,
        elem: param.typ.to.uaTo,
        wrap: if st.isKernel: "Wrap" & capitalizeAscii(sym.name)
              else: fnBufferWrap(param.typ.to.uaTo),
        binding: ordinal)
      inc ordinal
    of gtObject, gtGenericInst:
      # gtObject carries the plain type name, gtGenericInst the generic's name
      let tname = if param.typ.kind == gtGenericInst: param.typ.gName else: param.typ.name
      if tname in ["TensorView", "TensorOwned"]:
        error "HLSL: " & tname & " parameter " & sym.name & " in `" & fnName &
          "` is outside the HLSL subset (HLSL structs cannot hold resources)"
      if st.isKernel:
        error "HLSL: struct parameter " & sym.name & " in kernel `" & fnName & "` has no binding"
      # device functions take plain structs by value
    of gtInt64, gtUint64:
      error "HLSL: 64-bit integer parameter " & sym.name & " in `" & fnName &
        "` is rejected: int64_t is a parse error in the HLSL frontend and uint64_t pulls the SPIR-V Int64 capability"
    of gtArray, gtUncheckedArray:
      error "HLSL: " & $param.typ.kind & " parameter " & sym.name & " in `" & fnName &
        "` has no binding, buffers pass as `ptr UncheckedArray[T]`"
    of gtFloat16, gtBf16, gtF8E4M3, gtF8E5M2, gtUint8, gtUint16, gtInt16, gtSpan, gtString:
      error "HLSL: parameter " & sym.name & " of type " & $param.typ.kind & " in `" & fnName &
        "` has no HLSL type name"
    else:
      if st.isKernel:
        st.pushParams[sym.iSym] = param.typ
      # device function scalars pass by value, no per-param state

proc writesThrough(n: GpuAst, bufISym: string): bool =
  ## True when the statement tree stores through the buffer parameter iSym, direct indexed assignment only
  case n.kind
  of gpuBlock:
    for s in n.statements:
      if s.writesThrough(bufISym):
        return true
  of gpuAssign:
    if n.aLeft.kind == gpuIndex:
      let base = indexBase(n.aLeft)
      if base.kind == gpuIdent and base.symbol.iSym == bufISym:
        return true
  of gpuIf:
    if n.ifThen.writesThrough(bufISym):
      return true
    for el in n.ifElifs:
      if el.body.writesThrough(bufISym):
        return true
    if n.ifElse.kind != gpuDiscard and n.ifElse.writesThrough(bufISym):
      return true
  of gpuFor:
    if n.fBody.writesThrough(bufISym):
      return true
  of gpuWhile:
    if n.wBody.writesThrough(bufISym):
      return true
  else:
    discard

# ═══════════════════════════════════════════════════════════════════════
# Device function signatures
# ═══════════════════════════════════════════════════════════════════════

proc fnRetTypeText(fn: GpuAst): string =
  ## HLSL return type text of a function, gtVoid renders `void`
  if fn.pRetType.isNil or fn.pRetType.kind == gtVoid:
    result = "void"
  else:
    result = gpuTypeToString(fn.pRetType, allowEmptyIdent = true)

proc deviceFnSignature(st: HlslKernel, fn: GpuAst): string =
  ## `RetType name(params)` of a device function:
  ##
  ## - buffer params render as StructuredBuffer / RWStructuredBuffer resource params,
  ##   read-only vs RW from the function's own write-scan
  ## - scalars pass by value
  ## - large-struct params bind `inout`
  var params: seq[string]
  for p in fn.pParams:
    let sym = p.ident.symbol
    if sym.iSym in st.buffers:
      let buf = st.buffers[sym.iSym]
      let resTy = if buf.readOnly: "StructuredBuffer" else: "RWStructuredBuffer"
      params.add resTy & "<" & buf.wrap & "> " & sym.name
    elif p.passByRef:
      params.add "inout " & gpuTypeToString(p.typ, sym.name)
    else:
      params.add gpuTypeToString(p.typ, sym.name)
  result = fnRetTypeText(fn) & ' ' & fn.pName.ident() & '(' & params.join(", ") & ')'

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc getExprTypeHlsl(ctx: GpuContext, n: GpuAst): GpuType =
  ## getOperandType plus two fixups:
  ## - `min`/`max`/`abs` calls resolve to the operand type: the call site keeps the unmangled
  ##   Nim magic name while the materialized body sits in fnTab under its instantiation mangle,
  ##   so the fn-table lookup cannot see it (the intrinsic prints the plain name)
  ## - coordinate-builtin component reads are uint32, synthesized component idents carry no type
  if n.kind == gpuCall and not n.cName.symbol.isNil and
      n.cName.symbol.name in NimGpuNumericBuiltinsFunctions:
    return ctx.getOperandType(n.cArgs[0])
  result = ctx.getOperandType(n)
  if result.isNil and n.kind == gpuDot and n.dParent.kind == gpuIdent and
      n.dParent.symbol != nil and n.dParent.symbol.coordBuiltin != gbkNone:
    result = GpuType(kind: gtUint32)

proc emitCast(ctx: GpuContext, target: GpuType, exprText: string, source: GpuType): string =
  ## Explicit HLSL constructor cast when the IR carries a type change:
  ##
  ## - identical scalar types and sourceless expressions emit uncast
  ## - a gtVoid target emits uncast, the implicit `result` ident carries no type,
  ##   the frontend already converted its value to the declared result type
  if source.isNil or target.kind == gtVoid or source.kind == target.kind:
    return exprText
  if target.kind == gtArray:
    error "HLSL: array casts have no emission"
  gpuTypeToString(target, allowEmptyIdent = true) & "(" & exprText & ")"

proc genIdentRef(ctx: var GpuContext, st: HlslKernel, n: GpuAst): string =
  ## Text of an identifier read: kernel scalar parameters read through the push block,
  ## coordinate builtins join main's signature on first read,
  ## buffer parameters refuse bare use outside a call argument
  let sym = n.symbol
  case sym.coordBuiltin
  of gbkThreadIndexInThreadgroup:
    st.groupIndexName = sym.name
    return sym.name
  of gbkNone: discard
  else:
    error "HLSL: whole-vector use of coordinate builtin " & sym.name & " has no emission, index a component"
  if sym.iSym in st.buffers:
    error "HLSL: bare pointer value use of buffer parameter " & sym.name
  if sym.iSym in st.pushParams:
    return "push." & sym.name
  sym.name

proc isFp16ConversionCall(n: GpuAst): bool =
  ## True for the fp16 ↔ fp32 conversion builtins (toFp16, toFp32, asFp16, asU16):
  ## the per-backend name table carries no HLSL entries, the calls have no emission
  let name = n.cName.ident()
  for key in NimGpuFp16ConversionBuiltins.keys:
    if key[1] == name:
      return true
  false

proc genHlslExpr(ctx: var GpuContext, st: HlslKernel, n: GpuAst): string
proc genHlsl(ctx: var GpuContext, st: HlslKernel, ast: GpuAst, indent = 0): string
proc genBraceList(ctx: var GpuContext, st: HlslKernel, oc: GpuAst): string

proc genConstrValue(ctx: var GpuContext, st: HlslKernel, v: GpuAst): string =
  ## One field value of a brace initializer, nested constructions emit nested braces
  if v.kind == gpuObjConstr:
    ctx.genBraceList(st, v)
  else:
    ctx.genHlslExpr(st, v)

proc genBraceList(ctx: var GpuContext, st: HlslKernel, oc: GpuAst): string =
  ## Brace initializer `{ a, { b, c } }` for an object constructor, fields in declaration order, verified against glslang 16.6
  result = "{ "
  for i, f in oc.ocFields:
    if i > 0:
      result.add ", "
    result.add ctx.genConstrValue(st, f.value)
  result.add " }"

proc genHlslExpr(ctx: var GpuContext, st: HlslKernel, n: GpuAst): string =
  case n.kind
  of gpuLit:
    result = genLit(n)
  of gpuIdent:
    result = ctx.genIdentRef(st, n)
  of gpuBinOp:
    result = "(" & ctx.genHlslExpr(st, n.bLeft) & " " & n.bOp.ident() & " " &
      ctx.genHlslExpr(st, n.bRight) & ")"
  of gpuDot:
    if n.dParent.kind == gpuIdent and n.dParent.symbol != nil and
        n.dParent.symbol.coordBuiltin != gbkNone:
      case n.dParent.symbol.coordBuiltin
      of gbkThreadPositionInGrid:
        st.gridRead = true
        result = "gid." & n.dField.ident()
      of gbkThreadgroupPositionInGrid:
        st.groupIdName = n.dParent.ident()
        result = n.dParent.ident() & "." & n.dField.ident()
      else:
        error "HLSL: component access on " & n.dParent.ident() & " has no emission (scalar or unsupported coordinate builtin)"
    else:
      let fieldName = if n.dField.kind == gpuIdent: n.dField.ident()
                      else: ctx.genHlslExpr(st, n.dField)
      result = ctx.genHlslExpr(st, n.dParent) & '.' & fieldName
  of gpuIndex:
    let base = indexBase(n)
    if base.kind != gpuIdent:
      error "HLSL: indexed base is not a plain identifier, got " & $base.kind
    let sym = base.symbol
    if sym.iSym in st.buffers:
      # buffer element reads go through the wrapper member v
      result = sym.name & "[" & ctx.genHlslExpr(st, n.iIndex) & "].v"
    elif sym.typ != nil and sym.typ.kind == gtArray:
      result = sym.name & "[" & ctx.genHlslExpr(st, n.iIndex) & "]"
    else:
      error "HLSL: indexing something other than a buffer parameter or a fixed array: " & sym.name
  of gpuConv, gpuCast:
    let target = if n.kind == gpuConv: n.convTo else: n.cTo
    let src = if n.kind == gpuConv: n.convExpr else: n.cExpr
    result = ctx.emitCast(target, ctx.genHlslExpr(st, src), ctx.getExprTypeHlsl(src))
  of gpuPrefix:
    result = "(" & n.pOp & ctx.genHlslExpr(st, n.pVal) & ")"
  of gpuTernary:
    result = "(" & ctx.genHlslExpr(st, n.tCond) & " ? " & ctx.genHlslExpr(st, n.tThen) &
      " : " & ctx.genHlslExpr(st, n.tElse) & ")"
  of gpuCall:
    case n.cName.symbol.synchroBuiltin
    of gbkThreadgroupBarrier:
      result = "GroupMemoryBarrierWithGroupSync()"
    of gbkThreadgroupBarrierDevice:
      result = "DeviceMemoryBarrierWithGroupSync()"
    of gbkNone:
      if n.cName.symbol.reductionBuiltin != gbkNone:
        error "HLSL: subgroup reductions have no emission, wave intrinsics are outside this backend"
      if n.cName.symbol.atomicsBuiltin != gbkNone:
        error "HLSL: atomic builtins have no emission in this backend"
      if isFp16ConversionCall(n):
        error "HLSL: call to `" & n.cName.ident() & "` has no emission, float16 types are outside the HLSL subset"
      # materialized min/max/abs wrappers pass through under their mangled names like on every backend,
      # other plain names (exp2, rsqrt, device calls) emit verbatim
      var parts: seq[string]
      for arg in n.cArgs:
        if arg.kind == gpuIdent and arg.symbol.iSym in st.buffers:
          # a buffer resource passes by name, its identity flows from the call site
          parts.add arg.symbol.name
        elif arg.kind == gpuMaterialize:
          # a hidden const-reference argument unwraps to the value it materializes
          parts.add ctx.genHlslExpr(st, arg.mExpr)
        else:
          parts.add ctx.genHlslExpr(st, arg)
      result = ctx.getFnName(bkHlsl, n) & "(" & parts.join(", ") & ")"
  of gpuObjConstr:
    error "HLSL: object constructor outside a declaration initializer or member-wise assignment"
  of gpuAddr:
    error "HLSL: address-of has no emission, buffers are index-only resources"
  of gpuDeref:
    error "HLSL: bare dereference has no emission, buffers are index-only resources"
  else:
    error "HLSL: expression node " & $n.kind & " has no emission"

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc genMemberType(t: GpuType, path: seq[string]): GpuType =
  ## Field type reached by walking a struct type along a dotted path
  result = t
  for name in path:
    var found = false
    for f in result.fields:
      if f.name == name:
        result = f.typ
        found = true
        break
    doAssert found, "no field " & name & " in " & gpuTypeToString(result, allowEmptyIdent = true)

proc genObjAssignLeaves(ctx: var GpuContext, st: HlslKernel, targetType: GpuType,
    targetName: string, v: GpuAst, path: seq[string], lines: var seq[string]) =
  ## Appends one `targetName.leaf = rhs` line per leaf field of an object constructor,
  ## nested constructions recurse, field values cast to the walked member type
  for f in v.ocFields:
    let leafPath = path & f.name
    if f.value.kind == gpuObjConstr:
      ctx.genObjAssignLeaves(st, targetType, targetName, f.value, leafPath, lines)
    else:
      let member = genMemberType(targetType, leafPath)
      let rhs = ctx.emitCast(member, ctx.genHlslExpr(st, f.value), ctx.getExprTypeHlsl(f.value))
      lines.add targetName & "." & leafPath.join(".") & " = " & rhs

proc genObjAssign(ctx: var GpuContext, st: HlslKernel, target: GpuAst, value: GpuAst, indent: int): string =
  ## Member-wise leaf assignment of an object constructor to a declared struct local:
  ## glslang's HLSL frontend rejects compound literals on assignment
  let targetType = target.symbol.typ
  var lines: seq[string]
  ctx.genObjAssignLeaves(st, targetType, target.ident(), value, @[], lines)
  # Each member line carries its own `;` except the last, the block renderer terminates the statement with one
  for i, l in lines:
    if i > 0:
      result.add '\n'
    result.add "  ".repeat(indent) & l
    if i < lines.high:
      result.add ';'

proc genHlsl(ctx: var GpuContext, st: HlslKernel, ast: GpuAst, indent = 0): string =
  let indentStr = "  ".repeat(indent)
  case ast.kind
  of gpuDiscard: return

  of gpuBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "{ // " & ast.blockLabel & '\n'
    for i, el in ast.statements:
      result.add ctx.genHlsl(st, el, indent)
      if not el.isSelfTerminating() and not ctx.skipSemicolon: # nested blocks and emits carry their own terminators
        result.add ';'
      if i < ast.statements.high:
        result.add '\n'
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "} // " & ast.blockLabel & '\n'

  of gpuVar:
    if ast.addressSpace != asRMEM:
      error "HLSL: local address spaces other than register memory have no emission"
    doAssert not ast.vRequiresMemcpy, "memcpy variables are decomposed by the shared lowering pass"
    let name = ast.vName.ident()
    case ast.vType.kind
    of gtPtr:
      error "HLSL: pointer-valued local " & name & " has no emission"
    of gtObject, gtGenericInst:
      # gtObject carries the plain type name, gtGenericInst the generic's name
      let typeName = if ast.vType.kind == gtGenericInst: ast.vType.gName else: ast.vType.name
      if typeName in ["TensorView", "TensorOwned"]:
        error "HLSL: " & typeName & " local " & name &
          " is outside the HLSL subset (HLSL structs cannot hold resources)"
      let tname = gpuTypeToString(ast.vType, allowEmptyIdent = true)
      case ast.vInit.kind
      of gpuDiscard:
        result = indentStr & tname & ' ' & name
      of gpuObjConstr:
        result = indentStr & tname & ' ' & name & " = " & ctx.genBraceList(st, ast.vInit)
      else:
        # whole-struct copy declaration, verified against glslang 16.6
        result = indentStr & tname & ' ' & name & " = " & ctx.genHlslExpr(st, ast.vInit)
    else:
      result = indentStr & gpuTypeToString(ast.vType, name)
      if ast.vInit.kind != gpuDiscard:
        result &= " = " & ctx.emitCast(ast.vType, ctx.genHlslExpr(st, ast.vInit),
                                       ctx.getExprTypeHlsl(ast.vInit))

  of gpuAssign:
    if ast.aRight.kind == gpuObjConstr and ast.aLeft.kind == gpuIdent:
      result = ctx.genObjAssign(st, ast.aLeft, ast.aRight, indent)
    else:
      case ast.aLeft.kind
      of gpuIndex:
        let base = indexBase(ast.aLeft)
        if base.kind != gpuIdent:
          error "HLSL: assignment target base is not a plain identifier, got " & $base.kind
        let sym = base.symbol
        if sym.iSym in st.buffers:
          let buf = st.buffers[sym.iSym]
          if buf.readOnly:
            error "HLSL: assignment through read-only buffer parameter " & buf.name
          let rhs = ctx.emitCast(buf.elem, ctx.genHlslExpr(st, ast.aRight),
                                 ctx.getExprTypeHlsl(ast.aRight))
          result = indentStr & buf.name & "[" & ctx.genHlslExpr(st, ast.aLeft.iIndex) & "].v = " & rhs
        elif sym.typ != nil and sym.typ.kind == gtArray:
          let elem = sym.typ.aTyp
          let rhs = ctx.emitCast(elem, ctx.genHlslExpr(st, ast.aRight),
                                 ctx.getExprTypeHlsl(ast.aRight))
          result = indentStr & sym.name & "[" & ctx.genHlslExpr(st, ast.aLeft.iIndex) & "] = " & rhs
        else:
          error "HLSL: assignment through something other than a buffer parameter or a fixed array: " & sym.name
      of gpuIdent:
        let sym = ast.aLeft.symbol
        if sym.iSym in st.pushParams:
          error "HLSL: assignment to push-constant parameter " & sym.name & ", push constants are read-only inputs"
        if sym.iSym in st.buffers:
          error "HLSL: bare buffer parameter " & sym.name & " as an assignment target"
        let rhs = ctx.emitCast(sym.typ, ctx.genHlslExpr(st, ast.aRight),
                               ctx.getExprTypeHlsl(ast.aRight))
        result = indentStr & sym.name & ' ' & ast.aOp & ' ' & rhs
      else:
        error "HLSL: assignment target " & $ast.aLeft.kind & " has no emission"

  of gpuIf:
    if ast.ifIsExpr:
      error "HLSL: expression if has no emission, use a statement if or a ternary"
    ctx.withoutSemicolon:
      result = indentStr & "if (" & ctx.genHlslExpr(st, ast.ifCond) & ") {\n"
    result &= ctx.genHlsl(st, ast.ifThen, indent + 1) & '\n'
    result &= indentStr & '}'
    for el in ast.ifElifs:
      ctx.withoutSemicolon:
        result &= " else if (" & ctx.genHlslExpr(st, el.cond) & ") {\n"
      result &= ctx.genHlsl(st, el.body, indent + 1) & '\n'
      result &= indentStr & '}'
    if ast.ifElse.kind != gpuDiscard:
      result &= " else {\n"
      result &= ctx.genHlsl(st, ast.ifElse, indent + 1) & '\n'
      result &= indentStr & '}'

  of gpuFor:
    let i = ast.fVar.ident()
    let indTyp = gpuTypeToString(ast.fVar.symbol.typ, allowEmptyIdent = true)
    let cmp = if ast.fRangeKind == rkInclusive: " <= " else: " < "
    result = indentStr & "for(" & indTyp & ' ' & i & " = " & ctx.genHlslExpr(st, ast.fStart) &
             "; " & i & cmp & ctx.genHlslExpr(st, ast.fEnd) & "; " &
             i & " += " & ctx.genHlslExpr(st, ast.fStep) & ") {\n"
    result &= ctx.genHlsl(st, ast.fBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genHlslExpr(st, ast.wCond) & ") {\n"
    result &= ctx.genHlsl(st, ast.wBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuReturn:
    if ast.rValue.kind == gpuDiscard:
      result = indentStr & "return"
    elif st.isKernel:
      error "HLSL: non-void return has no emission, kernels are void"
    else:
      result = indentStr & "return " & ctx.genHlslExpr(st, ast.rValue)

  of gpuComment:
    result = indentStr & "/* " & ast.comment & " */"

  of gpuConstexpr:
    result = indentStr & "const " & gpuTypeToString(ast.cType, ast.cIdent.ident()) &
             " = " & ctx.genHlslExpr(st, ast.cValue)

  of gpuEmit:
    # Self-terminating raw text: the gpuBlock loop appends no `;`,
    # the emitted text carries its own terminators.
    proc renderEmitPart(c: var GpuContext, n: GpuAst): string =
      c.genHlslExpr(st, n)
    result = genEmitStmt(ctx, ast, renderEmitPart)

  of gpuInlineAsm:
    error "HLSL: inline assembly has no emission"

  of gpuAlias:
    error "HLSL: type aliases have no emission"

  of gpuArrayLit:
    error "HLSL: array literals have no emission"

  else:
    error "HLSL: statement node " & $ast.kind & " has no emission"

# ═══════════════════════════════════════════════════════════════════════
# Global declarations
# ═══════════════════════════════════════════════════════════════════════

proc emitStructDecl(byName: Table[string, GpuAst], emitted: var HashSet[string], name: string, outp: var string) =
  ## Emits one struct definition, member structs first, deduplicated by name.
  ## HLSL resolves types at their declaration site.
  if name in emitted or name notin byName:
    return
  emitted.incl name
  let blk = byName[name]
  for f in blk.tFields:
    case f.typ.kind
    of gtObject, gtGenericInst:
      emitStructDecl(byName, emitted, gpuTypeToString(f.typ, allowEmptyIdent = true), outp)
    of gtArray:
      if f.typ.aTyp.kind in {gtObject, gtGenericInst}:
        emitStructDecl(byName, emitted, gpuTypeToString(f.typ.aTyp, allowEmptyIdent = true), outp)
    else:
      discard
  outp.add "struct " & name & " {"
  if blk.tFields.len == 0:
    # HLSL requires at least one member in a struct
    outp.add "\n  uint _padding;"
  else:
    for f in blk.tFields:
      outp.add "\n  " & gpuTypeToString(f.typ, f.name) & ";"
  outp.add "\n};\n"

proc emitTypeDefs(ctx: GpuContext, outp: var string) =
  ## Emits the struct definitions from the global blocks in dependency order
  var byName = initTable[string, GpuAst]()
  for blk in ctx.globalBlocks:
    if blk.kind == gpuTypeDef:
      byName[gpuTypeToString(blk.tTyp, allowEmptyIdent = true)] = blk
  var emitted = initHashSet[string]()
  for name in byName.keys:
    emitStructDecl(byName, emitted, name, outp)

proc emitResources(st: HlslKernel, kernel: GpuAst, result: var string) =
  ## Emits the wrapper structs, the buffer resources with their explicit
  ## [[vk::binding]] attributes and the push-constant block
  for param in kernel.pParams:
    let sym = param.ident.symbol
    if sym.iSym notin st.buffers:
      continue
    let buf = st.buffers[sym.iSym]
    result.add "struct " & buf.wrap & " { " &
      gpuTypeToString(buf.elem, allowEmptyIdent = true) & " v; };\n"
  for param in kernel.pParams:
    let sym = param.ident.symbol
    if sym.iSym notin st.buffers:
      continue
    let buf = st.buffers[sym.iSym]
    let resTy = if buf.readOnly: "StructuredBuffer" else: "RWStructuredBuffer"
    result.add "[[vk::binding(" & $buf.binding & ", 0)]] " & resTy & "<" & buf.wrap & "> " & buf.name & ";\n"
  if st.pushParams.len > 0:
    result.add "struct Push {\n"
    for param in kernel.pParams:
      let sym = param.ident.symbol
      if sym.iSym in st.pushParams:
        result.add "  " & gpuTypeToString(st.pushParams[sym.iSym], allowEmptyIdent = true) &
          ' ' & sym.name & ";\n"
    result.add "};\n"
    result.add "[[vk::push_constant]] ConstantBuffer<Push> push;\n"

proc emitDeviceFnResources(fns: seq[GpuAst], outp: var string) =
  ## Emits the wrapper structs the device function resource params share,
  ## one per element type, deduplicated by name.
  ##
  ## Kernel resources keep per-parameter wrapper structs:
  ## MoltenVK merges identically typed storage blocks into one MSL struct,
  ## distinct wrapper types keep the kernel bindings apart.
  var emitted = initHashSet[string]()
  for fn in fns:
    for p in fn.pParams:
      if p.typ.kind != gtPtr or p.typ.to.kind != gtUncheckedArray:
        continue
      let wrap = fnBufferWrap(p.typ.to.uaTo)
      if wrap notin emitted:
        emitted.incl wrap
        outp.add "struct " & wrap & " { " &
          gpuTypeToString(p.typ.to.uaTo, allowEmptyIdent = true) & " v; };\n"

proc workgroupSize(kernel: GpuAst): tuple[x, y, z: int] =
  ## [numthreads] triple, (0, 0, 0) in the IR means the 256x1x1 default
  if kernel.pWorkgroupSize == (0, 0, 0):
    result = (256, 1, 1)
  else:
    result = kernel.pWorkgroupSize

proc collectDeviceCallTargets(ctx: GpuContext, n: GpuAst, targets: var seq[GpuAst]) =
  ## Device functions a statement tree calls directly or transitively,
  ## resolved through fnTab (dead-code elimination keeps the called subset there)
  if n.isNil:
    return
  if n.kind == gpuCall and not n.cName.isNil and not n.cName.symbol.isNil:
    let fn = ctx.fnTab.getOrDefault(n.cName, nil)
    # `!= nil` is wrong here: GpuAst's custom `==` returns false for (nil, nil), so a nil fn compares unequal to nil
    if not fn.isNil and not fn.isGlobal():
      targets.add fn
  for ch in n:
    if not ch.isNil:
      ctx.collectDeviceCallTargets(ch, targets)

proc checkDeviceFnRecursion(ctx: GpuContext) =
  ## Refuses recursion over the device-function call graph before emission:
  ## a recursive call would not terminate, and the HLSL compiler accepts
  ## recursive source with exit 0 while miscompiling it
  var state = initTable[string, int]() # fn iSym: 1 = on the visit stack, 2 = fully visited

  proc visit(fn: GpuAst) =
    let iSym = fn.pName.symbol.iSym
    state[iSym] = 1
    var callees: seq[GpuAst]
    ctx.collectDeviceCallTargets(fn.pBody, callees)
    for callee in callees:
      case state.getOrDefault(callee.pName.symbol.iSym, 0)
      of 1:
        error "HLSL: recursive device function `" & callee.pName.ident() &
          "` would not terminate, recursion is rejected"
      of 0:
        visit(callee)
      else:
        discard
    state[iSym] = 2

  for fn in ctx.fnTab.values:
    if fn.kind == gpuProc and not fn.isGlobal() and
        state.getOrDefault(fn.pName.symbol.iSym, 0) == 0:
      visit(fn)

proc deviceFnText(ctx: var GpuContext, fn: GpuAst, forwardDeclare: bool): string =
  ## Emitted text of one device function:
  ##
  ## - the signature from its classified params
  ## - `;` for a forward declaration, otherwise the body in braces
  ## - device function calls emit as plain calls, early returns emit natively
  if fn.pBody.isNil:
    error "HLSL: device function `" & fn.pName.ident() & "` has no IR body, extern declarations have no emission"
  var st = HlslKernel()
  st.classifyParams(fn)
  for iSym in st.buffers.keys:
    st.buffers[iSym].readOnly = not fn.pBody.writesThrough(iSym)
  result = deviceFnSignature(st, fn)
  if forwardDeclare:
    result.add ';'
  else:
    result.add " {\n"
    result.add ctx.genHlsl(st, fn.pBody, 1)
    result.add "\n}\n"

proc codegen*(ctx: var GpuContext): string =
  ## Renders the compiled IR as HLSL compute source for glslang's HLSL frontend.
  ## Returns: the complete source text.
  ##
  ## Layout: struct definitions, resources, device function forward declarations,
  ## device function definitions, `[numthreads]` + `void main(...)`.
  ##
  ## One global kernel per source, emitted as `main` to match the glslang
  ## invocation `-e main` (glslang compiles one entry point per invocation).

  var kernel: GpuAst = nil
  for _, fn in ctx.fnTab:
    if fn.kind != gpuProc:
      continue
    if fn.isGlobal():
      doAssert kernel.isNil, "HLSL compiles one kernel per source"
      kernel = fn
  doAssert kernel.isNil.not, "HLSL found no global kernel"
  if kernel.pRetType.kind != gtVoid:
    error "HLSL: kernel `" & kernel.pName.ident() & "` must return void, got " & $kernel.pRetType.kind

  var fns: seq[GpuAst]
  for _, fn in ctx.fnTab:
    if fn.kind == gpuProc and not fn.isGlobal():
      fns.add fn

  ctx.checkDeviceFnRecursion()

  var st = HlslKernel(isKernel: true)
  st.classifyParams(kernel)
  for iSym in st.buffers.keys:
    st.buffers[iSym].readOnly = not kernel.pBody.writesThrough(iSym)

  ctx.emitTypeDefs(result)
  result.add "\n"
  st.emitResources(kernel, result)
  emitDeviceFnResources(fns, result)

  # forward declarations first, a call resolves to a definition emitted later through them
  for fn in fns:
    result.add ctx.deviceFnText(fn, forwardDeclare = true) & '\n'
  if fns.len > 0:
    result.add "\n"
  for fn in fns:
    result.add ctx.deviceFnText(fn, forwardDeclare = false) & "\n\n"

  # the body emits first: the input variables join the signature only when a read demands them
  let body = ctx.genHlsl(st, kernel.pBody, 1)

  let lws = kernel.workgroupSize()
  result.add "\n[numthreads(" & $lws[0] & ", " & $lws[1] & ", " & $lws[2] & ")]\n"
  var inputs: seq[string]
  if st.gridRead:
    inputs.add "uint3 gid : SV_DispatchThreadID"
  if st.groupIdName.len > 0:
    inputs.add "uint3 " & st.groupIdName & " : SV_GroupID"
  if st.groupIndexName.len > 0:
    inputs.add "uint " & st.groupIndexName & " : SV_GroupIndex"
  result.add "void main(" & inputs.join(", ") & ")\n"
  result.add "{\n"
  result.add body
  result.add "\n}\n"
