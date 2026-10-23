# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal Shading Language (MSL) printer for the `bkMetal` backend.

import std / [macros, strformat, strutils, sequtils, tables]

import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./lang_utils

proc gpuTypeToString*(t: GpuType, ident: string = "", allowEmptyIdent = false): string

proc getInnerArrayType(t: GpuType): string =
  ## Returns the name of the innermost type for a nested array.
  case t.kind
  of gtArray:
    result = getInnerArrayType(t.aTyp)
  else:
    result = gpuTypeToString(t)

proc gpuTypeToString*(t: GpuTypeKind): string =
  ## MSL name of a scalar type kind (`long`/`ulong` for 64-bit, fp64 rejected loudly).
  case t
  of gtBool: "bool"
  of gtUint8: "uchar"
  of gtUint16: "ushort"
  of gtUint32: "uint"
  of gtUint64: "ulong"
  of gtInt16: "short"
  of gtInt32: "int"
  of gtInt64: "long"
  of gtFloat32: "float"
  of gtFloat16: "half"
  of gtBf16: "bfloat"
  of gtFloat64:
    error "Metal has no fp64 kernel type, MSL rejects `double`, rework the kernel on fp32/fp16"
  of gtVoid: "void"
  of gtSize_t: "size_t"
  of gtPtr: "*"
  of gtVoidPtr: "void*"
  of gtObject: "struct"
  of gtString: "const char*"
  of gtUncheckedArray: "" # `UncheckedArray` by itself is nothing in MSL
  of gtStatic: "int"
  else:
    error "Invalid type : " & $t

proc gpuTypeToString*(t: GpuType, ident: string = "", allowEmptyIdent = false): string =
  ## MSL name of `t`, with `ident` appended for array types, which require an identifier in the emitted declaration.
  ## `allowEmptyIdent` permits the identifier to be omitted (function return types and casts).
  ## Without it, an array type with an empty `ident` raises.
  var skipIdent = false
  case t.kind
  of gtPtr:
    let inner = if t.to.kind == gtUncheckedArray: t.to.uaTo else: t.to
    if inner.kind == gtArray: # ptr to array type
      # Need to pass `*` for the pointer into the identifier.
      # `state: var array[4, BigInt]` must become `BigInt (*state)[4]`.
      # So we pass `theIdent = (*<ident>)` and generate the type
      # for the internal array type, which yields `BigInt <theIdent>[4]`.
      let ptrStar = gpuTypeToString(t.kind)
      result = gpuTypeToString(inner, '(' & ptrStar & ident & ')')
      skipIdent = true
    else:
      let typ = gpuTypeToString(inner, allowEmptyIdent = allowEmptyIdent)
      let ptrStar = gpuTypeToString(t.kind)
      result = typ & ptrStar
  of gtArray:
    # empty idents happen in e.g. function return types or casts
    if ident.len == 0 and not allowEmptyIdent:
      when nimvm:
        error("Invalid call, got an array type but don't have an identifier: " & $t)
      else:
        raise newException(ValueError, "Invalid call, got an array type but don't have an identifier: " & $t)
    case t.aTyp.kind
    of gtArray: # nested array
      let typ = getInnerArrayType(t)
      let lengths = getInnerArrayLengths(t) # get lengths as `[X][Y][Z]...`
      result = typ & ' ' & ident & lengths
    else:
      # NOTE: Nested arrays don't have an inner identifier!
      if t.aLen == 0: # Zero-length arrays emit as flexible arrays (no length suffix).
        result = gpuTypeToString(t.aTyp, allowEmptyIdent = allowEmptyIdent) & ' ' & ident & "[]"
      else:
        result = gpuTypeToString(t.aTyp, allowEmptyIdent = allowEmptyIdent) & ' ' & ident & '[' & $t.aLen & ']'
    skipIdent = true
  of gtGenericInst:
    # NOTE: `foo[float32, uint32]` becomes `foo_f32_u32`,
    # short generic-arg names (uint32, int64), not C names (unsigned int, long long)
    result = t.gName
    if t.gArgs.len == 1:
      result.add gpuTypeToShortString(t.gArgs[0])
    elif t.gArgs.len > 1:
      # 2+ args collapse to a short hash
      result.add mangledHashName("", $hash(t))
  of gtObject:
    result = t.name
  of gtUncheckedArray:     result = gpuTypeToString(t.uaTo, allowEmptyIdent = allowEmptyIdent) ## unchecked array just T?
  of gtStatic: result = "int"
  else:        result = gpuTypeToString(t.kind)

  if ident.len > 0 and not skipIdent: # still need to add ident
    result.add ' ' & ident

proc genFunctionType*(typ: GpuType, fn: string, fnArgs: string): string =
  ## Function declaration with its return type. Pointers to statically sized arrays
  ## use the `(*fn(args))[N]` form. Everything else is a plain `RetType fn(args)`.
  if typ.kind == gtPtr and typ.to.kind == gtArray:
    # Syntax to return a pointer to a statically sized array:
    # `Foo (*fnName(fnArgs))[ArrayLen]`
    # where the return type is actually:
    # `Foo (*)[ArrayLen]`
    let arrayTyp = typ.to.aTyp
    let innerTyp = gpuTypeToString(arrayTyp, allowEmptyIdent = true)
    let innerLen = $typ.to.aLen
    result = &"{innerTyp} (*{fn}({fnArgs}))[{innerLen}]"
  else:
    result = &"{gpuTypeToString(typ, allowEmptyIdent = true)} {fn}({fnArgs})"

proc genLit*(ast: GpuAst): string =
  ## Lower a literal node for the MSL backend.
  if ast.lType.kind == gtString:
    result = '"' & ast.lValue & '"'
  elif ast.lValue == "DEFAULT":
    result = "{}"
  elif ast.lValue == "NULL":
    result = "nullptr"
  else:
    case ast.lType.kind
    of gtFloat32, gtFloat64:
      # ±Inf and NaN fold into literals whose text form is "inf"/"-inf"/"nan",
      # no such MSL token exists. INFINITY and NAN are the float-typed MSL macros
      # for the special values.
      case ast.lValue
      of "inf":  result = "INFINITY"
      of "-inf": result = "-INFINITY"
      of "nan", "-nan": result = "NAN"
      else:
        if ast.lType.kind == gtFloat64:
          error "Metal has no fp64 kernel type, MSL rejects `double` literals, rework the kernel on fp32/fp16"
        else:
          result = ast.lValue & "f"
    of gtFloat16: result = ast.lValue & "h"
    of gtBf16:    result = "bfloat(" & ast.lValue & ')'
    of gtUint32: result = ast.lValue & "U"
    of gtUint64: result = ast.lValue & "ULL"
    of gtInt64:  result = ast.lValue & "LL"
    of gtInt16, gtUint16, gtUint8, gtBool:
      result = '(' & gpuTypeToString(ast.lType, allowEmptyIdent = true) & ')' & ast.lValue
    else:
      result = ast.lValue

proc genKernelParams(ctx: var GpuContext, fn: GpuAst): string =
  ## Emits the kernel's MSL parameter list. The argument table is one slot
  ## sequence in parameter order:
  ## - pointers: `device T* name [[buffer(n)]]`, n is the parameter's slot
  ## - scalars: `constant T& name`, unattributed, auto-assigned the free slot
  ##   in declaration order, set inline via setBytes
  ## - coordinate builtins emit `uint3 name [[name]]`, no binding and no slot
  ##   `materializeIndexBuiltinParams` appends them
  var params: seq[string]
  var slot = 0
  for p in fn.pParams:
    let name = p.ident.ident()
    if p.ident.symbol.coordBuiltin != gbkNone:
      # The vector coordinate builtins are `uint3` attribute params.
      # The flat thread index `gbkThreadIndexInThreadgroup` is a scalar `uint` builtin.
      let attrType = if p.ident.symbol.coordBuiltin == gbkThreadIndexInThreadgroup:
        "uint" else: "uint3"
      params.add attrType & " " & name & " [[" & name & "]]"
      continue
    if p.typ.kind == gtPtr:
      var inner = p.typ.to
      if inner.kind == gtUncheckedArray:
        inner = inner.uaTo
      var elem = gpuTypeToString(inner, allowEmptyIdent = true)
      params.add "device " & elem & "* " & name & " [[buffer(" & $slot & ")]]"
    else:
      var elem = gpuTypeToString(p.typ, allowEmptyIdent = true)
      if p.typ.kind == gtBool:
        elem = "int"
      params.add "constant " & elem & "& " & name
    inc slot
  result = params.join(", ")

proc addrSpaceToMsl(space: AddressSpace): string =
  case space
  of asDevice: "device"
  of asConstant: "constant"
  of asSMEM: "threadgroup"
  of asRMEM: "thread"

proc genDeviceParam(ctx: var GpuContext, p: GpuParam): string =
  ## Pointer params carry an explicit MSL address space.
  ## `thread` names the caller's locals, `device` names kernel buffers,
  ## `threadgroup` names `{.smem.}` arenas.
  let name = p.ident.ident()
  if p.typ.kind == gtPtr:
    var inner = p.typ.to
    if inner.kind == gtUncheckedArray:
      inner = inner.uaTo
    if inner.kind == gtArray:
      inner = inner.aTyp
    var elem = gpuTypeToString(inner, allowEmptyIdent = true)
    if p.addressSpace == asSMEM:
      let op = if p.typ.to.kind in {gtObject, gtGenericInst}: "&" else: "*"
      result = "threadgroup " & elem & op & ' ' & name
    else:
      # the pointer type carries the space it accesses
      let space = addrSpaceToMsl(p.typ.addressSpace)
      result = space & ' ' & elem & "* " & name
  else:
    # By-value array params decay to pointers in MSL. They are read-only
    # (a plain `array[T]` param), so emit `const`: the call site passes
    # const lvalue arrays, and a non-const `T*` would reject them.
    if p.typ.kind == gtArray:
      result = "const " & gpuTypeToString(p.typ, name)
    else:
      result = gpuTypeToString(p.typ, name)

proc genMetal*(ctx: var GpuContext, ast: GpuAst, indent = 0): string
proc genMetalImpl(ctx: var GpuContext, ast: GpuAst, indent: int): string

proc genMetalImpl(ctx: var GpuContext, ast: GpuAst, indent: int): string =
  ## Generates the MSL for one AST node.
  let indentStr = "  ".repeat(indent)
  case ast.kind
  of gpuDiscard: return # nothing to emit
  of gpuProc:
    let isKernel = attGlobal in ast.pAttributes
    # Parameters
    var params: seq[string]
    if isKernel:
      params.add ctx.genKernelParams(ast)
    else:
      for p in ast.pParams:
        if p.passByRef:
          # Reference params carry the space of their argument: the caller's
          # threadgroup arena or the calling thread's memory.
          let space = if p.addressSpace == asSMEM: "threadgroup " else: "thread const "
          params.add space & gpuTypeToString(p.typ, allowEmptyIdent = true) & "& " & p.ident.ident()
        else:
          params.add ctx.genDeviceParam(p)
    let fnArgs = params.join(", ")
    let fnSig = genFunctionType(ast.pRetType, ast.pName.ident(), fnArgs)

    # `kernel` marks the compute entry point. `inline` is a hint
    # for device functions. Plain `device` functions need no qualifier in MSL.
    let prefix =
      if isKernel: "kernel "
      elif attForceInline in ast.pAttributes: "inline "
      else: ""
    result = indentStr & prefix & fnSig
    if ast.forwardDeclare:
      result.add ';'
    else:
      result.add "{\n"
      result &= ctx.genMetalImpl(ast.pBody, indent + 1)
      result &= '\n' & indentStr & '}'

  of gpuBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "{ // " & ast.blockLabel & '\n'
    for i, el in ast.statements:
      let code = ctx.genMetalImpl(el, indent + (if ast.blockLabel.len > 0: 1 else: 0))
      if code.len == 0:
        continue # skip gpuDiscard and empty statements
      result.add code
      if not el.isSelfTerminating() and not ctx.skipSemicolon:
        result.add ';'
      if i < ast.statements.high:
        result.add '\n'
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "} // " & ast.blockLabel & '\n'

  of gpuVar:
    let vName = ast.vName.ident()
    var attrs = ""
    if ast.addressSpace != asRMEM:
      attrs.add addrSpaceToMsl(ast.addressSpace) & ' '
    var typ = gpuTypeToString(ast.vType, vName)
    if ast.vType.kind == gtPtr and ast.addressSpace == asRMEM:
      # the pointer type carries the space it accesses
      typ = addrSpaceToMsl(ast.vType.addressSpace) & ' ' & typ
    result = indentStr & attrs & typ
    if ast.vInit.kind != gpuDiscard:
      result &= " = " & ctx.genMetalImpl(ast.vInit, 0)
  of gpuAssign:
    result = indentStr & ctx.genMetalImpl(ast.aLeft, 0) & ' ' & ast.aOp & ' ' &
             ctx.genMetalImpl(ast.aRight, 0)
  of gpuIf:
    # skip semicolon in the condition. Otherwise can lead to problematic code
    ctx.withoutSemicolon: # skip semicolon for if bodies
      result = indentStr & "if (" & ctx.genMetalImpl(ast.ifCond, 0) & ") {\n"
    result &= ctx.genMetalImpl(ast.ifThen, indent + 1) & '\n'
    result &= indentStr & '}'
    for el in ast.ifElifs:
      ctx.withoutSemicolon:
        result &= " else if (" & ctx.genMetalImpl(el.cond, 0) & ") {\n"
      result &= ctx.genMetalImpl(el.body, indent + 1) & '\n'
      result &= indentStr & '}'
    if ast.ifElse.kind != gpuDiscard:
      result &= " else {\n"
      result &= ctx.genMetalImpl(ast.ifElse, indent + 1) & '\n'
      result &= indentStr & '}'

  of gpuTernary:
    ctx.withoutSemicolon:
      result = '(' & ctx.genMetalImpl(ast.tCond, 0) & " ? " &
               ctx.genMetalImpl(ast.tThen, 0) & " : " &
               ctx.genMetalImpl(ast.tElse, 0) & ')'

  of gpuFor:
    let cmp = if ast.fRangeKind == rkInclusive: " <= " else: " < "
    result = indentStr & "for(int " & ast.fVar.ident() & " = " &
             ctx.genMetalImpl(ast.fStart, 0) & "; " &
             ast.fVar.ident() & cmp & ctx.genMetalImpl(ast.fEnd, 0) & "; " &
             ast.fVar.ident() & " += " & ctx.genMetalImpl(ast.fStep, 0) & ") {\n"
    result &= ctx.genMetalImpl(ast.fBody, indent + 1) & '\n'
    result &= indentStr & '}'
  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genMetalImpl(ast.wCond, 0) & "){\n"
    result &= ctx.genMetalImpl(ast.wBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuDot:
    if ast.dParent.kind == gpuIdent and ast.dParent.symbol != nil and
       ast.dParent.symbol.coordBuiltin != gbkNone and ast.dField.kind == gpuIdent:
      # Component of an unsigned MSL builtin, cast to standard Nim `int` from default uint.
      result = "(int)(" & ctx.genMetalImpl(ast.dParent, 0) & '.' & ctx.genMetalImpl(ast.dField, 0) & ')'
    else:
      result = ctx.genMetalImpl(ast.dParent, 0) & '.' &
               ctx.genMetalImpl(ast.dField, 0)

  of gpuIndex:
    let base = ctx.genMetalImpl(ast.iArr, 0)
    let wrapped = if ast.iArr.kind in {gpuCast, gpuConv}: '(' & base & ')' else: base
    result = wrapped & '[' & ctx.genMetalImpl(ast.iIndex, 0) & ']'

  of gpuCall:
    case ast.cName.symbol.synchroBuiltin
    of gbkThreadgroupBarrier:
      # Every backend lowers the barrier to an alias template that instantiates
      # to the canonical call, so only this kind reaches the IR.
      # MSL takes the barrier with its memory flags.
      result = indentStr & "threadgroup_barrier(mem_flags::mem_threadgroup)"
    of gbkThreadgroupBarrierDevice:
      # mem_device orders device-memory accesses, `mem_threadgroup` does not.
      result = indentStr & "threadgroup_barrier(mem_flags::mem_device)"
    of gbkNone:
      case ast.cName.symbol.reductionBuiltin
      of gbkReduceAdd:
        result = indentStr & "simd_sum(" & ctx.genMetalImpl(ast.cArgs[0], 0) & ')'
      of gbkReduceMax:
        result = indentStr & "simd_max(" & ctx.genMetalImpl(ast.cArgs[0], 0) & ')'
      of gbkNone:
        case ast.cName.symbol.atomicsBuiltin
        of gbkAtomicFetchAdd:
          # MSL atomics act on `atomic_*` pointees: cast the raw device
          # pointer at the call site. The only permitted order is relaxed.
          # fp32 needs Metal 3+ (__HAVE_ATOMIC_FLOAT__).
          let atomicTy = case getPointeeKind(ctx.getOperandType(ast.cArgs[0]))
                         of gtFloat32: "atomic_float"
                         of gtInt32: "atomic_int"
                         of gtUint32: "atomic_uint"
                         else: error "Metal atomicFetchAdd: unsupported pointee type"
          result = indentStr & "atomic_fetch_add_explicit((volatile device " & atomicTy & "*)" &
                   ctx.genMetalImpl(ast.cArgs[0], 0) & ", " &
                   ctx.genMetalImpl(ast.cArgs[1], 0) & ", memory_order_relaxed)"
        of gbkNone:
          var args: seq[string]
          for a in ast.cArgs:
            args.add ctx.genMetalImpl(a, 0)
          result = indentStr & ctx.getFnName(bkMetal, ast) & '(' & args.join(", ") & ')'

  of gpuBinOp:
    ctx.withoutSemicolon:
      let l = ctx.genMetalImpl(ast.bLeft, 0)
      let r = ctx.genMetalImpl(ast.bRight, 0)
      result = indentStr & '(' & l & ' ' &
               ctx.genMetalImpl(ast.bOp, 0) & ' ' &
               r & ')'

  of gpuIdent:
    # Identity naming: the source-level builtin names are the MSL attribute names,
    # so a builtin-marked identifier emits verbatim and binds to the attribute
    # param `genKernelParams` emits for a kernel, or to the plain param a device
    # function receives for the same builtin.
    # Unmarked identifiers never get a param, so an undeclared name stays a loud MSL error.
    case ast.symbol.coordBuiltin
    of gbkThreadIndexInThreadgroup:
      # Scalar MSL attribute param is uint,
      # the canonical Nim type is int, the printer casts.
      result = "(int)" & ast.ident()
    of gbkThreadPositionInGrid, gbkThreadgroupPositionInGrid,
       gbkThreadPositionInThreadgroup, gbkThreadsPerThreadgroup, gbkThreadgroupsPerGrid:
      # Whole-value use: MSL attr params are uint3 and the canonical Nim type is
      # int3 (distinct C++ struct types, no implicit conversion), construct.
      let n = ast.ident()
      result = "int3((int)" & n & ".x, (int)" & n & ".y, (int)" & n & ".z)"
    of gbkNone:
      result = ast.ident()

  of gpuLit:
      result = genLit(ast)

  of gpuArrayLit:
    result = "{"
    for i, el in ast.aValues:
      result.add '(' & gpuTypeToString(ast.aLitType) & ')' & ctx.genMetalImpl(el, 0)
      if i < ast.aValues.high:
        result.add ", "
    result.add '}'

  of gpuReturn:
    result = indentStr & "return " & ctx.genMetalImpl(ast.rValue, 0)

  of gpuPrefix:
    result = ast.pOp & ctx.genMetalImpl(ast.pVal, 0)

  of gpuTypeDef:
    result.add "struct " & gpuTypeToString(ast.tTyp) & "{\n"
    if ast.tFields.len == 0:
      # MSL requires at least one field in a struct.
      result.add "  char _;\n"
    else:
      for el in ast.tFields:
        # MSL has no string type, gtString fields drop at emission (no runtime strings reach MSL)
        if el.typ.kind == gtString:
          continue
        # MSL requires an explicit address-space qualifier on pointer-typed fields,
        # the field's own type carries the space
        if el.typ.kind == gtPtr:
          result.add "  " & addrSpaceToMsl(el.typ.addressSpace) & ' ' &
                     gpuTypeToString(el.typ, el.name) & ";\n"
        else:
          result.add "  " & gpuTypeToString(el.typ, el.name) & ";\n"
    result.add '}'

  of gpuAlias:
    # Aliases come from `ctx.types`, MSL emits them as C++11 `using`
    result = "using " & gpuTypeToString(ast.aTyp) & " = " &
             ctx.genMetalImpl(ast.aTo, 0) & ';'

  of gpuObjConstr:
    # Braced init list: `TypeName{val1, val2, ...}` is a functional-style cast, a valid C++
    # expression usable with member access (gpuDot). A bare `{val}` braced-init-list
    # is not an expression.
    result = gpuTypeToString(ast.ocType, allowEmptyIdent = true) & "{"
    for i, el in ast.ocFields:
      if el.value.kind == gpuDiscard:
        result.add "{}"
      else:
        result.add ctx.genMetalImpl(el.value, 0)
      if i < ast.ocFields.len - 1:
        result.add ", "
    result.add '}'

  of gpuInlineAsm:
    # Raw MSL statement, the gpuBlock loop appends the `;` terminator.
    # CUDA/OpenCL/Vulkan wrap the body in `asm(...)`, MSL has no such wrapper,
    # the simdgroup intrinsics are plain statements.
    result = indentStr & genAsmStmt(ast).strip

  of gpuEmit:
    # Self-terminating raw text: the gpuBlock loop appends no `;`
    # (the emitted text owns its own terminators).
    result = genEmitStmt(ctx, ast,
      proc(c: var GpuContext, n: GpuAst): string = c.genMetalImpl(n, 0))

  of gpuComment:
    result = indentStr & "/* " & ast.comment & " */"

  of gpuConv:
    var castTyp = gpuTypeToString(ast.convTo, allowEmptyIdent = true)
    # Pointer-typed casts need an explicit address-space qualifier,
    # the cast target's type carries the space
    if ast.convTo.kind == gtPtr:
      castTyp = addrSpaceToMsl(ast.convTo.addressSpace) & ' ' & castTyp
    result = '(' & castTyp & ')' & ctx.genMetalImpl(ast.convExpr, 0)
  of gpuCast:
    var castTyp = gpuTypeToString(ast.cTo, allowEmptyIdent = true)
    if ast.cTo.kind == gtPtr:
      castTyp = addrSpaceToMsl(ast.cTo.addressSpace) & ' ' & castTyp
    result = '(' & castTyp & ')' & ctx.genMetalImpl(ast.cExpr, 0)

  of gpuAddr:
    result = "(&" & ctx.genMetalImpl(ast.aOf, 0) & ')'

  of gpuDeref:
    result = "(*" & ctx.genMetalImpl(ast.dOf, 0) & ')'

  of gpuConstexpr:
    ## Emits a C++14 `constexpr` variable, MSL supports them. Array types need
    ## the length in the type, hence the two emission shapes.
    let cInit =
      if ast.cValue.kind == gpuDiscard: "{}"
      else: ctx.genMetalImpl(ast.cValue, 0)
    if ast.cType.kind == gtArray:
      result = indentStr & "constexpr " & gpuTypeToString(ast.cType, ctx.genMetalImpl(ast.cIdent, 0)) & " = " & cInit
    else:
      result = indentStr & "constexpr " & gpuTypeToString(ast.cType, allowEmptyIdent = true) & ' ' &
               ctx.genMetalImpl(ast.cIdent, 0) & " = " & cInit
  of gpuMaterialize:
    result = ctx.genMetalImpl(ast.mExpr, 0) # C++ const& binds implicitly to temporaries

  else:
    error "Unhandled node kind in genMetal: " & ast.repr

proc genMetal*(ctx: var GpuContext, ast: GpuAst, indent = 0): string =
  ## MSL code for `ast`, used by `codegen` for top-level nodes.
  ctx.genMetalImpl(ast, indent)

proc codegen*(ctx: var GpuContext): string =
  ## Emits the full MSL translation unit:
  ## - the `metal_stdlib` header
  ## - the global blocks (types, global variables)
  ## - forward declarations for device functions, then every function body.
  # Resolve the value-level address spaces from the var pragmas and struct
  # pointer fields before any emission needs them.
  result.add "#include <metal_stdlib>\n"
  result.add "#include <metal_atomic>\n"
  result.add "using namespace metal;\n\n"

  # 1. generate code for the global blocks (types, global vars etc).
  #    Empty blocks (e.g. a kernel source with no globals) emit nothing.
  for blk in ctx.globalBlocks:
    let code = ctx.genMetal(blk)
    if code.len > 0:
      result.add code & ";\n\n"

  # 2. Forward declarations for the device functions, a kernel may precede its callees.
  #    Kernels need no forward declaration, no shader code calls a kernel.
  let fns = toSeq(ctx.fnTab.pairs)
  for (fnIdent, fn) in fns:
    if attGlobal in fn.pAttributes:
      continue
    let fnC = fn.clone()
    fnC.forwardDeclare = true
    result.add ctx.genMetal(fnC) & '\n'
  result.add "\n\n"

  for fnIdent, fn in ctx.fnTab.pairs:
    result.add ctx.genMetal(fn) & "\n\n"
