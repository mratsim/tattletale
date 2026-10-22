# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, strformat, strutils, sugar, sequtils, tables]

import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./lang_utils

proc gpuTypeToString*(t: GpuType,
                      ident: string = "",
                      allowArrayToPtr = false,
                      allowEmptyIdent = false): string

proc getInnerArrayType(t: GpuType): string =
  ## Returns the name of the inner most type for a nested array.
  case t.kind
  of gtArray:
    result = getInnerArrayType(t.aTyp)
  else:
    result = gpuTypeToString(t)

# ═══════════════════════════════════════════════════════════════════════
# Type mapping – OpenCL C types
# ═══════════════════════════════════════════════════════════════════════

proc gpuTypeToString*(t: GpuTypeKind): string =
  case t
  of gtBool: "int"            # OpenCL C has no bool, use int
  of gtUint8: "uchar"
  of gtUint16: "ushort"
  of gtUint32: "uint"
  of gtUint64: "ulong"
  of gtInt16: "short"
  of gtInt32: "int"
  of gtInt64: "long"
  of gtFloat32: "float"
  of gtFloat64: "double"
  of gtFloat16: "half"
  of gtVoid: "void"
  of gtSize_t: "size_t"
  of gtPtr: "*"
  of gtVoidPtr: "void*"
  of gtObject: "struct"
  of gtString: "const char*"
  of gtUA: ""       # UncheckedArray used as buffer pointer
  of gtStatic: "int"
  else:
    error "Invalid type : " & $t

proc gpuTypeToString*(t: GpuType, ident: string = "", allowArrayToPtr = false,
                      allowEmptyIdent = false,
                    ): string =
  ## Given an optional identifier required for array types
  var skipIdent = false
  case t.kind
  of gtPtr:
    let inner = if t.to.kind == gtUA: t.to.uaTo else: t.to
    if inner.kind == gtArray: # ptr to array type
      let ptrStar = gpuTypeToString(t.kind)
      result = gpuTypeToString(inner, '(' & ptrStar & ident & ')')
      skipIdent = true
    else:
      let typ = gpuTypeToString(inner, allowEmptyIdent = allowEmptyIdent)
      let ptrStar = gpuTypeToString(t.kind)
      # Explicit `ptr T` types bind device buffers and must carry the
      # __global qualifier wherever they appear (params, struct fields,
      # casts). Implicit `var T` pointers are per-thread registers.
      let space = if t.implicit: "" else: "__global "
      result = space & typ & ptrStar
  of gtArray:
    if ident.len == 0 and not allowEmptyIdent:
      when nimvm:
        error("Invalid call, got an array type but don't have an identifier: " & $t)
      else:
        raise newException(ValueError, "Invalid call, got an array type but don't have an identifier: " & $t)
    case t.aTyp.kind
    of gtArray: # nested array
      let typ = getInnerArrayType(t)
      let lengths = getInnerArrayLengths(t)
      result = typ & ' ' & ident & lengths
    else:
      if t.aLen == 0:
        result = gpuTypeToString(t.aTyp, allowEmptyIdent = allowEmptyIdent) & ' ' & ident & "[]"
      else:
        result = gpuTypeToString(t.aTyp, allowEmptyIdent = allowEmptyIdent) & ' ' & ident & '[' & $t.aLen & ']'
    skipIdent = true
  of gtGenericInst:
    result = "struct " & t.gName
    if t.gArgs.len == 1:
      result.add gpuTypeToShortString(t.gArgs[0])
    elif t.gArgs.len > 1:
      # 2+ args collapse to a short hash
      result.add mangledHashName("", $hash(t))
  of gtObject: result = "struct " & t.name
  of gtUA:     result = gpuTypeToString(t.uaTo, allowEmptyIdent = allowEmptyIdent)
  of gtStatic: result = "int"
  else:        result = gpuTypeToString(t.kind)

  if ident.len > 0 and not skipIdent:
    result.add ' ' & ident

proc genFunctionType*(typ: GpuType, fn: string, fnArgs: string): string =
  ## Returns the correct function signature
  if typ.kind == gtPtr and typ.to.kind == gtArray:
    let arrayTyp = typ.to.aTyp
    let innerTyp = gpuTypeToString(arrayTyp, allowEmptyIdent = true)
    let innerLen = $typ.to.aLen
    result = &"{innerTyp} (*{fn}({fnArgs}))[{innerLen}]"
  else:
    result = &"{gpuTypeToString(typ, allowEmptyIdent = true)} {fn}({fnArgs})"


proc containsFloat64(t: GpuType): bool =
  ## Returns true if the type tree contains float64 (double) somewhere.
  case t.kind
  of gtPtr:
    result = t.to.containsFloat64()
  of gtArray:
    result = t.aTyp.containsFloat64()
  of gtUA:
    result = t.uaTo.containsFloat64()
  of gtFloat64: result = true
  else: result = false

proc containsFloat16(t: GpuType): bool =
  ## Returns true if the type tree contains float16 (half) somewhere.
  case t.kind
  of gtPtr:
    result = t.to.containsFloat16()
  of gtArray:
    result = t.aTyp.containsFloat16()
  of gtUA:
    result = t.uaTo.containsFloat16()
  of gtFloat16: result = true
  else: result = false


# ═══════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════

proc genOpenCL*(ctx: var GpuContext, ast: GpuAst, indent = 0): string

# ── genOpenCL ─────────────────────────────────────────────────────────

proc genLit*(ast: GpuAst): string =
  ## Lower a literal node for the OpenCL backend.
  if ast.lType.kind == gtString:
    result = '"' & ast.lValue & '"'
  elif ast.lValue == "DEFAULT":
    # A missing object-constructor field (e.g. a nil smem pointer that
    # preflight fills later). OpenCL C is C99: `{}` is invalid for a scalar
    # ("scalar initializer cannot be empty"). `0` zero-initializes scalars,
    # pointers and aggregates (remaining members initialize implicitly).
    result = "0"
  else:
    case ast.lType.kind
    of gtFloat32: result = ast.lValue & "f"
    of gtUint32: result = ast.lValue & "U"
    of gtUint64: result = ast.lValue & "ULL"
    of gtInt64:  result = ast.lValue & "LL"
    of gtInt16, gtUint16, gtUint8, gtBool:
      result = '(' & gpuTypeToString(ast.lType, allowEmptyIdent = true) & ')' & ast.lValue
    else:
      result = ast.lValue

proc openclCoordFieldAccess(kind: GpuCoordBuiltinKind, field: string): string =
  ## OpenCL C spelling of a canonical coordinate field access: each component
  ## maps to a `get_*(d)` call with the dimension as a literal.
  ## Returns "" for kinds without a per-component mapping
  ## (whole-value uses of the vector builtins have no OpenCL spelling).
  let d =
    case field
    of "x": "0"
    of "y": "1"
    of "z": "2"
    else: return ""
  case kind
  of gbkThreadPositionInGrid: "(int)get_global_id(" & d & ")"
  of gbkThreadgroupPositionInGrid: "(int)get_group_id(" & d & ")"
  of gbkThreadPositionInThreadgroup: "(int)get_local_id(" & d & ")"
  of gbkThreadsPerThreadgroup: "(int)get_local_size(" & d & ")"
  of gbkThreadgroupsPerGrid: "(int)get_num_groups(" & d & ")"
  of gbkThreadIndexInThreadgroup, gbkNone: ""

proc openclCoordIdent(kind: GpuCoordBuiltinKind, name: string): string =
  ## OpenCL spelling of a canonical scalar coordinate builtin referenced whole.
  case kind
  of gbkThreadIndexInThreadgroup:
    # x-major flat thread index, parenthesized so a trailing `* k`
    # cannot mis-associate into `+ get_local_id(0) * k`.
    # OpenCL index builtins are
    # size_t, the canonical Nim type is int, the printer casts.
    "(int)(get_local_id(2)*get_local_size(0)*get_local_size(1) + get_local_id(1)*get_local_size(0) + get_local_id(0))"
  of gbkThreadPositionInGrid, gbkThreadgroupPositionInGrid,
     gbkThreadPositionInThreadgroup, gbkThreadsPerThreadgroup,
     gbkThreadgroupsPerGrid:
    # Whole-value vector coordinates have no OpenCL spelling. Their names
    # are emitted verbatim (zero in-tree uses, non-goal).
    name
  of gbkNone:
    # Unreachable-by-construction: ident sites emit gbkNone verbatim, so this branch never fires.
    error "coordinate site with no coordinate builtin kind: " & name

proc genOpenCL*(ctx: var GpuContext, ast: GpuAst, indent = 0): string =
  ## The actual OpenCL C code generator.
  let indentStr = "  ".repeat(indent)
  case ast.kind
  of gpuDiscard: return

  of gpuProc:
    let attrs = collect:
      for att in ast.pAttributes:
        $att

    let isKernel = attGlobal in ast.pAttributes
    var params: seq[string]
    for p in ast.pParams:
      if p.passByRef and not isKernel:
        # const Type* _p_name — pointer to const for large structs (no C++ references)
        # Note: lowerByrefParamsImpl already renamed the param to _p_
        params.add "const " & gpuTypeToString(p.typ, allowEmptyIdent = true) & "* " & p.ident.ident()
      elif p.addressSpace == asSMEM:
        # __local T* — shared memory
        let inner = gpuTypeToString(p.typ.to, allowEmptyIdent = true)
        params.add &"__local {inner}* {p.ident.ident()}"
      elif p.typ.kind == gtPtr and p.addressSpace == asDevice:
        # __global T* restrict — explicit `ptr T` params bind device
        # buffers, in kernel signatures and device functions alike.
        # Implicit `var T` pointer params (private arrays) take the
        # plain unqualified spelling below.
        let inner = gpuTypeToString(p.typ.to, allowEmptyIdent = true)
        params.add &"__global {inner}* restrict {p.ident.ident()}"
      else:
        # By-value array params decay to pointers. They are read-only
        # (a plain `array[T]` param), so emit `const`: the call site
        # passes const lvalue arrays, and a non-const `T*` rejects them.
        if p.typ.kind == gtArray:
          params.add "const " & gpuTypeToString(p.typ, p.ident.ident(), allowEmptyIdent = false)
        else:
          params.add gpuTypeToString(p.typ, p.ident.ident(), allowEmptyIdent = false)
    let fnArgs = params.join(", ")
    let fnSig = genFunctionType(ast.pRetType, ast.pName.ident(), fnArgs)

    if isKernel:
      # Global kernel entry point → `__kernel void kernelName(...)`
      result = indentStr & "__kernel void " & ast.pName.ident() & '(' & fnArgs & ')'
    else:
      # Device function → `static inline retType fnName(...)`
      result = indentStr & "static inline " & fnSig

    if ast.forwardDeclare:
      result.add ';'
    else:
      result.add " {\n"
      # Local copies for byref params are emitted as gpuVar nodes in the body
      # by lowerByrefParamsImpl — no additional codegen needed here
      result &= ctx.genOpenCL(ast.pBody, indent + 1)
      result &= '\n' & indentStr & '}'

  of gpuBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "{ // " & ast.blockLabel & '\n'
    for i, el in ast.statements:
      result.add ctx.genOpenCL(el, indent)
      if not el.isSelfTerminating() and not ctx.skipSemicolon:
        result.add ';'
      if i < ast.statements.high:
        result.add '\n'
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "} // " & ast.blockLabel & '\n'


  of gpuVar:
    # The var's address-space keyword, prefixing the declaration ident for
    # asSMEM arrays (`__local uint scratch[8]`).
    var typeStr: string
    case ast.addressSpace
    of asSMEM:
      typeStr = "__local " & gpuTypeToString(ast.vType, ast.vName.ident())
    of asRMEM:
      typeStr = "__private " & gpuTypeToString(ast.vType, ast.vName.ident())
    of asConstant:
      typeStr = "__constant " & gpuTypeToString(ast.vType, ast.vName.ident())
    of asDevice:
      typeStr = gpuTypeToString(ast.vType, ast.vName.ident())

    result = indentStr & typeStr
    if ast.vInit.kind != gpuDiscard:
      result &= " = " & ctx.genOpenCL(ast.vInit)

  of gpuAssign:
    result = indentStr & ctx.genOpenCL(ast.aLeft) & ' ' & ast.aOp & ' ' &
             ctx.genOpenCL(ast.aRight)
  of gpuIf:
    ctx.withoutSemicolon:
      result = indentStr & "if (" & ctx.genOpenCL(ast.ifCond) & ") {\n"
    result &= ctx.genOpenCL(ast.ifThen, indent + 1) & '\n'
    result &= indentStr & '}'
    for el in ast.ifElifs:
      ctx.withoutSemicolon:
        result &= " else if (" & ctx.genOpenCL(el.cond) & ") {\n"
      result &= ctx.genOpenCL(el.body, indent + 1) & '\n'
      result &= indentStr & '}'
    if ast.ifElse.kind != gpuDiscard:
      result &= " else {\n"
      result &= ctx.genOpenCL(ast.ifElse, indent + 1) & '\n'
      result &= indentStr & '}'

  of gpuTernary:
    ctx.withoutSemicolon:
      result = '(' & ctx.genOpenCL(ast.tCond) & " ? " &
               ctx.genOpenCL(ast.tThen) & " : " &
               ctx.genOpenCL(ast.tElse) & ')'

  of gpuFor:
    let cmp = if ast.fRangeKind == rkInclusive: " <= " else: " < "
    result = indentStr & "for(int " & ast.fVar.ident() & " = " &
             ctx.genOpenCL(ast.fStart) & "; " &
             ast.fVar.ident() & cmp & ctx.genOpenCL(ast.fEnd) & "; " &
             ast.fVar.ident() & " += " & ctx.genOpenCL(ast.fStep) & ") {\n"
    result &= ctx.genOpenCL(ast.fBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genOpenCL(ast.wCond) & "){\n"
    result &= ctx.genOpenCL(ast.wBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuDot:
    if ast.dParent.kind == gpuIdent and ast.dParent.symbol != nil and
       ast.dParent.symbol.coordBuiltin != gbkNone and
       ast.dField.kind == gpuIdent:
      let mapped = openclCoordFieldAccess(ast.dParent.symbol.coordBuiltin, ast.dField.ident())
      if mapped.len > 0:
        result = mapped
      else:
        result = ctx.genOpenCL(ast.dParent) & '.' & ctx.genOpenCL(ast.dField)
    else:
      result = ctx.genOpenCL(ast.dParent) & '.' & ctx.genOpenCL(ast.dField)

  of gpuIndex:
    result = ctx.genOpenCL(ast.iArr) & '[' & ctx.genOpenCL(ast.iIndex) & ']'

  of gpuCall:
    case ast.cName.symbol.synchroBuiltin
    of gbkThreadgroupBarrier:
      # Every backend name of the barrier is an alias template that semantic analysis
      # expands to the canonical call, so only this kind reaches the IR.
      # OpenCL spells it `barrier(CLK_LOCAL_MEM_FENCE)`.
      result = indentStr & "barrier(CLK_LOCAL_MEM_FENCE)"
    of gbkThreadgroupBarrierDevice:
      # Global fence bit orders device-address-space accesses.
      result = indentStr & "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)"
    of gbkNone:
      case ast.cName.symbol.reductionBuiltin
      of gbkReduceAdd:
        result = indentStr & "sub_group_reduce_add(" & ctx.genOpenCL(ast.cArgs[0]) & ')'
      of gbkReduceMax:
        result = indentStr & "sub_group_reduce_max(" & ctx.genOpenCL(ast.cArgs[0]) & ')'
      of gbkNone:
        case ast.cName.symbol.atomicsBuiltin
        of gbkAtomicFetchAdd:
          # fp needs cl_ext_float_atomics, the pragma is emitted at
          # module scope when the AST contains an fp atomicFetchAdd.
          # int atomic_fetch_add is OpenCL 2.0 core.
          # atomic_fetch_add acts on `atomic_*` pointees: cast the raw
          # __global pointer at the call site.
          let atomicTy = case getPointeeKind(ctx.getOperandType(ast.cArgs[0]))
                         of gtFloat32: "atomic_float"
                         of gtInt32: "atomic_int"
                         of gtUint32: "atomic_uint"
                         else: error "OpenCL atomicFetchAdd: unsupported pointee type"
          result = indentStr & "atomic_fetch_add((volatile __global " & atomicTy & "*)" &
                   ctx.genOpenCL(ast.cArgs[0]) & ", " &
                   ctx.genOpenCL(ast.cArgs[1]) & ')'
        of gbkNone:
          var clArgs: seq[string]
          for arg in ast.cArgs:
            clArgs.add ctx.genOpenCL(arg)
          result = indentStr & ctx.getFnName(bkOpenCL, ast) & '(' & clArgs.join(", ") & ')'


  of gpuBinOp:
    ctx.withoutSemicolon:
      let l = ctx.genOpenCL(ast.bLeft)
      let r = ctx.genOpenCL(ast.bRight)
      result = indentStr & '(' & l & ' ' &
               ctx.genOpenCL(ast.bOp) & ' ' &
               r & ')'

  of gpuIdent:
    if ast.symbol != nil:
      case ast.symbol.coordBuiltin
      of gbkNone:
        # A local shadowing a canonical name, or a call-shaped builtin
        # (printf, cvtaGenericToShared): emit verbatim.
        result = ast.ident()
      else:
        # Canonical coordinates are the MSL vocabulary. This printer maps
        # each kind to its OpenCL spelling: get_global_id, get_group_id,
        # get_local_id, get_local_size, get_num_groups and the flat-index
        # linearization.
        result = openclCoordIdent(ast.symbol.coordBuiltin, ast.ident())
    else:
      result = ast.ident()

  of gpuLit:
      result = genLit(ast)

  of gpuArrayLit:
    result = "{"
    for i, el in ast.aValues:
      result.add '(' & gpuTypeToString(ast.aLitType) & ')' & ctx.genOpenCL(el)
      if i < ast.aValues.high:
        result.add ", "
    result.add '}'

  of gpuReturn:
    result = indentStr & "return " & ctx.genOpenCL(ast.rValue)

  of gpuPrefix:
    result = ast.pOp & ctx.genOpenCL(ast.pVal)

  of gpuTypeDef:
    result = gpuTypeToString(ast.tTyp) & " {\n"
    if ast.tFields.len == 0:
      # OpenCL C requires at least one field in a struct.
      result.add "  char _;\n"
    else:
      for el in ast.tFields:
        result.add "  " & gpuTypeToString(el.typ, el.name) & ";\n"
    result.add '}'

  of gpuObjConstr:
    # C99 compound literal: (TypeName){val1, val2, ...}
    # Using compound literal syntax ensures the result is a valid expression
    # (bare braced-init-lists cannot be used with member access).
    # OpenCL C is C99-based, so we use the C99 `(type){init}` syntax,
    # NOT C++ functional-style cast `Type{init}`.
    result = "(" & gpuTypeToString(ast.ocType, allowEmptyIdent = true) & "){"
    if ast.ocFields.len == 0:
      # Empty struct (e.g. EpiIdentity/EpiReLU): C99 rejects `{}` (empty
      # initializer list), `{0}` zero-initializes the single member.
      result.add "0"
    else:
      for i, el in ast.ocFields:
        if el.value.kind == gpuDiscard or
           (el.value.kind == gpuLit and el.value.lValue == "DEFAULT"):
          # Missing constructor field (e.g. a nil smem pointer that preflight
          # fills later). C99 forbids `{}` for scalars, and a bare `0` would
          # be absorbed by a leading array member, so brace aggregates.
          if el.typ.kind in {gtArray, gtObject, gtGenericInst}:
            result.add "{0}"
          else:
            result.add "0"
        else:
          result.add ctx.genOpenCL(el.value)
        if i < ast.ocFields.len - 1:
          result.add ", "
    result.add "}"
  of gpuInlineAsm:
    result = indentStr & "asm(" & genAsmStmt(ast).strip & ");"

  of gpuEmit:
    # Self-terminating raw text: the gpuBlock loop appends no `;`
    # (the emitted text owns its own terminators).
    result = genEmitStmt(ctx, ast,
      proc(c: var GpuContext; n: GpuAst): string = c.genOpenCL(n, 0))

  of gpuComment:
    result = indentStr & "/* " & ast.comment & " */"

  of gpuConv:
    result = '(' & gpuTypeToString(ast.convTo, allowEmptyIdent = true) & ')' & ctx.genOpenCL(ast.convExpr)

  of gpuCast:
    result = '(' & gpuTypeToString(ast.cTo, allowEmptyIdent = true) & ')' & ctx.genOpenCL(ast.cExpr)

  of gpuAddr:
    result = "(&" & ctx.genOpenCL(ast.aOf) & ')'

  of gpuDeref:
    result = "(*" & ctx.genOpenCL(ast.dOf) & ')'

  of gpuConstexpr:
    if ast.cType.kind == gtArray:
      result = indentStr & "const " & gpuTypeToString(ast.cType, ctx.genOpenCL(ast.cIdent)) & " = " & ctx.genOpenCL(ast.cValue)
    else:
      result = indentStr & "const " & gpuTypeToString(ast.cType, allowEmptyIdent = true) & ' ' & ctx.genOpenCL(ast.cIdent) & " = " & ctx.genOpenCL(ast.cValue)
  of gpuMaterialize:
    let typ = gpuTypeToString(ast.mType, allowEmptyIdent = true)
    result = "&(" & typ & "){" & ctx.genOpenCL(ast.mExpr) & "}"

  else:
    echo "Unhandled node kind in genOpenCL: ", ast.kind
    error "Unhandled node kind in genOpenCL: " & ast.repr

# ═══════════════════════════════════════════════════════════════════════
# Top-level codegen
# ═══════════════════════════════════════════════════════════════════════

proc usesFloatAtomic(ctx: GpuContext, n: GpuAst): bool =
  ## True when the AST contains an fp `atomicFetchAdd` call
  ##
  ## - the cl_ext_float_atomics pragma is emitted at module scope
  ## - int atomic_fetch_add is OpenCL 2.0 core
  if n.kind == gpuCall and n.cName.symbol != nil and
     n.cName.symbol.atomicsBuiltin == gbkAtomicFetchAdd and
     getPointeeKind(ctx.getOperandType(n.cArgs[0])) == gtFloat32:
    return true
  for ch in n:
    if ctx.usesFloatAtomic(ch):
      return true

proc usesSubgroupReduce(ctx: GpuContext, n: GpuAst): bool =
  if n.kind == gpuCall and n.cName.symbol != nil and
     n.cName.symbol.reductionBuiltin != gbkNone:
    return true
  for ch in n:
    if ctx.usesSubgroupReduce(ch):
      return true

proc codegen*(ctx: var GpuContext): string =
  ## Generate the actual code for all pieces of the puzzle.
  # Check if we need fp64/fp16 extensions
  var needsFp64 = false
  var needsFp16 = false
  var needsFloatAtomic = false
  var needsSubgroup = false
  for blk in ctx.globalBlocks:
    # We could scan types here, but for now we check in the generated code
    discard
  for fnIdent, fn in ctx.fnTab:
    # Check for float64 usage in function
    if fn.pRetType.kind == gtFloat64:
      needsFp64 = true
    if fn.pRetType.kind == gtFloat16:
      needsFp16 = true
    for p in fn.pParams:
      if p.typ.containsFloat64():
        needsFp64 = true
      if p.typ.containsFloat16():
        needsFp16 = true
    if ctx.usesFloatAtomic(fn.pBody):
      needsFloatAtomic = true
    if ctx.usesSubgroupReduce(fn.pBody):
      needsSubgroup = true

  if needsFp64:
    result = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n"
  if needsFp16:
    result = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
  if needsFloatAtomic:
    result.add "#pragma OPENCL EXTENSION cl_ext_float_atomics : enable\n\n"
  if needsSubgroup:
    result.add "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n\n"

  # 1. Generate code for the global blocks (types, global vars etc)
  for blk in ctx.globalBlocks:
    result.add ctx.genOpenCL(blk) & ";\n\n"

  # 2. Forward declarations
  let fns = toSeq(ctx.fnTab.pairs)
  for (fnIdent, fn) in fns:
    let fnC = fn.clone()
    fnC.forwardDeclare = true
    result.add ctx.genOpenCL(fnC) & '\n'
  result.add "\n\n"

  # 3. Full function definitions
  for fnIdent, fn in ctx.fnTab:
    result.add ctx.genOpenCL(fn) & "\n\n"
