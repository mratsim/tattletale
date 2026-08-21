# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, strformat, strutils, sugar, sequtils, tables]

import ../ir/gpu_types
import ./lang_utils
import ../passes/passes_preprocessing as pp

proc gpuTypeToString*(t: GpuType,
                      ident: string = "",
                      allowArrayToPtr = false,
                      allowEmptyIdent = false): string
proc size*(ctx: var GpuContext, a: GpuType): string = size(gpuTypeToString(a, allowEmptyIdent = true))

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
  of gtVoid: "void"
  of gtSize_t: "size_t"
  of gtPtr: "*"
  of gtVoidPtr: "void*"
  of gtObject: "struct"
  of gtString: "const char*"
  of gtUA: ""       # UncheckedArray used as buffer pointer
  of gtStatic: "int"
  else:
    raiseAssert "Invalid type : " & $t

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
      result = typ & ptrStar
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
    for i, g in t.gArgs:
      result.add gpuTypeToShortString(g)
      if i < t.gArgs.high:
        result.add 'x'
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


# ═══════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════

proc scanFunctions(ctx: var GpuContext, n: GpuAst) =
  ## Iterates over the given function and checks for all `gpuCall` nodes. Any function
  ## called in the scope is added to `fnTab`. This is a form of dead code elimination.
  case n.kind
  of gpuCall:
    let fn = n.cName
    if fn in ctx.allFnTab:
      if fn notin ctx.fnTab:
        let fnCalled = ctx.allFnTab[fn]
        ctx.fnTab[fn] = fnCalled
        for ch in fnCalled:
          ctx.scanFunctions(ch)
    for arg in n.cArgs:
      ctx.scanFunctions(arg)
  else:
    for ch in n:
      ctx.scanFunctions(ch)


# ═══════════════════════════════════════════════════════════════════════
# Code generation
# ═══════════════════════════════════════════════════════════════════════

proc genOpenCL*(ctx: var GpuContext, ast: GpuAst, indent = 0): string
proc size(ctx: var GpuContext, a: GpuAst): string = size(ctx.genOpenCL(a))
proc address(ctx: var GpuContext, a: GpuAst): string = address(ctx.genOpenCL(a))

# ── Preprocess ────────────────────────────────────────────────────────

proc preprocess*(ctx: var GpuContext, ast: GpuAst, kernel: string = "") =
  # 1. Add all data from `genericInsts` and `types` tables
  for k, v in pairs(ctx.genericInsts):
    ctx.allFnTab[k] = v
  for k, typ in pairs(ctx.types):
    ctx.globalBlocks.add typ

  # 2. Fill table with all *global* functions or *only* the specific `kernel`
  var varBlock = GpuAst(kind: gpuBlock)
  ctx.farmTopLevel(ast, kernel, varBlock)
  ctx.globalBlocks.add varBlock

  # 3. Traverse global functions for any `gpuCall` node and record in `fnTab`.
  let fns = toSeq(ctx.fnTab.pairs)
  for (fnIdent, fn) in fns:
    let fnOrig = ctx.allFnTab[fnIdent]
    ctx.scanFunctions(fn)

  # 4. Apply Phase 6 byref lowering passes
  # NOTE: genericInsts entries were already copied to allFnTab in step 1,
  # so we only iterate allFnTab (not genericInsts separately) to avoid
  # double-processing the same ref object.
  # lowerByrefParamsImpl: process allFnTab.
  # fnTab shares references with allFnTab for called functions (scanFunctions),
  # and kernels are skipped, so no separate fnTab loop needed.
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      pp.lowerByrefParamsImpl(ctx, fn)
  # insertByrefAddrsImpl: needs to visit call sites in fnTab entries too,
  # because fnTab has clones of top-level kernels from farmTopLevel.
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      pp.insertByrefAddrsImpl(ctx, fn)
  for fnIdent, fn in ctx.fnTab.mpairs:
    if fn.kind == gpuProc:
      pp.insertByrefAddrsImpl(ctx, fn)

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
  of gbkThreadPositionInGrid: "get_global_id(" & d & ")"
  of gbkThreadgroupPositionInGrid: "get_group_id(" & d & ")"
  of gbkThreadPositionInThreadgroup: "get_local_id(" & d & ")"
  of gbkThreadsPerThreadgroup: "get_local_size(" & d & ")"
  of gbkThreadgroupsPerGrid: "get_num_groups(" & d & ")"
  of gbkThreadIndexInThreadgroup, gbkNone: ""

proc openclCoordIdent(kind: GpuCoordBuiltinKind, name: string): string =
  ## OpenCL spelling of a canonical scalar coordinate builtin referenced whole.
  case kind
  of gbkThreadIndexInThreadgroup:
    # x-major flat thread index, parenthesized so a trailing `* k` cannot
    # mis-associate into `+ get_local_id(0) * k`.
    "(get_local_id(2)*get_local_size(0)*get_local_size(1) + get_local_id(1)*get_local_size(0) + get_local_id(0))"
  of gbkThreadPositionInGrid, gbkThreadgroupPositionInGrid,
     gbkThreadPositionInThreadgroup, gbkThreadsPerThreadgroup,
     gbkThreadgroupsPerGrid:
    # Whole-value vector coordinates have no OpenCL spelling. Their names
    # are emitted verbatim (zero in-tree uses, non-goal).
    name
  of gbkNone:
    # Unreachable-by-construction: ident sites emit gbkNone verbatim, so this branch never fires.
    raiseAssert "coordinate site with no coordinate builtin kind: " & name

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
      elif isKernel and p.typ.kind == gtPtr:
        # __global T* restrict — for kernel pointer parameters
        let inner = gpuTypeToString(p.typ.to, allowEmptyIdent = true)
        params.add &"__global {inner}* restrict {p.ident.ident()}"
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
    result = indentStr & ctx.genOpenCL(ast.aLeft) & " = " & ctx.genOpenCL(ast.aRight)
  of gpuIf:
    ctx.withoutSemicolon:
      result = indentStr & "if (" & ctx.genOpenCL(ast.ifCond) & ") {\n"
    result &= ctx.genOpenCL(ast.ifThen, indent + 1) & '\n'
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
             ast.fVar.ident() & "++) {\n"
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
      # Every backend spelling of the barrier is an alias template that sem
      # expands to the canonical call, so only this kind reaches the IR.
      # OpenCL spells it `barrier(CLK_LOCAL_MEM_FENCE)`.
      result = indentStr & "barrier(CLK_LOCAL_MEM_FENCE)"
    of gbkNone:
      var clArgs: seq[string]
      for arg in ast.cArgs:
        clArgs.add ctx.genOpenCL(arg)
      result = indentStr & ctx.getFnName(bkOpenCL, ast) & '(' & clArgs.join(", ") & ')'

  of gpuTemplateCall:
    when nimvm:
      error("Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `opencl` macro.")
    else:
      raise newException(ValueError, "Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `opencl` macro.")

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
    raiseAssert "Unhandled node kind in genOpenCL: " & ast.repr

# ═══════════════════════════════════════════════════════════════════════
# Top-level codegen
# ═══════════════════════════════════════════════════════════════════════

proc codegen*(ctx: var GpuContext): string =
  ## Generate the actual code for all pieces of the puzzle.
  # Check if we need fp64 extension
  var needsFp64 = false
  for blk in ctx.globalBlocks:
    # We could scan types here, but for now we check in the generated code
    discard
  for fnIdent, fn in ctx.fnTab:
    # Check for float64 usage in function
    if fn.pRetType.kind == gtFloat64:
      needsFp64 = true
    for p in fn.pParams:
      if p.typ.containsFloat64():
        needsFp64 = true

  if needsFp64:
    result = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n"

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
