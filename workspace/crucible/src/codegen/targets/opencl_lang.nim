# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, strformat, strutils, sugar, sequtils, tables]

import ../ir/gpu_types
import ./lang_utils

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

proc gpuTypeToShortString*(t: GpuType): string =
  ## Short, space-free type name for use in generic struct identifiers.
  case t.kind
  of gtUint8:   result = "u8"
  of gtUint16:  result = "u16"
  of gtUint32:  result = "u32"
  of gtUint64:  result = "u64"
  of gtInt16:   result = "i16"
  of gtInt32:   result = "i32"
  of gtInt64:   result = "i64"
  of gtFloat32: result = "f32"
  of gtFloat64: result = "f64"
  of gtBool:    result = "bool"
  of gtObject:
    result = $t.name
  of gtGenericInst:
    result = t.gName
    if t.gArgs.len > 0:
      result.add '_'
      for i, g in t.gArgs:
        if i > 0: result.add 'x'
        result.add gpuTypeToShortString(g)
  of gtVoidPtr: result = "void_ptr"
  of gtPtr:
    result = "ptr_" & gpuTypeToShortString(t.to)
  of gtStatic:
    result = $t.sValue
  else:
    result = $t.kind # fallback — safe but verbose

proc gpuTypeToString*(t: GpuType, ident: string = "", allowArrayToPtr = false,
                      allowEmptyIdent = false,
                    ): string =
  ## Given an optional identifier required for array types
  var skipIdent = false
  case t.kind
  of gtPtr:
    var t = t
    # Strip `gtUA` layer (ptr UncheckedArray[T] → ptr T)
    if t.to.kind == gtUA:
      t.to = t.to.uaTo

    if t.to.kind == gtArray: # ptr to array type
      let ptrStar = gpuTypeToString(t.kind)
      result = gpuTypeToString(t.to, '(' & ptrStar & ident & ')')
      skipIdent = true
    else:
      let typ = gpuTypeToString(t.to, allowEmptyIdent = allowEmptyIdent)
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
    if t.gArgs.len > 0:
      result.add '_'
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

proc genMemcpy(lhs, rhs, size: string): string =
  result = &"memcpy({lhs}, {rhs}, {size})"

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

proc getFieldType(t: GpuType, field: GpuAst): GpuType =
  doAssert field.kind == gpuIdent, "Field is not an ident: " & $field
  doAssert t.kind in [gtObject, gtGenericInst]
  let flds = if t.kind == gtObject: t.oFields
               else: t.gFields
  result = GpuType(kind: gtInvalid)
  for f in flds:
    if f.name == field.ident():
      return f.typ

proc getType(ctx: var GpuContext, arg: GpuAst, typeOfIndex = true): GpuType =
  template dfl(): untyped = GpuType(kind: gtInvalid)
  case arg.kind
  of gpuIdent: arg.iTyp
  of gpuAddr: GpuType(kind: gtPtr, to: ctx.getType(arg.aOf))
  of gpuDeref:
    let argTyp = ctx.getType(arg.dOf)
    doAssert argTyp.kind == gtPtr
    argTyp.to
  of gpuCall: dfl()
  of gpuIndex:
    let arrType = ctx.getType(arg.iArr)
    if typeOfIndex:
      case arrType.kind
      of gtPtr:   arrType.to
      of gtUA:    arrType.uaTo
      of gtArray: arrType.aTyp
      else: raiseAssert "`gpuIndex` cannot be of a non pointer / array type: " & $arrType
    else:
      arrType
  of gpuDot:
    let parentTyp = ctx.getType(arg.dParent)
    parentTyp.getFieldType(arg.dField)
  of gpuLit: arg.lType
  of gpuBinOp: dfl()
  of gpuPrefix: ctx.getType(arg.pVal)
  of gpuConv: arg.convTo
  of gpuCast: arg.cTo
  else:
    raiseAssert "Not implemented to determine type from node: " & $arg

proc makeCodeValid(ctx: var GpuContext, n: var GpuAst) =
  ## Addresses AST patterns that need to be rewritten for OpenCL C.
  ## Similar to CUDA – `Index(Deref(Ident))` → `Index(Ident)` for pointer types.
  case n.kind
  of gpuIndex:
    if n.iArr.kind == gpuDeref:
      let typ = ctx.getType(n, typeOfIndex = false)
      if typ.kind != gtArray:
        n = GpuAst(kind: gpuIndex, iArr: n.iArr.dOf, iIndex: n.iIndex)
    else:
      for ch in mitems(n):
        ctx.makeCodeValid(ch)
  else:
    for ch in mitems(n):
      ctx.makeCodeValid(ch)

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

  # 4. Finalize AST transformations
  for (fnIdent, fn) in mpairs(ctx.fnTab):
    ctx.makeCodeValid(fn)

# ── genOpenCL ─────────────────────────────────────────────────────────

proc genOpenCL*(ctx: var GpuContext, ast: GpuAst, indent = 0): string =
  ## The actual OpenCL C code generator.
  let indentStr = "  ".repeat(indent)
  case ast.kind
  of gpuVoid: return

  of gpuProc:
    let attrs = collect:
      for att in ast.pAttributes:
        $att

    let isKernel = attGlobal in ast.pAttributes
    var params: seq[string]
    for p in ast.pParams:
      if p.passByRef and not isKernel:
        # const Type* _p_name — pointer to const for large structs (no C++ references)
        params.add "const " & gpuTypeToString(p.typ, allowEmptyIdent = true) & "* _p_" & p.ident.ident()
      elif p.addressSpace == asWorkspace:
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
      # Local copies for byref params — const pointer is dereferenced into local
      for p in ast.pParams:
        if p.passByRef and not isKernel:
          let innerIndent = "  ".repeat(indent + 1)
          result.add innerIndent & gpuTypeToString(p.typ, p.ident.ident()) & " = *_p_" & p.ident.ident() & ";\n"
      result &= ctx.genOpenCL(ast.pBody, indent + 1)
      result &= '\n' & indentStr & '}'

  of gpuBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "{ // " & ast.blockLabel & '\n'
    for i, el in ast.statements:
      result.add ctx.genOpenCL(el, indent)
      if el.kind != gpuBlock and not ctx.skipSemicolon:
        result.add ';'
      if i < ast.statements.high:
        result.add '\n'
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "} // " & ast.blockLabel & '\n'

  of gpuVar:
    let attrs = if ast.vAttributes.len > 0: ast.vAttributes.join(" ") & ' '
                else: ""
    # Handle __local shared variables
    var typeStr: string
    if atvShared in ast.vAttributes:
      let inner = gpuTypeToString(ast.vType, allowEmptyIdent = true)
      typeStr = "__local " & inner & ' ' & ast.vName.ident()
    else:
      typeStr = attrs & gpuTypeToString(ast.vType, ast.vName.ident())

    result = indentStr & typeStr
    if ast.vInit.kind != gpuVoid and not ast.vRequiresMemcpy:
      result &= " = " & ctx.genOpenCL(ast.vInit)
    elif ast.vInit.kind != gpuVoid:
      result.add ";\n"
      result.add indentStr & genMemcpy(address(ast.vName.ident()), ctx.address(ast.vInit),
                                       size(ast.vName.ident()))

  of gpuAssign:
    if ast.aRequiresMemcpy:
      result = indentStr & genMemcpy(ctx.address(ast.aLeft), ctx.address(ast.aRight),
                                     ctx.size(ast.aLeft))
    else:
      result = indentStr & ctx.genOpenCL(ast.aLeft) & " = " & ctx.genOpenCL(ast.aRight)

  of gpuIf:
    ctx.withoutSemicolon:
      result = indentStr & "if (" & ctx.genOpenCL(ast.ifCond) & ") {\n"
    result &= ctx.genOpenCL(ast.ifThen, indent + 1) & '\n'
    result &= indentStr & '}'
    if ast.ifElse.kind != gpuVoid:
      result &= " else {\n"
      result &= ctx.genOpenCL(ast.ifElse, indent + 1) & '\n'
      result &= indentStr & '}'

  of gpuTernary:
    ctx.withoutSemicolon:
      result = '(' & ctx.genOpenCL(ast.tCond) & " ? " &
               ctx.genOpenCL(ast.tThen) & " : " &
               ctx.genOpenCL(ast.tElse) & ')'

  of gpuFor:
    result = indentStr & "for(int " & ast.fVar.ident() & " = " &
             ctx.genOpenCL(ast.fStart) & "; " &
             ast.fVar.ident() & " < " & ctx.genOpenCL(ast.fEnd) & "; " &
             ast.fVar.ident() & "++) {\n"
    result &= ctx.genOpenCL(ast.fBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genOpenCL(ast.wCond) & "){\n"
    result &= ctx.genOpenCL(ast.wBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuDot:
    result = ctx.genOpenCL(ast.dParent) & '.' & ctx.genOpenCL(ast.dField)

  of gpuIndex:
    result = ctx.genOpenCL(ast.iArr) & '[' & ctx.genOpenCL(ast.iIndex) & ']'

  of gpuCall:
    let fnName = ast.cName
    let fnParams = ctx.getFnParams(fnName)
    var clArgs: seq[string]
    for i, arg in ast.cArgs:
      if i < fnParams.len and fnParams[i].passByRef:
        clArgs.add "&" & ctx.genOpenCL(arg)
      else:
        clArgs.add ctx.genOpenCL(arg)
    result = indentStr & fnName.ident() & '(' & clArgs.join(", ") & ')'

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
    result = ast.ident()

  of gpuLit:
    if ast.lType.kind == gtString: result = '"' & ast.lValue & '"'
    elif ast.lValue == "DEFAULT": result = "{}"
    else: result = ast.lValue

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
    for el in ast.tFields:
      result.add "  " & gpuTypeToString(el.typ, el.name) & ";\n"
    result.add '}'

  of gpuObjConstr:
    result = "{"
    for i, el in ast.ocFields:
      result.add ctx.genOpenCL(el.value)
      if i < ast.ocFields.len - 1:
        result.add ", "
    result.add '}'

  of gpuInlineAsm:
    result = indentStr & "asm(" & ast.stmt.strip & ");"

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
