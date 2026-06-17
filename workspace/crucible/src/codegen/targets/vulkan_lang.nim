## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

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
# Type mapping – GLSL 450 (Vulkan compute)
# ═══════════════════════════════════════════════════════════════════════

proc gpuTypeToString*(t: GpuTypeKind): string =
  case t
  of gtBool: "bool"
  of gtUint8: "uint8_t"
  of gtUint16: "uint16_t"
  of gtUint32: "uint"
  of gtUint64: "uint64_t"
  of gtInt16: "int16_t"
  of gtInt32: "int"
  of gtInt64: "int64_t"
  of gtFloat32: "float"
  of gtFloat64: "double"
  of gtVoid: "void"
  of gtSize_t: "uint"
  of gtPtr: raiseAssert "Vulkan GLSL does not support raw pointers — lower to SSBO/indexing first"
  of gtVoidPtr: raiseAssert "Vulkan GLSL does not support void* pointers — lower to SSBO/indexing first"
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
    result = t.gName
    if t.gArgs.len > 0:
      result.add '_'
    for i, g in t.gArgs:
      result.add gpuTypeToShortString(g)
      if i < t.gArgs.high:
        result.add 'x'
  of gtObject: result = t.name
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


const glslReserved*: array[40, string] = [
  "output", "input", "in", "out", "attribute", "uniform", "varying",
  "buffer", "shared", "layout", "main", "void", "return",
  "if", "else", "for", "while", "break", "continue", "struct",
  "const", "true", "false", "bool", "int", "uint", "float", "double",
  "vec2", "vec3", "vec4", "mat2", "mat3", "mat4",
  "sampler", "texture", "image", "subroutine", "discard", "precise"
]

proc glslSafeName*(name: string): string =
  ## Returns a GLSL-safe version of the name (appends _vk if reserved).
  if name in glslReserved:
    result = name & "_vk"
  else:
    result = name

proc genSsboDeclaration(name: string, innerType: string, binding: int): string =
  ## Generates a GLSL SSBO declaration:
  ##   layout(set = 0, binding = N) buffer BufN { uint payload[]; };
  let safeName = glslSafeName(name)
  result = &"layout(set = 0, binding = {binding}) buffer Buf{binding} {{ {innerType} {safeName}[]; }};\n"

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
  ## Addresses AST patterns that need to be rewritten for GLSL.
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

proc genVulkan*(ctx: var GpuContext, ast: GpuAst, indent = 0): string
proc size(ctx: var GpuContext, a: GpuAst): string = size(ctx.genVulkan(a))
proc address(ctx: var GpuContext, a: GpuAst): string = address(ctx.genVulkan(a))

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
  # 5. (No SSBO counter reset needed — handled in codegen())

  # 5. Rename kernel parameters that clash with GLSL reserved words
  proc renameGlslReserved(n: var GpuAst, symToRename: Table[string, string]) =
    case n.kind
    of gpuIdent:
      if n.iSym in symToRename:
        n.iName = symToRename[n.iSym]
    else:
      for ch in mitems(n):
        renameGlslReserved(ch, symToRename)

  for (fnIdent, fn) in mpairs(ctx.fnTab):
    if fn.isGlobal():
      var renames = initTable[string, string]()
      for p in mitems(fn.pParams):
        let oldName = p.ident.ident()
        let safeName = glslSafeName(oldName)
        if oldName != safeName:
          renames[p.ident.iSym] = safeName
          p.ident.iName = safeName
      if renames.len > 0:
        renameGlslReserved(fn.pBody, renames)

# ── genVulkan ─────────────────────────────────────────────────────────

proc genVulkan*(ctx: var GpuContext, ast: GpuAst, indent = 0): string =
  ## The actual GLSL compute shader code generator.
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
      # For kernel parameters that are pointers, we lift them to SSBOs
      if isKernel and p.typ.kind == gtPtr:
        # Skip — these will be emitted as SSBO declarations at the top level
        discard
      elif p.addressSpace == asWorkspace:
        # shared memory
        let inner = gpuTypeToString(p.typ.to, allowEmptyIdent = true)
        params.add &"shared {inner} {p.ident.ident()}[]"
      else:
        params.add gpuTypeToString(p.typ, p.ident.ident(), allowEmptyIdent = false)
    let fnArgs = params.join(", ")

    if isKernel:
      # Global kernel entry point → uses actual kernel name (supports multiple kernels per module)
      result = indentStr & "void " & ast.pName.ident() & "()"
    else:
      # Device function → `retType fnName(...)` (GLSL doesn't have static/inline)
      result = indentStr & genFunctionType(ast.pRetType, ast.pName.ident(), fnArgs)

    if ast.forwardDeclare:
      result.add ';'
    else:
      result.add " {\n"
      result &= ctx.genVulkan(ast.pBody, indent + 1)
      result &= '\n' & indentStr & '}'

  of gpuBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "{ // " & ast.blockLabel & '\n'
    for i, el in ast.statements:
      result.add ctx.genVulkan(el, indent)
      if el.kind != gpuBlock and not ctx.skipSemicolon:
        result.add ';'
      if i < ast.statements.high:
        result.add '\n'
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "} // " & ast.blockLabel & '\n'

  of gpuVar:
    let attrs = if ast.vAttributes.len > 0: ast.vAttributes.join(" ") & ' '
                else: ""
    var typeStr: string
    if atvShared in ast.vAttributes:
      let inner = gpuTypeToString(ast.vType, allowEmptyIdent = true)
      typeStr = "shared " & inner & ' ' & ast.vName.ident()
    else:
      typeStr = attrs & gpuTypeToString(ast.vType, ast.vName.ident())

    result = indentStr & typeStr
    if ast.vInit.kind != gpuVoid and not ast.vRequiresMemcpy:
      result &= " = " & ctx.genVulkan(ast.vInit)
    elif ast.vInit.kind != gpuVoid:
      result.add ";\n"
      result.add indentStr & genMemcpy(address(ast.vName.ident()), ctx.address(ast.vInit),
                                       size(ast.vName.ident()))

  of gpuAssign:
    if ast.aRequiresMemcpy:
      result = indentStr & genMemcpy(ctx.address(ast.aLeft), ctx.address(ast.aRight),
                                     ctx.size(ast.aLeft))
    else:
      result = indentStr & ctx.genVulkan(ast.aLeft) & " = " & ctx.genVulkan(ast.aRight)

  of gpuIf:
    ctx.withoutSemicolon:
      result = indentStr & "if (" & ctx.genVulkan(ast.ifCond) & ") {\n"
    result &= ctx.genVulkan(ast.ifThen, indent + 1) & '\n'
    result &= indentStr & '}'
    if ast.ifElse.kind != gpuVoid:
      result &= " else {\n"
      result &= ctx.genVulkan(ast.ifElse, indent + 1) & '\n'
      result &= indentStr & '}'

  of gpuTernary:
    ctx.withoutSemicolon:
      result = '(' & ctx.genVulkan(ast.tCond) & " ? " &
               ctx.genVulkan(ast.tThen) & " : " &
               ctx.genVulkan(ast.tElse) & ')'

  of gpuFor:
    result = indentStr & "for(int " & ast.fVar.ident() & " = " &
             ctx.genVulkan(ast.fStart) & "; " &
             ast.fVar.ident() & " < " & ctx.genVulkan(ast.fEnd) & "; " &
             ast.fVar.ident() & "++) {\n"
    result &= ctx.genVulkan(ast.fBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genVulkan(ast.wCond) & "){\n"
    result &= ctx.genVulkan(ast.wBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuDot:
    result = ctx.genVulkan(ast.dParent) & '.' & ctx.genVulkan(ast.dField)

  of gpuIndex:
    result = ctx.genVulkan(ast.iArr) & '[' & ctx.genVulkan(ast.iIndex) & ']'

  of gpuCall:
    let fnName = ast.cName
    var vkArgs: seq[string]
    for arg in ast.cArgs:
      vkArgs.add ctx.genVulkan(arg)
    result = indentStr & fnName.ident() & '(' & vkArgs.join(", ") & ')'

  of gpuTemplateCall:
    when nimvm:
      error("Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `vulkan` macro.")
    else:
      raise newException(ValueError, "Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `vulkan` macro.")

  of gpuBinOp:
    ctx.withoutSemicolon:
      let l = ctx.genVulkan(ast.bLeft)
      let r = ctx.genVulkan(ast.bRight)
      result = indentStr & '(' & l & ' ' &
               ctx.genVulkan(ast.bOp) & ' ' &
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
      result.add gpuTypeToString(ast.aLitType) & '(' & ctx.genVulkan(el) & ')'
      if i < ast.aValues.high:
        result.add ", "
    result.add '}'

  of gpuReturn:
    result = indentStr & "return " & ctx.genVulkan(ast.rValue)

  of gpuPrefix:
    result = ast.pOp & ctx.genVulkan(ast.pVal)

  of gpuTypeDef:
    result = "struct " & gpuTypeToString(ast.tTyp) & " {\n"
    for el in ast.tFields:
      result.add "  " & gpuTypeToString(el.typ, el.name) & ";\n"
    result.add '}'

  of gpuObjConstr:
    result = "{"
    for i, el in ast.ocFields:
      result.add ctx.genVulkan(el.value)
      if i < ast.ocFields.len - 1:
        result.add ", "
    result.add '}'

  of gpuInlineAsm:
    result = indentStr & "asm(" & ast.stmt.strip & ");"

  of gpuComment:
    result = indentStr & "/* " & ast.comment & " */"

  of gpuConv:
    result = gpuTypeToString(ast.convTo, allowEmptyIdent = true) & '(' & ctx.genVulkan(ast.convExpr) & ')'

  of gpuCast:
    result = gpuTypeToString(ast.cTo, allowEmptyIdent = true) & '(' & ctx.genVulkan(ast.cExpr) & ')'

  of gpuAddr:
    raiseAssert "Vulkan GLSL does not support addr — lower to SSBO/indexing first"

  of gpuDeref:
    raiseAssert "Vulkan GLSL does not support deref — lower to SSBO/indexing first"

  of gpuConstexpr:
    if ast.cType.kind == gtArray:
      result = indentStr & "const " & gpuTypeToString(ast.cType, ctx.genVulkan(ast.cIdent)) & " = " & ctx.genVulkan(ast.cValue)
    else:
      result = indentStr & "const " & gpuTypeToString(ast.cType, allowEmptyIdent = true) & ' ' & ctx.genVulkan(ast.cIdent) & " = " & ctx.genVulkan(ast.cValue)

  else:
    echo "Unhandled node kind in genVulkan: ", ast.kind
    raiseAssert "Unhandled node kind in genVulkan: " & ast.repr

# ═══════════════════════════════════════════════════════════════════════
# Top-level codegen
# ═══════════════════════════════════════════════════════════════════════

proc renameIdentRefs(n: var GpuAst, symToRename: Table[string, string]) =
  ## Rename `gpuIdent` nodes whose `iSym` is in the rename table.
  case n.kind
  of gpuIdent:
    if n.iSym in symToRename:
      n.iName = symToRename[n.iSym]
  else:
    for ch in mitems(n):
      renameIdentRefs(ch, symToRename)

proc codegen*(ctx: var GpuContext): string =
  ## Generate the actual code for all pieces of the puzzle.

  # Check if we need fp64 extension
  var needsFp64 = false
  for fnIdent, fn in ctx.fnTab:
    if fn.pRetType.kind == gtFloat64:
      needsFp64 = true
    for p in fn.pParams:
      if p.typ.containsFloat64():
        needsFp64 = true

  # Emit GLSL header
  result = "#version 450\n"
  if needsFp64:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable\n"
  result.add "\n"

# ── Step 1: Pre-scan kernels — build SSBO dedup and collect push-const params ──
  # canonicalSsbo[i] = (name, innerType) for the i-th pointer param (dense index)
  var canonicalSsbo: seq[tuple[name: string, inner: string]]
  # Push-constant params for non-pointer, non-workspace kernel params
  var pushConstDecls: seq[string]

  for (fnIdent, fn) in mpairs(ctx.fnTab):
    if fn.isGlobal():
      var ssboIdx = 0  # dense index across pointer params only
      for p in fn.pParams:
        if p.typ.kind == gtPtr:
          let inner = gpuTypeToString(p.typ.to, allowEmptyIdent = true)
          if ssboIdx < canonicalSsbo.len:
            let (canonName, canonInner) = canonicalSsbo[ssboIdx]
            if canonInner != inner:
              raiseAssert &"Type mismatch at SSBO position {ssboIdx}: {canonInner} vs {inner}"
            if p.ident.ident() != canonName:
              var renames = initTable[string, string]()
              renames[p.ident.iSym] = canonName
              renameIdentRefs(fn.pBody, renames)
              p.ident.iName = canonName
          else:
            canonicalSsbo.add (p.ident.ident(), inner)
          inc ssboIdx
        elif p.addressSpace != asWorkspace:
          pushConstDecls.add gpuTypeToString(p.typ, p.ident.ident(), allowEmptyIdent = false)

  # ── Step 2: Emit push-constant block (if any) ──
  if pushConstDecls.len > 0:
    result.add "layout(push_constant) uniform KernelParams {\n"
    for decl in pushConstDecls:
      result.add "  " & decl & ";\n"
    result.add "};\n\n"

  # ── Step 3: Emit SSBO declarations by position ──
  for idx, (name, inner) in canonicalSsbo.pairs:
    result.add genSsboDeclaration(name, inner, idx)
  if canonicalSsbo.len > 0:
    result.add "\n"

  # ── Generate code for the global blocks (types, global vars etc)
  for blk in ctx.globalBlocks:
    let blkStr = ctx.genVulkan(blk)
    if blkStr.len > 0:
      result.add blkStr
      if blk.kind in {gpuTypeDef, gpuAlias}:
        result.add ";\n"
      result.add "\n"

  # 3. Workgroup layout for kernel entry points
  # Each kernel function gets the layout qualifier.
  for fnIdent, fn in ctx.fnTab:
    if fn.isGlobal():
      discard

  # 4. Forward declarations (only for device functions, not kernels)
  let fns = toSeq(ctx.fnTab.pairs)
  for (fnIdent, fn) in fns:
    if not fn.isGlobal():
      let fnC = fn.clone()
      fnC.forwardDeclare = true
      result.add ctx.genVulkan(fnC) & '\n'
  if fns.anyIt(not it[1].isGlobal()):
    result.add "\n"

  # 5. Full function definitions
  for fnIdent, fn in ctx.fnTab:
    if fn.isGlobal():
      # Kernel functions: place layout qualifier before entry point
      result.add "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
    result.add ctx.genVulkan(fn) & "\n\n"
