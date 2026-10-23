## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

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
# Type mapping – GLSL 450 (Vulkan compute)
# ═══════════════════════════════════════════════════════════════════════

proc refTypeName(inner: GpuType): string =
  ## GLSL name of the buffer-reference block binding `inner`:
  ## a pointer renders as its reference block type
  let pt = if inner.kind == gtUA: inner.uaTo else: inner
  let base = gpuTypeToString(pt, allowEmptyIdent = true)
  result = "Ref_" & base.multiReplace((" ", "_"), ("[", ""), ("]", ""),
                                      (",", "_"), ("(", ""), (")", ""))

proc refBlockDecl(inner: GpuType): string =
  ## `layout(buffer_reference, std430) buffer Ref_X { T x[]; } Ref_X;`
  ## for one pointee type.
  ##
  ## Member shapes:
  ## - runtime array (UncheckedArray): `T x[]`, fixed-array elements nest
  ##   after the runtime dimension (`uint32_t x[][32]`), the unsized
  ##   dimension is only legal in the outermost position
  ## - pointer to one fixed array: fixed member `uint32_t x[32]`

  proc arrayDims(t: GpuType): tuple[base: GpuType, dims: string] =
    ## Strips the array nesting, collecting the dimension suffixes outermost
    ## first (the Nim nesting order and the GLSL reading order agree)
    var t = t
    while t.kind == gtArray:
      doAssert t.aLen >= 0,
        "Unresolved static array length in a Vulkan ref block: lengths must resolve before emission"
      result.dims &= '[' & $t.aLen & ']'
      t = t.aTyp
    result.base = t

  doAssert inner.kind notin {gtVoid, gtVoidPtr, gtObject, gtGenericInst},
    "Pointing at " & (if inner.kind in {gtVoid, gtVoidPtr}: "void"
        else: gpuTypeToString(inner, allowEmptyIdent = true)) &
        " is currently unsupported, only integers, floats, bool and arrays of those are supported"
  let member =
    if inner.kind == gtUA:
      let el = inner.uaTo
      doAssert el.kind != gtUA,
        "Runtime array of runtime arrays is not expressible in a GLSL buffer block"
      if el.kind == gtArray:
        let (base, dims) = arrayDims(el)
        gpuTypeToString(base, "x", allowEmptyIdent = true) & "[]" & dims
      else:
        gpuTypeToString(el, "x", allowEmptyIdent = true) & "[]"
    elif inner.kind == gtArray:
      let (base, dims) = arrayDims(inner)
      gpuTypeToString(base, "x", allowEmptyIdent = true) & dims
    else:
      gpuTypeToString(inner, "x") & "[]"
  let name = refTypeName(inner)
  result = "layout(buffer_reference, std430) buffer " & name & " { " & member & "; };"

proc collectRefPointees(t: GpuType, pointees: var seq[GpuType]) =
  ## Collects the pointee of every pointer reachable from the type tree:
  ## through pointer targets, fixed-array elements, runtime-array elements
  ## and object fields. Dedup by reference-block name.
  case t.kind
  of gtPtr:
    if t.to != nil:
      let name = refTypeName(t.to)
      if not pointees.anyIt(refTypeName(it) == name):
        pointees.add t.to
      collectRefPointees(t.to, pointees)
  of gtArray:
    if t.aTyp != nil:
      collectRefPointees(t.aTyp, pointees)
  of gtUA:
    if t.uaTo != nil:
      collectRefPointees(t.uaTo, pointees)
  of gtObject, gtGenericInst:
    for f in t.fields:
      collectRefPointees(f.typ, pointees)
  else:
    discard

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
  of gtFloat16: "float16_t"
  of gtBf16: "bfloat16_t"
  of gtFloat64: "double"
  of gtVoid: "void"
  of gtSize_t: "uint"
  of gtPtr: error "Vulkan GLSL renders pointers as buffer references (buffer device addresses) — render the full type, not the bare kind"
  of gtVoidPtr: error "Vulkan GLSL renders void* as a void buffer reference — render the full type, not the bare kind"
  of gtString: "const char*"
  of gtUA: ""       # UncheckedArray used as buffer pointer
  of gtStatic: "int"
  of gtObject: "struct"
  else:
    error "Invalid type : " & $t

proc gpuTypeToString*(t: GpuType, ident: string = "", allowArrayToPtr = false, allowEmptyIdent = false): string =
  ## Struct types render their own (mangled) name
  ## Given an optional identifier required for array types
  var skipIdent = false
  case t.kind
  of gtPtr:
    result = refTypeName(t.to)
  of gtVoidPtr:
    result = refTypeName(GpuType(kind: gtVoid))
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
    if t.gArgs.len == 1:
      result.add gpuTypeToShortString(t.gArgs[0])
    elif t.gArgs.len > 1:
      # 2+ args collapse to a short hash
      result.add mangledHashName("", $hash(t))
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


proc usesReductionBuiltin(n: GpuAst): bool =
  ## True when the AST contains a reduction builtin call needing the GLSL subgroup extensions at module scope.
  if n.kind == gpuCall and n.cName.symbol != nil and
     n.cName.symbol.reductionBuiltin != gbkNone:
    return true
  for ch in n:
    if usesReductionBuiltin(ch):
      return true

proc usesFloatAtomic(ctx: GpuContext, n: GpuAst): bool =
  ## True when the AST contains an fp `atomicFetchAdd` call, which needs
  ## GL_EXT_shader_atomic_float at module scope. int atomicAdd is core.
  if n.kind == gpuCall and n.cName.symbol != nil and
     n.cName.symbol.atomicsBuiltin == gbkAtomicFetchAdd and
     getPointeeKind(ctx.getOperandType(n.cArgs[0])) == gtFloat32:
    return true
  for ch in n:
    if ctx.usesFloatAtomic(ch):
      return true


# ═══════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════

proc genGLSL*(ctx: var GpuContext, ast: GpuAst, indent = 0): string

# ── genGLSL ─────────────────────────────────────────────────────────

proc genLit*(ast: GpuAst): string =
  ## Lower a literal node for the Vulkan (GLSL) backend.
  if ast.lType.kind == gtString:
    result = '"' & ast.lValue & '"'
  elif ast.lValue == "DEFAULT":
    result = "{}"
  else:
    case ast.lType.kind
    of gtFloat32: result = ast.lValue & "f"
    of gtFloat16, gtBf16:
      result = gpuTypeToString(ast.lType, allowEmptyIdent = true) & '(' & ast.lValue & ')'
    of gtUint32: result = ast.lValue & "U"
    of gtUint64: result = ast.lValue & "ULL"
    of gtInt64:  result = ast.lValue & "LL"
    of gtInt16, gtUint16, gtUint8, gtBool:
      result = '(' & gpuTypeToString(ast.lType, allowEmptyIdent = true) & ')' & ast.lValue
    else:
      result = ast.lValue

proc glslCoordIdent(kind: GpuCoordBuiltinKind): string =
  ## GLSL spelling of a canonical coordinate builtin referenced whole.
  ## All six canonicals are native GLSL builtins, so this is a plain rename.
  case kind
  of gbkThreadPositionInGrid: "ivec3(int(gl_GlobalInvocationID.x), int(gl_GlobalInvocationID.y), int(gl_GlobalInvocationID.z))"
  of gbkThreadgroupPositionInGrid: "ivec3(int(gl_WorkGroupID.x), int(gl_WorkGroupID.y), int(gl_WorkGroupID.z))"
  of gbkThreadPositionInThreadgroup: "ivec3(int(gl_LocalInvocationID.x), int(gl_LocalInvocationID.y), int(gl_LocalInvocationID.z))"
  of gbkThreadsPerThreadgroup: "ivec3(int(gl_WorkGroupSize.x), int(gl_WorkGroupSize.y), int(gl_WorkGroupSize.z))"
  of gbkThreadgroupsPerGrid: "ivec3(int(gl_NumWorkGroups.x), int(gl_NumWorkGroups.y), int(gl_NumWorkGroups.z))"
  of gbkThreadIndexInThreadgroup: "int(gl_LocalInvocationIndex)"
  of gbkNone:
    # Unreachable-by-construction: ident sites emit gbkNone verbatim, so this branch never fires.
    error "coordinate site with no coordinate builtin kind"

proc genGLSL*(ctx: var GpuContext, ast: GpuAst, indent = 0): string =
  ## The actual GLSL compute shader code generator.
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
      if p.addressSpace == asSMEM:
        # shared memory
        let inner = gpuTypeToString(p.typ.to, allowEmptyIdent = true)
        params.add &"shared {inner} {p.ident.ident()}[]"
      else:
        let binding = if p.flavor == kVar: "inout " else: ""
        params.add binding & gpuTypeToString(p.typ, p.ident.ident(), allowEmptyIdent = false)
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
      result &= ctx.genGLSL(ast.pBody, indent + 1)
      result &= '\n' & indentStr & '}'

  of gpuBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "{ // " & ast.blockLabel & '\n'
    for i, el in ast.statements:
      result.add ctx.genGLSL(el, indent)
      if not el.isSelfTerminating() and not ctx.skipSemicolon:
        result.add ';'
      if i < ast.statements.high:
        result.add '\n'
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "} // " & ast.blockLabel & '\n'

  of gpuVar:
    # The var's address-space keyword, prefixing the declaration ident for asSMEM arrays (`shared uint scratch[8]`).
    # asConstant/asRMEM/asDevice have no GLSL per-variable keyword and emit none (`private` is not a valid GLSL storage qualifier).
    var typeStr: string
    case ast.addressSpace
    of asSMEM:
      typeStr = "shared " & gpuTypeToString(ast.vType, ast.vName.ident())
    of asRMEM, asConstant, asDevice:
      typeStr = gpuTypeToString(ast.vType, ast.vName.ident())

    result = indentStr & typeStr
    if ast.vInit.kind != gpuDiscard:
      result &= " = " & ctx.genGLSL(ast.vInit)

  of gpuAssign:
    result = indentStr & ctx.genGLSL(ast.aLeft) & ' ' & ast.aOp & ' ' &
             ctx.genGLSL(ast.aRight)
  of gpuIf:
    ctx.withoutSemicolon:
      result = indentStr & "if (" & ctx.genGLSL(ast.ifCond) & ") {\n"
    result &= ctx.genGLSL(ast.ifThen, indent + 1) & '\n'
    result &= indentStr & '}'
    for el in ast.ifElifs:
      ctx.withoutSemicolon:
        result &= " else if (" & ctx.genGLSL(el.cond) & ") {\n"
      result &= ctx.genGLSL(el.body, indent + 1) & '\n'
      result &= indentStr & '}'
    if ast.ifElse.kind != gpuDiscard:
      result &= " else {\n"
      result &= ctx.genGLSL(ast.ifElse, indent + 1) & '\n'
      result &= indentStr & '}'

  of gpuTernary:
    ctx.withoutSemicolon:
      result = '(' & ctx.genGLSL(ast.tCond) & " ? " &
               ctx.genGLSL(ast.tThen) & " : " &
               ctx.genGLSL(ast.tElse) & ')'

  of gpuFor:
    let cmp = if ast.fRangeKind == rkInclusive: " <= " else: " < "
    result = indentStr & "for(int " & ast.fVar.ident() & " = " &
             ctx.genGLSL(ast.fStart) & "; " &
             ast.fVar.ident() & cmp & ctx.genGLSL(ast.fEnd) & "; " &
             ast.fVar.ident() & " += " & ctx.genGLSL(ast.fStep) & ") {\n"
    result &= ctx.genGLSL(ast.fBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genGLSL(ast.wCond) & "){\n"
    result &= ctx.genGLSL(ast.wBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuDot:
    if ast.dParent.kind == gpuIdent and ast.dParent.symbol != nil and
       ast.dParent.symbol.coordBuiltin != gbkNone and ast.dField.kind == gpuIdent:
      # Component of an unsigned GLSL builtin, cast to standard Nim `int` from default uint.
      result = "int(" & ctx.genGLSL(ast.dParent) & '.' & ctx.genGLSL(ast.dField) & ")"
    else:
      result = ctx.genGLSL(ast.dParent) & '.' & ctx.genGLSL(ast.dField)

  of gpuIndex:
    if ast.iArr.kind == gpuDeref:
      # the ref block member is runtime-sized, the index applies to it directly.
      # Routing through the gpuDeref rendering would give `.x[0][i]`
      result = ctx.genGLSL(ast.iArr.dOf) & ".x[" & ctx.genGLSL(ast.iIndex) & ']'
    else:
      result = ctx.genGLSL(ast.iArr) & '[' & ctx.genGLSL(ast.iIndex) & ']'

  of gpuCall:
    case ast.cName.symbol.synchroBuiltin
    of gbkThreadgroupBarrier:
      # Every backend name of the barrier is an alias template that semantic analysis
      # expands to the canonical call, so only this kind reaches the IR.
      # Vulkan spells it `barrier()`.
      result = indentStr & "barrier()"
    of gbkThreadgroupBarrierDevice:
      # GLSL's barrier() orders shared memory only, memoryBarrierBuffer()
      # runs first, the buffer writes then ordered before the barrier
      result = indentStr & "memoryBarrierBuffer();" & "\n" & indentStr &
               "barrier()"
    of gbkNone:
      case ast.cName.symbol.reductionBuiltin
      of gbkReduceAdd:
        result = indentStr & "subgroupAdd(" & ctx.genGLSL(ast.cArgs[0]) & ')'
      of gbkReduceMax:
        result = indentStr & "subgroupMax(" & ctx.genGLSL(ast.cArgs[0]) & ')'
      of gbkNone:
        case ast.cName.symbol.atomicsBuiltin
        of gbkAtomicFetchAdd:
          # fp needs GL_EXT_shader_atomic_float, emitted at module scope
          # when the AST contains an fp atomicFetchAdd, int is core.
          # Pointers are GL_EXT_buffer_reference handles, dereference
          # the first element of the reference's member array.
          result = indentStr & "atomicAdd(" & ctx.genGLSL(ast.cArgs[0]) &
                   ".x[0], " & ctx.genGLSL(ast.cArgs[1]) & ')'
        of gbkNone:
          var vkArgs: seq[string]
          for arg in ast.cArgs:
            vkArgs.add ctx.genGLSL(arg)
          result = indentStr & ctx.getFnName(bkVulkan, ast) & '(' & vkArgs.join(", ") & ')'


  of gpuBinOp:
    ctx.withoutSemicolon:
      let l = ctx.genGLSL(ast.bLeft)
      let r = ctx.genGLSL(ast.bRight)
      result = indentStr & '(' & l & ' ' &
               ctx.genGLSL(ast.bOp) & ' ' &
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
        # each kind to its GLSL spelling (gl_GlobalInvocationID,
        # gl_WorkGroupID, gl_LocalInvocationID, gl_WorkGroupSize,
        # gl_NumWorkGroups, gl_LocalInvocationIndex).
        result = glslCoordIdent(ast.symbol.coordBuiltin)
    else:
      result = ast.ident()

  of gpuLit:
      result = genLit(ast)

  of gpuArrayLit:
    # GLSL uses array constructor syntax Type[N](val1, val2, ...), not C-style {val1, val2}
    # Explicit size required for proper struct field initialization in GLSL
    result = gpuTypeToString(ast.aLitType) & "[" & $ast.aValues.len & "]("
    for i, el in ast.aValues:
      result.add gpuTypeToString(ast.aLitType) & '(' & ctx.genGLSL(el) & ')'
      if i < ast.aValues.high:
        result.add ", "
    result.add ')'

  of gpuReturn:
    result = indentStr & "return " & ctx.genGLSL(ast.rValue)

  of gpuPrefix:
    result = ast.pOp & ctx.genGLSL(ast.pVal)

  of gpuTypeDef:
    result = "struct " & gpuTypeToString(ast.tTyp) & " {\n"
    if ast.tFields.len == 0:
      # GLSL requires at least one field in a struct.
      result.add "  uint _padding;\n"
    else:
      for el in ast.tFields:
        result.add "  " & gpuTypeToString(el.typ, el.name) & ";\n"
    result.add '}'

  of gpuObjConstr:
    if ast.ocFields.len == 0:
      # Empty ocFields: struct has backend-added padding, use {0} for zero-init
      result = gpuTypeToString(ast.ocType) & "(0)"
    else:
      # GLSL uses constructor syntax TypeName(val1, val2), not C-style {val1, val2}
      result = gpuTypeToString(ast.ocType) & "("
      for i, el in ast.ocFields:
        if el.value.kind == gpuDiscard:
          result.add gpuTypeToString(el.typ, allowEmptyIdent = true) & "(0)"
        elif el.value.kind == gpuLit and el.value.lValue == "DEFAULT":
          result.add gpuTypeToString(el.typ, allowEmptyIdent = true) & "(0)"
        else:
          result.add ctx.genGLSL(el.value)
        if i < ast.ocFields.len - 1:
          result.add ", "
      result.add ')'

  of gpuInlineAsm:
    result = indentStr & "asm(" & genAsmStmt(ast).strip & ");"

  of gpuEmit:
    # Self-terminating raw text: the gpuBlock loop appends no `;`
    # (the emitted text owns its own terminators).
    result = genEmitStmt(ctx, ast,
      proc(c: var GpuContext, n: GpuAst): string = c.genGLSL(n, 0))

  of gpuComment:
    result = indentStr & "/* " & ast.comment & " */"

  of gpuConv:
    result = gpuTypeToString(ast.convTo, allowEmptyIdent = true) & '(' & ctx.genGLSL(ast.convExpr) & ')'

  of gpuCast:
    result = gpuTypeToString(ast.cTo, allowEmptyIdent = true) & '(' & ctx.genGLSL(ast.cExpr) & ')'

  of gpuAddr:
    # Address-of element 0 of a pointer renders as the buffer-reference handle.
    # Locals have no address in GLSL, fail loudly.
    if ast.aOf.kind == gpuIndex and ast.aOf.iArr.kind != gpuDeref and
       ast.aOf.iIndex.kind == gpuLit and ast.aOf.iIndex.lValue == "0":
      result = ctx.genGLSL(ast.aOf.iArr)
    else:
      error "Vulkan GLSL cannot take the address of a local: buffer device addressing covers buffer memory only, not function storage. Promote the local (SROA) or inline the callee. Site: " &
        ast.repr

  of gpuDeref:
    # the ref block member is a runtime-sized array, a value read takes
    # element 0. Indexed and field-accessed uses route around this branch
    result = ctx.genGLSL(ast.dOf) & ".x[0]"

  of gpuConstexpr:
    if ast.cType.kind == gtArray:
      result = indentStr & "const " & gpuTypeToString(ast.cType, ctx.genGLSL(ast.cIdent)) & " = " & ctx.genGLSL(ast.cValue)
    else:
      result = indentStr & "const " & gpuTypeToString(ast.cType, allowEmptyIdent = true) & ' ' & ctx.genGLSL(ast.cIdent) & " = " & ctx.genGLSL(ast.cValue)
  of gpuMaterialize:
    error "gpuMaterialize should not reach Vulkan backend — passByRef is not used"

  else:
    echo "Unhandled node kind in genGLSL: ", ast.kind
    error "Unhandled node kind in genGLSL: " & ast.repr

# ═══════════════════════════════════════════════════════════════════════
# Top-level codegen
# ═══════════════════════════════════════════════════════════════════════

proc codegen*(ctx: var GpuContext): string =
  ## Generate the actual code for all pieces of the puzzle.

  # Check if we need fp64 / fp16 / bf16 extensions or the subgroup
  # shuffles (which need the KHR subgroup shuffle extensions)


  # ── Step 1: Build the push-constant block and reference block info ──
  # Pointers are BDA references: 64-bit addresses packed in the push block.
  # Ref blocks are declared for the pointer surfaces of every function,
  # kernels included. Device fns take pointer params, return pointers, hold
  # pointer locals and cast to pointers inside their bodies.
  var refPointees: seq[GpuType]
  var pushConstDecls: seq[string]
  for (fnIdent, fn) in mpairs(ctx.fnTab):
    for p in fn.pParams:
      collectRefPointees(p.typ, refPointees)
      if fn.isGlobal() and p.addressSpace != asSMEM:
        # multi-kernel sources share one block, same-name params dedup
        let decl = gpuTypeToString(p.typ, p.ident.ident(), allowEmptyIdent = false)
        if decl notin pushConstDecls:
          pushConstDecls.add decl
    if fn.pRetType != nil:
      collectRefPointees(fn.pRetType, refPointees)
    var ptrees = @[fn.pBody]
    for blk in ctx.globalBlocks:
      if blk.kind in {gpuTypeDef, gpuAlias}:
        ptrees.add blk
    while ptrees.len > 0:
      let n = ptrees.pop()
      if n.kind == gpuVar and n.vType != nil:
        collectRefPointees(n.vType, refPointees)
      if n.kind == gpuTypeDef:
        for f in n.tFields:
          collectRefPointees(f.typ, refPointees)
      if n.kind == gpuCast and n.cTo != nil:
        collectRefPointees(n.cTo, refPointees)
      if n.kind == gpuConv and n.convTo != nil:
        collectRefPointees(n.convTo, refPointees)
      for ch in n:
        if not ch.isNil:
          ptrees.add ch

  var needsFp64 = false
  var needsFp16 = false
  var needsBf16 = false
  var needsSubgroup = false
  var needsAtomicFloat = false
  var needsInt8 = false
  var needsInt64 = false
  for fnIdent, fn in ctx.fnTab:
    if fn.pRetType.containsKind(gtFloat64):
      needsFp64 = true
    if fn.pRetType.containsKind(gtFloat16):
      needsFp16 = true
    if fn.pRetType.containsKind(gtBf16):
      needsBf16 = true
    if fn.pRetType.containsKind(gtUint8):
      needsInt8 = true
    if fn.pRetType.containsKind(gtInt64) or fn.pRetType.containsKind(gtUint64):
      needsInt64 = true
    for p in fn.pParams:
      if p.typ.containsKind(gtFloat64):
        needsFp64 = true
      if p.typ.containsKind(gtFloat16):
        needsFp16 = true
      if p.typ.containsKind(gtBf16):
        needsBf16 = true
      if p.typ.containsKind(gtUint8):
        needsInt8 = true
      if p.typ.containsKind(gtInt64) or p.typ.containsKind(gtUint64):
        needsInt64 = true
    if usesReductionBuiltin(fn.pBody):
      needsSubgroup = true
    if ctx.usesFloatAtomic(fn.pBody):
      needsAtomicFloat = true
  for pt in refPointees:
    var b = pt
    while b.kind in {gtArray, gtUA}:
      b = if b.kind == gtArray: b.aTyp else: b.uaTo
    if b.kind == gtUint8:
      needsInt8 = true
    if b.kind == gtInt64 or b.kind == gtUint64:
      needsInt64 = true

  # Emit GLSL header
  result = "#version 450\n"
  result.add "#extension GL_EXT_buffer_reference : enable\n"
  result.add "#extension GL_EXT_buffer_reference2 : enable\n"
  if needsFp64:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable\n"
  if needsFp16:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n"
    result.add "#extension GL_EXT_shader_16bit_storage : enable\n"
  if needsBf16:
    result.add "#extension GL_EXT_bfloat16 : enable\n"
    result.add "#extension GL_EXT_shader_16bit_storage : enable\n"
  if needsInt64:
    # uint64_t members in push-constant and buffer-reference blocks
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable\n"
  if needsInt8:
    # uint8_t members in push-constant and buffer-reference blocks need
    # both extensions: explicit_int8 defines the type, 8bit_storage
    # admits it in storage-class blocks
    result.add "#extension GL_EXT_shader_explicit_int8 : enable\n"
    result.add "#extension GL_EXT_shader_8bit_storage : enable\n"
  if needsSubgroup:
    result.add "#extension GL_KHR_shader_subgroup_basic : enable\n"
    result.add "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"
  if needsAtomicFloat:
    result.add "#extension GL_EXT_shader_atomic_float : enable\n"
  result.add "\n"

  # ── Step 1b: Emit reference block typedefs, before the structs using them ──
  for pt in refPointees:
    result.add refBlockDecl(pt)
    result.add "\n"
  if refPointees.len > 0:
    result.add "\n"

  # ── Step 2: Emit struct typedefs the push-constant block references ──
  # GLSL resolves types at their declaration site. The push-constant block's
  # struct members need the typedef text above them.
  for blk in ctx.globalBlocks:
    if blk.kind in {gpuTypeDef, gpuAlias}:
      let blkStr = ctx.genGLSL(blk)
      if blkStr.len > 0:
        result.add blkStr
        if blk.kind in {gpuTypeDef, gpuAlias}:
          result.add ";\n"
        result.add "\n"

  # ── Step 3: Emit push-constant block (if any) ──
  if pushConstDecls.len > 0:
    result.add "layout(push_constant, std430) uniform KernelParams {\n"
    for decl in pushConstDecls:
      result.add "  " & decl & ";\n"
    result.add "};\n\n"

  # ── Generate code for the remaining global blocks (global vars etc) ──
  for blk in ctx.globalBlocks:
    if blk.kind in {gpuTypeDef, gpuAlias}:
      continue
    let blkStr = ctx.genGLSL(blk)
    if blkStr.len > 0:
      result.add blkStr
      result.add "\n"

  # 4. Forward declarations (only for device functions, not kernels)
  let fns = toSeq(ctx.fnTab.pairs)
  for (fnIdent, fn) in fns:
    if not fn.isGlobal():
      let fnC = fn.clone()
      fnC.forwardDeclare = true
      result.add ctx.genGLSL(fnC) & '\n'
  if fns.anyIt(not it[1].isGlobal()):
    result.add "\n"

  # 5. Device function definitions, before the kernels calling them
  for fnIdent, fn in ctx.fnTab:
    if not fn.isGlobal():
      result.add ctx.genGLSL(fn) & "\n\n"

  # 6. Full kernel definitions
  for fnIdent, fn in ctx.fnTab:
    if fn.isGlobal():
      # Kernel functions: place layout qualifier before entry point.
      # Workgroup size from the `{.workgroup: (X, Y, Z).}` annotation.
      # Default 256×1×1.
      let wg = fn.pWorkgroupSize
      let wx = if wg.x > 0: wg.x else: 256
      let wy = if wg.y > 0: wg.y else: 1
      let wz = if wg.z > 0: wg.z else: 1
      result.add "layout(local_size_x = " & $wx & ", local_size_y = " & $wy &
                 ", local_size_z = " & $wz & ") in;\n"
      result.add ctx.genGLSL(fn) & "\n\n"
