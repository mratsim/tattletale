# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal Shading Language (MSL) printer for the `bkMetal` backend.
##
## Lowers the shared GPU IR to MSL, modeled on the CUDA printer with Metal's ABI rules:
##   - kernel arguments are buffers (`device` pointers for arrays, `constant` references for scalars)
##   - referenced index builtins become attribute-qualified `uint3` parameters, one per use
##   - the threadgroup size stays host-side (dispatch-time) and is never baked into the shader.
##
## `MetalReservedKeywords` is the MSL reserved-word table minus the boolean literals
## `true`/`false` (emitted as valid MSL tokens), plus the `metal` namespace.
## It is a defensible superset of the MSL reserved words, not a hand-picked sample.

import std / [macros, strformat, strutils, sequtils, tables, sets]

import ../ir/gpu_types
import ./lang_utils

const MetalReservedKeywords = [
  # C++/MSL keyword table (MSL is C++14-based)
  "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
  "bool", "break", "case", "catch", "char", "char16_t", "char32_t", "class",
  "compl", "const", "const_cast", "constexpr", "continue", "decltype",
  "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
  "explicit", "export", "extern", "float", "for", "friend", "goto",
  "half", "if", "inline", "int", "long", "mutable", "namespace", "new",
  "noexcept", "not", "not_eq", "nullptr", "object_data", "operator", "or",
  "or_eq", "private", "protected", "public", "ray_data", "register",
  "reinterpret_cast", "return", "short", "signed", "sizeof", "static",
  "static_assert", "static_cast", "struct", "switch", "template", "this",
  "thread", "thread_local", "threadgroup", "threadgroup_imageblock", "throw",
  "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
  "vertex", "virtual", "void", "volatile", "wchar_t", "while", "xor",
  "xor_eq",
  # MSL address-space qualifiers and the printer's scalar spellings
  # `threadgroup` repeats the C++ group above: it is both a C++ keyword and an MSL address space.
  "kernel", "device", "constant", "threadgroup", "uchar", "ushort", "ulong",
  # `metal` is the MSL namespace, not a keyword
  "metal",
]
  ## MSL keywords the printer must never emit as identifiers. The Metal compiler
  ## rejects them with parser errors that do not name the cause, so the printer
  ## raises with a rename hint instead.

proc checkReservedIdent(name: string; what: string) =
  ## Raises when `name` is a reserved MSL keyword, so the shader source never contains a colliding identifier.
  if name == "metal":
    raiseAssert "'" & name & "' collides with the MSL `metal::` namespace. Rename the " & what & "."
  elif name in MetalReservedKeywords:
    raiseAssert "'" & name & "' is a reserved keyword in MSL. Rename the " & what & "."

proc gpuTypeToString*(t: GpuType,
                      ident: string = "",
                      allowEmptyIdent = false): string

proc getInnerArrayType(t: GpuType): string =
  ## Returns the name of the innermost type for a nested array.
  case t.kind
  of gtArray:
    result = getInnerArrayType(t.aTyp)
  else:
    result = gpuTypeToString(t)

proc gpuTypeToString*(t: GpuTypeKind): string =
  ## MSL spelling of a scalar type kind. MSL 64-bit integers are `long`
  ## and `ulong`, 8 bytes each. The compiler rejects the C `long long` spelling.
  ## fp64 does not exist on Metal and is rejected loudly, like the WGSL printer does.
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
  of gtFloat64:
    raiseAssert "The Metal target does not support 64-bit floating point (gtFloat64): " & $t
  of gtVoid: "void"
  of gtSize_t: "size_t"
  of gtPtr: "*"
  of gtVoidPtr: "void*"
  of gtObject: "struct"
  of gtString: "const char*"
  of gtUA: "" # `UncheckedArray` by itself is nothing in MSL
  of gtStatic: "int"
  else:
    raiseAssert "Invalid type : " & $t

proc gpuTypeToString*(t: GpuType, ident: string = "",
                      allowEmptyIdent = false): string =
  ## MSL spelling of `t`, with `ident` appended for array types, which require an identifier in the emitted declaration.
  ## `allowEmptyIdent` permits the identifier to be omitted (function return types and casts).
  ## Without it, an array type with an empty `ident` raises.
  var skipIdent = false
  case t.kind
  of gtPtr:
    var t = t # if `ptr UncheckedArray`, remove the `gtUA` layer. No meaning on MSL
    if t.to.kind == gtUA:
      t.to = t.to.uaTo

    if t.to.kind == gtArray: # ptr to array type
      # Need to pass `*` for the pointer into the identifier.
      # `state: var array[4, BigInt]` must become `BigInt (*state)[4]`.
      # So we pass `theIdent = (*<ident>)` and generate the type
      # for the internal array type, which yields `BigInt <theIdent>[4]`.
      let ptrStar = gpuTypeToString(t.kind)
      result = gpuTypeToString(t.to, '(' & ptrStar & ident & ')')
      skipIdent = true
    else:
      let typ = gpuTypeToString(t.to, allowEmptyIdent = allowEmptyIdent)
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
    # NOTE: We turn e.g. `foo[float32, uint32]` into `foo_f32_u32`.
    # use short names (uint32, int64) for generic args, not C names (unsigned int, long long)
    checkReservedIdent(t.gName, "type")
    result = t.gName
    for i, g in t.gArgs:
      result.add gpuTypeToShortString(g)
      if i < t.gArgs.high:
        result.add 'x'
  of gtObject:
    checkReservedIdent(t.name, "type")
    result = t.name
  of gtUA:     result = gpuTypeToString(t.uaTo, allowEmptyIdent = allowEmptyIdent) ## unchecked array just T?
  of gtStatic: result = "int"
  else:        result = gpuTypeToString(t.kind)

  if ident.len > 0 and not skipIdent: # still need to add ident
    result.add ' ' & ident

proc genFunctionType*(typ: GpuType, fn: string, fnArgs: string): string =
  ## Function declaration with its return type. Pointers to statically sized arrays
  ## use the `(*fn(args))[N]` spelling. Everything else is a plain `RetType fn(args)`.
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

proc scanFunctions(ctx: var GpuContext, n: GpuAst) =
  ## Iterates over the given function and checks for all `gpuCall` nodes.
  ## Any function called in the scope is added to `fnTab`.
  ## This is a form of dead code elimination.
  case n.kind
  of gpuCall:
    let fn = n.cName
    if fn in ctx.allFnTab:
      # Check if any of the parameters are pointers (otherwise non generic)
      if fn notin ctx.fnTab: # function not known, add to `fnTab` (i.e. avoid code elimination)
        let fnCalled = ctx.allFnTab[fn]
        ctx.fnTab[fn] = fnCalled
        # still "scan for functions", i.e. fill `fnTab` from inner calls
        for ch in fnCalled:
          ctx.scanFunctions(ch)
      # else we don't do anything for this function
    # Harvest functions from arguments to this call!
    for arg in n.cArgs:
      ctx.scanFunctions(arg)
  else:
    for ch in n:
      ctx.scanFunctions(ch)

# ── Metal builtins ──────────────────────────────────────────────────────────

const MetalAtomicCalls = ["atomic_add", "atomicAdd",
                          "atomic_sub", "atomicSub",
                          "atomic_xchg", "atomicExchange"]
  ## IR call names of the atomic builtin dummies (opencl_builtins, vulkan_builtins), mapped to MSL `atomic_fetch_*_explicit`.

proc metalAtomicFnName(name: string): string =
  ## MSL atomic function for an IR atomic builtin call name.
  case name
  of "atomic_add", "atomicAdd": "atomic_fetch_add_explicit"
  of "atomic_sub", "atomicSub": "atomic_fetch_sub_explicit"
  of "atomic_xchg", "atomicExchange": "atomic_exchange_explicit"
  else: ""

proc atomicElemName(t: GpuType): string =
  ## MSL atomic element type for a (possibly pointer- or UA-wrapped) numeric type.
  ## MSL atomics operate on `atomic_uint` / `atomic_int` /
  ## `atomic_float` / `atomic_ulong` / `atomic_long` memory, declared
  ## as kernel pointer params or as shared/private variables.
  var e = t
  if e.kind == gtPtr:
    e = e.to
  if e.kind == gtUA:
    e = e.uaTo
  case e.kind
  of gtUint32: "atomic_uint"
  of gtInt32: "atomic_int"
  of gtFloat32: "atomic_float"
  of gtUint64: "atomic_ulong"
  of gtInt64: "atomic_long"
  else:
    raiseAssert "Metal atomics support 32/64-bit integers and 32-bit floats, got: " & $e.kind

proc collectAtomicTargets(ctx: GpuContext, n: GpuAst, targets: var HashSet[string]) =
  ## Records the first-argument identifier of every atomic builtin call.
  ## The printer uses the set to declare the matching buffer or variable
  ## with an `atomic_<T>` element type, since MSL atomic functions require atomic-typed pointers.
  case n.kind
  of gpuCall:
    if n.cName.kind == gpuIdent and n.cName.ident() in MetalAtomicCalls and n.cArgs.len > 0:
      var arg = n.cArgs[0]
      if arg.kind == gpuAddr and arg.aOf.kind == gpuIdent:
        arg = arg.aOf
      if arg.kind == gpuIdent:
        targets.incl arg.ident()
    for ch in n:
      ctx.collectAtomicTargets(ch, targets)
  else:
    for ch in n:
      ctx.collectAtomicTargets(ch, targets)

proc genLit*(ast: GpuAst): string =
  ## Lower a literal node for the MSL backend.
  if ast.lType.kind == gtString:
    result = '"' & ast.lValue & '"'
  elif ast.lValue == "DEFAULT":
    result = "{}"
  else:
    case ast.lType.kind
    of gtFloat32: result = ast.lValue & "f"
    of gtUint32: result = ast.lValue & "U"
    of gtUint64: result = ast.lValue & "ULL"
    of gtInt64:  result = ast.lValue & "LL"
    of gtInt16, gtUint16, gtUint8, gtBool:
      result = '(' & gpuTypeToString(ast.lType, allowEmptyIdent = true) & ')' & ast.lValue
    of gtFloat64:
      raiseAssert "The Metal target does not support 64-bit floating point (gtFloat64 literal): " & ast.lValue
    else:
      result = ast.lValue

proc exprType(ctx: GpuContext, n: GpuAst): GpuType =
  ## Best-effort type of an expression node (nil when unknown). The printer uses it
  ## to detect array-typed operands of `addr`, which need pointer decay rather than `&`.
  case n.kind
  of gpuIdent: n.symbol.typ
  of gpuLit: n.lType
  of gpuBinOp: n.bType
  of gpuPrefix: ctx.exprType(n.pVal)
  of gpuCall: ctx.getFnReturnType(n.cName)
  of gpuAddr: ctx.exprType(n.aOf)
  of gpuDeref: ctx.exprType(n.dOf)
  of gpuIndex:
    let arrT = ctx.exprType(n.iArr)
    if arrT.isNil:
      nil
    elif arrT.kind == gtArray: arrT.aTyp
    elif arrT.kind == gtPtr: arrT.to
    elif arrT.kind == gtUA: arrT.uaTo
    else: nil
  of gpuObjConstr: n.ocType
  of gpuConv: n.convTo
  of gpuCast: n.cTo
  of gpuMaterialize: ctx.exprType(n.mExpr)
  else: nil

proc collectAttrIdents(n: GpuAst, attrIdents: var seq[string]) =
  ## Records gsBuiltin-marked identifiers in first-use order, deduped.
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.symKind == gsBuiltin and
       n.ident() notin attrIdents:
      attrIdents.add n.ident()
  else:
    for ch in n:
      collectAttrIdents(ch, attrIdents)

proc genKernelParams(ctx: var GpuContext, fn: GpuAst,
                     atomics: HashSet[string]): string =
  ## MSL kernel parameter list:
  ## - output buffer first: `device T*` at `[[buffer(0)]]`
  ## - input buffers: `device const T*`
  ## - scalars: `constant T&`
  ## - the attribute params for the index builtins the kernel body references
  ## The workgroup size is dispatch-time, hence no baked threadgroup-size attribute.
  ## `bool` element types become `int` to match the host's 4-byte i32 marshalling (arg_blobs blobOf).
  ## Atomic-used pointers are declared `device atomic_<T>*` and never const, since atomics mutate.
  var attrIdents: seq[string]
  collectAttrIdents(fn.pBody, attrIdents)
  var params: seq[string]
  var bufferIdx = 0
  for i, p in fn.pParams:
    let name = p.ident.ident()
    checkReservedIdent(name, "parameter")
    let binding = " [[buffer(" & $bufferIdx & ")]]"
    if p.typ.kind == gtPtr:
      var inner = p.typ.to
      if inner.kind == gtUA:
        inner = inner.uaTo
      if name in atomics:
        params.add "device " & atomicElemName(p.typ) & "* " & name & binding
      else:
        var elem = gpuTypeToString(inner, allowEmptyIdent = true)
        if inner.kind == gtBool:
          elem = "int"
        if i == 0: # the engine binds the output at index 0
          params.add "device " & elem & "* " & name & binding
        else:
          params.add "device const " & elem & "* " & name & binding
    else:
      var elem = gpuTypeToString(p.typ, allowEmptyIdent = true)
      if p.typ.kind == gtBool:
        elem = "int"
      params.add "constant " & elem & "& " & name & binding
    inc bufferIdx
  for attr in attrIdents:
    params.add "uint3 " & attr & " [[" & attr & "]]"
  result = params.join(", ")

proc genDeviceParam(ctx: var GpuContext, p: GpuParam,
                    atomics: HashSet[string]): string =
  ## MSL parameter for a device function (non-kernel). Pointer params carry an address space:
  ## `thread` for implicit `var T` params (locals), `device` for explicit `ptr T` params (kernel buffers).
  ## Atomic-used pointers get the matching `atomic_<T>` element type.
  ## Large passByRef structs are emitted `thread const T&`.
  ## Such a reference binds thread-space values and temporaries, like a C++ const reference.
  let name = p.ident.ident()
  checkReservedIdent(name, "parameter")
  if p.typ.kind == gtPtr:
    var inner = p.typ.to
    if inner.kind == gtUA:
      inner = inner.uaTo
    if name in atomics:
      result = "device " & atomicElemName(p.typ) & "* " & name
    else:
      var elem = gpuTypeToString(inner, allowEmptyIdent = true)
      if inner.kind == gtBool:
        elem = "int"
      let space = if p.typ.implicit: "thread" else: "device"
      result = space & ' ' & elem & "* " & name
  else:
    result = gpuTypeToString(p.typ, name)

proc metalVarAttr(a: GpuVarAttribute): string =
  ## MSL address-space or qualifier keyword for a GPU variable attribute.
  case a
  of atvShared: "threadgroup"
  of atvConstant: "constant"
  of atvExtern: "extern"
  of atvVolatile: "volatile"
  of atvPrivate: "thread"

proc genAtomicVarType(t: GpuType, ident: string): string =
  ## `atomic_<T>` MSL declaration for a numeric or static-array variable.
  if t.kind == gtArray:
    result = atomicElemName(t.aTyp) & ' ' & ident & '[' & $t.aLen & ']'
  else:
    result = atomicElemName(t) & ' ' & ident

proc genMetal*(ctx: var GpuContext, ast: GpuAst, indent = 0): string

proc preprocess*(ctx: var GpuContext, ast: GpuAst, kernel: string = "") =
  ## MSL-specific IR preprocessing before codegen, mirroring the CUDA pipeline.
  ## Type definitions land before any global that uses them.
  ## MSL is a C-family language, and types come first.
  ## Global functions fill `fnTab`. Reachable device functions are harvested
  ## via `scanFunctions`.
  # 1. Add all data from `genericInsts` and `types` tables.
  #    In MSL the types have to appear before any global variables using them.
  for k, v in pairs(ctx.genericInsts):
    ctx.allFnTab[k] = v
  for k, typ in pairs(ctx.types):
    ctx.globalBlocks.add typ

  # 2. Fill the table with all *global* functions, or *only* the specific `kernel`
  #    if one is given.
  var varBlock = GpuAst(kind: gpuBlock)
  ctx.farmTopLevel(ast, kernel, varBlock)
  ctx.globalBlocks.add varBlock

  # 3. Traverse every global function's AST for `gpuCall` nodes, and record the called functions in `fnTab`.
  let fns = toSeq(ctx.fnTab.pairs)
  for (fnIdent, fn) in fns: # everything in `fnTab` at this point is a global function
    ctx.scanFunctions(fn)

proc genMetalImpl(ctx: var GpuContext, ast: GpuAst, indent: int,
                  atomics: HashSet[string]): string =
  ## The actual MSL code generator. `atomics` holds the identifiers
  ## of the atomic-call targets in the enclosing function. Params and variables
  ## in that set declare `atomic_<T>` element types.
  let indentStr = "  ".repeat(indent)
  case ast.kind
  of gpuDiscard: return # nothing to emit
  of gpuProc:
    checkReservedIdent(ast.pName.ident(), "function")
    var fnAtomics: HashSet[string]
    ctx.collectAtomicTargets(ast.pBody, fnAtomics)
    let isKernel = attGlobal in ast.pAttributes
    # Parameters
    var params: seq[string]
    if isKernel:
      params.add ctx.genKernelParams(ast, fnAtomics)
    else:
      for p in ast.pParams:
        if p.passByRef:
          # MSL requires an explicit address-space qualifier on reference params.
          # The referenced struct lives in the calling thread's memory.
          checkReservedIdent(p.ident.ident(), "parameter")
          params.add "thread const " & gpuTypeToString(p.typ, allowEmptyIdent = true) & "& " & p.ident.ident()
        else:
          params.add ctx.genDeviceParam(p, fnAtomics)
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
      result &= ctx.genMetalImpl(ast.pBody, indent + 1, fnAtomics)
      result &= '\n' & indentStr & '}'

  of gpuBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "{ // " & ast.blockLabel & '\n'
    for i, el in ast.statements:
      let code = ctx.genMetalImpl(el, indent + (if ast.blockLabel.len > 0: 1 else: 0), atomics)
      if code.len == 0:
        continue # skip gpuDiscard and empty statements
      result.add code
      if el.kind != gpuBlock and not ctx.skipSemicolon:
        result.add ';'
      if i < ast.statements.high:
        result.add '\n'
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "} // " & ast.blockLabel & '\n'

  of gpuVar:
    let vName = ast.vName.ident()
    checkReservedIdent(vName, "variable")
    if vName in atomics and ast.vInit.kind != gpuDiscard:
      raiseAssert "Atomic variable '" & vName & "' must be declared without an initializer. " &
        "MSL atomic types with value initializers are not device-verified."
    var attrs = ""
    for a in ast.vAttributes:
      attrs.add metalVarAttr(a) & ' '
    let typ =
      if vName in atomics: genAtomicVarType(ast.vType, vName)
      else: gpuTypeToString(ast.vType, vName)
    result = indentStr & attrs & typ
    if ast.vInit.kind != gpuDiscard:
      result &= " = " & ctx.genMetalImpl(ast.vInit, 0, atomics)
  of gpuAssign:
    result = indentStr & ctx.genMetalImpl(ast.aLeft, 0, atomics) & " = " &
             ctx.genMetalImpl(ast.aRight, 0, atomics)
  of gpuIf:
    # skip semicolon in the condition. Otherwise can lead to problematic code
    ctx.withoutSemicolon: # skip semicolon for if bodies
      result = indentStr & "if (" & ctx.genMetalImpl(ast.ifCond, 0, atomics) & ") {\n"
    result &= ctx.genMetalImpl(ast.ifThen, indent + 1, atomics) & '\n'
    result &= indentStr & '}'
    if ast.ifElse.kind != gpuDiscard:
      result &= " else {\n"
      result &= ctx.genMetalImpl(ast.ifElse, indent + 1, atomics) & '\n'
      result &= indentStr & '}'

  of gpuTernary:
    ctx.withoutSemicolon:
      result = '(' & ctx.genMetalImpl(ast.tCond, 0, atomics) & " ? " &
               ctx.genMetalImpl(ast.tThen, 0, atomics) & " : " &
               ctx.genMetalImpl(ast.tElse, 0, atomics) & ')'

  of gpuFor:
    checkReservedIdent(ast.fVar.ident(), "loop variable")
    let cmp = if ast.fRangeKind == rkInclusive: " <= " else: " < "
    result = indentStr & "for(int " & ast.fVar.ident() & " = " &
             ctx.genMetalImpl(ast.fStart, 0, atomics) & "; " &
             ast.fVar.ident() & cmp & ctx.genMetalImpl(ast.fEnd, 0, atomics) & "; " &
             ast.fVar.ident() & "++) {\n"
    result &= ctx.genMetalImpl(ast.fBody, indent + 1, atomics) & '\n'
    result &= indentStr & '}'
  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genMetalImpl(ast.wCond, 0, atomics) & "){\n"
    result &= ctx.genMetalImpl(ast.wBody, indent + 1, atomics) & '\n'
    result &= indentStr & '}'

  of gpuDot:
    result = ctx.genMetalImpl(ast.dParent, 0, atomics) & '.' &
             ctx.genMetalImpl(ast.dField, 0, atomics)

  of gpuIndex:
    result = ctx.genMetalImpl(ast.iArr, 0, atomics) & '[' &
             ctx.genMetalImpl(ast.iIndex, 0, atomics) & ']'

  of gpuCall:
    let name = ast.cName.ident()
    if name == "__syncthreads" or name == "syncthreads":
      # The `{.cudaName: "__syncthreads".}` pragma on the shared builtin dummy renames the call.
      # MSL spells the barrier with its memory flags.
      result = indentStr & "threadgroup_barrier(mem_flags::mem_threadgroup)"
    elif name in MetalAtomicCalls:
      # The first argument must be an atomic target that `collectAtomicTargets` recorded:
      # a plain identifier or `addr <ident>`. Any other shape (e.g. `addr arr[i]`)
      # would emit a non-atomic pointer that MSL rejects, so raise instead of emitting invalid code.
      if ast.cArgs.len == 0:
        raiseAssert "Atomic call '" & name & "' requires a target argument"
      let first = ast.cArgs[0]
      let target = if first.kind == gpuAddr: first.aOf else: first
      if target.kind != gpuIdent or target.ident() notin atomics:
        raiseAssert "Atomic call '" & name & "' targets a non-atomic identifier. " &
          "Pass a plain identifier or `addr <ident>` so the atomic declaration is emitted."
      var args: seq[string]
      for a in ast.cArgs:
        args.add ctx.genMetalImpl(a, 0, atomics)
      args.add "memory_order_relaxed" # MSL atomics require an explicit order
      result = indentStr & metalAtomicFnName(name) & '(' & args.join(", ") & ')'
    else:
      var args: seq[string]
      for a in ast.cArgs:
        args.add ctx.genMetalImpl(a, 0, atomics)
      result = indentStr & ctx.getFnName(bkMetal, ast) & '(' & args.join(", ") & ')'
  of gpuTemplateCall:
    when nimvm:
      error("Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `metal` macro.")
    else:
      raise newException(ValueError, "Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `metal` macro.")

  of gpuBinOp:
    ctx.withoutSemicolon:
      let l = ctx.genMetalImpl(ast.bLeft, 0, atomics)
      let r = ctx.genMetalImpl(ast.bRight, 0, atomics)
      result = indentStr & '(' & l & ' ' &
               ctx.genMetalImpl(ast.bOp, 0, atomics) & ' ' &
               r & ')'

  of gpuIdent:
    # Identity naming: the source-level builtin names are the MSL attribute names,
    # so a builtin-marked identifier emits verbatim
    # and binds to the attribute param `genKernelParams` injects for that reference.
    # Unmarked identifiers never get a param, so an undeclared name stays a loud MSL error.
    checkReservedIdent(ast.ident(), "identifier")
    result = ast.ident()

  of gpuLit:
      result = genLit(ast)

  of gpuArrayLit:
    result = "{"
    for i, el in ast.aValues:
      result.add '(' & gpuTypeToString(ast.aLitType) & ')' & ctx.genMetalImpl(el, 0, atomics)
      if i < ast.aValues.high:
        result.add ", "
    result.add '}'

  of gpuReturn:
    result = indentStr & "return " & ctx.genMetalImpl(ast.rValue, 0, atomics)

  of gpuPrefix:
    result = ast.pOp & ctx.genMetalImpl(ast.pVal, 0, atomics)

  of gpuTypeDef:
    result = "struct " & gpuTypeToString(ast.tTyp) & "{\n"
    if ast.tFields.len == 0:
      # MSL requires at least one field in a struct.
      result.add "  char _;\n"
    else:
      for el in ast.tFields:
        checkReservedIdent(el.name, "field")
        result.add "  " & gpuTypeToString(el.typ, el.name) & ";\n"
    result.add '}'

  of gpuAlias:
    # Aliases come from `ctx.types`. MSL spells them as C++11 `using`
    result = "using " & gpuTypeToString(ast.aTyp) & " = " &
             ctx.genMetalImpl(ast.aTo, 0, atomics) & ';'

  of gpuObjConstr:
    # Braced init list: TypeName{val1, val2, ...}
    # Using `TypeName{...}` (functional-style cast) instead of bare `{val}`
    # ensures the result is a valid C++ expression. Bare braced-init-lists
    # are not expressions and cannot be used with member access (gpuDot).
    result = gpuTypeToString(ast.ocType, allowEmptyIdent = true) & "{"
    for i, el in ast.ocFields:
      if el.value.kind == gpuDiscard:
        result.add "{}"
      else:
        result.add ctx.genMetalImpl(el.value, 0, atomics)
      if i < ast.ocFields.len - 1:
        result.add ", "
    result.add '}'

  of gpuInlineAsm:
    raiseAssert "Inline assembly is not supported on the Metal target."

  of gpuComment:
    result = indentStr & "/* " & ast.comment & " */"

  of gpuConv:
    result = '(' & gpuTypeToString(ast.convTo, allowEmptyIdent = true) & ')' &
             ctx.genMetalImpl(ast.convExpr, 0, atomics)
  of gpuCast:
    result = '(' & gpuTypeToString(ast.cTo, allowEmptyIdent = true) & ')' &
             ctx.genMetalImpl(ast.cExpr, 0, atomics)

  of gpuAddr:
    let t = ctx.exprType(ast.aOf)
    if not t.isNil and t.kind == gtArray:
      # Pointer decay: an array name is already a `T*` rvalue, while `&arr` would produce a pointer-to-array
      # that MSL rejects where a `T*` is expected.
      result = ctx.genMetalImpl(ast.aOf, 0, atomics)
    else:
      result = "(&" & ctx.genMetalImpl(ast.aOf, 0, atomics) & ')'

  of gpuDeref:
    result = "(*" & ctx.genMetalImpl(ast.dOf, 0, atomics) & ')'

  of gpuConstexpr:
    ## MSL supports C++14 `constexpr` variables. Arrays need the length in the type.
    ## Hence the two emission shapes (mirrors the CUDA printer).
    let cInit =
      if ast.cValue.kind == gpuDiscard: "{}"
      else: ctx.genMetalImpl(ast.cValue, 0, atomics)
    if ast.cType.kind == gtArray:
      result = indentStr & "constexpr " & gpuTypeToString(ast.cType, ctx.genMetalImpl(ast.cIdent, 0, atomics)) & " = " & cInit
    else:
      result = indentStr & "constexpr " & gpuTypeToString(ast.cType, allowEmptyIdent = true) & ' ' &
               ctx.genMetalImpl(ast.cIdent, 0, atomics) & " = " & cInit
  of gpuMaterialize:
    result = ctx.genMetalImpl(ast.mExpr, 0, atomics) # C++ const& binds implicitly to temporaries

  else:
    raiseAssert "Unhandled node kind in genMetal: " & ast.repr

proc genMetal*(ctx: var GpuContext, ast: GpuAst, indent = 0): string =
  ## MSL code for `ast`, used by `codegen` for top-level nodes. The entry point
  ## passes an empty atomic target set, because each `gpuProc` computes its own set from its body.
  var none: HashSet[string]
  ctx.genMetalImpl(ast, indent, none)

proc codegen*(ctx: var GpuContext): string =
  ## Emits the full MSL translation unit:
  ## - the `metal_stdlib` header
  ## - the global blocks (types, global variables)
  ## - forward declarations for device functions, then every function body.
  result.add "#include <metal_stdlib>\n"
  result.add "using namespace metal;\n\n"

  # 1. generate code for the global blocks (types, global vars etc).
  #    Empty blocks (e.g. a kernel source with no globals) emit nothing.
  for blk in ctx.globalBlocks:
    let code = ctx.genMetal(blk)
    if code.len > 0:
      result.add code & ";\n\n"

  # 2. Generate forward declarations for the device functions
  #    so that kernels defined before the functions they call still compile.
  #    Kernels need no forward declaration, since nothing calls a kernel from inside a shader.
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
