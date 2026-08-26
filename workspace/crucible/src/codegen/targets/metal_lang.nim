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
##   - coordinate builtins ride in the param list: the `materializeIndexBuiltinParams` pass
##     appends one param per referenced builtin, marked on the symbol's `coordBuiltin`.
##     Kernels emit the attribute-qualified form (`uint3 name [[name]]`), device functions
##     the plain form, and call sites forward the names
##     (MSL device functions have no implicit thread index)
##   - the threadgroup size stays host-side (dispatch-time) and is never baked into the shader.
##
## `MetalReservedKeywords` is the MSL reserved-word table minus the boolean literals
## `true`/`false` (emitted as valid MSL tokens), plus the `metal` namespace.
## It is a defensible superset of the MSL reserved words, not a hand-picked sample.

import std / [macros, strformat, strutils, sequtils, tables]

import ../ir/gpu_types
import ./lang_utils
import ../passes/passes_preprocessing as pp

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
  of gtFloat16: "half"
  of gtBf16: "bfloat"
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
    let inner = if t.to.kind == gtUA: t.to.uaTo else: t.to
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
    of gtFloat32: result = ast.lValue & "f"
    of gtFloat16: result = ast.lValue & "h"
    of gtBf16:    result = "bfloat(" & ast.lValue & ')'
    of gtUint32: result = ast.lValue & "U"
    of gtUint64: result = ast.lValue & "ULL"
    of gtInt64:  result = ast.lValue & "LL"
    of gtInt16, gtUint16, gtUint8, gtBool:
      result = '(' & gpuTypeToString(ast.lType, allowEmptyIdent = true) & ')' & ast.lValue
    of gtFloat64:
      raiseAssert "The Metal target does not support 64-bit floating point (gtFloat64 literal): " & ast.lValue
    else:
      result = ast.lValue

proc genKernelParams(ctx: var GpuContext, fn: GpuAst): string =
  ## MSL kernel parameter list:
  ## - buffer params: `device T*`, never const — matches device-fn params, and
  ##   MSL accepts non-const input buffers
  ## - scalars: `constant T&`
  ## - a param whose symbol carries a coordinate builtin kind emits the
  ##   attribute form (`uint3 name [[name]]`, scalar `uint` for the flat thread
  ##   index): no `[[buffer(n)]]` binding, and it does not advance the buffer
  ##   index. The `materializeIndexBuiltinParams` pass appends these after the
  ##   declared params, so declared params keep their `[[buffer(n)]]` positions.
  ## The workgroup size is dispatch-time, hence no baked threadgroup-size attribute.
  ## Scalars stay 4 bytes on the host (arg_blobs blobOf), so scalar `bool` is declared `int`.
  ## Buffer elements marshal at their Nim width, so bool buffers declare `bool` (1 byte).
  var params: seq[string]
  var bufferIdx = 0
  for p in fn.pParams:
    let name = p.ident.ident()
    checkReservedIdent(name, "parameter")
    if p.ident.symbol.coordBuiltin != gbkNone:
      # The vector coordinate builtins are `uint3` attribute params.
      # The flat thread index `gbkThreadIndexInThreadgroup` is a scalar `uint` builtin.
      let attrType = if p.ident.symbol.coordBuiltin == gbkThreadIndexInThreadgroup:
        "uint" else: "uint3"
      params.add attrType & " " & name & " [[" & name & "]]"
      continue
    let binding = " [[buffer(" & $bufferIdx & ")]]"
    if p.typ.kind == gtPtr:
      var inner = p.typ.to
      if inner.kind == gtUA:
        inner = inner.uaTo
      var elem = gpuTypeToString(inner, allowEmptyIdent = true)
      params.add "device " & elem & "* " & name & binding
    else:
      var elem = gpuTypeToString(p.typ, allowEmptyIdent = true)
      if p.typ.kind == gtBool:
        elem = "int"
      params.add "constant " & elem & "& " & name & binding
    inc bufferIdx
  result = params.join(", ")

proc genDeviceParam(ctx: var GpuContext, p: GpuParam): string =
  ## MSL parameter for a device function (non-kernel). Pointer params carry an address space:
  ## `thread` for implicit `var T` params (locals), `device` for explicit `ptr T` params (kernel buffers).
  ## Large passByRef structs are emitted `thread const T&`.
  ## Such a reference binds thread-space values and temporaries, like a C++ const reference.
  let name = p.ident.ident()
  checkReservedIdent(name, "parameter")
  if p.typ.kind == gtPtr:
    var inner = p.typ.to
    if inner.kind == gtUA:
      inner = inner.uaTo
    var elem = gpuTypeToString(inner, allowEmptyIdent = true)
    let space = if p.typ.implicit: "thread" else: "device"
    result = space & ' ' & elem & "* " & name
  else:
    # By-value array params decay to pointers in MSL. They are read-only
    # (a plain `array[T]` param), so emit `const`: the call site passes
    # const lvalue arrays, and a non-const `T*` would reject them.
    if p.typ.kind == gtArray:
      result = "const " & gpuTypeToString(p.typ, name)
    else:
      result = gpuTypeToString(p.typ, name)

proc addrSpaceToMsl(space: AddressSpace): string =
  case space
  of asDevice: "device"
  of asConstant: "constant"
  of asSMEM: "threadgroup"
  of asRMEM: "thread"

proc genMetal*(ctx: var GpuContext, ast: GpuAst, indent = 0): string
proc genMetalImpl(ctx: var GpuContext, ast: GpuAst, indent: int): string

proc structVariantName(ctx: GpuContext, t: GpuType,
                       spaces: seq[AddressSpace]): string =
  ## MSL struct name of a space tuple: base name for the first observed
  ## tuple, otherwise base name suffixed with the tuple's space tags.
  result = gpuTypeToString(t, allowEmptyIdent = true)
  if ctx.ptrFieldVariants.getOrDefault(t, @[]).find(spaces) > 0:
    result.add pp.variantSuffix(spaces)

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

proc genMetalImpl(ctx: var GpuContext, ast: GpuAst, indent: int): string =
  ## The actual MSL code generator.
  let indentStr = "  ".repeat(indent)
  case ast.kind
  of gpuDiscard: return # nothing to emit
  of gpuProc:
    checkReservedIdent(ast.pName.ident(), "function")
    let isKernel = attGlobal in ast.pAttributes
    # Parameters
    var params: seq[string]
    if isKernel:
      params.add ctx.genKernelParams(ast)
    else:
      for p in ast.pParams:
        if p.passByRef:
          # MSL requires an explicit address-space qualifier on reference params.
          # The referenced struct lives in the calling thread's memory.
          checkReservedIdent(p.ident.ident(), "parameter")
          params.add "thread const " & gpuTypeToString(p.typ, allowEmptyIdent = true) & "& " & p.ident.ident()
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
    checkReservedIdent(vName, "variable")
    var attrs = ""
    if ast.addressSpace != asRMEM:
      attrs.add addrSpaceToMsl(ast.addressSpace) & ' '
    var typ = gpuTypeToString(ast.vType, vName)
    if ast.vInit.kind == gpuObjConstr:
      # The var's decl type must match the objconstr's variant name, or MSL
      # rejects the type mismatch.
      let variantName = ctx.structVariantName(ast.vInit.ocType,
                                              pp.siteSpaceTuple(ctx, ast.vInit))
      let base = gpuTypeToString(ast.vInit.ocType, allowEmptyIdent = true)
      if variantName != base:
        typ = variantName & ' ' & vName
    result = indentStr & attrs & typ
    if ast.vInit.kind != gpuDiscard:
      result &= " = " & ctx.genMetalImpl(ast.vInit, 0)
  of gpuAssign:
    result = indentStr & ctx.genMetalImpl(ast.aLeft, 0) & " = " &
             ctx.genMetalImpl(ast.aRight, 0)
  of gpuIf:
    # skip semicolon in the condition. Otherwise can lead to problematic code
    ctx.withoutSemicolon: # skip semicolon for if bodies
      result = indentStr & "if (" & ctx.genMetalImpl(ast.ifCond, 0) & ") {\n"
    result &= ctx.genMetalImpl(ast.ifThen, indent + 1) & '\n'
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
    checkReservedIdent(ast.fVar.ident(), "loop variable")
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
    result = ctx.genMetalImpl(ast.dParent, 0) & '.' &
             ctx.genMetalImpl(ast.dField, 0)

  of gpuIndex:
    result = ctx.genMetalImpl(ast.iArr, 0) & '[' &
             ctx.genMetalImpl(ast.iIndex, 0) & ']'

  of gpuCall:
    case ast.cName.symbol.synchroBuiltin
    of gbkThreadgroupBarrier:
      # Every backend spelling of the barrier is an alias template that sem
      # expands to the canonical call, so only this kind reaches the IR.
      # MSL spells the barrier with its memory flags.
      result = indentStr & "threadgroup_barrier(mem_flags::mem_threadgroup)"
    of gbkNone:
      case ast.cName.symbol.reductionBuiltin
      of gbkSimdShuffleDown:
        # SIMD-group gather from lane + delta: `simd_shuffle_down(v, delta)`.
        result = indentStr & "simd_shuffle_down(" &
                 ctx.genMetalImpl(ast.cArgs[0], 0) & ", " &
                 ctx.genMetalImpl(ast.cArgs[1], 0) & ')'
      of gbkSimdShuffle:
        # SIMD-group gather from an absolute lane: `simd_shuffle(v, lane)`.
        result = indentStr & "simd_shuffle(" &
                 ctx.genMetalImpl(ast.cArgs[0], 0) & ", " &
                 ctx.genMetalImpl(ast.cArgs[1], 0) & ')'
      of gbkNone:
        var args: seq[string]
        for a in ast.cArgs:
          args.add ctx.genMetalImpl(a, 0)
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
    checkReservedIdent(ast.ident(), "identifier")
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
    let ptrNames = pp.ptrFieldNames(ast.tTyp)
    var tuples = ctx.ptrFieldVariants.getOrDefault(ast.tTyp, @[])
    if tuples.len == 0:
      # Never constructed: one struct with per-thread pointer fields.
      tuples = @[newSeqWith(ptrNames.len, asRMEM)]
    for i, spaces in tuples:
      let suffix = if i == 0: "" else: pp.variantSuffix(spaces)
      result.add "struct " & gpuTypeToString(ast.tTyp) & suffix & "{\n"
      if ast.tFields.len == 0:
        # MSL requires at least one field in a struct.
        result.add "  char _;\n"
      else:
        for el in ast.tFields:
          checkReservedIdent(el.name, "field")
          # MSL has no string type: every gtString field of every struct
          # is dropped here, not just the tile layer's atom descriptors.
          # The invariant is that a struct reaching MSL emission carries
          # no runtime string data, because the drop shifts every later
          # field's offset and a consumer would silently read the next
          # field. The MmaAtom name/instr strings are compile-time-only:
          # the atom travels as a static generic param and its struct is
          # emitted for type-name completeness but never instantiated,
          # so the dropped fields carry no runtime data.
          if el.typ.kind == gtString:
            continue
          # MSL requires an explicit address-space qualifier on pointer-typed
          # fields.
          if el.typ.kind == gtPtr:
            let fi = ptrNames.find(el.name)
            let space = if fi >= 0 and fi < spaces.len: spaces[fi] else: asRMEM
            result.add "  " & addrSpaceToMsl(space) & ' ' &
                       gpuTypeToString(el.typ, el.name) & ";\n"
          else:
            result.add "  " & gpuTypeToString(el.typ, el.name) & ";\n"
      result.add '}'
      if i < tuples.high:
        # MSL requires `;` after every struct declaration, not just the last
        # variant of the type.
        result.add ";\n"

  of gpuAlias:
    # Aliases come from `ctx.types`. MSL spells them as C++11 `using`
    result = "using " & gpuTypeToString(ast.aTyp) & " = " &
             ctx.genMetalImpl(ast.aTo, 0) & ';'

  of gpuObjConstr:
    # Braced init list: TypeName{val1, val2, ...}
    # Using `TypeName{...}` (functional-style cast) instead of bare `{val}`
    # ensures the result is a valid C++ expression. Bare braced-init-lists
    # are not expressions and cannot be used with member access (gpuDot).
    result = ctx.structVariantName(ast.ocType, pp.siteSpaceTuple(ctx, ast)) & "{"
    for i, el in ast.ocFields:
      if el.value.kind == gpuDiscard:
        result.add "{}"
      else:
        result.add ctx.genMetalImpl(el.value, 0)
      if i < ast.ocFields.len - 1:
        result.add ", "
    result.add '}'

  of gpuInlineAsm:
    # Raw MSL statement: the gpuBlock loop appends the `;` terminator
    # (CUDA/OpenCL/Vulkan wrap the body in `asm(...)`; MSL has no such
    # wrapper — the simdgroup intrinsics are plain statements).
    result = indentStr & genAsmStmt(ast).strip

  of gpuEmit:
    # Self-terminating raw text: the gpuBlock loop appends no `;`
    # (the emitted text owns its own terminators).
    result = genEmitStmt(ctx, ast,
      proc(c: var GpuContext; n: GpuAst): string = c.genMetalImpl(n, 0))

  of gpuComment:
    result = indentStr & "/* " & ast.comment & " */"

  of gpuConv:
    var castTyp = gpuTypeToString(ast.convTo, allowEmptyIdent = true)
    # Pointer-typed casts need an explicit address-space qualifier. The
    # space is resolved from the cast operand's value (`asRMEM` default).
    if ast.convTo.kind == gtPtr:
      castTyp = addrSpaceToMsl(pp.resolveValueAddressSpace(ctx, ast.convExpr)) & ' ' & castTyp
    result = '(' & castTyp & ')' & ctx.genMetalImpl(ast.convExpr, 0)
  of gpuCast:
    var castTyp = gpuTypeToString(ast.cTo, allowEmptyIdent = true)
    if ast.cTo.kind == gtPtr:
      castTyp = addrSpaceToMsl(pp.resolveValueAddressSpace(ctx, ast.cExpr)) & ' ' & castTyp
    result = '(' & castTyp & ')' & ctx.genMetalImpl(ast.cExpr, 0)

  of gpuAddr:
    result = "(&" & ctx.genMetalImpl(ast.aOf, 0) & ')'

  of gpuDeref:
    result = "(*" & ctx.genMetalImpl(ast.dOf, 0) & ')'

  of gpuConstexpr:
    ## MSL supports C++14 `constexpr` variables. Arrays need the length in the type.
    ## Hence the two emission shapes (mirrors the CUDA printer).
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
    raiseAssert "Unhandled node kind in genMetal: " & ast.repr

proc genMetal*(ctx: var GpuContext, ast: GpuAst, indent = 0): string =
  ## MSL code for `ast`, used by `codegen` for top-level nodes.
  ctx.genMetalImpl(ast, indent)

proc codegen*(ctx: var GpuContext): string =
  ## Emits the full MSL translation unit:
  ## - the `metal_stdlib` header
  ## - the global blocks (types, global variables)
  ## - forward declarations for device functions, then every function body.
  # Resolve the value-level address spaces (var pragmas and struct pointer
  # fields) before any spelling needs them.
  pp.collectValueAddressSpaces(ctx)
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
