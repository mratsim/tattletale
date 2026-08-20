# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, strformat, strutils, sugar, sequtils, tables, options]

import ../ir/gpu_types

import ./lang_utils
import ../passes/passes_preprocessing as pp

proc gpuTypeToString*(t: GpuType,
                      ident: string = "",
                      allowArrayToPtr = false,
                      allowEmptyIdent = false,
                      addrSpace: AddressSpace = asRMEM): string

proc size*(ctx: var GpuContext, a: GpuType): string = size(gpuTypeToString(a, allowEmptyIdent = true))


proc addrSpaceToWgsl(space: AddressSpace): string =
  ## WGSL keyword for `ptr<space, T>` types.
  case space
  of asDevice: "storage"
  of asConstant: "uniform"
  of asSMEM: "workgroup"
  of asRMEM: "function"

proc constructPtrSignature(addrSpace: AddressSpace, idTyp: GpuType, ptrStr, typStr: string): string =
  ## Constructs the `ptr<addressSpace, typStr, [read / read_write]>` string, which only includes
  ## the RW string if the address space is `storage`
  let rw = if idTyp.kind != gtVoid: idTyp.mutable else: false # symbol is a pointer -> mutable (can be implicit via `var T`)
  let rwStr = if rw: "read_write" else: "read"
  let space = addrSpaceToWgsl(addrSpace)
  if addrSpace == asDevice:
    result = &"{ptrStr}<{space}, {typStr}, {rwStr}>"
  else:
    result = &"{ptrStr}<{space}, {typStr}>"

proc gpuTypeToString*(t: GpuTypeKind): string =
  case t
  of gtBool: "bool"
  of gtUint32: "u32"
  of gtInt32: "i32"
  of gtFloat32: "f32"
  of gtVoid: ""
  of gtSize_t: "u32" ##: Acceptable mapping?
  of gtPtr: "ptr" ## XXX: needs address space and target type, `ptr<address_space, target_type>`
  of gtUA: "array"
  of gtObject: "struct"
  of gtUint8, gtUint16, gtUint64, gtInt16, gtInt64, gtFloat64, gtVoidPtr, gtString:
    raiseAssert "The type " & $t & " does not exist on the WebGPU target."
  of gtStatic: "i32"
  else:
    raiseAssert "Invalid type : " & $t

proc gpuTypeToString*(t: GpuType, ident: string = "", allowArrayToPtr = false,
                           allowEmptyIdent = false,
                           addrSpace: AddressSpace = asRMEM
                    ): string =
  ## WebGPU type generation is a bit more complicated than CUDA, due to their pointer semantics.
  var skipIdent = false
  case t.kind
  of gtPtr:
    # Let `foo` be the symbol `id`. If for example we generate code for `addr(foo)`, the type
    # `t` will be `ptr typeof(foo)`. The type of the symbol `id` though is static.
    # Thus, can use `id's` type to determine if we need mutability or not. If `id` was a
    # pointer, `mutable` will be true and `false` otherwise.
    # If code called with default `id`, type will be nil
    let ptrStr = gpuTypeToString(t.kind)
    let typStr = gpuTypeToString(t.to, allowEmptyIdent = true)
    let idTyp = if t.to.kind == gtPtr: t.to else: t
    result = constructPtrSignature(addrSpace, idTyp, ptrStr, typStr)
  of gtArray:
    # empty idents happen in e.g. function return types or casts
    if ident.len == 0 and not allowEmptyIdent: # and not allowArrayToPtr:
      #error("Invalid call, got an array type but don't have an identifier: " & $t)

      when nimvm:
        error("Invalid call, got an array type but don't have an identifier: " & $t)
      else:
        raise newException(ValueError, "Invalid call, got an array type but don't have an identifier: " & $t)

    let identPrefix = if ident.len > 0: ident & ": " else: ""
    let typ = gpuTypeToString(t.aTyp, allowEmptyIdent = true)
    if t.aLen == 0:
      result = &"{identPrefix}array<{typ}>"
    else:
      result = &"{identPrefix}array<{typ}, {t.aLen}>"
    skipIdent = true
  of gtGenericInst:
    # NOTE: WGSL does not support actual custom generic types. And as we only anyway deal with generic instantiations
    # we simply turn e.g. `foo[float32, uint32]` into `foo_f32_u32`.
    result = t.gName
    for i, g in t.gArgs:
      result.add gpuTypeToShortString(g)
      if i < t.gArgs.high:
        result.add 'x'
  of gtObject: result = t.name
  of gtUA:     result = gpuTypeToString(t.kind) & '<' & gpuTypeToString(t.uaTo, allowEmptyIdent = allowEmptyIdent) & '>'
  of gtStatic: result = "i32"
  else:        result = gpuTypeToString(t.kind)

  if ident.len > 0 and not skipIdent: # still need to add ident
    result = ident & ": " & result

proc genFunctionType*(typ: GpuType, fn: string, fnArgs: string): string =
  ## Returns the correct function with its return type
  if typ.kind == gtPtr and typ.to.kind == gtArray:
    ## TODO!
    # crazy stuff. Syntax to return a pointer to a statically sized array:
    # `Foo (*fnName(fnArgs))[ArrayLen]`
    # where the return type is actually:
    # `Foo (*)[ArrayLen]` (which already is hideous)
    let arrayTyp = typ.to.aTyp
    let innerTyp = gpuTypeToString(arrayTyp, allowEmptyIdent = true)
    let innerLen = $typ.to.aLen
    ## XXX: wrong
    result = &"{innerTyp} (*{fn}({fnArgs}))[{innerLen}]"
  else:
    # normal stuff
    result = &"{fn}({fnArgs})"
    let typ = gpuTypeToString(typ, allowEmptyIdent = true)
    if typ.len > 0:
      result.add &" -> {typ}"

proc patchType(t: GpuType): GpuType =
  ## Applies patches needed for WGSL support. E.g. `bool` cannot be a storage variable.
  result = t
  if result.kind == gtBool:
    result.kind = gtInt32
  elif result.kind == gtPtr and result.to.kind == gtBool:
    result.to.kind = gtInt32

proc patchSymbol(n: GpuAst): GpuAst =
  ## Applies patches needed for WGSL support. E.g. `bool` cannot be a storage variable.
  doAssert n.kind == gpuIdent, "Must be an ident, is: " & $n.kind
  result = n
  if n.symbol != nil and n.symbol.symKind == gsGlobalKernelParam:
    result.symbol.typ = patchType(result.symbol.typ)

proc shortAddrSpace(addrSpace: AddressSpace): string =
  ## Shortens the address space to a single letter
  case addrSpace
  of asDevice: "s"
  of asConstant: "u"
  of asSMEM: "w"
  of asRMEM: "l"

proc determineIdent(arg: GpuAst): GpuAst =
  ## Tries to determine the underlying ident that is contained in this node.
  ## The issue is the argument to a `gpuCall` can be a complicated expression.
  ## Depending on the node it may be possible to extract a simple identifier,
  ## e.g. for `addr(foo)` (`gpuAddr` of `gpuIdent` node) we can get the ident.
  ## If this fails, we return a `gpuDiscard` node.
  ##
  ## TODO: Think about if it ever makes sense to extract the ident underlying
  ## e.g. `deref` and use _that_ to determine mutability & address space.
  template dfl(): untyped = GpuAst(kind: gpuDiscard)
  case arg.kind
  of gpuIdent: arg
  of gpuAddr: arg.aOf.determineIdent()
  of gpuDeref: arg.dOf.determineIdent()
  of gpuCall: dfl()
  of gpuIndex: arg.iArr.determineIdent()
  of gpuDot: arg.dParent.determineIdent()
  of gpuLit: dfl()
  of gpuBinOp: dfl()
  of gpuBlock: arg.statements[^1].determineIdent()
  of gpuPrefix: dfl()
  of gpuConv: dfl()
  of gpuCast: arg.cExpr.determineIdent() # ident of the thing we cast
  of gpuObjConstr: dfl()  # constructor expressions have no single ident
  of gpuMaterialize: arg.mExpr.determineIdent()
  else:
    raiseAssert "Not implemented to determine ident from node: " & $arg

proc genWebGpu*(ctx: var GpuContext, ast: GpuAst, indent = 0): string

proc preprocess*(ctx: var GpuContext, ast: GpuAst, kernel: string = "") =
  ## If `kernel` is a global function, we *only* generate code for that kernel.
  ## This is useful if your GPU code contains multiple kernels with differing
  ## parameters to avoid having to fill dummy buffers for all the unused parameters
  ## or to work around conflicting paremeters.
  # 1. Fill table with all *global* functions or *only* the specific `kernel`
  #    if any given
  var varBlock = GpuAst(kind: gpuBlock)
  ctx.farmTopLevel(ast, kernel, varBlock)
  ctx.globalBlocks.add varBlock

  # Now add the generics to the `allFnTab`
  for k, v in pairs(ctx.genericInsts):
    ctx.allFnTab[k] = v
  # And all the known types
  var typBlock = GpuAst(kind: gpuBlock)
  for k, typ in pairs(ctx.types):
    typBlock.statements.add typ
  ctx.globalBlocks.add typBlock

  # Delegate to passes_preprocessing passes

  # 2. Remove all arguments from global functions, as none are allowed in WGSL
  for (fnIdent, fn) in mpairs(ctx.fnTab):
    if (fn.isGlobal() and kernel.len > 0 and fn.pName.ident() == kernel) or
        (kernel.len == 0 and fn.isGlobal()):
      for p in fn.pParams:
        ctx.globals[p.ident.symbol.iSym] = p
      fn.pParams.setLen(0)
      pp.updateSymsInGlobalsImpl(ctx, fn)
    else:
      discard

  # 2.b filter out all `var foo {.const_mem.}: dtype`
  if ctx.globalBlocks.len > 0:
    pp.pullConstantPragmaVarsImpl(ctx, ctx.globalBlocks[0])
  # 2.c remove all fields of structs, which have pointer type
  if ctx.globalBlocks.len > 1:
    pp.removeStructPointerFieldsImpl(ctx.globalBlocks[1])

  # 3. Collect value address spaces before scanGenerics derives generic
  # argument spaces from the authoritative map.
  pp.collectValueAddressSpaces(ctx)

  # 3.b Scan generics
  let fns = toSeq(ctx.fnTab.pairs)
  for (fnIdent, fn) in fns:
    let fnOrig = ctx.allFnTab[fnIdent]
    var callParams = initTable[string, GpuParam]()
    for p in fnOrig.pParams:
      callParams[p.ident.symbol.iSym] = p
    pp.scanGenericsImpl(ctx, fn, callParams)

  # 4. injectAddressOf for globals
  for (fnIdent, fn) in mpairs(ctx.fnTab):
    if fn.isGlobal():
      pp.injectAddressOfImpl(ctx, fn)

  # 5. makeCodeValid
  for (fnIdent, fn) in mpairs(ctx.fnTab):
    pp.makeCodeValidImpl(ctx, fn, inGlobal = fn.isGlobal())

  # 6. checkCodeValid
  for (fnIdent, fn) in pairs(ctx.fnTab):
    pp.checkCodeValidImpl(ctx, fn)

proc size(ctx: var GpuContext, a: GpuAst): string = size(ctx.genWebGpu(a))
proc address(ctx: var GpuContext, a: GpuAst): string = address(ctx.genWebGpu(a))

proc genLit*(ast: GpuAst): string =
  ## Lower a literal node for the WGSL backend.
  ## Bare literals (no suffix) are abstract in WGSL and work in any typed context.
  ## Type constructors like `u32(x)` or suffixes like `lf` are used for non-default types.
  if ast.lType.kind == gtString:
    result = '"' & ast.lValue & '"'
  elif ast.lValue == "DEFAULT":
    result = ""
  else:
    case ast.lType.kind
    of gtUint32: result = ast.lValue & "u"
    of gtFloat64: result = ast.lValue & "lf"
    of gtInt16, gtUint16, gtUint8, gtBool:
      result = gpuTypeToString(ast.lType, allowEmptyIdent = true) & '(' & ast.lValue & ')'
    else:
      result = ast.lValue

type WgslBuiltinParam = tuple[canonical, builtin, param, wgslType: string]
  ## Canonical coordinate builtin -> WGSL `@builtin(...)` kernel param.
  ## - `canonical`: the MSL-vocabulary coordinate name.
  ## - `builtin`: the WGSL builtin attribute.
  ## - `param`: the name the kernel body binds it to.
  ## - `wgslType`: the WGSL type (scalar for the flat index).

const WgslBuiltinParams: array[5, WgslBuiltinParam] = [
  ("thread_position_in_grid",        "global_invocation_id",  "global_id",             "vec3<u32>"),
  ("threadgroup_position_in_grid",   "workgroup_id",          "workgroup_id",          "vec3<u32>"),
  ("thread_position_in_threadgroup", "local_invocation_id",   "local_invocation_id",   "vec3<u32>"),
  ("threadgroups_per_grid",          "num_workgroups",        "num_workgroups",        "vec3<u32>"),
  ("thread_index_in_threadgroup",    "local_invocation_index","local_invocation_index","u32"),
]
  ## The five real WGSL builtins. `threads_per_threadgroup` is deliberately
  ## absent: WGSL has no `workgroup_size` builtin
  ## (only the `@workgroup_size` attribute exists), so it has no WGSL
  ## emission and is rejected at the ident site.

proc wgslBuiltinParam(canonical: string): WgslBuiltinParam =
  ## The WGSL param for a canonical coordinate builtin, or an empty tuple when
  ## the canonical has no WGSL builtin.
  for p in WgslBuiltinParams:
    if p.canonical == canonical:
      return p

proc wgslBuiltinParamName(canonical: string): string =
  ## The kernel-body spelling of a canonical coordinate builtin: the injected
  ## `@builtin` param name. Unknown names pass through unchanged.
  let p = wgslBuiltinParam(canonical)
  if p.canonical.len > 0: p.param else: canonical

proc collectWgslBuiltins(n: GpuAst, builtins: var seq[string]) =
  ## Records the coordinate builtins the body references, in first-use order, deduped (the Metal `collectAttrIdents` pattern).
  ## Only builtins with a real WGSL `@builtin` param are recorded. The `gpuCall` name is excluded from the walk.
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.coordBuiltin != gbkNone:
      let name = n.ident()
      if wgslBuiltinParam(name).canonical.len > 0 and name notin builtins:
        builtins.add name
  else:
    for ch in n:
      collectWgslBuiltins(ch, builtins)

proc genWebGpu*(ctx: var GpuContext, ast: GpuAst, indent = 0): string =
  #echo "AST: ", $ast
  let indentStr = "  ".repeat(indent)
  case ast.kind
  of gpuDiscard: return # nothing to emit
  of gpuProc:
    let attrs = collect:
      for att in ast.pAttributes:
        $att

    var params: seq[string]
    for p in ast.pParams:
      params.add gpuTypeToString(p.typ, p.ident.ident(), allowEmptyIdent = false, addrSpace = p.addressSpace)
    var fnArgs = params.join(", ")
    if $attGlobal in attrs:
      doAssert fnArgs.len == 0, "Global function `" & $ast.pName.ident() & "` still has arguments!"
      # Inject one `@builtin(...)` param per canonical coordinate the kernel
      # body references, in first-use order. Only real WGSL builtins exist:
      # there is no `workgroup_size` builtin, so `threads_per_threadgroup`
      # is never injected and is rejected at the ident site.
      var builtins: seq[string]
      collectWgslBuiltins(ast.pBody, builtins)
      var builtinArgs: seq[string]
      for name in builtins:
        let p = wgslBuiltinParam(name)
        builtinArgs.add "@builtin(" & p.builtin & ") " & p.param & ": " & p.wgslType
      fnArgs = builtinArgs.join(", ")
    let fnSig = genFunctionType(ast.pRetType, ast.pName.ident(), fnArgs)

    result = indentStr & "fn " & fnSig & " {\n"

    result &= ctx.genWebGpu(ast.pBody, indent + 1)
    result &= '\n' & indentStr & '}'

  of gpuBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "{ // " & ast.blockLabel & '\n'
    for i, el in ast.statements:
      result.add ctx.genWebGpu(el, indent)
      if not el.isSelfTerminating() and not ctx.skipSemicolon: # nested blocks and emits carry their own terminators
        result.add ';'
      if i < ast.statements.high:
        result.add '\n'
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "} // " & ast.blockLabel & '\n'

  of gpuVar:
    let letOrVar = if ast.vMutable: "var" else: "let"
    var addrSpaceAttr = ""
    case ast.addressSpace
    of asSMEM: addrSpaceAttr = "<workgroup>"
    of asRMEM, asDevice, asConstant: discard
    let vSpace = ctx.varAddressSpaces.getOrDefault(ast.vName.symbol.iSym, asRMEM)
    result = &"{indentStr}{letOrVar}{addrSpaceAttr} {gpuTypeToString(ast.vType, ast.vName.ident(), addrSpace = vSpace)}"
    # If there is an initialization, the type might require a memcpy
    doAssert not ast.vInit.isNil, "Variable initialization is nil. Should not happen."
    if ast.vInit.kind != gpuDiscard and not ast.vRequiresMemcpy:
      result &= " = " & ctx.genWebGpu(ast.vInit)
    elif ast.vInit.kind != gpuDiscard:
      when nimvm:
        error("Types that require memcpy not supported on WGSL. Probably a better solution.")
      else:
        raise newException(ValueError, "Types that require memcpy not supported on WGSL. Probably a better solution.")
      when false:
        result.add ";\n"
        result.add indentStr & genMemcpy(address(ast.vName.ident()), ctx.address(ast.vInit),
                                         size(ast.vName.ident()))

  of gpuAssign:
    if ast.aRequiresMemcpy:
      when nimvm:
        error("Types that require memcpy not supported on WGSL. Probably a better solution.")
      else:
        raise newException(ValueError, "Types that require memcpy not supported on WGSL. Probably a better solution.")
      when false:
        result = indentStr & genMemcpy(ctx.address(ast.aLeft), ctx.address(ast.aRight),
                                       ctx.size(ast.aLeft))
    else:
      let leftId = ast.aLeft.determineIdent()
      if leftId.kind != gpuDiscard and leftId.symbol.typ.kind == gtPtr and leftId.symbol.typ.to.kind == gtInt32:
        # If the LHS is `i32` then a conversion to `i32` is either a no-op, if the left always was
        # `i32` (and the Nim compiler type checked it for us) *OR* the RHS is a boolean expression and
        # we patched the `bool -> i32` and thus need to convert it.
        result = indentStr & ctx.genWebGpu(ast.aLeft) & " = i32(" & ctx.genWebGpu(ast.aRight) & ')'
      else:
        result = indentStr & ctx.genWebGpu(ast.aLeft) & " = " & ctx.genWebGpu(ast.aRight)

  of gpuIf:
    # skip semicolon in the condition. Otherwise can lead to problematic code
    ctx.withoutSemicolon: # skip semicolon for if bodies
      ## Compile time `bool` is turned into int literals 0 and 1 in typed AST
      if ast.ifCond.kind == gpuLit and ast.ifCond.lType.kind == gtInt32 and ast.ifCond.lValue == "1":
        result = indentStr & "if (true) {\n"
      elif ast.ifCond.kind == gpuLit and ast.ifCond.lType.kind == gtInt32 and ast.ifCond.lValue == "0":
        result = indentStr & "if (false) {\n"
      else:
        result = indentStr & "if (" & ctx.genWebGpu(ast.ifCond) & ") {\n"
    result &= ctx.genWebGpu(ast.ifThen, indent + 1) & '\n'
    result &= indentStr & '}'
    if ast.ifElse.kind != gpuDiscard:
      result &= " else {\n"
      result &= ctx.genWebGpu(ast.ifElse, indent + 1) & '\n'
      result &= indentStr & '}'

  of gpuTernary:
    # WGSL has no ternary ?: operator, but supports select(f, t, cond).
    # select returns t when cond is true, f when cond is false.
    result = "select(" & ctx.genWebGpu(ast.tElse) & ", " &
        ctx.genWebGpu(ast.tThen) & ", " &
        ctx.genWebGpu(ast.tCond) & ")"

  of gpuFor:
    let i = ast.fVar.ident()
    let cmp = if ast.fRangeKind == rkInclusive: " <= " else: " < "
    result = indentStr & "for(var " & i & ": i32 = " &
             ctx.genWebGpu(ast.fStart) & "; " &
             i & cmp & ctx.genWebGpu(ast.fEnd) & "; " &
             i & " = " & i & " + 1) {\n"
    result &= ctx.genWebGpu(ast.fBody, indent + 1) & '\n'
    result &= indentStr & '}'
  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genWebGpu(ast.wCond) & "){\n"
    result &= ctx.genWebGpu(ast.wBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuDot:
    result = ctx.genWebGpu(ast.dParent) & '.' & ctx.genWebGpu(ast.dField)

  of gpuIndex:
    result = ctx.genWebGpu(ast.iArr) & '[' & ctx.genWebGpu(ast.iIndex) & ']'

  of gpuCall:
    case ast.cName.symbol.synchroBuiltin
    of gbkThreadgroupBarrier:
      # Every backend spelling of the barrier is an alias template that sem
      # expands to the canonical call, so only this kind reaches the IR.
      # WGSL spells it `workgroupBarrier()`.
      result = indentStr & "workgroupBarrier()"
    of gbkNone:
      ctx.withoutSemicolon:
        result = indentStr & ctx.getFnName(bkWGSL, ast) & '(' &
                 ast.cArgs.mapIt(ctx.genWebGpu(it)).join(", ") & ')'

  of gpuTemplateCall:
    when nimvm:
      error("Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `webgpu` macro.")

    else:
      raise newException(ValueError, "Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `webgpu` macro.")


  of gpuBinOp:
    ctx.withoutSemicolon:
      let l = ctx.genWebGpu(ast.bLeft)
      let r = ctx.genWebGpu(ast.bRight)
      result = indentStr & '(' & l & ' ' &
               ctx.genWebGpu(ast.bOp) & ' ' &
               r & ')'

  of gpuIdent:
    if ast.symbol != nil:
      case ast.symbol.coordBuiltin
      of gbkNone:
        # A local shadowing a canonical name, or a call-shaped builtin
        # (printf, cvtaGenericToShared): emit verbatim.
        result = ast.ident()
      of gbkThreadsPerThreadgroup:
        raiseAssert "`threads_per_threadgroup` has no WGSL builtin: WGSL has " &
          "only the `@workgroup_size` attribute, never a `workgroup_size` " &
          "builtin, so the threadgroup size is not addressable from WGSL."
      of gbkThreadPositionInGrid, gbkThreadgroupPositionInGrid,
         gbkThreadPositionInThreadgroup, gbkThreadgroupsPerGrid,
         gbkThreadIndexInThreadgroup:
        # Canonical coordinates map to the injected `@builtin` param names
        # (global_id, workgroup_id, local_invocation_id, num_workgroups,
        # local_invocation_index).
        result = wgslBuiltinParamName(ast.ident())
    else:
      result = ast.ident()

  of gpuLit:
      result = genLit(ast)
  of gpuArrayLit:
    result = "array("
    for i, el in ast.aValues:
      result.add gpuTypeToString(ast.aLitType) & '(' & ctx.genWebGpu(el) & ')'
      if i < ast.aValues.high:
        result.add ", "
    result.add ')'

  of gpuReturn:
    result = indentStr & "return " & ctx.genWebGpu(ast.rValue)

  of gpuPrefix:
    result = ast.pOp & ctx.genWebGpu(ast.pVal)

  of gpuTypeDef:
    result = "struct " & gpuTypeToString(ast.tTyp) & " {\n"
    if ast.tFields.len == 0:
      # WGSL requires at least one member in a struct.
      result.add "  _padding: u32,\n"
    else:
      for el in ast.tFields:
        result.add "  " & gpuTypeToString(el.typ, el.name) & ",\n"
    result.add '}'

  of gpuAlias:
    # Aliases come from `ctx.types` and due to implementation details currently are _not_ wrapped
    # in a `block` (as they are handled like regular `structs`). However, WebGPU requires semicolons
    # after alias definitions, but not after `struct`. Hence we add `;` manually here
    result = "alias " & gpuTypeToString(ast.aTyp) & " = " & ctx.genWebGpu(ast.aTo) & ';'

  of gpuObjConstr:
    result = gpuTypeToString(ast.ocType) & '('
    for i, el in ast.ocFields:
      if el.value.kind == gpuLit and el.value.lValue == "DEFAULT":
        # use type to construct a default value
        let typStr = gpuTypeToString(el.typ, allowEmptyIdent = true)
        result.add typStr & "()"
      else:
        result.add ctx.genWebGpu(el.value)
      if i < ast.ocFields.len - 1:
        result.add ", "
    result.add ')'

  of gpuInlineAsm:
    raiseAssert "Inline assembly not supported on the WebGPU target."

  of gpuEmit:
    # Self-terminating raw text: the gpuBlock loop appends no `;`
    # (the emitted text owns its own terminators).
    result = genEmitStmt(ctx, ast,
      proc(c: var GpuContext; n: GpuAst): string = c.genWebGpu(n, 0))

  of gpuComment:
    result = indentStr & "/* " & ast.comment & " */"

  of gpuConv:
    result = gpuTypeToString(ast.convTo, allowEmptyIdent = true) & '(' & ctx.genWebGpu(ast.convExpr) & ')'

  of gpuCast:
    result = "bitcast<" & gpuTypeToString(ast.cTo, allowEmptyIdent = true) & ">(" & ctx.genWebGpu(ast.cExpr) & ')'

  of gpuAddr:
    result = "(&" & ctx.genWebGpu(ast.aOf) & ')'

  of gpuDeref:
    result = "(*" & ctx.genWebGpu(ast.dOf) & ')'

  of gpuConstexpr:
    result = indentStr & "const " & ctx.genWebGpu(ast.cIdent) & ": " & gpuTypeToString(ast.cType, allowEmptyIdent = true) & " = " & ctx.genWebGpu(ast.cValue)
  of gpuMaterialize:
    raiseAssert "gpuMaterialize should not reach WGSL backend — passByRef is not used"

proc codegen*(ctx: var GpuContext): string =
  ## Generate the actual code for all pieces of the puzzle
  ##
  ## NOTE: WGSL does not require forward declarations / does not care about
  ## the order in which functions are defined

  var bindingCounter = 0
  proc mutateToAllowedTypes(p: GpuType): GpuType =
    ## We strip pointer types `ptr T` to only emit `T`. This is because all global parameters
    ## must be global storage buffers. These cannot be of `ptr` type. If an implicit, runtime
    ## sized array is desired, use `ptr UncheckedArray[T]`, which will emit `array<T>`.
    ##
    ## If we have a `bool` type, we need to convert it to a `i32` (also applies to `ptr bool`)
    case p.kind
    of gtPtr: mutateToAllowedTypes(p.to) # if it is `ptr bool`
    of gtBool: ## boolean must become `i32`. Will inject `bool(foo)` into globals
      GpuType(kind: gtInt32)
    else: p
  proc genGlobal(p: GpuParam): string =
    ## XXX: deduce read or read_write based on argument type!
    let rw = if p.typ.kind == gtPtr: "read_write" else: "read"
    result = &"@group(0) @binding({bindingCounter}) var<storage, " & rw & "> "
    let typ = mutateToAllowedTypes(p.typ)
    result.add gpuTypeToString(typ, p.ident.ident(), allowEmptyIdent = false, addrSpace = p.addressSpace) & ";\n"
    inc bindingCounter

  # 1. Generate the header for all global variables (deduped by param name)
  var emitted: seq[string]
  for id, g in ctx.globals:
    if g.ident.ident() notin emitted:
      emitted.add g.ident.ident()
      result.add genGlobal(g)
  result.add '\n'

  # 2. generate code for the global blocks (types, global vars etc)
  for blk in ctx.globalBlocks:
    result.add ctx.genWebGpu(blk) & "\n\n"

  # 3. Kernel + device functions.
  # Workgroup size from the `{.workgroup: (X, Y, Z).}` annotation.
  # Default 64×1×1.
  for fnIdent, fn in ctx.fnTab:
    if fn.isGlobal():
      let wg = fn.pWorkgroupSize
      let wx = if wg.x > 0: wg.x else: 64
      let wy = if wg.y > 0: wg.y else: 1
      let wz = if wg.z > 0: wg.z else: 1
      result.add "@compute @workgroup_size(" & $wx & ", " & $wy & ", " & $wz & ")\n"
    result.add ctx.genWebGpu(fn) & "\n\n"
