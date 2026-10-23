# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, strformat, strutils, sugar, sequtils, sets, tables, options]

import ../ir/ast_datatypes
import ../ir/ast_type_resolvers

import ./lang_utils


proc gpuTypeToString*(t: GpuType,
                      ident: string = "",
                      allowArrayToPtr = false,
                      allowEmptyIdent = false,
                      addrSpace: AddressSpace = asRMEM): string



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
  of gtFloat16: "f16"
  of gtVoid: ""
  of gtSize_t: "u32" ##: Acceptable mapping?
  of gtPtr: "ptr" ## XXX: needs address space and target type, `ptr<address_space, target_type>`
  of gtUA: "array"
  of gtObject: "struct"
  of gtUint8, gtUint16, gtUint64, gtInt16, gtInt64, gtFloat64, gtVoidPtr, gtString:
    error "The type " & $t & " does not exist on the WebGPU target."
  of gtStatic: "i32"
  else:
    error "Invalid type : " & $t

proc gpuTypeToString*(t: GpuType, ident: string = "", allowArrayToPtr = false,
                           allowEmptyIdent = false,
                           addrSpace: AddressSpace = asRMEM
                    ): string =
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
    if t.gArgs.len == 1:
      result.add gpuTypeToShortString(t.gArgs[0])
    elif t.gArgs.len > 1:
      # 2+ args collapse to a short hash
      result.add mangledHashName("", $hash(t))
  of gtObject: result = t.name
  of gtUA:     result = gpuTypeToString(t.kind) & '<' & gpuTypeToString(t.uaTo, allowEmptyIdent = allowEmptyIdent) & '>'
  of gtStatic: result = "i32"
  else:        result = gpuTypeToString(t.kind)

  if ident.len > 0 and not skipIdent:
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
    result = &"{fn}({fnArgs})"
    let typ = gpuTypeToString(typ, allowEmptyIdent = true)
    if typ.len > 0:
      result.add &" -> {typ}"
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
    error "Not implemented to determine ident from node: " & $arg

proc genWebGpu*(ctx: var GpuContext, ast: GpuAst, indent = 0): string



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
    of gtFloat16: result = ast.lValue & "h"
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
  ## Unknown names pass through unchanged.
  let p = wgslBuiltinParam(canonical)
  if p.canonical.len > 0: p.param else: canonical

proc collectWgslBuiltins(n: GpuAst, builtins: var seq[string]) =
  ## Records the coordinate builtins the body references:
  ## - first-use order, deduped
  ## - only builtins with a real WGSL `@builtin` param
  ## - the `gpuCall` name is excluded from the walk
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
  let indentStr = "  ".repeat(indent)
  case ast.kind
  of gpuDiscard: return
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
      when defined(debugPasses):
        echo "[print] stmt kind=", el.kind, " selfTerm=", el.isSelfTerminating(), " skipSem=", ctx.skipSemicolon
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
    # the pointer type carries the space it accesses
    let vSpace = ast.vType.addressSpace
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

  of gpuAssign:
    if ast.aRequiresMemcpy:
      when nimvm:
        error("Types that require memcpy not supported on WGSL. Probably a better solution.")
      else:
        raise newException(ValueError, "Types that require memcpy not supported on WGSL. Probably a better solution.")
    else:
      result = indentStr & ctx.genWebGpu(ast.aLeft) & ' ' & ast.aOp & ' ' &
               ctx.genWebGpu(ast.aRight)

  of gpuIf:
    template genCond(c: GpuAst): untyped =
      # Compile time `bool` is turned into int literals 0 and 1 in typed AST
      (if c.kind == gpuLit and c.lType.kind == gtInt32 and c.lValue == "1": "true"
       elif c.kind == gpuLit and c.lType.kind == gtInt32 and c.lValue == "0": "false"
       else: ctx.genWebGpu(c))
    # skip semicolon in the condition. Otherwise can lead to problematic code
    ctx.withoutSemicolon:
      result = indentStr & "if (" & genCond(ast.ifCond) & ") {\n"
    result &= ctx.genWebGpu(ast.ifThen, indent + 1) & '\n'
    result &= indentStr & '}'
    for el in ast.ifElifs:
      ctx.withoutSemicolon:
        result &= " else if (" & genCond(el.cond) & ") {\n"
      result &= ctx.genWebGpu(el.body, indent + 1) & '\n'
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
             i & " = " & i & " + " & (if ast.fStep == nil: "1" else: ctx.genWebGpu(ast.fStep)) & ") {\n"
    result &= ctx.genWebGpu(ast.fBody, indent + 1) & '\n'
    result &= indentStr & '}'
  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genWebGpu(ast.wCond) & "){\n"
    result &= ctx.genWebGpu(ast.wBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuDot:
    if ast.dParent.kind == gpuIdent and ast.dParent.symbol != nil and
       ast.dParent.symbol.coordBuiltin != gbkNone and ast.dField.kind == gpuIdent:
      # A component use takes the bare builtin param, the whole-value
      # construction may name an ivec3 alias this shader never declared
      result = "i32(" & wgslBuiltinParamName(ast.dParent.ident()) & '.' &
               ctx.genWebGpu(ast.dField) & ')'
    else:
      let fieldName = if ast.dField.kind == gpuIdent: ast.dField.ident()
                      else: ctx.genWebGpu(ast.dField)
      result = ctx.genWebGpu(ast.dParent) & '.' & fieldName

  of gpuIndex:
    result = ctx.genWebGpu(ast.iArr) & '[' & ctx.genWebGpu(ast.iIndex) & ']'

  of gpuCall:
    case ast.cName.symbol.synchroBuiltin
    of gbkThreadgroupBarrier:
      result = indentStr & "workgroupBarrier()"
    of gbkThreadgroupBarrierDevice:
      # WGSL has no combined barrier: workgroupBarrier(), then storageBarrier()
      result = indentStr & "workgroupBarrier();" & "\n" & indentStr &
               "storageBarrier()"
    of gbkNone:
      case ast.cName.symbol.reductionBuiltin
      of gbkReduceAdd:
        ctx.withoutSemicolon:
          result = indentStr & "subgroupAdd(" & ctx.genWebGpu(ast.cArgs[0]) & ')'
      of gbkReduceMax:
        ctx.withoutSemicolon:
          result = indentStr & "subgroupMax(" & ctx.genWebGpu(ast.cArgs[0]) & ')'
      of gbkNone:
        ctx.withoutSemicolon:
          result = indentStr & ctx.getFnName(bkWebGPU, ast) & '(' &
                   ast.cArgs.mapIt(ctx.genWebGpu(it)).join(", ") & ')'



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
        error "`threads_per_threadgroup` has no WGSL builtin: WGSL has " &
          "only the `@workgroup_size` attribute, never a `workgroup_size` " &
          "builtin, so the threadgroup size is not addressable from WGSL."
      of gbkThreadPositionInGrid, gbkThreadgroupPositionInGrid,
         gbkThreadgroupsPerGrid, gbkThreadPositionInThreadgroup:
        # u32 @builtin coords, whole-value use casts to the canonical ivec3
        let n = wgslBuiltinParamName(ast.ident())
        result = "ivec3(i32(" & n & ".x), i32(" & n & ".y), i32(" & n & ".z))"
      of gbkThreadIndexInThreadgroup:
        # `local_invocation_index` is u32 in WGSL, cast to the int domain.
        result = "i32(" & wgslBuiltinParamName(ast.ident()) & ')'
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
    # Aliases emit like structs, WGSL wants `;` after a type alias,
    # not after a struct
    result = "alias " & gpuTypeToString(ast.aTyp) & " = " & ctx.genWebGpu(ast.aTo) & ';'

  of gpuObjConstr:
    result = gpuTypeToString(ast.ocType) & '('
    for i, el in ast.ocFields:
      if el.value.kind == gpuLit and el.value.lValue == "DEFAULT":
        let typStr = gpuTypeToString(el.typ, allowEmptyIdent = true)
        result.add typStr & "()"
      else:
        result.add ctx.genWebGpu(el.value)
      if i < ast.ocFields.len - 1:
        result.add ", "
    result.add ')'

  of gpuInlineAsm:
    error "Inline assembly not supported on the WebGPU target."

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
    error "gpuMaterialize should not reach WGSL backend — passByRef is not used"

proc codegen*(ctx: var GpuContext): string =
  # NOTE: WGSL does not require forward declarations / does not care about
  # the order in which functions are defined

  var needsF16 = false
  for fnIdent, fn in ctx.fnTab:
    if fn.pRetType.containsKind(gtFloat16):
      needsF16 = true
    for p in fn.pParams:
      if p.typ.containsKind(gtFloat16):
        needsF16 = true
  for id, g in ctx.globals:
    if g.typ.containsKind(gtFloat16):
      needsF16 = true
  if needsF16:
    result.add "enable f16;\n\n"
  # No `enable subgroups` emission: native wgpu runs the subgroup builtins
  # without the directive when the device requests the subgroups feature
  # (gfx-rs/wgpu#8202, #5555). The directive is not implemented in Naga
  # and would fail shader-module creation.

  var bindingCounter = 0
  proc genGlobal(p: GpuParam): string =
    let rw = if p.flavor == kVar: "read_write" else: "read"
    result = &"@group(0) @binding({bindingCounter}) var<storage, " & rw & "> "
    result.add gpuTypeToString(p.typ, p.ident.ident(), allowEmptyIdent = false, addrSpace = p.addressSpace) & ";\n"
    inc bindingCounter

  # deduped by param name
  var emitted: seq[string]
  for id, g in ctx.globals:
    if g.ident.ident() notin emitted:
      emitted.add g.ident.ident()
      result.add genGlobal(g)
  result.add '\n'

  for blk in ctx.globalBlocks:
    result.add ctx.genWebGpu(blk) & "\n\n"

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
