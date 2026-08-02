# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
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

proc gpuTypeToString*(t: GpuTypeKind): string =
  case t
  of gtBool: "bool"
  of gtUint8: "unsigned char"
  of gtUint16: "unsigned short"
  of gtUint32: "unsigned int"
  of gtUint64: "unsigned long long"
  of gtInt16: "short"
  of gtInt32: "int"
  of gtInt64: "long long"
  of gtFloat32: "float"
  of gtFloat64: "double"
  of gtVoid: "void"
  of gtSize_t: "size_t"
  of gtPtr: "*"
  of gtVoidPtr: "void*"
  of gtObject: "struct"
  of gtString: "const char*"
  of gtUA: "" # `UncheckedArray` by itself is nothing in CUDA
  of gtStatic: "int"
  else:
    raiseAssert "Invalid type : " & $t

proc gpuTypeToString*(t: GpuType, ident: string = "", allowArrayToPtr = false,
                      allowEmptyIdent = false,
                    ): string =
  ## Given an optional identifier required for array types
  ##
  ## XXX: we don't support this at the moment, it occured to me as something that
  ## could be useful sometimes...
  ## If `allowArrayToPtr` we allow casting a statically sized array to a pointer
  var skipIdent = false
  case t.kind
  of gtPtr:
    var t = t # if `ptr UncheckedArray`, remove the `gtUA` layer. No meaning on CUDA
    if t.to.kind == gtUA:
      t.to = t.to.uaTo

    if t.to.kind == gtArray: # ptr to array type
      # need to pass `*` for the pointer into the identifier, i.e.
      # `state: var array[4, BigInt]`
      # must become
      # `BigInt (*state)[4]`
      # so as our ident we pass `theIdent = (*<ident>)` and generate the type for the internal
      # array type, which yields e.g. `BigInt <theIdent>[4]`.
      let ptrStar = gpuTypeToString(t.kind)
      result = gpuTypeToString(t.to, '(' & ptrStar & ident & ')')
      skipIdent = true
    else:
      let typ = gpuTypeToString(t.to, allowEmptyIdent = allowEmptyIdent)
      let ptrStar = gpuTypeToString(t.kind)
      result = typ & ptrStar
  of gtArray:
    # empty idents happen in e.g. function return types or casts
    if ident.len == 0 and not allowEmptyIdent: # and not allowArrayToPtr:
      when nimvm:
        error("Invalid call, got an array type but don't have an identifier: " & $t)
      else:
        raise newException(ValueError, "Invalid call, got an array type but don't have an identifier: " & $t)
    case t.aTyp.kind
    of gtArray: # nested array
      let typ = getInnerArrayType(t)        # get inner most type
      let lengths = getInnerArrayLengths(t) # get lengths as `[X][Y][Z]...`
      result = typ & ' ' & ident & lengths
    else:
      # NOTE: Nested arrays don't have an inner identifier!
      if t.aLen == 0: ## XXX: for the moment for 0 length arrays we generate flexible arrays instead
        result = gpuTypeToString(t.aTyp, allowEmptyIdent = allowEmptyIdent) & ' ' & ident & "[]"
      else:
        result = gpuTypeToString(t.aTyp, allowEmptyIdent = allowEmptyIdent) & ' ' & ident & '[' & $t.aLen & ']'
    skipIdent = true
  of gtGenericInst:
    # NOTE: WGSL does not support actual custom generic types. And as we only anyway deal with generic instantiations
    # we simply turn e.g. `foo[float32, uint32]` into `foo_f32_u32`.
    # use short names (uint32, int64) for generic args, not C names (unsigned int, long long)
    result = t.gName
    for i, g in t.gArgs:
      result.add gpuTypeToShortString(g)
      if i < t.gArgs.high:
        result.add 'x'
  of gtObject: result = t.name
  of gtUA:     result = gpuTypeToString(t.uaTo, allowEmptyIdent = allowEmptyIdent) ## XXX: unchecked array just T?
  of gtStatic: result = "int"
  else:        result = gpuTypeToString(t.kind)

  if ident.len > 0 and not skipIdent: # still need to add ident
    result.add ' ' & ident

proc genFunctionType*(typ: GpuType, fn: string, fnArgs: string): string =
  ## Returns the correct function with its return type. Kept for backward compat
  ## during Phase 5; will be removed when codegen reads sigString from FnTableEntry.
  if typ.kind == gtPtr and typ.to.kind == gtArray:
    # crazy stuff. Syntax to return a pointer to a statically sized array:
    # `Foo (*fnName(fnArgs))[ArrayLen]`
    # where the return type is actually:
    # `Foo (*)[ArrayLen]` (which already is hideous)
    let arrayTyp = typ.to.aTyp
    let innerTyp = gpuTypeToString(arrayTyp, allowEmptyIdent = true)
    let innerLen = $typ.to.aLen
    result = &"{innerTyp} (*{fn}({fnArgs}))[{innerLen}]"
  else:
    # normal stuff
    result = &"{gpuTypeToString(typ, allowEmptyIdent = true)} {fn}({fnArgs})"
proc scanFunctions(ctx: var GpuContext, n: GpuAst) =
  ## Iterates over the given function and checks for all `gpuCall` nodes. Any function
  ## called in the scope is added to `fnTab`. This is a form of dead code elimination.
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


proc genCuda*(ctx: var GpuContext, ast: GpuAst, indent = 0): string
proc size(ctx: var GpuContext, a: GpuAst): string = size(ctx.genCuda(a))
proc address(ctx: var GpuContext, a: GpuAst): string = address(ctx.genCuda(a))

proc preprocess*(ctx: var GpuContext, ast: GpuAst, kernel: string = "") =

  # 1. Add all data from `genericInsts` and `types` tables
  #    In CUDA the types have to be before any possible global variables using
  #    them!
  for k, v in pairs(ctx.genericInsts):
    ctx.allFnTab[k] = v
  # And all the known types
  for k, typ in pairs(ctx.types):
    ctx.globalBlocks.add typ

  # 2. Fill table with all *global* functions or *only* the specific `kernel`
  #    if any given
  var varBlock = GpuAst(kind: gpuBlock)
  ctx.farmTopLevel(ast, kernel, varBlock)
  ctx.globalBlocks.add varBlock

  # 3. Using all global functions, we traverse their AST for any `gpuCall` node. We inspect
  #    the functions called and record them in `fnTab`.
  let fns = toSeq(ctx.fnTab.pairs)
  for (fnIdent, fn) in fns: # everything in `fnTab` at this point is a global function
    # Get the original arguments (before lifting them) of this function. Needed in scan
    # to check if `gpuCall` argument is a parameter.
    let fnOrig = ctx.allFnTab[fnIdent]
    ctx.scanFunctions(fn)


proc genLit*(ast: GpuAst): string =
  ## Lower a literal node for the CUDA backend.
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
    else:
      result = ast.lValue

proc genCuda*(ctx: var GpuContext, ast: GpuAst, indent = 0): string =
  ## The actual CUDA code generator.
  let indentStr = "  ".repeat(indent)
  case ast.kind
  of gpuDiscard: return # nothing to emit
  of gpuProc:
    let attrs = collect:
      for att in ast.pAttributes:
        $att

    # Parameters
    var params: seq[string]
    for p in ast.pParams:
      if p.passByRef:
        # const Type& name — C++ reference, no body changes needed
        params.add "const " & gpuTypeToString(p.typ, allowEmptyIdent = true) & "& " & p.ident.ident()
      else:
        params.add gpuTypeToString(p.typ, p.ident.ident(), allowEmptyIdent = false)
    let fnArgs = params.join(", ")
    let fnSig = genFunctionType(ast.pRetType, ast.pName.ident(), fnArgs)

    # extern "C" is needed on __global__ kernels so the host-side CUDA
    # runtime can look them up by unmangled name (e.g. nv.execute("foo", ...)).
    # __device__ functions are only called within the compilation unit and
    # don't need it — C++ name mangling is invisible inside a single
    # translation unit, and omitting it avoids interfering with overloads.
    let linkage = if attGlobal in ast.pAttributes: "extern \"C\" " else: ""
    result = indentStr & linkage & attrs.join(" ") & ' ' & fnSig
    if ast.forwardDeclare:
      result.add ';'
    else:
      result.add "{\n"
      result &= ctx.genCuda(ast.pBody, indent + 1)
      result &= '\n' & indentStr & '}'

  of gpuBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add '\n' & indentStr & "{ // " & ast.blockLabel & '\n'
    for i, el in ast.statements:
      let code = ctx.genCuda(el, indent + (if ast.blockLabel.len > 0: 1 else: 0))
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
    let attrs = if ast.vAttributes.len > 0: ast.vAttributes.join(" ") & ' '
                else: ""
    result = indentStr & attrs & gpuTypeToString(ast.vType, ast.vName.ident())
    if ast.vInit.kind != gpuDiscard:
      result &= " = " & ctx.genCuda(ast.vInit)
  of gpuAssign:
    result = indentStr & ctx.genCuda(ast.aLeft) & " = " & ctx.genCuda(ast.aRight)
  of gpuIf:
    # skip semicolon in the condition. Otherwise can lead to problematic code
    ctx.withoutSemicolon: # skip semicolon for if bodies
      result = indentStr & "if (" & ctx.genCuda(ast.ifCond) & ") {\n"
    result &= ctx.genCuda(ast.ifThen, indent + 1) & '\n'
    result &= indentStr & '}'
    if ast.ifElse.kind != gpuDiscard:
      result &= " else {\n"
      result &= ctx.genCuda(ast.ifElse, indent + 1) & '\n'
      result &= indentStr & '}'

  of gpuTernary:
    ctx.withoutSemicolon:
      result = '(' & ctx.genCuda(ast.tCond) & " ? " &
               ctx.genCuda(ast.tThen) & " : " &
               ctx.genCuda(ast.tElse) & ')'

  of gpuFor:
    let cmp = if ast.fRangeKind == rkInclusive: " <= " else: " < "
    result = indentStr & "for(int " & ast.fVar.ident() & " = " &
             ctx.genCuda(ast.fStart) & "; " &
             ast.fVar.ident() & cmp & ctx.genCuda(ast.fEnd) & "; " &
             ast.fVar.ident() & "++) {\n"
    result &= ctx.genCuda(ast.fBody, indent + 1) & '\n'
    result &= indentStr & '}'
  of gpuWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.genCuda(ast.wCond) & "){\n"
    result &= ctx.genCuda(ast.wBody, indent + 1) & '\n'
    result &= indentStr & '}'

  of gpuDot:
    result = ctx.genCuda(ast.dParent) & '.' & ctx.genCuda(ast.dField)

  of gpuIndex:
    result = ctx.genCuda(ast.iArr) & '[' & ctx.genCuda(ast.iIndex) & ']'

  of gpuCall:
    let fnName = ast.cName.ident()
    var cudaArgs: seq[string]
    for i, arg in ast.cArgs:
      cudaArgs.add ctx.genCuda(arg)
    result = indentStr & fnName & '(' & cudaArgs.join(", ") & ')'
  of gpuTemplateCall:
    when nimvm:
      error("Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `cuda` macro.")
    else:
      raise newException(ValueError, "Template calls are not supported at the moment. In theory there shouldn't even _be_ any template " &
        "calls in the expanded body of the `cuda` macro.")

  of gpuBinOp:
    ctx.withoutSemicolon:
      let l = ctx.genCuda(ast.bLeft)
      let r = ctx.genCuda(ast.bRight)
      result = indentStr & '(' & l & ' ' &
               ctx.genCuda(ast.bOp) & ' ' &
               r & ')'

  of gpuIdent:
    result = ast.ident()

  of gpuLit:
      result = genLit(ast)

  of gpuArrayLit:
    result = "{"
    for i, el in ast.aValues:
      result.add '(' & gpuTypeToString(ast.aLitType) & ')' & ctx.genCuda(el)
      if i < ast.aValues.high:
        result.add ", "
    result.add '}'

  of gpuReturn:
    result = indentStr & "return " & ctx.genCuda(ast.rValue)

  of gpuPrefix:
    result = ast.pOp & ctx.genCuda(ast.pVal)

  of gpuTypeDef:
    result = "struct " & gpuTypeToString(ast.tTyp) & "{\n"
    if ast.tFields.len == 0:
      # CUDA requires at least one field in a struct.
      result.add "  char _;\n"
    else:
      for el in ast.tFields:
        result.add "  " & gpuTypeToString(el.typ, el.name) & ";\n"
    result.add '}'

  of gpuObjConstr:
    # Braced init list: {val1, val2, ...}
    # Note: bare `{val, ...}` is used instead of `(TypeName){val}`
    # because NVRTC compiles in C++ mode where C99 compound literals
    # are not valid.
    # Braced init list: TypeName{val1, val2, ...}
    # Using `TypeName{...}` (functional-style cast) instead of bare `{val}`
    # ensures the result is a valid C++ expression — bare braced-init-lists
    # are not expressions and cannot be used with member access (gpuDot).
    result = gpuTypeToString(ast.ocType, allowEmptyIdent = true) & "{"
    for i, el in ast.ocFields:
      if el.value.kind == gpuDiscard:
        result.add "{}"
      else:
        result.add ctx.genCuda(el.value)
      if i < ast.ocFields.len - 1:
        result.add ", "
    result.add '}'

  of gpuInlineAsm:
    result = indentStr & "asm(" & ast.stmt.strip & ");"

  of gpuComment:
    result = indentStr & "/* " & ast.comment & " */"

  of gpuConv:
    result = '(' & gpuTypeToString(ast.convTo, allowEmptyIdent = true) & ')' & ctx.genCuda(ast.convExpr)
  of gpuCast:
    result = '(' & gpuTypeToString(ast.cTo, allowEmptyIdent = true) & ')' & ctx.genCuda(ast.cExpr)

  of gpuAddr:
    result = "(&" & ctx.genCuda(ast.aOf) & ')'

  of gpuDeref:
    result = "(*" & ctx.genCuda(ast.dOf) & ')'

  of gpuConstexpr:
    ## TODO: We need to change the code such that we emit `constexpr` inside of procs and
    ## `__constant__` outside of procs. The point is we want to support mapping to `__constant__
    ## for `const foo = bar` Nim declarations to evaluate values at Nim's compile time.
    ## Alternatively, make user write `const foo {.constant.} = bar` to produce a global
    ## `__constant__` value.
    let cInit = if ast.cValue.kind == gpuDiscard: "{}" else: ctx.genCuda(ast.cValue)
    if ast.cType.kind == gtArray:
      result = indentStr & "constexpr " & gpuTypeToString(ast.cType, ctx.genCuda(ast.cIdent)) & " = " & cInit
    else:
      result = indentStr & "constexpr " & gpuTypeToString(ast.cType, allowEmptyIdent = true) & ' ' & ctx.genCuda(ast.cIdent) & " = " & cInit
  of gpuMaterialize:
    result = ctx.genCuda(ast.mExpr)  # C++ const& binds implicitly to temporaries

  else:
    echo "Unhandled node kind in genCuda: ", ast.kind
    raiseAssert "Unhandled node kind in genCuda: " & ast.repr

proc codegen*(ctx: var GpuContext): string =
  ## Generate the actual code for all pieces of the puzzle
  # 1. generate code for the global blocks (types, global vars etc)
  for blk in ctx.globalBlocks:
    result.add ctx.genCuda(blk) & ";\n\n"

  # 2. generate all regular functions
  let fns = toSeq(ctx.fnTab.pairs)
  for (fnIdent, fn) in fns:
    let fnC = fn.clone()
    fnC.forwardDeclare = true
    result.add ctx.genCuda(fnC) & '\n'
  result.add "\n\n"

  for fnIdent, fn in ctx.fnTab:
    result.add ctx.genCuda(fn) & "\n\n"
