# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [tables, sets, hashes, strutils, strformat]
import std/macros

type
  BackendKind* = enum
    bkCuda,   ## CUDA backend
    bkWGSL,   ## WebGPU WGSL backend
    bkOpenCL, ## OpenCL backend
    bkVulkan  ## Vulkan (SPIR-V) backend

  GpuNodeKind* = enum
    gpuDiscard         # Just an empty statement. Useful to not emit anything
    gpuProc         # Function definition (both device and global)
    gpuCall         # Function call
    gpuTemplateCall # Call to a Nim template
    gpuIf           # If statement
    gpuTernary      # Ternary expression (cond ? then : else)
    gpuFor          # For loop
    gpuWhile        # While loop
    gpuBinOp        # Binary operation
    gpuVar          # Variable declaration
    gpuAssign       # Assignment
    gpuIdent        # Identifier
    gpuLit          # Literal value
    gpuArrayLit     # Literal array constructor `[1, 2, 3]`
    gpuPrefix       # Prefix e.g. `-`
    gpuBlock        # Block of statements
    gpuReturn       # Return statement
    gpuDot          # Member access (a.b)
    gpuIndex        # Array indexing (a[b])
    gpuTypeDef      # Type definition
    gpuAlias        # A type alias
    gpuObjConstr    # Object (struct) constructor
    gpuInlineAsm    # Inline assembly (PTX)
    gpuAddr         # Address of an expression
    gpuDeref        # Dereferences an expression
    gpuConv         # A type conversion, i.e. `let x = 5; x.float`
    gpuCast         # Cast expression
    gpuComment      # Just a comment
    gpuConstexpr    # A `constexpr`, i.e. compile time constant (Nim `const`)
    gpuMaterialize  # Force materialization of a non-lvalue expr for pass-by-ref

  GpuTypeKind* = enum
    gtVoid,
    gtBool, gtUint8, gtUint16, gtInt16, gtUint32, gtInt32, gtUint64, gtInt64, gtFloat32, gtFloat64, gtSize_t, # atomics
    gtStatic       # Static integer value (used for generic params)
    gtArray,       # Static array `array[N, dtype]` -> `dtype[N]`
    gtString,
    gtObject,      # Struct types
    gtPtr,         # Pointer type, carries inner type
    gtUA,          # UncheckedArray (UA) mapped to runtime sized arrays
    gtSpan,         # Runtime-length array view (openArray/varargs)
    gtGenericInst, # Instantiated generic type with one or more generic arguments (instantiated!)
    gtVoidPtr      # Opaque void pointer
    gtInvalid      # Can be returned to indicate a call to `nimToGpuType` failed to determine a type
                   ## XXX: make this the default value and replace all `gtVoid` placeholders by it


  GpuTypeField* = object
    name*: string
    typ*: GpuType

  GpuType* = ref object
    builtin*: bool ## Whether the type refers to a builtin type or not
    case kind*: GpuTypeKind
    of gtPtr:
      to*: GpuType # `ptr T` points to `to`
      implicit*: bool # Whether the type was implicitly a pointer, i.e. `var T`.
      mutable*: bool # WebGPU "Generics" only: Mutable (read write) or immutable pointer (read only)?
                    # If a function is called with a raw pointer as an argument or `implicit == true / var T` argument,
                    # `mutable` will be true. If on the other hand we have a non pointer type and take its address
                    # via `foo.addr`, mutable will be false.
    of gtUA: uaTo*: GpuType # `UncheckedArray[T]`
    of gtObject:
      name*: string
      oFields*: seq[GpuTypeField]
    of gtArray:
      aTyp*: GpuType # the inner type (must be some atomic base type at the moment)
      aLen*: int     # The length of the array. If `aLen == -1` we look at a generic (static) array. Will be given at instantiation time
                    # On both CUDA and WebGPU a length of `0` is also used to generate `int foo[]` (CUDA)
                    # `array<foo>` (WebGPU) (runtime sized arrays), which are generated from `ptr UncheckedArray[float32]` for example.
    of gtStatic:
      sValue*: int  # The actual static integer value
    of gtSpan:
      sKind*: GpuSpanKind  # kOpenArray or kVarargs
      sElemTyp*: GpuType   # element type T
    of gtGenericInst:
      gName*: string # name of the generic type
      gArgs*: seq[GpuType] # list of the instantiated generic arguments e.g. `vec3<f32>` on WGSL backend
      gFields*: seq[GpuTypeField] # same as `oFields` for `gtObject`
    else: discard

  GpuSpanKind* = enum
    kOpenArray   # Nim openArray[T] — ptr + runtime length
    kVarargs     # Unsupported at th moment

  GpuRangeKind* = enum
    rkInclusive    # Nim `a..b` — emitted as `i <= end` or equivalent
    rkExclusive    # Nim `a..<b` — emitted as `i < end`

  FunctionKind* = enum
    fkGenericInst    # Generic instantiation (body from Nim AST)
    fkExternal       # External/imported function (ident only, body nil)
    fkBuiltin        # Magic builtin (toOpenArray, etc.)
    fkDefined        # Defined in GPU block (body available)
    fkInstantiated   # Generic that has been instantiated

  NamePolicy* = enum
    npUnassigned    # Not yet processed
    npClean         # Emit plain name (C++ mangling handles disambiguation)
    npHashSuffix    # Append base58 short hash
    npPatch         # Operator rename (+ → add, etc.)

  FnTableEntry* = object
    ident*: GpuAst        # call-site identifier (iSym)
    body*: GpuAst          # function definition AST (nil for externals/builtins)
    kind*: set[FunctionKind]
    namePolicy*: NamePolicy  # set by mangleNames pass later
    sigString*: string       # pre-computed function signature (set by emitFunctionSignatures pass)
  GpuAttribute* = enum
    attDevice = "__device__"
    attGlobal = "__global__"
    attForceInline = "__forceinline__"

  GpuVarAttribute* = enum
    atvExtern = "extern"
    atvShared = "__shared__"
    atvPrivate = "private" # WebGPU only
    atvVolatile = "volatile"
    atvConstant = "__constant__" # use `{.constant.}` pragma, e.g. `var foo {.constant.}`

  GpuAst* = ref object
    case kind*: GpuNodeKind
    of gpuDiscard: discard
    of gpuProc:
      pName*: GpuAst ## Will be a `GpuIdent`
      pRetType*: GpuType
      pParams*: seq[GpuParam]
      pBody*: GpuAst
      pAttributes*: set[GpuAttribute] # order not important, hence set
      pRawPragmas*: seq[string]  ## Raw pragma names from Nim AST (preserved for filterPragmas pass)
      forwardDeclare*: bool ## can be set to true to _only_ generate a forward declaration
    of gpuCall:
      cIsExpr*: bool ## If the call returns a value
      cName*: GpuAst ## Will be a `GpuIdent`
      cArgs*: seq[GpuAst]
    of gpuTemplateCall:
      tcName*: GpuAst ## Will be a `GpuIdent`
      tcArgs*: seq[GpuAst]  # Arguments for template instantiation
    of gpuIf:
      ifCond*: GpuAst
      ifThen*: GpuAst
      ifIsExpr*: bool     ## True if from nnkIfExpr (expression-if), False if from nnkIfStmt
      ifElse*: GpuAst # will be `GpuAst(kind*: gpuDiscard)` if no else branch
    of gpuTernary:
      tCond*: GpuAst  # condition
      tThen*: GpuAst  # then-expression
      tElse*: GpuAst  # else-expression
    of gpuFor:
      fVar*: GpuAst ## Will be a `GpuIdent`
      fStart*, fEnd*: GpuAst
      fBody*: GpuAst
      fRangeKind*: GpuRangeKind
    of gpuWhile:
      wCond*: GpuAst
      wBody*: GpuAst
    of gpuBinOp:
      bOp*: GpuAst # `gpuIdent` of the binary operation
      bLeft*, bRight*: GpuAst
      bIsOverloaded*: bool  ## True if operands are non-primitive types (pass converts to gpuCall)
      bType*: GpuType ## result type of this binop, derived at construction; MUST be non-nil on any gpuBinOp that can be the tail of a block-expression.
    of gpuVar:
      vName*: GpuAst ## Will be a `GpuIdent`
      vType*: GpuType
      vInit*: GpuAst
      vRequiresMemcpy*: bool
      vMutable*: bool # `true == var`, `false == let`
      vAttributes*: seq[GpuVarAttribute] # order is important, hence seq
    of gpuAssign:
      aLeft*, aRight*: GpuAst
      aRequiresMemcpy*: bool
    of gpuIdent:
      symbol*: Symbol
    of gpuLit:
      lValue*: string
      lType*: GpuType
    of gpuConstexpr:
      cIdent*: GpuAst # the identifier
      cValue*: GpuAst # not just a string to support different types easily
      cType*: GpuType
    of gpuMaterialize:
      mExpr*: GpuAst      # the non-lvalue expression to materialize
      mType*: GpuType     # the type to materialize as
    of gpuArrayLit:
      aValues*: seq[GpuAst]
      aLitType*: GpuType # type of first element
    of gpuBlock:
      isExpr*: bool ## Whether this block represents an expression, i.e. it returns something
      blockLabel*: string # optional name of the block. If any given, will open a `{ }` scope in CUDA
      statements*: seq[GpuAst]
    of gpuReturn:
      rValue*: GpuAst
    of gpuDot:
      dParent*: GpuAst
      dField*: GpuAst #string
    of gpuIndex:
      iArr*: GpuAst
      iIndex*: GpuAst
    of gpuPrefix:
      pOp*: string
      pVal*: GpuAst
    of gpuTypeDef:
      tTyp*: GpuType ## the actual type. Used to generate the name
      tFields*: seq[GpuTypeField]
    of gpuAlias:
      aTyp*: GpuType ## Name of the type alias
      aTo*: GpuAst ## Type the alias maps to
      aDistinct*: bool ## If the alias is a distinct type in Nim.
    of gpuObjConstr:
      ocType*: GpuType  # type we construct
      ## XXX: it would be better if we already fill the fields with default values here
      ocFields*: seq[GpuFieldInit] # the fields we initialize
    of gpuInlineAsm:
      stmt*: string
      ops*: seq[GpuAst] ## Operand symbols (gpuIdent) for backtick names
    of gpuComment:
      comment*: string
    of gpuConv:
      convTo*: GpuType # type to cast to
      convExpr*: GpuAst # expression we convert
    of gpuCast:
      cTo*: GpuType # type to cast to
      cExpr*: GpuAst # expression we cast
    of gpuAddr:
      aOf*: GpuAst
    of gpuDeref:
      dOf*: GpuAst

  GpuSymbolKind* = enum
    gsNone,              ## Default to mark not explicitly set
    gsDeviceKernelParam, ## Parameter of a device kernel (`function`)
    gsGlobalKernelParam, ## Parameter of a global kernel (`storage`) for WebGPU
    gsLocal,             ## Local variable (`function`)
    gsProc,              ## Kernel
    gsShared,            ## A shared variable (`{.shared.}` / `workspace`)
    gsPrivate,           ## A private variable (to each thread)

  Symbol* = ref object
    name*: string       ## Display name -- may be mangled for collision safety
    iSym*: string       ## IMMUTABLE fingerprint -- used as FnTable key (stays forever)
    typ*: GpuType       ## Type of the symbol
    symKind*: GpuSymbolKind ## Symbol kind
    module*: string     ## Module provenance (optional)

  ## WebGPU only: Address space of a variable.
  ## - storage: Storage buffer allocated on host and passed to device
  ## - function: Local variable within a function
  ## - workspace: Shared variable for all execution units in a block (like CUDA `shared`)
  ## - uniform: ??
  ## - private: Each thread has its own instance of the variable, e.g. useful for `carry`
  ## On the CUDA backend the address space is ignored.
  AddressSpace* = enum
    asFunction = "function"
    asStorage = "storage"
    asWorkspace = "workspace"
    asUniform = "uniform"
    asPrivate = "private"

  ## XXX: maybe merge into `GpuAst`, then can be kept in same table as `gpuVar` for locals
  GpuParam* = object
    ident*: GpuAst ## The actual parameter symbol, `GpuIdent`
    typ*: GpuType
    addressSpace*: AddressSpace
    passByRef*: bool   ## Pass by hidden const reference (large structs > 24 bytes)

  GpuFieldInit* = object
    name*: string
    value*: GpuAst
    typ*: GpuType

  GpuProcSignature* = object
    params*: seq[GpuParam]
    retType*: GpuType
    staticParamPositions*: seq[int] = @[] ## original arg-order positions of dropped
                                        ## static VALUE params (compile-time only,
                                        ## no CUDA value — see parseProcParameters)

  GpuContext* = object
    ## XXX: need table for generic invocations. Then when we encounter a type, need to map to
    ## the specific version
    ## However, also need to keep every *generic procedure*. In their bodies the types are
    ## only defined once they are called after all.
    skipSemicolon*: bool # whether we *currently* add semicolons at the end of a block or not
    allFnTab*: OrderedTable[GpuAst, GpuAst] ## map of all function definitions. For easy lookup by identifier
                                 ## Key is the `GpuAst` of the functions identifier / symbol
    fnTab*: OrderedTable[GpuAst, GpuAst] ## Map only of those function we generate code for. Includes
                                        ## generically instantiated functions.
    globalBlocks*: seq[GpuAst] ## Blocks in the global space. E.g. type defs or global variables.
    ## XXX: for now globals only store parameters, but we need to store `GpuAst` so that we can
    ## also add manually added globals or lifted `{.shared.}` variables!
    ## NOTE: The `globals` must store the type *AS IT WAS WRITTEN* in the Nim code. Any potential
    ## modifications we make locally for WebGPU (e.g. convert `bool` to `i32` for a global
    ## argument), must not be made to them. `globals` is used precisely to handle the *result* of
    ## that kind of transformation.
    ## As a result, the `globals` also *ONLY* contains the unique symbol as a key and not a `GpuAst`.
    globals*: OrderedTable[string, GpuParam] #Table[GpuAst, GpuAst] ## Maps global symbols (`{.shared.}` lifted to global, manually defined in global,
                         ## or `storage` buffer identifiers to the type? XXX to what?
    sigTab*: Table[string, GpuAst] ## Map the `nnkSym.signatureHash` to a `GpuAst` of kind `GpuIdent`
    currentScope*: GpuAst  ## Current scope block for variable registration during toGpuAst
    currentScopeSyms*: seq[(string, Symbol)]  ## Current scope's symbol table (parallel to currentScope)
    scopeSymsStack*: seq[seq[(string, Symbol)]]  ## Stack of parent scope symbol tables for push/pop
    genSymCount*: int ## increases for every generated identifier (currently only underscore `_`), hence the basic solution
    ## Maps a struct type and field name, which is of pointer type to the value the user assigns
    ## in the constructor. Allows us to later replace `foo.ptrField` by the assignment in the `Foo()`
    ## constructor (WebGPU only).
    structsWithPtrs*: Table[(GpuType, string), GpuAst]
    ## Set of all generic proc names we have encountered in Nim -> GpuAst. When
    ## we see an `nnkCall` we check if we call a generic function. If so, look up
    ## the instantiated generic, parse it and store in `genericInsts` below.
    generics*: HashSet[string]

    ## Phase 3: Unified function table. Keyed by iSym string.
    ## Contains ALL known functions (defined, generic-inst, external, builtin).
    fnTable*: OrderedTable[string, FnTableEntry]

    ## Stores the unique identifiers (keys) and the implementations of the
    ## precise generic instantiations that are called.
    genericInsts*: OrderedTable[GpuAst, GpuAst]

    ## Table of procs and their signature to avoid looping infinitely for recursive procs
    ## Arguments are:
    ## - Key: ident of the proc
    ## - Value: signature of the (possibly generic) instantiation
    processedProcs*: OrderedTable[GpuAst, GpuProcSignature]

    ## Storse all builtin / nimonly / importc / ... functions we encounter so that we can
    ## check if they return a value when we encounter them in a `gpuCall`
    builtins*: OrderedTable[GpuAst, GpuAst]

    ## Table of all known types. Filled during Nim -> GpuAst. Includes generic
    ## instantiations, but also all other types.
    ## Key: the raw type. Value: a full `gpuTypeDef`
    types*: OrderedTable[GpuType, GpuAst]

    ## This is _effectively_ just a set of all already produced function symbols.
    ## We use it to determine if when encountering another function with the same
    ## name, but different arguments to instead of using `iName` to use `iSym` as
    ## the function name. This is to avoid overload issues in backends that don't
    ## allow overloading by function signatures.
    symChoices*: HashSet[string]

    ## Phase 6: Vulkan SSBO canonical slots (name, inner type) per indexed position
    ssboCanonicalInfo*: seq[tuple[name: string, innerType: GpuType]]
    ## Phase 6: Vulkan push-constant param info (type, name strings stored as comments
    ## in globalBlocks for codegen to read)
  ## only need the `genericInsts` field data (the values). Trying to `newLit` the full `GpuContext`
  ## causes trouble.
  GpuGenericsInfo* = object
    procs*: seq[GpuAst]
    types*: seq[GpuAst]

  GenericArg* = object
    addrSpace*: AddressSpace ## We store the address space, because that's what matters
    mutable*: bool # if the argument is mutable or not
  GenericInst* = object
    name*: string # unique name of this generic variant
    args*: seq[GenericArg] # kind of symbols passed in at the call site. To determine ptr types, if args are ptrs
    # types are not stored in the instantiation, because we look up the types from the original function when generating the code


type
  TypeRegistry* = object
    types*: OrderedTable[GpuType, GpuAst]  ## type definition dedup

const GpuNumericTypes* = {gtBool, gtUint8, gtUint16, gtInt16,
                         gtUint32, gtInt32, gtUint64, gtInt64,
                         gtFloat32, gtFloat64, gtSize_t}
  ## Set of numeric (scalar) GpuTypeKind variants.

const TAG_IDENT_IN_ASM* = "\x01"
  ## Marker byte for a backtick identifier inside an inline-asm statement.
  ## The frontend stores the operand symbol (gpuIdent) in `ops` and emits
  ## this byte + index into `stmt`.
  ## The printers substitute the symbol's (mangled) display name at codegen time.
  ## A control byte is used so the marker can never collide with valid asm text.

proc newSymbol*(name: string, iSym: string = "", typ: GpuType = GpuType(kind: gtVoid), symKind: GpuSymbolKind = gsNone, module: string = ""): Symbol =
  new(result)
  result.name = name
  result.iSym = if iSym == "": name else: iSym
  result.typ = typ
  result.symKind = symKind
  result.module = module

proc newGpuIdent*(ident: string = "", symKind: GpuSymbolKind = gsNone): GpuAst =
  var sym = newSymbol(ident, symKind = symKind)
  result = GpuAst(kind: gpuIdent, symbol: sym)

proc scopeAdd*(scope: var seq[(string, Symbol)]; name: string; sym: Symbol) =
  ## Add a name->Symbol mapping to a scope table (seq-based for newLit compat).
  scope.add((name, sym))

proc scopeHas*(scope: seq[(string, Symbol)]; name: string): bool =
  ## Check if a name exists in a scope table.
  for (n, _) in scope:
    if n == name:
      return true

proc scopeGet*(scope: seq[(string, Symbol)]; name: string): Symbol =
  ## Look up a name in a scope table. Raises if not found.
  for (n, s) in scope:
    if n == name:
      return s
  raiseAssert "Scope lookup failed: '" & name & "' not found"

proc scopeGetOrDefault*(scope: seq[(string, Symbol)]; name: string): Symbol =
  ## Look up a name in a scope table. Returns nil if not found.
  for (n, s) in scope:
    if n == name:
      return s

proc clone*(typ: GpuType): GpuType =
  ## Returns a clone of the input type
  result = GpuType(kind: typ.kind)
  case result.kind
  of gtPtr:
    result.to = typ.to.clone()
    result.implicit = typ.implicit
    result.mutable = typ.mutable
  of gtUA:
    result.uaTo = typ.uaTo.clone()
  of gtObject:
    result.name = typ.name
    for f in typ.oFields:
      result.oFields.add GpuTypeField(name: f.name, typ: f.typ.clone())
  of gtArray:
    result.aTyp = typ.aTyp.clone()
    result.aLen = typ.aLen
  of gtStatic:
    result.sValue = typ.sValue
  of gtGenericInst:
    result.gName = typ.gName
    for g in typ.gArgs:
      result.gArgs.add g.clone()
    for f in typ.gFields:
      result.gFields.add GpuTypeField(name: f.name, typ: f.typ.clone())
  else: discard

proc clone*(ast: GpuAst): GpuAst =
  if ast.isNil: return nil
  case ast.kind
  of gpuDiscard: result = GpuAst(kind: gpuDiscard)
  of gpuProc:
    result = GpuAst(kind: gpuProc)
    result.pName = ast.pName.clone()
    result.pRetType = ast.pRetType.clone()
    for p in ast.pParams:
      let clonedParam = GpuParam(
        ident: p.ident.clone(),
        typ: p.typ.clone(),
        addressSpace: p.addressSpace,
        passByRef: p.passByRef
      )
      result.pParams.add(clonedParam)
    result.pBody = ast.pBody.clone()
    result.pAttributes = ast.pAttributes
    result.pRawPragmas = ast.pRawPragmas
    result.forwardDeclare = ast.forwardDeclare
  of gpuCall:
    result = GpuAst(kind: gpuCall)
    result.cIsExpr = ast.cIsExpr
    result.cName = ast.cName.clone()
    for arg in ast.cArgs:
      result.cArgs.add(arg.clone())
  of gpuTemplateCall:
    result = GpuAst(kind: gpuTemplateCall)
    result.tcName = ast.tcName.clone()
    for arg in ast.tcArgs:
      result.tcArgs.add(arg.clone())
  of gpuIf:
    result = GpuAst(kind: gpuIf)
    result.ifCond = ast.ifCond.clone()
    result.ifThen = ast.ifThen.clone()
    result.ifIsExpr = ast.ifIsExpr
    result.ifElse = ast.ifElse.clone()
  of gpuTernary:
    result = GpuAst(kind: gpuTernary)
    result.tCond = ast.tCond.clone()
    result.tThen = ast.tThen.clone()
    result.tElse = ast.tElse.clone()
  of gpuFor:
    result = GpuAst(kind: gpuFor)
    result.fVar = ast.fVar.clone()
    result.fStart = ast.fStart.clone()
    result.fEnd = ast.fEnd.clone()
    result.fBody = ast.fBody.clone()
    result.fRangeKind = ast.fRangeKind
  of gpuWhile:
    result = GpuAst(kind: gpuWhile)
    result.wCond = ast.wCond.clone()
    result.wBody = ast.wBody.clone()
  of gpuBinOp:
    result = GpuAst(kind: gpuBinOp)
    result.bOp = ast.bOp.clone()
    result.bLeft = ast.bLeft.clone()
    result.bRight = ast.bRight.clone()
    result.bIsOverloaded = ast.bIsOverloaded
    result.bType = ast.bType # clone fidelity: keep the self-carried result type
  of gpuVar:
    result = GpuAst(kind: gpuVar)
    result.vName = ast.vName.clone()
    result.vType = ast.vType.clone()
    result.vInit = ast.vInit.clone()
    result.vRequiresMemcpy = ast.vRequiresMemcpy
    result.vMutable = ast.vMutable
    result.vAttributes = ast.vAttributes
  of gpuAssign:
    result = GpuAst(kind: gpuAssign)
    result.aLeft = ast.aLeft.clone()
    result.aRight = ast.aRight.clone()
    result.aRequiresMemcpy = ast.aRequiresMemcpy
  of gpuIdent:
    result = GpuAst(kind: gpuIdent, symbol: ast.symbol) ## Share the Symbol ref!
  of gpuLit:
    result = GpuAst(kind: gpuLit)
    result.lValue = ast.lValue
    result.lType = ast.lType.clone()
  of gpuConstexpr:
    result = GpuAst(kind: gpuConstexpr)
    result.cIdent = ast.cIdent.clone()
    result.cValue = ast.cValue.clone()
    result.cType = ast.cType.clone()
  of gpuMaterialize:
    result = GpuAst(kind: gpuMaterialize)
    result.mExpr = ast.mExpr.clone()
    result.mType = ast.mType.clone()
  of gpuArrayLit:
    result = GpuAst(kind: gpuArrayLit)
    for a in ast.aValues:
      result.aValues.add a.clone()
    result.aLitType = ast.aLitType.clone()
  of gpuPrefix:
    result = GpuAst(kind: gpuPrefix)
    result.pOp = ast.pOp
    result.pVal = ast.pVal.clone()
  of gpuBlock:
    result = GpuAst(kind: gpuBlock)
    result.isExpr = ast.isExpr
    result.blockLabel = ast.blockLabel
    for stmt in ast.statements:
      result.statements.add(stmt.clone())
  of gpuReturn:
    result = GpuAst(kind: gpuReturn)
    result.rValue = ast.rValue.clone()
  of gpuDot:
    result = GpuAst(kind: gpuDot)
    result.dParent = ast.dParent.clone()
    result.dField = ast.dField.clone()
  of gpuIndex:
    result = GpuAst(kind: gpuIndex)
    result.iArr = ast.iArr.clone()
    result.iIndex = ast.iIndex.clone()
  of gpuTypeDef:
    result = GpuAst(kind: gpuTypeDef)
    result.tTyp = ast.tTyp.clone()
    for f in ast.tFields:
      result.tFields.add(GpuTypeField(name: f.name, typ: f.typ.clone()))
  of gpuAlias:
    result = GpuAst(kind: gpuAlias)
    result.aTyp = ast.aTyp.clone()
    result.aTo = ast.aTo.clone()
    result.aDistinct = ast.aDistinct
  of gpuObjConstr:
    result = GpuAst(kind: gpuObjConstr)
    result.ocType = ast.ocType.clone()
    for f in ast.ocFields:
      result.ocFields.add(
        GpuFieldInit(
          name: f.name,
          value: f.value.clone(),
          typ: f.typ.clone()
        )
      )
  of gpuInlineAsm:
    result = GpuAst(kind: gpuInlineAsm)
    result.stmt = ast.stmt
    for op in ast.ops:
      result.ops.add op.clone()
  of gpuAddr:
    result = GpuAst(kind: gpuAddr)
    result.aOf = ast.aOf.clone()
  of gpuDeref:
    result = GpuAst(kind: gpuDeref)
    result.dOf = ast.dOf.clone()
  of gpuConv:
    result = GpuAst(kind: gpuConv)
    result.convTo = ast.convTo.clone()
    result.convExpr = ast.convExpr.clone()
  of gpuCast:
    result = GpuAst(kind: gpuCast)
    result.cTo = ast.cTo.clone()
    result.cExpr = ast.cExpr.clone()
  of gpuComment:
    result = GpuAst(kind: gpuComment)
    result.comment = ast.comment

proc hash*(t: GpuType): Hash =
  var h = 0
  h = h !& hash(t.kind)
  case t.kind
  of gtPtr:
    h = h !& hash(t.to)
    h = h !& hash(t.implicit)
    h = h !& hash(t.mutable)
  of gtUA:
    h = h !& hash(t.uaTo)
  of gtObject:
    h = h !& hash(t.name)
    for f in t.oFields:
      h = h !& hash(f)
  of gtArray:
    h = h !& hash(t.aTyp)
    h = h !& hash(t.aLen)
  of gtStatic:
    h = h !& hash(t.sValue)
  of gtGenericInst:
    h = h !& hash(t.gName)
    for g in t.gArgs:
      h = h !& hash(g)
    for f in t.gFields:
      h = h !& hash(f)
  of gtInvalid: h = h !& hash("gtInvalid")
  else: discard
  result = !$ h

proc hash*(n: GpuAst): Hash =
  doAssert n.kind == gpuIdent, "Cannot hash a value other than `gpuIdents`! Input is: " & $n.kind
  var h = 0
  if n.symbol != nil:
    h = h !& hash(n.symbol.iSym) # In theory the only thing relevant is the `iSym`, as it is unique per Nim symbol
                                 # but if we fail to update a type or symbolkind, we'd produce a different hash, which is good
    if n.symbol.typ != nil: # can be nil, e.g. `gpuProc` symbols don't define it
      h = h !& hash(n.symbol.typ)
    h = h !& hash(n.symbol.symKind)
  result = !$ h

proc `==`*(a, b: GpuType): bool =
  # If either or both are nil, they don't match
  if a.isNil or b.isNil: result = false
  elif a.kind != b.kind: result = false
  else:
    result = true
    case a.kind
    of gtPtr: result = a.to == b.to and a.implicit == b.implicit and a.mutable == b.mutable
    of gtUA:  result = a.uaTo == b.uaTo
    of gtObject:
      result = a.name == b.name
      if a.oFields.len != b.oFields.len: result = false
      else:
        for i in 0 ..< a.oFields.len:
          result = result and (a.oFields[i] == b.oFields[i])
    of gtGenericInst:
      result = a.gName == b.gName
      if a.gArgs.len != b.gArgs.len: result = false
      elif a.gFields.len != b.gFields.len: result = false
      else:
        for i in 0 ..< a.gArgs.len:
          result = result and (a.gArgs[i] == b.gArgs[i])
        for i in 0 ..< a.gFields.len:
          result = result and (a.gFields[i] == b.gFields[i])
    of gtArray: result = a.aTyp == b.aTyp and a.aLen == b.aLen
    of gtStatic: result = a.sValue == b.sValue
    of gtInvalid: result = false
    else: discard

proc `==`*(a, b: GpuAst): bool =
  if a.isNil or b.isNil: return false
  if a.kind != b.kind: result = false
  elif a.kind != gpuIdent:
    raiseAssert "Unsupported equality for GpuAst that are not idents"
  else:
    result = a.symbol == b.symbol and a.symbol.iSym == b.symbol.iSym

proc `==`*(a, b: GpuParam): bool =
  ## Value equality for GpuParam: compares ident (via Symbol ref),
  ## typ (structural), addressSpace, and passByRef.
  if a.ident.isNil or b.ident.isNil:
    result = a.ident.isNil and b.ident.isNil
  else:
    result = a.ident.symbol == b.ident.symbol
  result = result and
    a.typ == b.typ and
    a.addressSpace == b.addressSpace and
    a.passByRef == b.passByRef

proc `==`*(a, b: GpuProcSignature): bool =
  if a.retType != b.retType: result = false
  elif a.params.len != b.params.len:
    result = false
  else:
    result = true
    for i in 0 ..< a.params.len:
      let ap = a.params[i]
      let bp = b.params[i]
      result = result and (ap == bp)

proc len*(ast: GpuAst): int =
  case ast.kind
  of gpuProc:      1
  of gpuCall:      1 + ast.cArgs.len
  of gpuBlock:     ast.statements.len
  of gpuIf:
    if ast.ifElse.kind != gpuDiscard: 3
    else:          2
  of gpuTernary:   3
  of gpuFor:       3
  of gpuWhile:     2
  of gpuBinOp:     2
  of gpuVar:       1
  of gpuAssign:    2
  of gpuPrefix:    1
  of gpuReturn:    1
  of gpuDot:       2
  of gpuIndex:     2
  of gpuObjConstr: ast.ocFields.len
  of gpuAddr:      1
  of gpuDeref:     1
  of gpuConv:      1
  of gpuCast:      1
  of gpuConstexpr: 2
  of gpuMaterialize: 1
  else: 0

proc `$`*(x: GpuType): string =
  if x == nil:
    result = "GpuType(nil)"
  else:
    result = $x[]

proc removePrefix(s, p: string): string =
  result = s
  result.removePrefix(p)

proc pretty*(t: GpuType): string =
  ## returns a flat (but lossy) string representation of the type
  if t == nil:
    result = "GpuType(nil)"
  else:
    case t.kind
    of gtPtr:
      result = if t.implicit: "var " else: "ptr "
      result.add pretty(t.to)
    of gtUA:
      result = "UncheckedArray[" & t.uaTo.pretty() & "]"
    of gtObject:
      result = t.name # just the name
    of gtArray:
      result = "array[" & $t.aLen & ", " & t.aTyp.pretty() & "]"
    of gtGenericInst:
      result = t.gName & "["
      for i, g in t.gArgs:
        result.add pretty(g)
        if i < t.gArgs.high:
          result.add ", "
      result.add "]"
    of gtStatic:
      result = "static(" & $t.sValue & ")"
    of gtInvalid:
      result = "Invalid"
    else:
      result = ($t.kind).removePrefix("gt")
proc pretty*(n: GpuAst, indent: int = 0): string =
  template id(): untyped = repeat(" ", indent)
  template idn(x): untyped = repeat(" ", indent) & $x
  template iddn(x): untyped = repeat(" ", indent + 2) & $x
  template id(x): untyped = idn(x) & "\n"
  template idd(x): untyped = iddn(x) & "\n"
  template id(x,y): untyped = repeat(" ", indent) & $x & " " & $y & "\n"
  template idd(x,y): untyped = repeat(" ", indent + 2) & $x & " " & $y & "\n"
  template spl(x): untyped = " " & $x & "\n"
  if n.isNil: return id("nil")

  result = idn(($n.kind).removePrefix("gpu"))
  if n.len > 0: result.add "\n"
  case n.kind
  of gpuDiscard: result.add "\n"
  of gpuProc:
    result.add pretty(n.pName, indent + 2)
    result.add idd("RetType", n.pRetType)
    result.add idd("Params")
    for p in n.pParams:
      result.add pretty(p.ident, indent + 4)
    result.add pretty(n.pBody, indent + 2)
    if n.pAttributes.len > 0:
      result.add idd("Attributes")
      for attr in n.pAttributes:
        let indent = indent + 2
        result.add idd(attr)
  of gpuCall:
    result.add pretty(n.cName, indent + 2)
    for arg in n.cArgs:
      result.add pretty(arg, indent + 2)
  of gpuTemplateCall: discard
  of gpuIf:
    result.add idd("IfCond")
    result.add pretty(n.ifCond, indent + 4)
    result.add idd("IfThen")
    result.add pretty(n.ifThen, indent + 4)
    if n.ifElse.kind != gpuDiscard:
      result.add idd("IfElse")
      result.add pretty(n.ifElse, indent + 4)
  of gpuTernary:
    result.add idd("TCond")
    result.add pretty(n.tCond, indent + 4)
    result.add idd("TThen")
    result.add pretty(n.tThen, indent + 4)
    result.add idd("TElse")
    result.add pretty(n.tElse, indent + 4)
  of gpuFor:
    result.add pretty(n.fVar, indent + 2)
    result.add pretty(n.fStart, indent + 2)
    result.add pretty(n.fEnd, indent + 2)
    result.add pretty(n.fBody, indent + 2)
    result.add id("RangeKind", n.fRangeKind)
  of gpuWhile:
    result.add pretty(n.wCond, indent + 2)
    result.add pretty(n.wBody, indent + 2)
  of gpuBinOp:
    result.add pretty(n.bOp, indent + 2)
    result.add pretty(n.bLeft, indent + 2)
    result.add pretty(n.bRight, indent + 2)
  of gpuVar:
    result.add pretty(n.vName, indent + 2)
    result.add pretty(n.vInit, indent + 2)
    if n.vAttributes.len > 0:
      result.add idd("Attributes")
      for attr in n.vAttributes:
        let indent = indent + 2
        result.add idd(attr)
  of gpuAssign:
    result.add pretty(n.aLeft, indent + 2)
    result.add pretty(n.aRight, indent + 2)
  of gpuIdent:
    result.add spl(n.symbol.name & "(" & n.symbol.iSym & ")")
  of gpuLit:
    result.add spl(n.lValue)
  of gpuConstexpr:
    result.add pretty(n.cIdent, indent + 2)
    result.add pretty(n.cValue, indent + 2)
  of gpuMaterialize:
    result.add pretty(n.mExpr, indent + 2)
  of gpuArrayLit:
    for el in n.aValues:
      result.add pretty(el, indent + 2)
  of gpuBlock:
    if n.blockLabel.len > 0:
      result.add id("Label", n.blockLabel)
    for stmt in n.statements:
      result.add pretty(stmt, indent + 2)
  of gpuReturn:
    result.add pretty(n.rValue, indent + 2)
  of gpuDot:
    result.add pretty(n.dParent, indent + 2)
    result.add pretty(n.dField, indent + 2)
  of gpuIndex:
    result.add pretty(n.iArr, indent + 2)
    result.add pretty(n.iIndex, indent + 2)
  of gpuPrefix:
    result.add id("Op", n.pOp)
    result.add pretty(n.pVal, indent + 2)
  of gpuTypeDef:
    result.add id("Type", pretty(n.tTyp))
    result.add id("Fields")
    for t in n.tFields:
      let indent = indent + 2
      result.add id(t.name)
  of gpuAlias:
    result.add id("Alias", pretty(n.aTyp))
    result.add pretty(n.aTo, indent + 2)
  of gpuObjConstr:
    result.add idd("Ident", pretty(n.ocType))
    result.add idd("Fields")
    for f in n.ocFields:
      var indent = indent + 2
      result.add idd("Field")
      indent = indent + 2
      result.add idd("Name", f.name)
      result.add pretty(f.value, indent + 2)
  of gpuInlineAsm:
    result.add id(n.stmt)
  of gpuComment:
    result.add id(n.comment)
  of gpuConv:
    result.add id($n.convTo)
    result.add pretty(n.convExpr, indent + 2)
  of gpuCast:
    result.add id($n.cTo)
    result.add pretty(n.cExpr, indent + 2)
  of gpuAddr:
    result.add pretty(n.aOf, indent + 2)
  of gpuDeref:
    result.add pretty(n.dOf, indent + 2)

proc `$`*(n: GpuAst): string =
  result = pretty(n, 0)

template iterImpl(ast: untyped, mutable: static bool): untyped =
  template ya(field: untyped): untyped =
    yield ast.field
  case ast.kind
  of gpuProc: # body
    ya(pBody)
  of gpuCall: # args
    when mutable:
      for el in mitems(ast.cArgs):
        yield el
    else:
      for el in ast.cArgs:
        yield el
  of gpuIf:
    ya(ifCond)
    ya(ifThen)
    if ast.ifElse.kind != gpuDiscard:
      yield ast.ifElse
  of gpuTernary:
    ya(tCond)
    ya(tThen)
    ya(tElse)
  of gpuFor:
    ya(fStart)
    ya(fEnd)
    ya(fBody)
  of gpuWhile:
    ya(wCond)
    ya(wBody)
  of gpuBinOp:
    ya(bLeft)
    ya(bRight)
  of gpuVar:
    ya(vInit)
  of gpuAssign:
    ya(aLeft)
    ya(aRight)
  of gpuPrefix:
    ya(pVal)
  of gpuBlock:
    when mutable:
      for ch in mitems(ast.statements):
        yield ch
    else:
      for ch in ast.statements:
        yield ch
  of gpuReturn:
    ya(rValue)
  of gpuDot:
    ya(dParent)
    ya(dField)
  of gpuIndex:
    ya(iArr)
    ya(iIndex)
  of gpuObjConstr:
    when mutable:
      for el in mitems(ast.ocFields):
        yield el.value
    else:
      for el in ast.ocFields:
        yield el.value
  of gpuAddr:
    ya(aOf)
  of gpuDeref:
    ya(dOf)
  of gpuConv:
    ya(convExpr)
  of gpuCast:
    ya(cExpr)
  of gpuConstexpr:
    ya(cIdent)
    ya(cValue)
  of gpuMaterialize:
    ya(mExpr)
  else:
    discard # nothing to yield

iterator mitems*(ast: var GpuAst): var GpuAst =
  ## Iterate over all child nodes of the given AST
  iterImpl(ast, mutable = true)

iterator items*(ast: GpuAst): GpuAst =
  iterImpl(ast, mutable = false)

iterator mpairs*(ast: var GpuAst): (int, var GpuAst) =
  ## Iterate over all child nodes of the given AST and the index
  var i = 0
  for el in mitems(ast):
    yield (i, el)
    inc i

iterator pairs*(ast: GpuAst): (int, GpuAst) =
  var i = 0
  for el in items(ast):
    yield (i, el)
    inc i



func size*(t: GpuType): int =
  ## Compute the byte size of a GpuType.
  case t.kind
  of gtVoid:
    result = 0
  of gtBool, gtUint8:
    result = 1
  of gtUint16, gtInt16:
    result = 2
  of gtUint32, gtInt32, gtFloat32:
    result = 4
  of gtUint64, gtInt64, gtFloat64:
    result = 8
  of gtSize_t:
    result = 8
  of gtPtr, gtUA, gtVoidPtr:
    result = 8  # pointer size on 64-bit
  of gtString:
    result = 8  # pointer
  of gtStatic:
    result = 0  # compile-time value, no runtime size
  of gtArray:
    # TODO: for generic/unresolved array, size is set to -1, but can that happen in practice?
    # Due to constant folding I would expect array length to always be resolved.
    result = t.aLen * size(t.aTyp)
  of gtObject:
    for f in t.oFields:
      result += size(f.typ)
  of gtGenericInst:
    for f in t.gFields:
      result += size(f.typ)
  of gtInvalid:
    result = 0
  else:
    result = 4  # default fallback

func isLargeStruct*(t: GpuType): bool =
  ## Returns true if the type should be passed by hidden const reference.
  ## Uses Nim's C backend 3-pointer threshold. For GPU, also catch structs
  ## with embedded arrays (some Vulkan impls reject struct-by-value with arrays).
  const threshold = 24
  case t.kind
  of gtObject, gtGenericInst:
    result = size(t) >= threshold
  else:
    result = false

proc getFnParams*(ctx: GpuContext, fn: GpuAst): seq[GpuParam] =
  ## Look up the parameters of a function by its identifier.
  ## Phase 3: checks fnTable first, then falls back to old tables.
  let key = fn.symbol.iSym
  if key in ctx.fnTable:
    let entry = ctx.fnTable[key]
    if not entry.body.isNil and entry.body.kind == gpuProc:
      result = entry.body.pParams
    else:
      result = @[]
  elif fn in ctx.allFnTab:
    result = ctx.allFnTab[fn].pParams
  elif fn in ctx.fnTab:
    result = ctx.fnTab[fn].pParams
  elif fn in ctx.genericInsts:
    result = ctx.genericInsts[fn].pParams
  elif fn in ctx.builtins:
    result = ctx.builtins[fn].pParams
  elif fn in ctx.processedProcs:
    result = ctx.processedProcs[fn].params
proc ident*(n: GpuAst): string =
  ## Returns the associated identifier (string) of the given symbol. The input
  ## must be a `gpuIdent`
  doAssert n.kind == gpuIdent, "The input is not a `gpuIdent`, but a " & $n.kind
  result = n.symbol.name

proc getFnReturnType*(ctx: GpuContext, fn: GpuAst): GpuType =
  ## Look up the return type of a function by its identifier.
  ## Phase 3: checks fnTable first, then falls back to old tables.
  let key = fn.symbol.iSym
  if key in ctx.fnTable:
    let entry = ctx.fnTable[key]
    if not entry.body.isNil and entry.body.kind == gpuProc:
      result = entry.body.pRetType
    else:
      result = GpuType(kind: gtVoid)
  elif fn in ctx.allFnTab:
    result = ctx.allFnTab[fn].pRetType
  elif fn in ctx.fnTab:
    result = ctx.fnTab[fn].pRetType
  elif fn in ctx.genericInsts:
    result = ctx.genericInsts[fn].pRetType
  elif fn in ctx.builtins:
    result = ctx.builtins[fn].pRetType
  elif fn in ctx.processedProcs:
    result = ctx.processedProcs[fn].retType
  else:
    raiseAssert "Function not found: " & $fn & " (name=" & fn.ident() & ")"

template withoutSemicolon*(ctx: var GpuContext, body: untyped): untyped =
  if not ctx.skipSemicolon: # if we are already skipping, leave true
    ctx.skipSemicolon = true
    body
    ctx.skipSemicolon = false
  else:
    body

proc getInnerArrayLengths*(t: GpuType): string =
  ## Returns the lengths of the inner array types for a nested array.
  case t.kind
  of gtArray:
    let inner = getInnerArrayLengths(t.aTyp)
    result = &"[{$t.aLen}]"
    if inner.len > 0:
      result.add &"{inner}"
  else:
    result = ""

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
    result = $t.kind # fallback M-bM-^@M-^T safe but verbose

const Base58* = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  ## Base58 alphabet (no 0, O, I, l for readability and ambiguity avoidance).

func shortHash*(sigHash: int64): string =
  ## Encode a 64-bit signature hash as a 7-character base58 string.
  ## 58^7 = 2,204,715,403,072 (~2.2T namespace), sufficient for collision
  ## avoidance across all symbols in a GPU compilation unit.
  var n = uint64(sigHash)
  if n == 0:
    return "1111111"
  var chars: array[7, char]
  for i in countdown(6, 0):
    let rem = int(n mod 58)
    chars[i] = Base58[rem]
    n = n div 58
    if n == 0 and i == 0:
      break
    if n == 0:
      for j in 0 ..< i:
        chars[j] = '1'
      break
  result = cast[string](@chars)
