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
    bkWebGPU,   ## WebGPU WGSL backend
    bkOpenCL, ## OpenCL backend
    bkGlsl,  ## GLSL backend, executed on Vulkan
    bkMetal,  ## Metal (MSL) backend
    bkHlsl    ## HLSL backend, compiled through the glslang HLSL frontend

  GpuNodeKind* = enum
    gpuDiscard         # Just an empty statement. Useful to not emit anything
    gpuProc         # Function definition (both device and global)
    gpuCall         # Function call
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
    gpuEmit         # Raw target-source injection ({.emit.})
    gpuAddr         # Address of an expression
    gpuDeref        # Dereferences an expression
    gpuConv         # A type conversion, i.e. `let x = 5; x.float`
    gpuCast         # Cast expression
    gpuComment      # Just a comment
    gpuConstexpr    # A `constexpr`, i.e. compile time constant (Nim `const`)
    gpuMaterialize  # Force materialization of a non-lvalue expr for pass-by-ref

  GpuTypeKind* = enum
    gtVoid,
    gtBool, gtUint8, gtUint16, gtInt16, gtUint32, gtInt32, gtUint64, gtInt64, gtFloat32, gtFloat64,
    gtFloat16, gtBf16, gtF8E4M3, gtF8E5M2, gtSize_t,
    gtStatic       # Static integer value (used for generic params)
    gtArray,       # Static array `array[N, dtype]` -> `dtype[N]`
    gtString,
    gtObject,      # Struct types
    gtPtr,         # Pointer type, carries inner type
    gtUncheckedArray,          # UncheckedArray, mapped to runtime sized arrays
    gtSpan,         # Runtime-length array view (openArray/varargs)
    gtGenericInst, # Instantiated generic type with one or more generic arguments (instantiated!)
    gtVoidPtr      # Opaque void pointer
    gtInvalid      # Can be returned to indicate a call to `nimToGpuType` failed to determine a type
    gtCompileTimeValue  # the parameter's type slot holds the compile-time argument, not a type


  GpuTypeField* = object
    name*: string
    typ*: GpuType

  GpuType* = ref object
    builtin*: bool ## Whether the type refers to a builtin type or not
    isTuple*: bool ## Object or generic inst stands for a Nim tuple
    fields*: seq[GpuTypeField] ## Fields of gtObject and gtGenericInst, declaration order
    addressSpace*: AddressSpace ## Memory a gtPtr/gtSpan/gtArray accesses, untagged kinds keep asRMEM.
                  ## Variables carry their own storage space on `GpuParam.addressSpace`/`gpuVar.addressSpace`
    case kind*: GpuTypeKind
    of gtPtr:
      to*: GpuType # `ptr T` points to `to`
      implicit*: bool # Whether the type was implicitly a pointer, i.e. `var T`.
      mutable*: bool # WebGPU "Generics" only: Mutable (read write) or immutable pointer (read only)?
                    # True for raw pointer and `var T` arguments, false for `foo.addr` of a non pointer type.
    of gtUncheckedArray: uaTo*: GpuType # `UncheckedArray[T]`
    of gtObject:
      name*: string
    of gtArray:
      aTyp*: GpuType # the inner type
      aLen*: int     # The length of the array, `-1` for a generic (static) array resolved at instantiation time
                    # `0` generates runtime-sized arrays on CUDA (`int foo[]`) and WebGPU (`array<foo>`),
                    # produced from `ptr UncheckedArray[float32]` for example.
    of gtStatic:
      sValue*: int  # The actual static integer value
    of gtSpan:
      sElemTyp*: GpuType   # element type T
    of gtGenericInst:
      gName*: string # name of the generic type
      gArgs*: seq[GpuType] # list of the instantiated generic arguments e.g. `vec3<f32>` on WGSL backend
    else: discard

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
  GpuAttribute* = enum
    attDevice = "__device__"
    attGlobal = "__global__"
    attForceInline = "__forceinline__"

  GpuEmitPartKind* = enum
    peLiteral  ## Verbatim raw text run
    peExpr     ## Expression part rendered through the target codegen

  GpuEmitPart* = object
    ## One ordered part of a `gpuEmit` statement: a verbatim literal run or an expression rendered by the target codegen.
    case kind*: GpuEmitPartKind
    of peLiteral: literal*: string
    of peExpr: expr*: GpuAst

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
      pWorkgroupSize*: tuple[x, y, z: int]  ## `{.workgroup: (X, Y, Z).}` annotation, (0,0,0) = unset
      forwardDeclare*: bool ## can be set to true to _only_ generate a forward declaration
    of gpuCall:
      cName*: GpuAst ## Will be a `GpuIdent`
      cArgs*: seq[GpuAst]
    of gpuIf:
      ifCond*: GpuAst
      ifThen*: GpuAst
      ifIsExpr*: bool     ## True if from nnkIfExpr (expression-if), False if from nnkIfStmt
      ifElifs*: seq[tuple[cond, body: GpuAst]] ## Flat elif branches (from nnkIfStmt), empty for if-exprs
      ifElse*: GpuAst # will be `GpuAst(kind*: gpuDiscard)` if no else branch
    of gpuTernary:
      tCond*: GpuAst  # condition
      tThen*: GpuAst  # then-expression
      tElse*: GpuAst  # else-expression
    of gpuFor:
      fVar*: GpuAst ## Will be a `GpuIdent`
      fStart*, fEnd*: GpuAst
      fStep*: GpuAst ## The loop increment; always set (1 for ranges, the countup step)
      fBody*: GpuAst
      fRangeKind*: GpuRangeKind
      fUnroll*: bool ## unroll-hinted loop, backends emit an unroll pragma
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
      addressSpace*: AddressSpace
        ## Resolved from the `{.smem.}`/`{.rmem.}`/`{.const_mem.}` pragma, unannotated locals default to asRMEM
    of gpuAssign:
      aLeft*, aRight*: GpuAst
      aOp*: string = "="
        ## Compound-assignment token the target applies or the plain `"="`
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
    of gpuObjConstr:
      ocType*: GpuType  # the constructed type
      ## XXX: fields could be pre-filled with default values here
      ocFields*: seq[GpuFieldInit] # the initialized fields
    of gpuInlineAsm:
      stmt*: string
      ops*: seq[GpuAst] ## Operand symbols (gpuIdent) for backtick names
    of gpuEmit:
      parts*: seq[GpuEmitPart] ## Ordered parts: literal runs verbatim, expr parts via target codegen
    of gpuComment:
      comment*: string
    of gpuConv:
      convTo*: GpuType # type to cast to
      convExpr*: GpuAst # the converted expression
    of gpuCast:
      cTo*: GpuType # type to cast to
      cExpr*: GpuAst # the cast expression
    of gpuAddr:
      aOf*: GpuAst
    of gpuDeref:
      dOf*: GpuAst

  GpuSymbolKind* = enum
    gsNone,              ## Default to mark not explicitly set
    gsBuiltin,           ## Reference to a backend index builtin (e.g. MSL thread_position_in_grid).
    gsDeviceKernelParam, ## Parameter of a device kernel (`function`)
    gsGlobalKernelParam, ## Parameter of a global kernel (`storage`) for WebGPU
    gsLocal,             ## Local variable (`function`)
    gsProc,              ## Kernel

  GpuCoordBuiltinKind* = enum
    gbkNone,                      ## Not a coordinate builtin (locals, printf, cvtaGenericToShared)
    gbkThreadPositionInGrid,      ## Canonical `thread_position_in_grid`
    gbkThreadgroupPositionInGrid, ## Canonical `threadgroup_position_in_grid`
    gbkThreadPositionInThreadgroup, ## Canonical `thread_position_in_threadgroup`
    gbkThreadsPerThreadgroup,     ## Canonical `threads_per_threadgroup`
    gbkThreadgroupsPerGrid,       ## Canonical `threadgroups_per_grid`
    gbkThreadIndexInThreadgroup  ## Canonical `thread_index_in_threadgroup`

  GpuSynchroBuiltinKind* = enum
    gbkNone,              ## Not a synchronization builtin
    gbkThreadgroupBarrier ## Canonical `threadgroup_barrier()`, the threadgroup fence
    gbkThreadgroupBarrierDevice ## Canonical `threadgroup_barrier_device()`, the device fence

  GpuReductionBuiltinKind* = enum
    gbkNone,              ## Not a reduction builtin
    gbkReduceAdd,         ## Canonical `reduceAdd`: cross-lane sum, broadcast to all lanes
    gbkReduceMax          ## Canonical `reduceMax`: cross-lane max, broadcast to all lanes

  GpuShuffleBuiltinKind* = enum
    gbkNone
    gbkSimdShuffle
    gbkSimdShuffleXor

  GpuAtomicsBuiltinKind* = enum
    gbkNone,              ## Not an atomics builtin
    gbkAtomicFetchAdd     ## Canonical `atomicFetchAdd`: atomic add through a device pointer

  Symbol* = ref object
    name*: string       ## Display name -- may be mangled for collision safety
    iSym*: string       ## IMMUTABLE fingerprint -- used as FnTable key (stays forever)
    typ*: GpuType       ## Type of the symbol
    symKind*: GpuSymbolKind ## Symbol kind
    coordBuiltin*: GpuCoordBuiltinKind ## Coordinate builtin kind -> gbkNone when not a coordinate
    synchroBuiltin*: GpuSynchroBuiltinKind ## Synchronization builtin kind -> gbkNone when not a barrier
    reductionBuiltin*: GpuReductionBuiltinKind ## Reduction builtin kind -> gbkNone when not a reduction intrinsic
    atomicsBuiltin*: GpuAtomicsBuiltinKind ## Atomics builtin kind -> gbkNone when not an atomic intrinsic
    shuffleBuiltin*: GpuShuffleBuiltinKind ## Shuffle builtin kind -> gbkNone when not a shuffle intrinsic

  AddressSpace* = enum
    ## Per-target spellings:
    ## - asDevice:   MSL `device` / WGSL `storage` / CUDA `__global__` / OpenCL `__global`
    ## - asConstant: MSL `constant` / WGSL `uniform` / CUDA `__constant__`
    ## - asSMEM:     MSL `threadgroup` / WGSL `workgroup` / CUDA `__shared__`
    ## - asRMEM:     per-thread (register) memory: MSL `thread` / WGSL `function` / CUDA unqualified (automatic)
    asRMEM
    asSMEM
    asConstant
    asDevice

  ParamFlavor* = enum
    kIn
    kVar

  GpuParam* = object
    ident*: GpuAst ## The actual parameter symbol, `GpuIdent`
    typ*: GpuType
    addressSpace*: AddressSpace
    passByRef*: bool   ## Pass by hidden const reference (large structs > 24 bytes)
    flavor*: ParamFlavor

  GpuFieldInit* = object
    name*: string
    value*: GpuAst
    typ*: GpuType

  GpuProcSignature* = object
    params*: seq[GpuParam]
    retType*: GpuType
    staticParamPositions*: seq[int] = @[] ## original arg-order positions of dropped static VALUE params (compile-time only, no CUDA value)

  GpuContext* = object
    ## XXX: generic invocation table missing, call-site types must map to the instantiated version.
    skipSemicolon*: bool # when true, block statements emit without trailing semicolons
    allFnTab*: OrderedTable[GpuAst, GpuAst] ## All function definitions, keyed by identifier `GpuAst`
    fnTab*: OrderedTable[GpuAst, GpuAst] ## Functions codegen generates, generic instantiations included
    globalBlocks*: seq[GpuAst] ## Blocks in the global space, e.g. type defs or global variables
      ## Vulkan push-constant param info: type and name strings stored as comments in globalBlocks for codegen to read
    ## XXX: globals hold only parameters, the `GpuAst` storage exists for manually added globals and lifted `{.smem.}` variables
    ## NOTE: `globals` stores params as written in the Nim source, WebGPU-local rewrites must not touch it
    globals*: OrderedTable[string, GpuParam] ## `{.smem.}`-lifted, manually defined and `storage` buffer globals, keys are symbol strings
    sigTab*: Table[string, GpuAst] ## Map the `nnkSym.signatureHash` to a `GpuAst` of kind `GpuIdent`
    currentScope*: GpuAst  ## Current scope block for variable registration during toCrucibleAst
    currentScopeSyms*: seq[(string, Symbol)]  ## Current scope's symbol table (parallel to currentScope)
    scopeSymsStack*: seq[seq[(string, Symbol)]]  ## Stack of parent scope symbol tables for push/pop
    genSymCount*: int ## Counter for generated identifiers

    structsWithPtrs*: Table[(GpuType, string), GpuAst] ## WGSL: (type, field) -> kernel-param ident at construction, printer rewrites `foo.field` to it
    ptrFieldBindings*: Table[(string, string), GpuAst] ## WGSL: per-variable (variable, field) binding, the type-keyed table would alias same-struct variables

    generics*: HashSet[string] ## Generic proc names seen in Nim -> GpuAst, triggers instantiation recording into `genericInsts`
    ## All known functions (defined, generic-inst, external, builtin), keyed by iSym
    fnTable*: OrderedTable[string, FnTableEntry]

    ## Called generic instantiations, key is the instantiation ident, value the implementation
    genericInsts*: OrderedTable[GpuAst, GpuAst] ## Called generic instantiations, key the instantiation ident, value the implementation
    processedProcs*: OrderedTable[GpuAst, GpuProcSignature] ## Recursion guard: procs already processed, value the instantiation signature
    builtinFns*: OrderedTable[GpuAst, GpuAst] ## Builtin/importc functions encountered, consulted for value-return at `gpuCall` sites
    types*: OrderedTable[GpuType, GpuAst] ## All known types from Nim -> GpuAst, key the raw type, value a full `gpuTypeDef`
    symChoices*: HashSet[string] ## Produced function symbol names, a name collision switches the display name from `iName` to `iSym`

type
  TypeRegistry* = object
    types*: OrderedTable[GpuType, GpuAst]  ## type definition dedup

let GpuCoordBuiltinKindByName* {.compileTime.}: Table[string, GpuCoordBuiltinKind] = {
  "thread_position_in_grid": gbkThreadPositionInGrid,
  "threadgroup_position_in_grid": gbkThreadgroupPositionInGrid,
  "thread_position_in_threadgroup": gbkThreadPositionInThreadgroup,
  "threads_per_threadgroup": gbkThreadsPerThreadgroup,
  "threadgroups_per_grid": gbkThreadgroupsPerGrid,
  "thread_index_in_threadgroup": gbkThreadIndexInThreadgroup,
}.toTable()

let GpuSynchroBuiltinKindByName* {.compileTime.}: Table[string, GpuSynchroBuiltinKind] = {
  "threadgroup_barrier": gbkThreadgroupBarrier,
  "threadgroup_barrier_device": gbkThreadgroupBarrierDevice,
}.toTable()

let GpuReductionBuiltinKindByName* {.compileTime.}: Table[string, GpuReductionBuiltinKind] = {
  "reduceAdd": gbkReduceAdd,
  "reduceMax": gbkReduceMax,
}.toTable()

let GpuAtomicsBuiltinKindByName* {.compileTime.}: Table[string, GpuAtomicsBuiltinKind] = {
  "atomicFetchAdd": gbkAtomicFetchAdd,
}.toTable()

let GpuShuffleBuiltinKindByName* {.compileTime.}: Table[string, GpuShuffleBuiltinKind] = {
  "simd_shuffle": gbkSimdShuffle,
  "simd_shuffle_xor": gbkSimdShuffleXor,
}.toTable()

proc coordBuiltinKind*(name: string): GpuCoordBuiltinKind =
  ## Returns the IR kind of a canonical coordinate name, unmapped names (printf, cvtaGenericToShared, shadowing locals) resolve to `gbkNone`.
  GpuCoordBuiltinKindByName.getOrDefault(name, gbkNone)

proc synchroBuiltinKind*(name: string): GpuSynchroBuiltinKind =
  ## Returns the IR kind of a canonical synchronization name, unmapped names resolve to `gbkNone`.
  GpuSynchroBuiltinKindByName.getOrDefault(name, gbkNone)

proc reductionBuiltinKind*(name: string): GpuReductionBuiltinKind =
  GpuReductionBuiltinKindByName.getOrDefault(name, gbkNone)

proc atomicsBuiltinKind*(name: string): GpuAtomicsBuiltinKind =
  GpuAtomicsBuiltinKindByName.getOrDefault(name, gbkNone)

proc shuffleBuiltinKind*(name: string): GpuShuffleBuiltinKind =
  GpuShuffleBuiltinKindByName.getOrDefault(name, gbkNone)

const GpuNumericTypes* = {gtBool, gtUint8, gtUint16, gtInt16,
                         gtUint32, gtInt32, gtUint64, gtInt64,
                         gtFloat32, gtFloat64, gtFloat16, gtBf16,
                         gtF8E4M3, gtF8E5M2, gtSize_t}

const TAG_IDENT_INLINE_ASM* = "\x01"

func newSymbol*(name: string, iSym: string = "", typ: GpuType = GpuType(kind: gtVoid), symKind: GpuSymbolKind = gsNone): Symbol =
  new(result)
  result.name = name
  result.iSym = if iSym == "": name else: iSym
  result.typ = typ
  result.symKind = symKind

func newGpuIdent*(ident: string = "", symKind: GpuSymbolKind = gsNone): GpuAst =
  var sym = newSymbol(ident, symKind = symKind)
  result = GpuAst(kind: gpuIdent, symbol: sym)

func scopeAdd*(scope: var seq[(string, Symbol)], name: string, sym: Symbol) =
  ## Add a name->Symbol mapping to a scope table (seq-based for newLit compat).
  scope.add((name, sym))

func scopeHas*(scope: seq[(string, Symbol)], name: string): bool =
  ## Check if a name exists in a scope table.
  for (n, _) in scope:
    if n == name:
      return true

func scopeGet*(scope: seq[(string, Symbol)], name: string): Symbol =
  ## Look up a name in a scope table. Raises if not found.
  for (n, s) in scope:
    if n == name:
      return s
  error "Scope lookup failed: '" & name & "' not found"

func clone*(typ: GpuType): GpuType =
  ## Returns a clone of the input type
  result = GpuType(kind: typ.kind)
  result.addressSpace = typ.addressSpace
  case result.kind
  of gtPtr:
    result.to = typ.to.clone()
    result.implicit = typ.implicit
    result.mutable = typ.mutable
  of gtUncheckedArray:
    result.uaTo = typ.uaTo.clone()
  of gtObject:
    result.name = typ.name
    for f in typ.fields:
      result.fields.add GpuTypeField(name: f.name, typ: f.typ.clone())
  of gtArray:
    result.aTyp = typ.aTyp.clone()
    result.aLen = typ.aLen
  of gtStatic:
    result.sValue = typ.sValue
  of gtSpan:
    result.sElemTyp = typ.sElemTyp.clone()
  of gtGenericInst:
    result.gName = typ.gName
    for g in typ.gArgs:
      result.gArgs.add g.clone()
    for f in typ.fields:
      result.fields.add GpuTypeField(name: f.name, typ: f.typ.clone())
  else: discard

func clone*(ast: GpuAst): GpuAst =
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
        passByRef: p.passByRef,
        flavor: p.flavor
      )
      result.pParams.add(clonedParam)
    result.pBody = ast.pBody.clone()
    result.pAttributes = ast.pAttributes
    result.pRawPragmas = ast.pRawPragmas
    result.pWorkgroupSize = ast.pWorkgroupSize
    result.forwardDeclare = ast.forwardDeclare
  of gpuCall:
    result = GpuAst(kind: gpuCall)
    result.cName = ast.cName.clone()
    for arg in ast.cArgs:
      result.cArgs.add(arg.clone())
  of gpuIf:
    result = GpuAst(kind: gpuIf)
    result.ifCond = ast.ifCond.clone()
    result.ifThen = ast.ifThen.clone()
    result.ifIsExpr = ast.ifIsExpr
    for el in ast.ifElifs:
      result.ifElifs.add (cond: el.cond.clone(), body: el.body.clone())
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
    result.fStep = ast.fStep.clone()
    result.fUnroll = ast.fUnroll
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
    result.addressSpace = ast.addressSpace
  of gpuAssign:
    result = GpuAst(kind: gpuAssign)
    result.aLeft = ast.aLeft.clone()
    result.aRight = ast.aRight.clone()
    result.aOp = ast.aOp
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
  of gpuEmit:
    result = GpuAst(kind: gpuEmit)
    for part in ast.parts:
      case part.kind
      of peLiteral:
        result.parts.add GpuEmitPart(kind: peLiteral, literal: part.literal)
      of peExpr:
        result.parts.add GpuEmitPart(kind: peExpr, expr: part.expr.clone())
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

func hash*(t: GpuType): Hash =
  var h = 0
  h = h !& hash(t.kind)
  case t.kind
  of gtPtr:
    h = h !& hash(t.to)
    h = h !& hash(t.implicit)
    h = h !& hash(t.mutable)
    h = h !& hash(t.addressSpace)
  of gtUncheckedArray:
    h = h !& hash(t.uaTo)
  of gtObject:
    h = h !& hash(t.name)
    for f in t.fields:
      h = h !& hash(f)
  of gtArray:
    h = h !& hash(t.aTyp)
    h = h !& hash(t.aLen)
    h = h !& hash(t.addressSpace)
  of gtStatic:
    h = h !& hash(t.sValue)
  of gtGenericInst:
    h = h !& hash(t.gName)
    for g in t.gArgs:
      h = h !& hash(g)
    for f in t.fields:
      h = h !& hash(f)
  of gtSpan:
    h = h !& hash(t.sElemTyp)
    h = h !& hash(t.addressSpace)
  of gtInvalid: h = h !& hash("gtInvalid")
  else: discard
  result = !$ h

func hash*(n: GpuAst): Hash =
  doAssert n.kind == gpuIdent, "Cannot hash a value other than `gpuIdents`! Input is: " & $n.kind
  var h = 0
  if n.symbol != nil:
    h = h !& hash(n.symbol.iSym) # `iSym` is unique per Nim symbol, folding the type and symbol kind
                                 # in keeps distinct symbols hashing differently
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
    of gtPtr: result = a.to == b.to and a.implicit == b.implicit and a.mutable == b.mutable and
      a.addressSpace == b.addressSpace
    of gtUncheckedArray:  result = a.uaTo == b.uaTo
    of gtObject:
      result = a.name == b.name
      if a.fields.len != b.fields.len: result = false
      else:
        for i in 0 ..< a.fields.len:
          result = result and (a.fields[i] == b.fields[i])
    of gtGenericInst:
      result = a.gName == b.gName
      if a.gArgs.len != b.gArgs.len: result = false
      elif a.fields.len != b.fields.len: result = false
      else:
        for i in 0 ..< a.gArgs.len:
          result = result and (a.gArgs[i] == b.gArgs[i])
        for i in 0 ..< a.fields.len:
          result = result and (a.fields[i] == b.fields[i])
    of gtArray: result = a.aTyp == b.aTyp and a.aLen == b.aLen and
      a.addressSpace == b.addressSpace
    of gtSpan: result = a.sElemTyp == b.sElemTyp and a.addressSpace == b.addressSpace
    of gtStatic: result = a.sValue == b.sValue
    of gtInvalid: result = false
    else: discard

proc `==`*(a, b: Symbol): bool =
  ## Compares symbols by content (iSym, kind, type), not by pointer.
  ##
  ## Two Symbol objects can describe the same symbol: a function's name and a call to that function are created independently.
  ##
  ## Pointer comparison reports such pairs as different. The code emitter
  ## then fails to find the callee definition in the function tables.
  if a.isNil or b.isNil:
    result = a.isNil and b.isNil
  else:
    result = a.iSym == b.iSym and a.symKind == b.symKind and
      ((a.typ.isNil and b.typ.isNil) or
       (not a.typ.isNil and not b.typ.isNil and a.typ == b.typ))

proc `==`*(a, b: GpuAst): bool =
  if a.isNil or b.isNil: return false
  if a.kind != b.kind: result = false
  elif a.kind != gpuIdent:
    error "Unsupported equality for GpuAst that are not idents"
  else:
    result = a.symbol == b.symbol and a.symbol.iSym == b.symbol.iSym

proc `==`*(a, b: GpuParam): bool =
  ## Value equality for GpuParam: ident (via Symbol ref), typ (structural), addressSpace, passByRef
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

func len*(ast: GpuAst): int =
  case ast.kind
  of gpuProc:      1
  of gpuCall:      1 + ast.cArgs.len
  of gpuBlock:     ast.statements.len
  of gpuIf:
    (if ast.ifElse.kind != gpuDiscard: 3 else: 2) + ast.ifElifs.len
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
    of gtUncheckedArray:
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
  of gpuIf:
    result.add idd("IfCond")
    result.add pretty(n.ifCond, indent + 4)
    result.add idd("IfThen")
    result.add pretty(n.ifThen, indent + 4)
    for i, el in n.ifElifs:
      result.add idd("IfElif" & $i & "Cond")
      result.add pretty(el.cond, indent + 4)
      result.add idd("IfElif" & $i & "Body")
      result.add pretty(el.body, indent + 4)
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
    if n.fStep != nil:
      result.add pretty(n.fStep, indent + 2)
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
    if n.addressSpace != asRMEM:
      result.add idd("AddressSpace")
      result.add idd(n.addressSpace)
  of gpuAssign:
    if n.aOp != "=":
      result.add id("Op", n.aOp)
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
  of gpuEmit:
    for part in n.parts:
      case part.kind
      of peLiteral:
        result.add id(part.literal)
      of peExpr:
        result.add pretty(part.expr, indent + 2)
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
    when mutable:
      for el in mitems(ast.ifElifs):
        yield el.cond
        yield el.body
    else:
      for el in ast.ifElifs:
        yield el.cond
        yield el.body
    if ast.ifElse.kind != gpuDiscard:
      yield ast.ifElse
  of gpuTernary:
    ya(tCond)
    ya(tThen)
    ya(tElse)
  of gpuFor:
    ya(fStart)
    ya(fEnd)
    ya(fStep)
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
  of gpuEmit:
    when mutable:
      for part in mitems(ast.parts):
        if part.kind == peExpr:
          yield part.expr
    else:
      for part in ast.parts:
        if part.kind == peExpr:
          yield part.expr
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
  of gtBool, gtUint8, gtF8E4M3, gtF8E5M2:
    result = 1
  of gtUint16, gtInt16, gtFloat16, gtBf16:
    result = 2
  of gtUint32, gtInt32, gtFloat32:
    result = 4
  of gtUint64, gtInt64, gtFloat64:
    result = 8
  of gtSize_t:
    result = 8
  of gtPtr, gtUncheckedArray, gtVoidPtr:
    result = 8  # pointer size on 64-bit
  of gtString:
    result = 8  # pointer
  of gtStatic, gtCompileTimeValue:
    result = 0  # compile-time value, no runtime size
  of gtArray:
    # TODO: for generic/unresolved array, size is set to -1, but can that happen in practice?
    # Due to constant folding I would expect array length to always be resolved.
    result = t.aLen * size(t.aTyp)
  of gtObject, gtGenericInst:
    for f in t.fields:
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

func ident*(n: GpuAst): string =
  ## Returns the associated identifier (string) of the given symbol, a `gpuIdent` input
  doAssert n.kind == gpuIdent, "The input is not a `gpuIdent`, but a " & $n.kind
  result = n.symbol.name


template withoutSemicolon*(ctx: var GpuContext, body: untyped): untyped =
  if not ctx.skipSemicolon: # no-op when already skipping
    ctx.skipSemicolon = true
    body
    ctx.skipSemicolon = false
  else:
    body

func getInnerArrayLengths*(t: GpuType): string =
  ## Returns the lengths of the inner array types for a nested array.
  case t.kind
  of gtArray:
    let inner = getInnerArrayLengths(t.aTyp)
    result = &"[{$t.aLen}]"
    if inner.len > 0:
      result.add &"{inner}"
  else:
    result = ""

const Base58* = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  ## Base58 alphabet (no 0, O, I, l for readability and ambiguity avoidance).

func shortHash*(sigHash: int64): string =
  ## Encode a 64-bit signature hash as a 7-character base58 string.
  ## 58^7 = 2,204,715,403,072 (~2.2T namespace), sufficient for collision
  ## avoidance across all symbols in a GPU compilation unit.
  ##
  ## Returns: the 7-character string, leading '1' run for short hashes
  var n = uint64(sigHash)
  result = newString(7)
  for i in countdown(6, 0):
    result[i] = Base58[int(n mod 58)]
    n = n div 58
    if n == 0 and i == 0:
      break
    if n == 0:
      for j in 0 ..< i:
        result[j] = '1'
      break

func mangledHashName*(name, iSym: string): string =
  ## npHashSuffix display name, `name` + the 7-char base58 hash of `iSym`
  ## FNV-1a accumulation plus a splitmix64 finalizer: XOR into byte lanes
  ## let chars 8 positions apart cancel out
  var h: uint64 = 0xcbf29ce484222325'u64
  for c in iSym:
    h = (h xor uint64(ord(c))) * 0x100000001b3'u64
  h = (h xor (h shr 30)) * 0xbf58476d1ce4e5b9'u64
  h = (h xor (h shr 27)) * 0x94d049bb133111eb'u64
  h = h xor (h shr 31)
  if h == 0:
    h = 1
  result = name & "_" & shortHash(cast[int64](h))

func gpuTypeToShortString*(t: GpuType): string =
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
  of gtFloat16: result = "f16"
  of gtBf16:    result = "bf16"
  of gtF8E4M3:  result = "f8e4m3"
  of gtF8E5M2:  result = "f8e5m2"
  of gtBool:    result = "bool"
  of gtObject:
    if t.isTuple:
      result = mangledHashName("Tuple", $hash(t))
    else:
      result = $t.name
  of gtGenericInst:
    let base = if t.isTuple: "Tuple" else: t.gName
    result = mangledHashName(base, $hash(t))
  of gtVoidPtr: result = "void_ptr"
  of gtPtr:
    result = "ptr_" & gpuTypeToShortString(t.to)
  of gtStatic:
    result = $t.sValue
  else:
    result = $t.kind # fallback M-bM-^@M-^T safe but verbose

func isGlobal*(fn: GpuAst): bool =
  doAssert fn.kind == gpuProc, "Not a function, but: " & $fn.kind
  result = attGlobal in fn.pAttributes
