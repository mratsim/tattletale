# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cirv2 datatypes: the Crucible IR v2 tree, Final-IR shaped.
##
## CIRv2 is a plain Nim object-variant tree usable at macro time.
## The shape encodes the normal form:
##
## - control flow is statement-based
## - `loop` takes an optional condition, a nil loopCond keeps the conditionless
##   shape that every target already emits
## - a call is an ordinary expression node: the builders bind statement calls
##   through `sCall` and value calls appear in any operand position
## - every typed operation states its type in the `(add Type a b)` discipline of the Leng grammar
##
## The frontend refuses a construct that it cannot lower. The validation passes
## check the semantics that the shape cannot encode.

import std/[macros, hashes, sequtils, sets, strutils, tables]

type
  AddressSpace* = enum
    ## Memory class a pointer or aptr addresses.
    caThread
    caThreadgroup
    caConstant
    caDevice

  CoordBuiltinKind* = enum
    ## Canonical coordinate builtin category. The alias canonicalization pass
    ## assigns the kind from the canonical name table below, `cbkNone` marks every
    ## symbol that is not a coordinate builtin.
    cbkNone
    cbkThreadPositionInGrid
    cbkThreadgroupPositionInGrid
    cbkThreadPositionInThreadgroup
    cbkThreadsPerThreadgroup
    cbkThreadgroupsPerGrid
    cbkThreadIndexInThreadgroup

  SynchroBuiltinKind* = enum
    ## Canonical synchronization builtin category, `sbkNone` when not a barrier.
    sbkNone
    sbkThreadgroupBarrier
    sbkThreadgroupBarrierDevice

  ReductionBuiltinKind* = enum
    ## Canonical subgroup-reduction builtin category, `rbkNone` when not a reduction.
    rbkNone
    rbkReduceAdd
    rbkReduceMax

  AtomicsBuiltinKind* = enum
    ## Canonical atomic builtin category, `abkNone` when not an atomic.
    abkNone
    abkAtomicFetchAdd

  ShuffleBuiltinKind* = enum
    ## Canonical cross-lane shuffle builtin category, `shkNone` when not a shuffle.
    shkNone
    shkSimdShuffle
    shkSimdShuffleXor

  CirTypeKind* = enum
    ## Closed set of type kinds, derived once at the frontend boundary and never
    ## re-derived by passes.
    ctVoid
    ctBool
    ctInt
    ctUint
    ctFloat
    ctVector
    ctPtr
    ctAptr
    ctStruct
    ctArray
    ctSpan
    ctProc

  CirField* = object
    ## One named field of a nominal struct type.
    name*: string
    typ*: CirType

  CirType* = ref object
    ## A CIRv2 type.
    ##
    ## Contract:
    ## - nominal struct types identify by `name`, all other kinds compare
    ##   structurally through `sameType`
    ## - `bits` is the bit width: 8, 16, 32, 64, or -1 for the machine word
    case kind*: CirTypeKind
    of ctInt, ctUint, ctFloat:
      bits*: int
    of ctVector:
      vN*: int
      vElem*: CirType
    of ctPtr, ctAptr:
      space*: AddressSpace
      pElem*: CirType
    of ctStruct:
      name*: string
      fields*: seq[CirField]
    of ctArray:
      aN*: int
      aElem*: CirType
    of ctSpan:
      sSpace*: AddressSpace
      sElem*: CirType
    of ctProc:
      params*: seq[CirType]
      ret*: CirType
    of ctVoid, ctBool:
      discard

  CirExprKind* = enum
    ## Expression vocabulary, Leng names.
    ##
    ## `eCall` is an ordinary value node, statement-position calls stay `sCall`
    ## and a call result may appear in any operand position:
    ##
    ## - a comparison operand
    ## - an `and`/`or` operand
    ## - a ternary branch
    ## - a loop condition
    eNil
    eTrue
    eFalse
    eSym
    eIntLit
    eFloatLit
    eDeref
    eAt
    ePat
    eDot
    eAdd
    eSub
    eMul
    eDiv
    eMod
    eShl
    eShr
    eBitAnd
    eBitOr
    eBitXor
    eEq
    eNeq
    eLt
    eLe
    eNot
    eNeg
    eAnd
    eOr
    eTernary
    eCast
    eConv
    eAddr
    eSizeof
    eOconstr
    eAconstr
    eCall

  CirExpr* = ref object
    ## A CIRv2 expression tree.
    ##
    ## Typed ops state their type in `opType` or the per-branch type field,
    ## comparisons state their operand type and yield `bool`,
    ## `ternary` states its own type,
    ## and `and`/`or` combine bool operands into a bool result value.
    case kind*: CirExprKind
    of eNil:
      nilType*: CirType
    of eSym:
      name*: string
      typ*: CirType
      builtin*: bool
      ## True for a symbol born from a `{.builtin.}` read or call. The frontend records
      ## the pragma there:
      ##
      ## - the alias canonicalization pass renames only builtin symbols, a local
      ##   or parameter that happens to share a builtin's name stays untouched
      ## - the pass assigns the five typed kinds below from the canonical name tables
      ## - downstream recognition dispatches on the kinds, never on names
      coordBuiltin*: CoordBuiltinKind
      ## Coordinate builtin kind, `cbkNone` when not a coordinate.
      synchroBuiltin*: SynchroBuiltinKind
      ## Synchronization builtin kind, `sbkNone` when not a barrier.
      reductionBuiltin*: ReductionBuiltinKind
      ## Reduction builtin kind, `rbkNone` when not a reduction intrinsic.
      atomicsBuiltin*: AtomicsBuiltinKind
      ## Atomic builtin kind, `abkNone` when not an atomic intrinsic.
      shuffleBuiltin*: ShuffleBuiltinKind
      ## Shuffle builtin kind, `shkNone` when not a shuffle intrinsic.
    of eIntLit:
      value*: int64
      litType*: CirType
    of eFloatLit:
      fValue*: float64
      fLitType*: CirType
    of eDeref:
      derefOperand*: CirExpr
    of eAt:
      atBase*, atIndex*: CirExpr
    of ePat:
      patBase*, patIndex*: CirExpr
    of eDot:
      dotBase*: CirExpr
      fieldName*: string
    of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
      opType*: CirType
      left*, right*: CirExpr
    of eNeg:
      unType*: CirType
      unOperand*: CirExpr
    of eEq, eNeq, eLt, eLe:
      cmpType*: CirType
      a*, b*: CirExpr
    of eNot:
      notOperand*: CirExpr
    of eAnd, eOr:
      andLeft*, andRight*: CirExpr
    of eTernary:
      ternType*: CirType
      cond*, thenE*, elseE*: CirExpr
    of eCast, eConv:
      castType*: CirType
      castOperand*: CirExpr
    of eAddr:
      addrType*: CirType
      addrOperand*: CirExpr
    of eSizeof:
      ofType*: CirType
    of eOconstr:
      ocType*: CirType
      kvFields*: seq[CirKv]
    of eAconstr:
      acType*: CirType
      elems*: seq[CirExpr]
    of eCall:
      callee*: CirExpr
      args*: seq[CirExpr]
    of eTrue, eFalse:
      discard

  CirKv* = object
    ## One `name = value` member of an object constructor.
    name*: string
    value*: CirExpr

  CirStmtKind* = enum
    ## Statement vocabulary, Leng names. Inline assembly is first-class (`sAsm`):
    ## a verbatim template spliced with named symbol operands.
    ## `{.emit.}` lowers to the same shape. It injects raw target-source text
    ## with operand symbols spliced by name.
    sVar
    sConst
    sAsgn
    sIte
    sLoop
    sRet
    sDiscard
    sCall
    sAsm
    sEmit

  CirStmt* = ref object
    ## A CIRv2 statement.
    ##
    ## Control flow is statement-based:
    ## - `ite` takes a bool condition and two statement lists
    ## - `loop` takes an optional bool condition in `loopCond`, a nil loopCond
    ##   keeps the conditionless shape that every target already emits
    ## - a `var` states the memory class of its declaration in `space`, `const` ignores `space`
    ##   (`caThread` is per-thread registers and the unannotated default)
    case kind*: CirStmtKind
    of sVar, sConst:
      name*: string
      declType*: CirType
      init*: CirExpr
      space*: AddressSpace
    of sAsgn:
      lhs*, rhs*: CirExpr
    of sIte:
      cond*: CirExpr
      thenBody*, elseBody*: seq[CirStmt]
    of sLoop:
      loopCond*: CirExpr
      ## Only `whileS` sets `loopCond`.
      loopBody*: seq[CirStmt]
      unroll*: bool
      ## `{.unroll.}` statement in the loop body: every printer emits a full-unroll directive in front of the loop.
      ## - Metal: `#pragma clang loop unroll(full)`
      ## - OpenCL: `#pragma unroll`
    of sRet:
      retVal*: CirExpr
    of sDiscard:
      discardVal*: CirExpr
    of sCall:
      callCallee*: CirExpr
      callArgs*: seq[CirExpr]
    of sAsm:
      asmParts*: seq[string]
      ## Verbatim template text, one element more than `asmOperands` has.
      ## Piece `asmParts[i]` precedes operand `asmOperands[i]`.
      asmOperands*: seq[CirExpr]
      ## Spliced operands, each a typed CIRv2 symbol.
    of sEmit:
      emitParts*: seq[string]
      ## Verbatim target-source text, one element more than `emitOperands` has.
      ## Piece `emitParts[i]` precedes operand `emitOperands[i]`. The text owns
      ## its own terminators. The printer appends no `;`.
      emitOperands*: seq[CirExpr]
      ## Spliced operands, each a typed CIRv2 symbol.

  CirParam* = object
    ## One kernel or proc parameter. Pointer parameters name their memory
    ## class inside the `ptr`/`aptr` type (`space`).
    name*: string
    typ*: CirType
    varParam*: bool
    ## Nim `var T` formal: the argument binds by reference at the call boundary.
    ##
    ## Contract:
    ## - the frontend marks it while the type is still the thread pointer
    ## - the GLSL lowering pass retypes the parameter to its pointee
    ## - the GLSL printer emits the `inout` qualifier on the mark

  CirProc* = ref object
    ## One proc or kernel: name, typed params, return type, statement body.
    name*: string
    params*: seq[CirParam]
    ret*: CirType
    body*: seq[CirStmt]
    isKernel*: bool
    ## `{.global.}` proc: a compute entry point. The printers emit the backend's kernel qualifier and its argument ABI for it.
    inline*: bool
    ## `{.inline.}` proc: the printers emit the target inline qualifier on it.
    exportName*: string
    ## `{.exportc: "name".}` override of the emitted entry-point name, "" when the proc's own name is used.
    workgroup*: array[3, int]
    ## `{.workgroup(x, y, z)}` baked launch shape for the targets that compile it into the shader (Vulkan, WebGPU), all 0 when absent.

  ModuleFeatures* = object
    ## Feature metadata of one module, input to every text target's extension block.
    ##
    ## - the feature-scan pass fills the record after the target's last
    ##   transform
    ## - each printer maps the flags it reads to its own extension block
    ## - flags a printer does not read stay unset
    fp64*, fp16*: bool
    ## A 64-bit / 16-bit float value occurs in the module.
    int8*, int16*, int64*: bool
    ## An 8-, 16-, or 64-bit integer value occurs in the module.
    subgroup*: bool
    ## A subgroup reduction (`reduceAdd`/`reduceMax`) occurs in the module.
    subgroupShuffle*: bool
    ## A cross-lane shuffle (`simd_shuffle`/`simd_shuffle_xor`) occurs in the module.
    atomicFloat*: bool
    ## A floating-point `atomicFetchAdd` on 32-bit float storage occurs in the module.
    controlFlow*: bool
    ## A `{.unroll.}`-marked loop occurs in the module.
    reductions*: set[ReductionBuiltinKind]
    ## Subgroup reduction kinds in use.
    ##
    ## - the warp-reduction template list of the targets that emit one
    ##   template per kind

  StructLayout* = object
    ## Packed layout of one nominal struct type, computed under the launch packing rules.
    ##
    ## - each member starts at the first offset that satisfies the member
    ##   alignment
    ## - the struct size rounds up to the struct alignment
    size*: int
    ## Padded struct size in bytes. The value `eSizeof` reports.
    align*: int
    ## Strictest member alignment in bytes, at least 1.

  CallGraph* = object
    ## Call graph over the device functions of one module, the stored record of the target's call-graph analysis pass.
    ##
    ## - `reached` marks every `fn/param` key stored through its proc directly or through a transitive callee
    ## - a buffer parameter whose `fn/param` key is in `reached` is read-write
    stored*: HashSet[string]
    ## `fn/param` keys whose parameter the proc stores through directly.
    reached*: HashSet[string]
    ## `fn/param` keys reachable from `stored` over the argument-flow edges.

  CirModule* = ref object
    ## One CIRv2 module: nominal type declarations plus the proc set.
    name*: string
    types*: seq[CirType]
    procs*: seq[CirProc]
    globals*: seq[CirStmt]
    ## Program-scope declarations hoisted out of a kernel body by a target
    ## lowering pass: `var` statements of a threadgroup or constant memory class.
    ## Populated in kernel-body order, emitted before the proc set.
    ## Empty on every module a lowering pass has not touched.
    features*: ModuleFeatures
    ## Feature scan result of the target's feature-scan analysis pass,
    ## filled after that target's last transform.
    ##
    ## - valid only if no transform runs after the scan, which runPasses enforces
    ## - the printer reads it, it never rescans
    callGraph*: CallGraph
    ## Call-graph analysis record: the read-write classification of every buffer parameter.
    ##
    ## - filled by the target's call-graph analysis pass after the last transform
    ## - valid only if no transform runs after the analysis, which runPasses enforces
    ## - the printer reads the stored record, it never rescans
    refBlocks*: OrderedTable[string, CirType]
    ## GLSL buffer-reference blocks, block name -> pointee type.
    ## One entry per distinct block, in registration order.
    ## Filled by the GLSL ref-block analysis pass after its last transform.
    ##
    ## - valid only if no transform runs after the pass, which runPasses enforces
    ## - the printer emits one `layout(buffer_reference, std430) buffer`
    ##   declaration per entry in order, it never registers a block itself
    layouts*: Table[string, StructLayout]
    ## Packed layout of each nominal struct type, keyed by struct name,
    ## filled by the target's layout analysis pass after its last transform.
    ##
    ## - valid only if no transform runs after the layout pass, which runPasses enforces
    ## - the printer reads the stored records, it never recomputes a struct
    ##   layout

## Canonical name -> builtin-kind tables. The alias canonicalization pass reads
## them after it renames an alias to its canonical name. The cross-check against
## the catalog declarations lives in the pass module, which is the only importer of both sides.

let CoordBuiltinKindByName* {.compileTime.}: Table[string, CoordBuiltinKind] = {
  "thread_position_in_grid": cbkThreadPositionInGrid,
  "threadgroup_position_in_grid": cbkThreadgroupPositionInGrid,
  "thread_position_in_threadgroup": cbkThreadPositionInThreadgroup,
  "threads_per_threadgroup": cbkThreadsPerThreadgroup,
  "threadgroups_per_grid": cbkThreadgroupsPerGrid,
  "thread_index_in_threadgroup": cbkThreadIndexInThreadgroup,
}.toTable()

## Canonical name of each coordinate builtin kind, "" for `cbkNone`.
## Inverse of `CoordBuiltinKindByName`.
##
## Consumers that synthesize a coordinate-builtin reference read the name
## from here: the Metal builtin-threading pass names its parameters
## after the builtin.
let CoordBuiltinCanonicalName* {.compileTime.}: array[CoordBuiltinKind, string] = [
  cbkNone: "",
  cbkThreadPositionInGrid: "thread_position_in_grid",
  cbkThreadgroupPositionInGrid: "threadgroup_position_in_grid",
  cbkThreadPositionInThreadgroup: "thread_position_in_threadgroup",
  cbkThreadsPerThreadgroup: "threads_per_threadgroup",
  cbkThreadgroupsPerGrid: "threadgroups_per_grid",
  cbkThreadIndexInThreadgroup: "thread_index_in_threadgroup",
]

let SynchroBuiltinKindByName* {.compileTime.}: Table[string, SynchroBuiltinKind] = {
  "threadgroup_barrier": sbkThreadgroupBarrier,
  "threadgroup_barrier_device": sbkThreadgroupBarrierDevice,
}.toTable()

let ReductionBuiltinKindByName* {.compileTime.}: Table[string, ReductionBuiltinKind] = {
  "reduceAdd": rbkReduceAdd,
  "reduceMax": rbkReduceMax,
}.toTable()

let AtomicsBuiltinKindByName* {.compileTime.}: Table[string, AtomicsBuiltinKind] = {
  "atomicFetchAdd": abkAtomicFetchAdd,
}.toTable()

let ShuffleBuiltinKindByName* {.compileTime.}: Table[string, ShuffleBuiltinKind] = {
  "simd_shuffle": shkSimdShuffle,
  "simd_shuffle_xor": shkSimdShuffleXor,
}.toTable()

proc coordBuiltinKind*(name: string): CoordBuiltinKind =
  ## Returns the coordinate builtin kind of a canonical name, `cbkNone` for every
  ## other name (locals, math intrinsics, shadowing symbols).
  CoordBuiltinKindByName.getOrDefault(name, cbkNone)

proc synchroBuiltinKind*(name: string): SynchroBuiltinKind =
  ## Returns the synchronization builtin kind of a canonical name, `sbkNone` otherwise.
  SynchroBuiltinKindByName.getOrDefault(name, sbkNone)

proc reductionBuiltinKind*(name: string): ReductionBuiltinKind =
  ## Returns the reduction builtin kind of a canonical name, `rbkNone` otherwise.
  ReductionBuiltinKindByName.getOrDefault(name, rbkNone)

proc atomicsBuiltinKind*(name: string): AtomicsBuiltinKind =
  ## Returns the atomic builtin kind of a canonical name, `abkNone` otherwise.
  AtomicsBuiltinKindByName.getOrDefault(name, abkNone)

proc shuffleBuiltinKind*(name: string): ShuffleBuiltinKind =
  ## Returns the shuffle builtin kind of a canonical name, `shkNone` otherwise.
  ShuffleBuiltinKindByName.getOrDefault(name, shkNone)

proc withAddressSpace*(t: CirType, space: AddressSpace): CirType =
  ## Returns a copy of pointer type `t` re-spaced to `space`, with the pointee shared.
  ## CirType values are shared refs: a caller never re-spaces in place, so the copy is the per-site instance.
  doAssert t != nil and t.kind in {ctPtr, ctAptr}
  result = CirType(kind: t.kind)
  result.space = space
  result.pElem = t.pElem

proc structuralEqual*(a, b: CirType): bool =
  ## Returns true when `a` and `b` carry the same structure:
  ## a struct compares by its field list, every other kind field by field.
  ## The emitted struct name never participates.
  if a.isNil != b.isNil:
    return false
  if a.isNil:
    return true
  if a.kind != b.kind:
    return false
  case a.kind
  of ctVoid, ctBool:
    result = true
  of ctInt, ctUint, ctFloat:
    result = a.bits == b.bits
  of ctVector:
    result = a.vN == b.vN and structuralEqual(a.vElem, b.vElem)
  of ctPtr, ctAptr:
    result = a.space == b.space and structuralEqual(a.pElem, b.pElem)
  of ctStruct:
    if a.fields.len != b.fields.len:
      return false
    result = true
    for i in 0 ..< a.fields.len:
      if a.fields[i].name != b.fields[i].name or
          not structuralEqual(a.fields[i].typ, b.fields[i].typ):
        return false
  of ctArray:
    result = a.aN == b.aN and structuralEqual(a.aElem, b.aElem)
  of ctSpan:
    result = a.sSpace == b.sSpace and structuralEqual(a.sElem, b.sElem)
  of ctProc:
    if a.params.len != b.params.len:
      return false
    for i in 0 ..< a.params.len:
      if not structuralEqual(a.params[i], b.params[i]):
        return false
    result = structuralEqual(a.ret, b.ret)

proc sameType*(a, b: CirType): bool =
  ## Returns true when `a` and `b` are the same type.
  ##
  ## Contract:
  ## - nominal struct types compare by name only, a struct is compatible with itself
  ## - every other kind compares field by field, recursively
  if a.isNil != b.isNil:
    return false
  if a.isNil:
    return true
  if a.kind != b.kind:
    return false
  case a.kind
  of ctVoid, ctBool:
    result = true
  of ctInt, ctUint, ctFloat:
    result = a.bits == b.bits
  of ctVector:
    result = a.vN == b.vN and sameType(a.vElem, b.vElem)
  of ctPtr, ctAptr:
    result = a.space == b.space and sameType(a.pElem, b.pElem)
  of ctStruct:
    result = a.name == b.name
  of ctArray:
    result = a.aN == b.aN and sameType(a.aElem, b.aElem)
  of ctSpan:
    result = a.sSpace == b.sSpace and sameType(a.sElem, b.sElem)
  of ctProc:
    if a.params.len != b.params.len:
      return false
    for i in 0 ..< a.params.len:
      if not sameType(a.params[i], b.params[i]):
        return false
    result = sameType(a.ret, b.ret)

proc getExprType*(e: CirExpr): CirType =
  ## Returns the expression's type, or nil when the node's shape does not determine one.
  ## Nil arises for a `deref` of a non-pointer, an `at` of a non-array, a `dot` naming no field of its struct.
  ##
  ## The shape-enforced kinds always produce a type: `eTrue`/`eFalse` yield `bool`,
  ## comparisons yield `bool`, `eSizeof` yields the machine word.
  case e.kind
  of eSym:
    result = e.typ
  of eIntLit:
    result = e.litType
  of eFloatLit:
    result = e.fLitType
  of eNil:
    result = e.nilType
  of eTrue, eFalse:
    result = CirType(kind: ctBool)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    result = e.opType
  of eNeg:
    result = e.unType
  of eEq, eNeq, eLt, eLe, eNot, eAnd, eOr:
    result = CirType(kind: ctBool)
  of eTernary:
    result = e.ternType
  of eCast, eConv:
    result = e.castType
  of eAddr:
    result = e.addrType
  of eSizeof:
    result = CirType(kind: ctUint, bits: -1)
  of eOconstr:
    result = e.ocType
  of eAconstr:
    result = e.acType
  of eDeref:
    if e.derefOperand != nil:
      let t = getExprType(e.derefOperand)
      if t != nil and t.kind == ctPtr:
        result = t.pElem
  of eAt:
    if e.atBase != nil:
      let t = getExprType(e.atBase)
      if t != nil and t.kind == ctArray:
        result = t.aElem
  of ePat:
    if e.patBase != nil:
      let t = getExprType(e.patBase)
      if t != nil and t.kind == ctAptr:
        result = t.pElem
      elif t != nil and t.kind == ctArray:
        # A pat over a base that a deref of a pointer-to-array produced reads one element of that array.
        result = t.aElem
      elif t != nil and t.kind == ctPtr and t.pElem != nil and
          t.pElem.kind == ctArray:
        # A pat over a pointer-to-array base reads one element of the array that the pointer names.
        result = t.pElem.aElem
      elif t != nil and t.kind == ctSpan:
        # A pat over a span base reads one element of the view. The span split
        # pass retypes the base symbol to the element pointer first.
        result = t.sElem
  of eDot:
    if e.dotBase != nil:
      let t = getExprType(e.dotBase)
      if t != nil and t.kind == ctStruct:
        for f in t.fields:
          if f.name == e.fieldName:
            return f.typ
  of eCall:
    if e.callee != nil:
      let t = getExprType(e.callee)
      if t != nil and t.kind == ctProc:
        result = t.ret

proc getExprChildren*(e: CirExpr): seq[CirExpr] =
  ## Returns the child expressions of `e`, one entry per operand position.
  case e.kind
  of eAt: result = @[e.atBase, e.atIndex]
  of ePat: result = @[e.patBase, e.patIndex]
  of eDot: result = @[e.dotBase]
  of eDeref: result = @[e.derefOperand]
  of eAddr: result = @[e.addrOperand]
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    result = @[e.left, e.right]
  of eEq, eNeq, eLt, eLe: result = @[e.a, e.b]
  of eAnd, eOr: result = @[e.andLeft, e.andRight]
  of eTernary: result = @[e.cond, e.thenE, e.elseE]
  of eNot: result = @[e.notOperand]
  of eNeg: result = @[e.unOperand]
  of eCast, eConv: result = @[e.castOperand]
  of eOconstr:
    for f in e.kvFields:
      result.add f.value
  of eAconstr: result = e.elems
  of eCall:
    result = @[e.callee]
    for a in e.args:
      result.add a
  of eNil, eTrue, eFalse, eIntLit, eFloatLit, eSym, eSizeof:
    result = @[]

proc hasCallDeep*(e: CirExpr): bool =
  ## Returns true when a call occurs anywhere in `e`.
  ##
  ## Eager-evaluation forms (a conditional value, WGSL's `select`)
  ## run both variants, a call in either would run unconditionally.
  if e.isNil:
    return false
  if e.kind == eCall:
    return true
  for c in e.getExprChildren():
    if c.hasCallDeep():
      return true

func resolveAddressSpace*(e: CirExpr): AddressSpace =
  ## Address space of the storage an expression's pointer provenance names:
  ## the innermost pointer type along the access chain.
  ##
  ## - provenance survives element reads, field reads, value casts, and integer
  ##   arithmetic over cast pointers
  ## - `caDevice` reads as no provenance: a plain integer or an array value
  ##   derives the unprovenanced default
  ##
  ## One walker serves both consumers:
  ##
  ## - the frontend stamps a pointer cast's target space from this derivation at the cast's birth
  ## - the cast validation reads the same derivation over the module's expressions
  result = caDevice
  case e.kind
  of eSym:
    let t = e.typ
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.space
  of eDeref, eAt, ePat:
    # element reads live where the base's storage lives
    let base = case e.kind
      of eDeref: e.derefOperand
      of eAt: e.atBase
      else: e.patBase
    if base != nil:
      result = resolveAddressSpace(base)
  of eDot:
    let t = getExprType(e)
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.space
    elif e.dotBase != nil:
      result = resolveAddressSpace(e.dotBase)
  of eCast, eConv:
    let t = e.castType
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.space
    elif e.castOperand != nil:
      result = resolveAddressSpace(e.castOperand)
  of eAddr:
    if e.addrType != nil and e.addrType.kind in {ctPtr, ctAptr}:
      result = e.addrType.space
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    result = resolveAddressSpace(e.left)
    if result == caDevice:
      result = resolveAddressSpace(e.right)
  else:
    discard

func isLvalue*(e: CirExpr): bool =
  ## An expression that names storage: a symbol, a dereference, an index,
  ## a pointer index, or a field access. `addr` and assignment take lvalues.
  e.kind in {eSym, eDeref, eAt, ePat, eDot}

# ═══════════════════════════════════════════════════════════════════════
# Emitted-name vocabulary
# ═══════════════════════════════════════════════════════════════════════

const Base58Alphabet* = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  ## Base58 alphabet: no 0/O/I/l for readability in emitted device code.

func sanitizeIdent*(s: string): string =
  ## Returns `s` with every character invalid in a target identifier replaced
  ## by an underscore: Nim gensym backticks and the punctuation of structural
  ## type text (`(int, int)` derives `_int__int_`).
  result = newStringOfCap(s.len)
  for c in s:
    result.add(if c in {'a'..'z', 'A'..'Z', '0'..'9', '_'}: c else: '_')

func mangleOperatorName*(s: string): string =
  ## Returns `s` with operator punctuation replaced by words:
  ## `*=` becomes `star_eq`, `+` becomes `plus`.
  ## Any other non-identifier character becomes `X` plus its two-digit hex code.
  ## An operator is not a legal identifier in the target languages.
  result = newStringOfCap(s.len + 1)
  if s.len > 0 and s[0] in Digits:
    result.add "X"
  template special(word: string) =
    if result.len > 0 and result[^1] != '_':
      result.add '_'
    result.add word
  for c in s:
    case c
    of 'a'..'z', '0'..'9', 'A'..'Z', '_':
      result.add c
    of '$': special "dollar"
    of '%': special "percent"
    of '&': special "amp"
    of '^': special "roof"
    of '!': special "emark"
    of '?': special "qmark"
    of '*': special "star"
    of '+': special "plus"
    of '-': special "minus"
    of '/': special "slash"
    of '\\': special "backslash"
    of '=': special "eq"
    of '<': special "lt"
    of '>': special "gt"
    of '~': special "tilde"
    of ':': special "colon"
    of '.': special "dot"
    of '@': special "at"
    of '|': special "bar"
    else:
      special "X" & toHex(ord(c), 2)

func base58Short*(h: uint64): string =
  ## Returns the 7-character base58 encoding of `h`, left-padded with `1` while `h` is small.
  ## 58^7 is about 2.2e12: enough collision avoidance for one compilation unit's symbol names.
  var n = h
  result = newString(7)
  for i in countdown(6, 0):
    result[i] = Base58Alphabet[int(n mod 58)]
    n = n div 58
    if n == 0 and i == 0:
      break
    if n == 0:
      for j in 0 ..< i:
        result[j] = '1'
      break

func typeIdentityHash*(s: string): uint64 =
  ## Returns the 64-bit FNV-1a hash of `s` with a splitmix64 finalizer:
  ## the value distributes uniformly over all 64 bits so the base58 digits
  ## never cluster on short inputs.
  var h: uint64 = 0xcbf29ce484222325'u64
  for c in s:
    h = (h xor uint64(ord(c))) * 0x100000001b3'u64
  h = (h xor (h shr 30)) * 0xbf58476d1ce4e5b9'u64
  h = (h xor (h shr 27)) * 0x94d049bb133111eb'u64
  h = h xor (h shr 31)
  if h == 0:
    h = 1
  result = h
