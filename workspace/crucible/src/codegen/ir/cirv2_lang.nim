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
## - `loop` has no condition slot
## - calls bind to their destination: a call exists only as a statement or as the RHS of `asgn`
## - every typed operation states its type in the `(add Type a b)` discipline of the Leng grammar
##
## The verifier in `cirv2_verify` enforces what the shape cannot.

import std/[macros, hashes, sequtils, sets, tables]

type
  CirAddrSpace* = enum
    ## Memory class a pointer or aptr addresses.
    ## Value-compatible with the codegen `AddressSpace` order (thread=RMEM, threadgroup=SMEM).
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
    ## - `origin` stores the frontend's source NimNode
    ##   (a fidelity and debugging aid, macro-time only, nil in hand-built fixtures)
    case kind*: CirTypeKind
    of ctInt, ctUint, ctFloat:
      bits*: int
    of ctVector:
      vN*: int
      vElem*: CirType
    of ctPtr, ctAptr:
      space*: CirAddrSpace
      pElem*: CirType
    of ctStruct:
      name*: string
      fields*: seq[CirField]
    of ctArray:
      aN*: int
      aElem*: CirType
    of ctProc:
      params*: seq[CirType]
      ret*: CirType
    of ctVoid, ctBool:
      discard
    origin*: NimNode

  CirExprKind* = enum
    ## Expression vocabulary, Leng names. `eCall` exists so a call can bind
    ## to its destination as the RHS of `asgn`. The verifier rejects it everywhere else.
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
    eBitNot
    eEq
    eNeq
    eLt
    eLe
    eNot
    eNeg
    eAnd
    eOr
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
    ## comparisons state their operand type and yield `bool`, and `and`/`or`
    ## are pure bool operators over side-effect-free operands.
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
      ## - the pass assigns the four typed kinds below from the canonical name tables
      ## - downstream recognition dispatches on the kinds, never on names
      coordBuiltin*: CoordBuiltinKind
      ## Coordinate builtin kind, `cbkNone` when not a coordinate.
      synchroBuiltin*: SynchroBuiltinKind
      ## Synchronization builtin kind, `sbkNone` when not a barrier.
      reductionBuiltin*: ReductionBuiltinKind
      ## Reduction builtin kind, `rbkNone` when not a reduction intrinsic.
      atomicsBuiltin*: AtomicsBuiltinKind
      ## Atomic builtin kind, `abkNone` when not an atomic intrinsic.
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
      inherit*: int
    of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
      opType*: CirType
      left*, right*: CirExpr
    of eBitNot, eNeg:
      unType*: CirType
      unOperand*: CirExpr
    of eEq, eNeq, eLt, eLe:
      cmpType*: CirType
      a*, b*: CirExpr
    of eNot:
      notOperand*: CirExpr
    of eAnd, eOr:
      andLeft*, andRight*: CirExpr
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

  CirRange* = object
    ## One inclusive literal range of a case branch.
    lo*: int64
    hi*: int64

  CirCaseBranch* = object
    ## One `of` variant of a case statement: disjoint literal ranges plus a statement body.
    ranges*: seq[CirRange]
    body*: seq[CirStmt]

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
    sCase
    sLab
    sJmp
    sScope
    sAssume
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
    ## - `loop` is conditionless: the shape has no condition slot
    ## - `lab`/`jmp` are forward-only and scoped, `scope` gives destructor discipline
    ## - a `var` states the memory class of its declaration in `space`, `const` ignores `space`
    ##   (`caThread` is per-thread registers and the unannotated default)
    case kind*: CirStmtKind
    of sVar, sConst:
      name*: string
      declType*: CirType
      init*: CirExpr
      space*: CirAddrSpace
    of sAsgn:
      lhs*, rhs*: CirExpr
    of sIte:
      cond*: CirExpr
      thenBody*, elseBody*: seq[CirStmt]
    of sLoop:
      loopBody*: seq[CirStmt]
      unroll*: bool
      ## `{.unroll.}` statement in the loop body: every printer emits a full-unroll directive in front of the loop.
      ## - Metal: `#pragma clang loop unroll(full)`
      ## - OpenCL: `#pragma unroll`
    of sCase:
      selector*: CirExpr
      branches*: seq[CirCaseBranch]
      caseElse*: seq[CirStmt]
    of sLab:
      labName*: string
    of sJmp:
      jmpTarget*: string
    of sScope:
      scopeBody*: seq[CirStmt]
    of sAssume:
      assumeCond*: CirExpr
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

  CirProc* = ref object
    ## One proc or kernel: name, typed params, return type, statement body.
    name*: string
    params*: seq[CirParam]
    ret*: CirType
    body*: seq[CirStmt]
    isKernel*: bool
    ## `{.global.}` proc: a compute entry point. The printers emit the backend's kernel qualifier and its argument ABI for it.
    inline*: bool
    ## `{.inline.}` proc: the inline-splice pass splices its call sites into
    ## the callers before printing.
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
    memberOffsets*: seq[int]
    ## Start offset of each member in declaration order.

  CallGraph* = object
    ## Call graph over the device functions of one module, the stored record of the target's call-graph analysis pass.
    ##
    ## - each edge records one argument flow: `caller` passes the argument rooted at parameter `arg` to callee parameter `param`
    ## - `reached` marks every `fn/param` key stored through its proc directly or through a transitive callee
    ## - a buffer parameter whose `fn/param` key is in `reached` is read-write
    procs*: Table[string, CirProc]
    ## Module procs by name.
    edges*: seq[tuple[caller, arg, callee, param: string]]
    ## One argument flow: `caller` passes the argument rooted at parameter
    ## `arg` to the `param` parameter of `callee`.
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
    ## - valid only if no transform runs after the scan: the pass registry
    ##   has no invalidation
    ## - the printer reads it, it never rescans
    callGraph*: CallGraph
    ## Call-graph analysis record: the argument-flow edges and the read-write classification of every buffer parameter.
    ##
    ## - filled by the target's call-graph analysis pass after the last transform
    ## - valid only if no transform runs after the analysis: the pass registry has no invalidation
    ## - the printer reads the stored record, it never rescans
    layouts*: Table[string, StructLayout]
    ## Packed layout of each nominal struct type, keyed by struct name,
    ## filled by the target's layout analysis pass after its last transform.
    ##
    ## - valid only if no transform runs after the layout pass: the pass
    ##   registry has no invalidation
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

proc withSpace*(t: CirType, space: CirAddrSpace): CirType =
  ## Returns a copy of pointer type `t` re-spaced to `space`, with the pointee shared.
  ## CirType values are shared refs: a caller never re-spaces in place, so the copy is the per-site instance.
  doAssert t != nil and t.kind in {ctPtr, ctAptr}
  result = CirType(kind: t.kind)
  result.space = space
  result.pElem = t.pElem
  result.origin = t.origin

proc sameType*(a, b: CirType): bool =
  ## Returns true when `a` and `b` are the same type.
  ##
  ## Contract:
  ## - nominal struct types compare by name only, a struct is compatible with itself
  ## - every other kind compares field by field, recursively
  ## - the `origin` NimNode never participates
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
  ## The verifier turns a nil result into a positioned rejection.
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
  of eBitNot, eNeg:
    result = e.unType
  of eEq, eNeq, eLt, eLe, eNot, eAnd, eOr:
    result = CirType(kind: ctBool)
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
  of eDot:
    if e.dotBase != nil:
      let t = getExprType(e.dotBase)
      if t != nil and t.kind == ctStruct and e.inherit == 0:
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
  of eNot: result = @[e.notOperand]
  of eNeg, eBitNot: result = @[e.unOperand]
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

func isLvalue*(e: CirExpr): bool =
  ## An expression that names storage: a symbol, a dereference, an index,
  ## a pointer index, or a field access. `addr` and assignment take lvalues.
  e.kind in {eSym, eDeref, eAt, ePat, eDot}
