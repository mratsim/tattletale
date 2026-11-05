# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, hashes, sequtils, sets, strutils, tables]
import ../../builtins/builtins_catalog
export builtins_catalog

# ############################################################
#
#            CIRv2 language datatypes
#
# ############################################################

type
  AddressSpace* = enum
    ## Memory class a pointer or aptr addresses.
    caThread
    caThreadgroup
    caConstant
    caDevice

  CirTypeKind* = enum
    ## Closed set of type kinds, derived once at the frontend boundary and never
    ## re-derived by passes.
    ctVoid
    ctBool
    ctString
    ctInt
    ctUint
    ctFloat
    ctBf16
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
    of ctInt, ctUint, ctFloat, ctBf16:
      bits*: int
    of ctVector:
      vN*: int
      vElem*: CirType
    of ctPtr, ctAptr:
      addressSpace*: AddressSpace
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
    of ctVoid, ctBool, ctString:
      discard

  CirExprKind* = enum
    ## Expression vocabulary, Leng names.
    ##
    ## `eCall` is an ordinary value node, statement-position calls stay `sCall`
    ## and a call result may appear in any operand position.
    eNil
    eTrue
    eFalse
    eSym
    eIntLit
    eFloatLit
    eStrLit
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
    eStmts

  CirExpr* = ref object
    ## A CIRv2 expression tree. Comparisons and `and`/`or` yield `bool`.
    case kind*: CirExprKind
    of eNil:
      nilType*: CirType
    of eSym:
      name*: string
      typ*: CirType
      builtin*: bool
      ## True for a symbol born from a `{.builtin.}` read or call, false for a local or parameter sharing a builtin's name.
      coordBuiltin*: CirCoordBuiltinKind
      synchroBuiltin*: CirSynchroBuiltinKind
      reductionBuiltin*: CirReductionBuiltinKind
      shuffleBuiltin*: CirShuffleBuiltinKind
      atomicsBuiltin*: CirAtomicsBuiltinKind
    of eIntLit:
      value*: int64
      litType*: CirType
    of eFloatLit:
      fValue*: float64
      fLitType*: CirType
    of eStrLit:
      sValue*: string
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
    of eStmts:
      stmts*: seq[CirStmt]
      valueExpr*: CirExpr
    of eTrue, eFalse:
      discard

  CirKv* = object
    ## One `name = value` member of an object constructor.
    name*: string
    value*: CirExpr

  CirStmtKind* = enum
    ## Statement vocabulary, Leng names.
    sVar
    sAsgn
    sIte
    sWhile
    sRet
    sDiscard
    sCall
    sAsm
    sEmit

  CirStmt* = ref object
    ## A CIRv2 statement.
    case kind*: CirStmtKind
    of sVar:
      name*: string
      declType*: CirType
      init*: CirExpr
      addressSpace*: AddressSpace
    of sAsgn:
      lhs*, rhs*: CirExpr
    of sIte:
      cond*: CirExpr
      thenBody*, elseBody*: seq[CirStmt]
    of sWhile:
      whileCond*: CirExpr
      whileBody*: seq[CirStmt]
      whileUnroll*: bool
      ## Every printer emits a full-unroll directive in front of the loop.
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
    ## class inside the `ptr`/`aptr` type (`addressSpace`).
    name*: string
    typ*: CirType
    varParam*: bool
    ## Nim `var T` formal: the argument binds by reference at the call boundary.
    ##
    ## Contract:
    ## - the frontend marks it while the type is still the thread pointer
    ## - the GLSL lowering pass retypes the parameter to its pointee
    ## - the GLSL printer emits the `inout` qualifier on the mark
    passByRef*: bool
    ## Pass by hidden const reference (large structs > 24 bytes), buffers and pointers never mark.

  CirProc* = ref object
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
    fp64*, fp16*: bool
    int8*, int16*, int64*: bool
    subgroup*: bool
    ## A subgroup reduction (`reduceAdd`/`reduceMax`) occurs in the module.
    subgroupShuffle*: bool
    ## A cross-lane shuffle (`simd_shuffle`/`simd_shuffle_xor`) occurs in the module.
    atomicFloat*: bool
    ## A floating-point `atomicFetchAdd` on 32-bit float storage occurs in the module.
    controlFlow*: bool
    ## A `{.unroll.}`-marked loop occurs in the module.
    reductions*: HashSet[string]
    ## Canonical names of the subgroup reductions in use.

  CallGraph* = object
    ## Call graph over the device functions of one module, filled by the target's call-graph analysis pass.
    ##
    ## - `reached` marks every `fn/param` key stored through its proc directly or through a transitive callee
    ## - a buffer parameter whose `fn/param` key is in `reached` is read-write
    stored*: HashSet[string]
    ## `fn/param` keys whose parameter the proc stores through directly.
    reached*: HashSet[string]
    ## `fn/param` keys reachable from `stored` over the argument-flow edges.

  CirModule* = ref object
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
    ## valid only if no transform runs after the scan, which runPasses enforces.
    callGraph*: CallGraph
    ## Call-graph analysis record, filled after the last transform,
    ## runPasses enforces the timing (see `features` for the contract).
    refBlocks*: OrderedTable[string, CirType]
    ## GLSL buffer-reference blocks, block name -> pointee type.
    ## One entry per distinct block in registration order,
    ## filled by the GLSL ref-block analysis pass, runPasses enforces the timing.
    ## The printer emits one `layout(buffer_reference, std430) buffer` per entry, it never registers a block itself.
    verifyIR*: bool
    ## Pass-boundary verification, on by default under `-d:CIR_verify`.

proc computeCirSizeBytes(t: CirType): int =
  ## Byte size under the C layout with no alignment padding: a struct sums its fields, an array its elements, a pointer is one machine word.
  case t.kind
  of ctInt, ctUint, ctFloat, ctBf16:
    result = if t.bits > 0: t.bits div 8 else: 8
  of ctBool:
    result = 1
  of ctString:
    # String values store as `const char*`, the pointer is the storage
    result = 8
  of ctPtr, ctAptr:
    result = 8
  of ctArray:
    result = t.aN * computeCirSizeBytes(t.aElem)
  of ctVector:
    result = t.vN * computeCirSizeBytes(t.vElem)
  of ctStruct:
    result = 0
    for f in t.fields:
      result += computeCirSizeBytes(f.typ)
  of ctVoid, ctProc, ctSpan:
    result = 0

func isLargeStruct*(t: CirType): bool =
  ## True when the struct passes by hidden const reference: the threshold is 24 bytes, matching the C backend's 3-pointer rule.
  t != nil and t.kind == ctStruct and computeCirSizeBytes(t) > 24

proc withAddressSpace*(t: CirType, addressSpace: AddressSpace): CirType =
  ## Returns a copy of pointer type `t` re-spaced to `addressSpace`, with the pointee shared.
  ## CirType values are shared refs: a caller never re-spaces in place, so the copy is the per-site instance.
  doAssert t != nil and t.kind in {ctPtr, ctAptr}
  result = CirType(kind: t.kind)
  result.addressSpace = addressSpace
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
  of ctVoid, ctBool, ctString:
    result = true
  of ctInt, ctUint, ctFloat, ctBf16:
    result = a.bits == b.bits
  of ctVector:
    result = a.vN == b.vN and structuralEqual(a.vElem, b.vElem)
  of ctPtr, ctAptr:
    result = a.addressSpace == b.addressSpace and structuralEqual(a.pElem, b.pElem)
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
  of ctVoid, ctBool, ctString:
    result = true
  of ctInt, ctUint, ctFloat, ctBf16:
    result = a.bits == b.bits
  of ctVector:
    result = a.vN == b.vN and sameType(a.vElem, b.vElem)
  of ctPtr, ctAptr:
    result = a.addressSpace == b.addressSpace and sameType(a.pElem, b.pElem)
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

func isScalarType*(t: CirType): bool =
  ## True for the primitive scalar kinds a primitive operator takes: bool, int, uint, float, bfloat16.
  t != nil and t.kind in {ctBool, ctInt, ctUint, ctFloat, ctBf16}

proc getExprType*(e: CirExpr): CirType =
  ## Returns the expression's type, or nil when the node's shape does not determine one.
  ## Nil arises for a `deref` of a non-pointer, an `at` of a non-array, a `dot` naming no field of its struct.
  case e.kind
  of eSym:
    result = e.typ
  of eIntLit:
    result = e.litType
  of eFloatLit:
    result = e.fLitType
  of eStrLit:
    result = CirType(kind: ctString)
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
  of eStmts:
    result = getExprType(e.valueExpr)

proc getExprChildren*(e: CirExpr): seq[CirExpr] =
  ## Returns the child expressions of `e`, one entry per operand position.
  case e.kind
  of eAt: result = @[e.atBase, e.atIndex]
  of ePat: result = @[e.patBase, e.patIndex]
  of eStrLit: discard
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
  of eStmts:
    result = @[e.valueExpr]
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
  ## Address space of the storage an expression names:
  ## the innermost pointer type along the access chain.
  ##
  ## - the space survives element reads, field reads, value casts, and integer
  ##   arithmetic over cast pointers
  ## - `caDevice` reads as no space: a scalar integer or an array value
  ##   keeps the device default
  result = caDevice
  case e.kind
  of eSym:
    let t = e.typ
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.addressSpace
  of eDeref, eAt, ePat:
    let base = case e.kind
      of eDeref: e.derefOperand
      of eAt: e.atBase
      else: e.patBase
    if base != nil:
      result = resolveAddressSpace(base)
  of eDot:
    let t = getExprType(e)
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.addressSpace
    elif e.dotBase != nil:
      result = resolveAddressSpace(e.dotBase)
  of eCast, eConv:
    let t = e.castType
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.addressSpace
    elif e.castOperand != nil:
      result = resolveAddressSpace(e.castOperand)
  of eAddr:
    if e.addrType != nil and e.addrType.kind in {ctPtr, ctAptr}:
      result = e.addrType.addressSpace
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    result = resolveAddressSpace(e.left)
    if result == caDevice:
      result = resolveAddressSpace(e.right)
  else:
    discard

func isLvalue*(e: CirExpr): bool =
  ## An expression that names storage, `addr` and assignment take lvalues.
  e.kind in {eSym, eDeref, eAt, ePat, eDot}

# ═══════════════════════════════════════════════════════════════════════
#  Emitted-name vocabulary
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
