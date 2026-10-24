# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-AST type resolution: expression and function type lookups and pointer/array peeling.

import std/[macros, tables]
import ./ast_datatypes
import ./ast_type_constructors

func getPointeeKind*(t: GpuType): GpuTypeKind =
  ## Scalar kind at the end of a pointer, array or UncheckedArray chain,
  ## gtInvalid when the type is unknown.
  if t.isNil:
    return gtInvalid
  var cur = t
  while cur.kind in {gtPtr, gtArray, gtUncheckedArray}:
    cur = case cur.kind
          of gtPtr: cur.to
          of gtArray: cur.aTyp
          else: cur.uaTo
  result = cur.kind

proc getFnParams*(ctx: GpuContext, fn: GpuAst): seq[GpuParam] =
  ## Look up the parameters of a function by its identifier.
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
  elif fn in ctx.builtinFns:
    result = ctx.builtinFns[fn].pParams
  elif fn in ctx.processedProcs:
    result = ctx.processedProcs[fn].params

proc getFnReturnType*(ctx: GpuContext, fn: GpuAst): GpuType =
  ## Look up the return type of a function by its identifier.
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
  elif fn in ctx.builtinFns:
    result = ctx.builtinFns[fn].pRetType
  elif fn in ctx.processedProcs:
    result = ctx.processedProcs[fn].retType
  else:
    error "Function not found: " & $fn & " (name=" & fn.ident() & ")"

proc getExprType*(ctx: GpuContext, n: GpuAst): GpuType =
  ## Type of an expression node, erroring when no type is derivable.
  if n == nil:
    error "Cannot get type of nil node"
  case n.kind
  of gpuIdent:
    result = n.symbol.typ
    # nil/void symbol type is a construction defect, error at the read site.
    if result.isNil or result.kind == gtVoid:
      error "gpuIdent with nil or void symbol type: Cannot determine type for blit temp in " &
        "block expression (nil/void symbol.typ on a gpuIdent is a defect — idents reaching getExprType must carry a type)"
  of gpuLit: result = n.lType
  of gpuCall: result = ctx.getFnReturnType(n.cName)
  of gpuObjConstr: result = n.ocType
  of gpuBinOp:
    result = n.bType
    if result.isNil or result.kind == gtVoid:
      error "gpuBinOp with nil or void bType: Cannot determine type for blit temp in " &
        "block expression (nil/void bType on a gpuBinOp is a defect — all construction sites must populate bType)"
  of gpuConstexpr: result = n.cType
  of gpuMaterialize: result = n.mType
  of gpuConv: result = n.convTo
  of gpuCast: result = n.cTo
  of gpuBlock:
    if n.statements.len > 0:
      result = ctx.getExprType(n.statements[^1])
    else:
      error "Empty block expression"
  of gpuIndex:
    # `iArr` may be `gpuDeref(p)`, a by-ref var-array access `(*p)[i]`,
    # or a pointer-to-scalar `p[i]`. getExprType(gpuDeref) unwraps gtPtr
    # to the pointee, so dispatch on the pointee: a pointer-to-array
    # pointee is gtArray and indexing yields the element type, a scalar
    # pointee passes through (pointer arithmetic).
    # Direct array-ident indexing dispatches gtArray the same way.
    # Peeking at the deref operand instead would return the pointer,
    # and with it the array type for ptr-to-array, not the element.
    let arrType = ctx.getExprType(n.iArr)
    if arrType != nil:
      if n.iArr.kind == gpuDeref:
        # pointee already unwrapped, index it
        case arrType.kind
        of gtPtr: result = arrType.to      # ptr-to-ptr: (*p) is ptr T
        of gtArray: result = arrType.aTyp  # (*p)[i] → element type
        of gtUncheckedArray: result = arrType.uaTo
        else: result = arrType             # scalar pointee: p[i] is the pointee
      else:
        case arrType.kind
        of gtPtr:
          result = arrType.to
          if result.kind == gtArray:
            result = result.aTyp
          elif result.kind == gtUncheckedArray:
            result = result.uaTo
        of gtArray: result = arrType.aTyp
        of gtUncheckedArray: result = arrType.uaTo
        else:
          error "getExprType(gpuIndex): cannot get element type of " & $arrType.kind
  of gpuDeref:
    let dType = ctx.getExprType(n.dOf)
    if dType != nil and dType.kind == gtPtr:
      result = dType.to       # by-ref params: ident typed as ptr, deref is the pointee
    else:
      result = dType
  of gpuDot:
    let parentType = ctx.getExprType(n.dParent)
    if parentType != nil and parentType.kind in {gtObject, gtGenericInst}:
      let fields = parentType.fields
      for f in fields:
        if f.name == n.dField.symbol.name:
          result = f.typ
          break
    if result.isNil:
      error "getExprType(gpuDot): field '" & n.dField.symbol.name & "' not found in " & $n.dParent.kind
  of gpuAddr:
    result = ctx.getExprType(n.aOf)
  of gpuTernary:
    result = ctx.getExprType(n.tThen)
    if result.isNil or result.kind == gtVoid:
      error "gpuTernary with nil or void type: Cannot determine type for blit temp in " &
        "block expression (gpuTernary with an untyped then-branch is a defect — " &
        "if-expr branches are single typed expressions)"
  else:
    error "getExprType: unhandled node kind " & $n.kind & ". Caller must provide type from context."

proc getDotParentTypeImpl(ctx: GpuContext, n: GpuAst): GpuType =
  ## Type of an expression node, nil when unknown.
  ## Private helper of getDotParentType, which must tolerate
  ## partially-typed preprocessing ASTs where getExprType would error.
  case n.kind
  of gpuIdent: n.symbol.typ
  of gpuLit: n.lType
  of gpuBinOp: n.bType
  of gpuPrefix: ctx.getDotParentTypeImpl(n.pVal)
  of gpuCall: ctx.getFnReturnType(n.cName)
  of gpuAddr: ctx.getDotParentTypeImpl(n.aOf)
  of gpuDeref: ctx.getDotParentTypeImpl(n.dOf)
  of gpuIndex:
    let arrT = ctx.getDotParentTypeImpl(n.iArr)
    if arrT.isNil:
      nil
    elif arrT.kind == gtArray:
      arrT.aTyp
    elif arrT.kind == gtPtr:
      arrT.to
    elif arrT.kind == gtUncheckedArray:
      arrT.uaTo
    else:
      nil
  of gpuObjConstr: n.ocType
  of gpuConv: n.convTo
  of gpuCast: n.cTo
  of gpuMaterialize: ctx.getDotParentTypeImpl(n.mExpr)
  of gpuDot:
    # Field type on the parent struct type (ptr/UncheckedArray layers stripped),
    # for field-access chains.
    block:
      var pT = ctx.getDotParentTypeImpl(n.dParent)
      if pT != nil and pT.kind == gtPtr:
        pT = pT.to
      if pT != nil and pT.kind == gtUncheckedArray:
        pT = pT.uaTo
      if pT == nil or n.dField.kind != gpuIdent or n.dField.symbol == nil:
        nil
      else:
        let fields = pT.fields
        var found: GpuType = nil
        for f in fields:
          if f.name == n.dField.symbol.name:
            found = f.typ
            break
        found
  else: nil

proc getDotParentType*(ctx: GpuContext, n: GpuAst): GpuType =
  ## Struct type of a field-access base, with the pointer/UncheckedArray layer stripped.
  var t = ctx.getDotParentTypeImpl(n)
  if t != nil and t.kind == gtPtr:
    t = t.to
  if t != nil and t.kind == gtUncheckedArray:
    t = t.uaTo
  result = t

proc getOperandType*(ctx: GpuContext, n: GpuAst): GpuType

proc getTypeAccessedSpace*(t: GpuType): AddressSpace =
  ## Storage a value of this type lives in, the untagged default reads
  ## as thread. Scalar types carry a space when written (constant globals)
  if not t.isNil and t.addressSpace != asRMEM:
    result = t.addressSpace
  else:
    result = asRMEM

proc resolveAddressSpace*(ctx: GpuContext, n: GpuAst): AddressSpace =
  ## Memory the storage designated by an expression accesses
  ## Returns:
  ## - the address space of the innermost pointer/span/array type along the access chain, `p[i]` and `p` resolve to the same address space
  ## - the address space of a pointer-cast target, its operand derives it at birth
  ## - the storage a value-cast (`cast[uint64](p)`) or integer arithmetic over
  ##   one derives from, pointer provenance survives the round-trip
  ## - asRMEM for plain values
  case n.kind
  of gpuIndex:
    var container = n.iArr
    if container.kind == gpuDeref:
      container = container.dOf
    if container.kind == gpuDot:
      result = ctx.resolveAddressSpace(container)
    else:
      result = getTypeAccessedSpace(ctx.getOperandType(container))
  of gpuDot:
    let fieldTyp = ctx.getOperandType(n)
    if not fieldTyp.isNil and fieldTyp.kind in {gtPtr, gtSpan}:
      result = getTypeAccessedSpace(fieldTyp)
    else:
      result = ctx.resolveAddressSpace(n.dParent)
  of gpuDeref:
    result = getTypeAccessedSpace(ctx.getOperandType(n.dOf))
  of gpuCast:
    # a pointer/span/array target keeps its own address space, a value cast keeps the operand's storage provenance
    if n.cTo.kind in {gtPtr, gtSpan, gtArray}:
      result = getTypeAccessedSpace(n.cTo)
    else:
      result = ctx.resolveAddressSpace(n.cExpr)
  of gpuBinOp:
    result = getTypeAccessedSpace(ctx.getOperandType(n))
    if result == asRMEM:
      # integer arithmetic over cast pointers keeps the pointer's storage provenance
      result = ctx.resolveAddressSpace(n.bLeft)
      if result == asRMEM:
        result = ctx.resolveAddressSpace(n.bRight)
  else:
    result = getTypeAccessedSpace(ctx.getOperandType(n))

proc getOperandType*(ctx: GpuContext, n: GpuAst): GpuType =
  ## Type of an expression node, nil when undeterminable, never errors.
  ##
  ## - synthesized idents and blit temps may carry no type, those are
  ##   never pointers
  ## - callers skip the transform when nil
  case n.kind
  of gpuIdent:
    if n.symbol != nil: n.symbol.typ else: nil
  of gpuLit: n.lType
  of gpuBinOp: n.bType
  of gpuPrefix: ctx.getOperandType(n.pVal)  # `-x` keeps x's type
  of gpuCall: ctx.getFnReturnType(n.cName)
  of gpuCast: n.cTo
  of gpuConv: n.convTo
  of gpuObjConstr: n.ocType
  of gpuAddr:
    let inner = ctx.getOperandType(n.aOf)
    if inner.isNil or inner.kind == gtInvalid: nil
    else: initGpuPtrType(inner, false, ctx.resolveAddressSpace(n.aOf))
  of gpuDeref:
    let d = ctx.getOperandType(n.dOf)
    if not d.isNil and d.kind == gtPtr: d.to else: d
  of gpuIndex:
    let t = ctx.getOperandType(n.iArr)
    if t.isNil: nil
    else:
      case t.kind
      of gtArray: t.aTyp
      of gtUncheckedArray: t.uaTo
      of gtPtr: t.to
      else: nil
  of gpuDot:
    # Synthesized field idents carry no type: resolve through the parent struct
    if n.dField.kind == gpuIdent and n.dField.symbol != nil and
        n.dField.symbol.typ != nil and n.dField.symbol.typ.kind != gtVoid:
      n.dField.symbol.typ
    else:
      let parent = ctx.getOperandType(n.dParent)
      if parent.isNil or parent.kind notin {gtObject, gtGenericInst} or
          n.dField.kind != gpuIdent:
        nil
      else:
        for f in parent.fields:
          if f.name == n.dField.symbol.name:
            return f.typ
        nil
  else:
    nil

proc getLoweringExprType*(n: GpuAst): GpuType =
  ## Type of a lowering expression for span ptr/len args, never nil.
  ##
  ## - lowering sites build fresh gpuBinOps whose bType is read downstream,
  ##   the constructed `+ 1` literal's lType likewise:
  ##   a nil GpuType there is a FieldDefect
  ## - unresolvable shapes fall back to gtInt32 with a warning
  case n.kind
  of gpuIdent: result = n.symbol.typ
  of gpuLit: result = n.lType
  of gpuBinOp: result = n.bType
  of gpuConv: result = n.convTo
  else:
    discard
  if result.isNil or result.kind == gtVoid:
    # Span ptr/len args are ints in practice, a wrong-width fallback
    # must surface when it fires.
    result = initGpuType(gtInt32)
    warning "[lowerSpans] WARNING: getLoweringExprType could not derive a type for " & $n.kind &
      "; assuming gtInt32 for span index/pointer arithmetic"

func getStructType*(n: GpuAst): GpuType =
  ## Given an ident, return the struct type or void if not a struct.
  doAssert n.kind in [gpuIdent, gpuDeref]
  var p = n
  if p.kind == gpuDeref:
    p = n.dOf
  result = if p.symbol.typ.kind == gtPtr and p.symbol.typ.to.kind == gtObject:
             p.symbol.typ.to
           elif p.symbol.typ.kind == gtObject:
             p.symbol.typ
           else: GpuType(kind: gtVoid)

func containsKind*(t: GpuType, kind: GpuTypeKind): bool =
  ## True if the type tree contains `kind` somewhere.
  case t.kind
  of gtPtr:
    result = t.to.containsKind(kind)
  of gtArray:
    result = t.aTyp.containsKind(kind)
  of gtUncheckedArray:
    result = t.uaTo.containsKind(kind)
  else:
    result = t.kind == kind

