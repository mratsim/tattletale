# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-AST type resolution: expression and function type lookups and pointer/array peeling.

import std/[macros, tables]
import ./gpu_ast_datatypes
import ./gpu_type_constructors

func getPointeeKind*(t: GpuType): GpuTypeKind =
  ## Scalar kind at the end of a pointer/array/UA chain,
  ## gtInvalid when the type is unknown.
  if t.isNil:
    return gtInvalid
  var cur = t
  while cur.kind in {gtPtr, gtArray, gtUA}:
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
        of gtUA: result = arrType.uaTo
        else: result = arrType             # scalar pointee: p[i] is the pointee
      else:
        case arrType.kind
        of gtPtr: result = arrType.to
        of gtArray: result = arrType.aTyp
        of gtUA: result = arrType.uaTo
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
    elif arrT.kind == gtArray: arrT.aTyp
    elif arrT.kind == gtPtr: arrT.to
    elif arrT.kind == gtUA: arrT.uaTo
    else: nil
  of gpuObjConstr: n.ocType
  of gpuConv: n.convTo
  of gpuCast: n.cTo
  of gpuMaterialize: ctx.getDotParentTypeImpl(n.mExpr)
  of gpuDot:
    # Field type on the parent struct type (ptr/UA layers stripped),
    # for field-access chains.
    block:
      var pT = ctx.getDotParentTypeImpl(n.dParent)
      if pT != nil and pT.kind == gtPtr: pT = pT.to
      if pT != nil and pT.kind == gtUA: pT = pT.uaTo
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
  ## Struct type of a field-access base, with the pointer/UA layer stripped.
  var t = ctx.getDotParentTypeImpl(n)
  if t != nil and t.kind == gtPtr:
    t = t.to
  if t != nil and t.kind == gtUA:
    t = t.uaTo
  result = t

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
    else: initGpuPtrType(inner, false)
  of gpuDeref:
    let d = ctx.getOperandType(n.dOf)
    if not d.isNil and d.kind == gtPtr: d.to else: d
  of gpuIndex:
    let t = ctx.getOperandType(n.iArr)
    if t.isNil: nil
    else:
      case t.kind
      of gtArray: t.aTyp
      of gtUA: t.uaTo
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
    stderr.writeLine "[lowerSpans] WARNING: getLoweringExprType could not derive a type for " & $n.kind &
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
  of gtUA:
    result = t.uaTo.containsKind(kind)
  else:
    result = t.kind == kind

# ═══════════════════════════════════════════════════════════════════════
# Value address spaces
# ═══════════════════════════════════════════════════════════════════════

func getPtrFieldNames*(t: GpuType): seq[string] =
  case t.kind
  of gtObject, gtGenericInst:
    for f in t.fields:
      if f.typ.kind == gtPtr:
        result.add f.name
  else: discard

proc resolveIdentAddressSpace*(ctx: GpuContext, n: GpuAst): AddressSpace =
  var base = n
  while base != nil and base.kind != gpuIdent:
    case base.kind
    of gpuAddr: base = base.aOf
    of gpuDeref: base = base.dOf
    of gpuIndex: base = base.iArr
    of gpuDot: base = base.dParent
    of gpuCast: base = base.cExpr
    of gpuBlock: base = base.statements[^1]
    of gpuMaterialize: base = base.mExpr
    else: return asRMEM
  if base == nil or base.symbol == nil:
    return asRMEM
  ctx.symAddressSpaces.getOrDefault(base.symbol.iSym, asRMEM)

proc resolveFieldValueSpace(ctx: GpuContext, n: GpuAst): AddressSpace =
  case n.kind
  of gpuDot: ctx.resolveIdentAddressSpace(n)
  of gpuCast: ctx.resolveFieldValueSpace(n.cExpr)
  of gpuConv: ctx.resolveFieldValueSpace(n.convExpr)
  of gpuAddr: ctx.resolveFieldValueSpace(n.aOf)
  of gpuIndex: ctx.resolveFieldValueSpace(n.iArr)
  of gpuDeref: ctx.resolveFieldValueSpace(n.dOf)
  of gpuPrefix: ctx.resolveFieldValueSpace(n.pVal)
  of gpuBinOp: ctx.resolveFieldValueSpace(n.bLeft)
  of gpuMaterialize: ctx.resolveFieldValueSpace(n.mExpr)
  else: ctx.resolveIdentAddressSpace(n)

proc resolvePtrFieldAddressSpaces*(ctx: GpuContext, n: GpuAst): seq[AddressSpace] =
  for fname in getPtrFieldNames(n.ocType):
    var space = asRMEM
    for f in n.ocFields:
      if f.name == fname and f.value.kind != gpuDiscard:
        space = ctx.resolveFieldValueSpace(f.value)
        break
    result.add space

proc resolveInstanceVariantSpaces(ctx: GpuContext, n: GpuAst): seq[AddressSpace] =
  ## Pointer-field address-space tuple of the struct instance a dot accesses:
  ## - constructions carry it directly
  ## - variables through their init, else their storage space
  ## - other parents resolve to an empty tuple, no variant match
  case n.kind
  of gpuObjConstr:
    return ctx.resolvePtrFieldAddressSpaces(n)
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in ctx.objFieldAddressSpaces:
      return ctx.objFieldAddressSpaces[n.symbol.iSym]
    let space = ctx.resolveIdentAddressSpace(n)
    let ptype = ctx.getDotParentType(n)
    let cnt = if ptype != nil: getPtrFieldNames(ptype).len else: 0
    result = newSeq[AddressSpace](cnt)
    for i in 0 ..< cnt:
      result[i] = space
  else:
    result = @[]

proc resolveValueAddressSpace*(ctx: GpuContext, n: GpuAst): AddressSpace =
  if n == nil:
    return asRMEM
  case n.kind
  of gpuIdent:
    if n.symbol == nil:
      asRMEM
    elif n.symbol.iSym in ctx.symAddressSpaces:
      ctx.symAddressSpaces[n.symbol.iSym]
    elif n.symbol.symKind in {gsDeviceKernelParam, gsGlobalKernelParam}:
      # explicit `ptr T` is a device pointer, implicit `var T` is thread
      if n.symbol.typ != nil and n.symbol.typ.kind == gtPtr and
         not n.symbol.typ.implicit:
        asDevice
      else:
        asRMEM
    else:
      asRMEM
  of gpuAddr: ctx.resolveValueAddressSpace(n.aOf)
  of gpuCast: ctx.resolveValueAddressSpace(n.cExpr)
  of gpuConv: ctx.resolveValueAddressSpace(n.convExpr)
  of gpuBinOp: ctx.resolveValueAddressSpace(n.bLeft)
  of gpuDot:
    let ptype = ctx.getDotParentType(n.dParent)
    if ptype != nil and ptype in ctx.ptrFieldVariants and
        n.dParent.kind in {gpuIdent, gpuObjConstr}:
      let fieldIdx = getPtrFieldNames(ptype).find(n.dField.ident())
      if fieldIdx >= 0:
        let expected = ctx.resolveInstanceVariantSpaces(n.dParent)
        let vi = ctx.ptrFieldVariants[ptype].find(expected)
        if vi >= 0:
          let base = ctx.ptrFieldVariants[ptype][vi]
          if fieldIdx < base.len:
            return base[fieldIdx] # the instance variant's space of the field
    ctx.resolveValueAddressSpace(n.dParent)
  of gpuIndex: ctx.resolveValueAddressSpace(n.iArr)
  of gpuDeref: ctx.resolveValueAddressSpace(n.dOf)
  of gpuPrefix: ctx.resolveValueAddressSpace(n.pVal)
  of gpuMaterialize: ctx.resolveValueAddressSpace(n.mExpr)
  else: asRMEM

proc getParamAddressSpace(p: GpuParam): AddressSpace =
  if p.addressSpace != asRMEM:
    p.addressSpace
  elif p.typ != nil and p.typ.kind == gtPtr and not p.typ.implicit:
    asDevice
  else:
    asRMEM

proc collectValueAddressSpacesImpl(ctx: var GpuContext, n: GpuAst) =
  if n == nil:
    return
  case n.kind
  of gpuVar:
    if n.vName.symbol != nil:
      ctx.symAddressSpaces[n.vName.symbol.iSym] = n.addressSpace
      if n.vInit.kind == gpuObjConstr:
        ctx.objFieldAddressSpaces[n.vName.symbol.iSym] =
          ctx.resolvePtrFieldAddressSpaces(n.vInit)
  of gpuObjConstr:
    let spaces = ctx.resolvePtrFieldAddressSpaces(n)
    var variants = ctx.ptrFieldVariants.getOrDefault(n.ocType, @[])
    if spaces notin variants:
      variants.add spaces
    ctx.ptrFieldVariants[n.ocType] = variants
  else: discard
  for ch in n:
    ctx.collectValueAddressSpacesImpl(ch)

proc collectValueAddressSpaces*(ctx: var GpuContext) =
  ## Generic-fn params get authoritative entries at their clone sites (makeFnGeneric)
  for (_, fn) in ctx.fnTab.pairs:
    for p in fn.pParams:
      if p.ident.symbol != nil:
        ctx.symAddressSpaces[p.ident.symbol.iSym] = getParamAddressSpace(p)
    ctx.collectValueAddressSpacesImpl(fn)
  for (iSym, g) in ctx.globals.pairs:
    ctx.symAddressSpaces[iSym] = asDevice
  for blk in ctx.globalBlocks:
    ctx.collectValueAddressSpacesImpl(blk)
