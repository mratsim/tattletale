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

proc getPointeeKind*(t: GpuType): GpuTypeKind =
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
    raiseAssert "Function not found: " & $fn & " (name=" & fn.ident() & ")"

proc getExprType*(ctx: GpuContext, n: GpuAst): GpuType =
  ## Type of an expression node, erroring when no type is derivable.
  if n == nil: error "Cannot get type of nil node"
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

proc exprType*(ctx: GpuContext, n: GpuAst): GpuType =
  ## Best-effort type of an expression node, nil when unknown
  ##
  ## - printers use it to detect array-typed operands of `addr`
  ## - array-typed `addr` operands need pointer decay, not `&`
  case n.kind
  of gpuIdent: n.symbol.typ
  of gpuLit: n.lType
  of gpuBinOp: n.bType
  of gpuPrefix: ctx.exprType(n.pVal)
  of gpuCall: ctx.getFnReturnType(n.cName)
  of gpuAddr: ctx.exprType(n.aOf)
  of gpuDeref: ctx.exprType(n.dOf)
  of gpuIndex:
    let arrT = ctx.exprType(n.iArr)
    if arrT.isNil:
      nil
    elif arrT.kind == gtArray: arrT.aTyp
    elif arrT.kind == gtPtr: arrT.to
    elif arrT.kind == gtUA: arrT.uaTo
    else: nil
  of gpuObjConstr: n.ocType
  of gpuConv: n.convTo
  of gpuCast: n.cTo
  of gpuMaterialize: ctx.exprType(n.mExpr)
  of gpuDot:
    # Field type on the parent struct type (ptr/UA layers stripped),
    # generic field-type inference for field-access chains, consumed by
    # the value address-space resolution for nested pointer-field chains.
    block:
      var pT = ctx.exprType(n.dParent)
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
  var t = ctx.exprType(n)
  if t != nil and t.kind == gtPtr:
    t = t.to
  if t != nil and t.kind == gtUA:
    t = t.uaTo
  result = t

proc operandType*(ctx: GpuContext, n: GpuAst): GpuType =
  ## Best-effort type of a pattern operand.
  ## Returns nil when the pass cannot determine it,
  ## and the fold then conservatively skips.
  case n.kind
  of gpuIdent: n.symbol.typ
  of gpuLit: n.lType
  of gpuBinOp: n.bType
  of gpuPrefix: ctx.operandType(n.pVal)  # `-x` keeps x's type
  of gpuCall: ctx.getFnReturnType(n.cName)
  else: nil

proc getOperandType*(n: GpuAst): GpuType =
  ## Type of an expression for deref insertion, nil when undeterminable:
  ## synthesized idents and blit temps may carry no type, those are never pointers.
  case n.kind
  of gpuIdent:
    if n.symbol != nil: n.symbol.typ else: nil
  of gpuLit: n.lType
  of gpuCast: n.cTo
  of gpuConv: n.convTo
  of gpuObjConstr: n.ocType
  of gpuAddr:
    let inner = getOperandType(n.aOf)
    if inner.isNil or inner.kind == gtInvalid: nil
    else: initGpuPtrType(inner, false)
  of gpuDeref:
    let d = getOperandType(n.dOf)
    if not d.isNil and d.kind == gtPtr: d.to else: d
  of gpuIndex:
    let t = getOperandType(n.iArr)
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
      let parent = getOperandType(n.dParent)
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

proc exprGpuType*(n: GpuAst): GpuType =
  ## Best-effort GpuType of a lowering expression (span ptr/len args)
  ##
  ## - never nil: the lowering sites build fresh gpuBinOps whose bType
  ##   and the constructed `+ 1` literal's lType are read downstream,
  ##   a nil GpuType there raises a FieldDefect
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
    warning "[lowerSpans] WARNING: exprGpuType could not derive a type for " & $n.kind &
      "; assuming gtInt32 for span index/pointer arithmetic"

proc getStructType*(n: GpuAst): GpuType =
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

proc containsKind*(t: GpuType, kind: GpuTypeKind): bool =
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
