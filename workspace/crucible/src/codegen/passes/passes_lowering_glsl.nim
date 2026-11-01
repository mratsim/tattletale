# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## GLSL lowering: per-target transforms for the GLSL storage model.
## The pipeline table in `gpu_compiler.nim` declares the registration order:
##
##   legalize → normalize → lowerSpans → glslReservedIdentifiers → glslQualifyVarParams
##   → glslExplodePointerLocals → smemInitRewrite → glslHoistProgramScopeLocals
##   → instantiateAddressSpaceGenerics → glslRegisterRefBlocks → glslScanModuleFeatures
##   → validation
##
## GLSL has no function-storage addressing, a local cannot be addressed.
## Threadgroup and constant-storage locals must live at program scope:
##
## - `glslQualifyVarParams`: `var T` device-function parameters bind as `inout`
## - the qualify pass retypes each marked parameter to its pointee, unwraps
##   the body derefs and rewrites the call sites to pass the plain lvalue
## - `glslExplodePointerLocals`: pointer locals rewrite to their pointee,
##   reads through the alias rewrite to the pointee symbol
## - `glslHoistProgramScopeLocals`: `{.smem.}` kernel locals and `{.const_mem.}`
##   locals relocate into `CirModule.globals` in encounter order
##
## `glslRegisterRefBlocks` fills `CirModule.refBlocks` with one entry per
## distinct buffer-reference block name, in registration order.
##
## Kernel and device-function pointer parameters stay for the printer to refuse.
##
## Program-scope `const` storage is visible to all functions, a device function
## reads the constants of the kernels that it serves.
##
## Refusal homes:
## - the pointer-local explosion pass refuses a pointer read before
##   its address-of binding, a value read of a pointer, or a rebinding
## - the printer refuses a device-space local and a nested-scope
##   or device-function threadgroup local, GLSL locals are private
## - the hoist pass refuses a const_mem local without an initializer,
##   the printer refuses it again as a shape guard

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ../backends/glsl_lang
import ./pass_datatypes
import ./pass_helpers
import ./passes_lowering_cfamily
import ./passes_low_pointers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerGlslQualifyVarParams*(reg: var PassRegistry) =
  reg.register("glslQualifyVarParams", pkTransform,
    "Retypes var parameters to their pointee, unwraps their body derefs and" &
    " the call-site address arguments",
    run = qualifyVarParams
  )

func registerGlslExplodePointerLocals*(reg: var PassRegistry) =
  registerExplodePointerLocals(reg, "glslExplodePointerLocals",
    "Rewrites pointer locals to their pointee locals, reads through the" &
    " alias to the pointee symbol",
    newExplodeCtx, explodeVarDecl, explodeAsgn,
    explodeDropsVar, explodeDropsAsgn, rewritePointerExpr)

func registerGlslHoistProgramScopeLocals*(reg: var PassRegistry) =
  ## Relocated declarations use the renamed names that their body already references.
  registerCfamilyHoistProgramScopeLocals(reg, "glslHoistProgramScopeLocals",
    "Relocates smem and const_mem kernel locals into the module's" &
    " program scope")

# ═══════════════════════════════════════════════════════════════════════
# var-parameter qualification
# ═══════════════════════════════════════════════════════════════════════

proc splicePlainSym(e: CirExpr, name: string, typ: CirType) =
  ## Expressions are refs, so the splice applies to the node every sharer sees.
  e.kind = eSym
  e.name = name
  e.typ = typ
  e.builtin = false

proc unwrapVarDerefsExpr(e: CirExpr, varParams: HashSet[string]) =
  ## Every deref of a `var` parameter splices to the parameter itself at the pointee type. Other derefs survive.
  if e.isNil:
    return
  case e.kind
  of eDeref:
    let op = e.derefOperand
    if op.kind == eSym and op.name in varParams and
        op.typ != nil and op.typ.kind == ctPtr:
      e.spliceSym(op.name, op.typ.pElem)
    else:
      op.unwrapVarDerefsExpr(varParams)
  else:
    for ch in getExprChildren(e):
      ch.unwrapVarDerefsExpr(varParams)

proc unwrapVarDerefsStmts(ss: seq[CirStmt], varParams: HashSet[string]) =
  ss.walkStmts(nil, proc(e: CirExpr) =
    e.unwrapVarDerefsExpr(varParams))

proc rewriteVarCallArgsExpr(fn: CirProc, e: CirExpr, varParams: HashSet[string], procs: Table[string, CirProc]): string =
  ## Anything at a `var` parameter position that is neither an address-of
  ## wrapper nor the enclosing proc's own `var` parameter must be an lvalue,
  ## the rewrite refuses otherwise.
  if e.isNil:
    return ""
  if e.kind == eCall and e.callee.kind == eSym:
    let callee = procs.getOrDefault(e.callee.name)
    if callee != nil and e.args.len == callee.params.len:
      var hasVarParams = false
      for p in callee.params:
        hasVarParams = hasVarParams or p.varParam
      for i, arg in mpairs(e.args):
        if not callee.params[i].varParam:
          continue
        if arg.kind == eAddr:
          e.args[i] = arg.addrOperand
        elif arg.kind == eSym and arg.name in varParams and
            arg.typ != nil and arg.typ.kind == ctPtr:
          arg.typ = arg.typ.pElem
        elif arg.kind notin {eSym, eAt, ePat, eDot}:
          return refuseAt(fn, "var parameter call argument must be an" &
            " lvalue, got " & $arg.kind)
      if hasVarParams:
        # a call site's callee type still states the pre-lowering signature, so the rewrite re-types it to the retyped parameters
        var ps: seq[CirType]
        for p in callee.params:
          ps.add p.typ
        e.callee.typ = freshProcType(ps, callee.ret)
    for ch in getExprChildren(e):
      let err = rewriteVarCallArgsExpr(fn, ch, varParams, procs)
      if err.len > 0:
        return err
    return ""
  for ch in getExprChildren(e):
    let err = rewriteVarCallArgsExpr(fn, ch, varParams, procs)
    if err.len > 0:
      return err
  return ""

proc rewriteVarCallArgsStmts(fn: CirProc, ss: var seq[CirStmt], varParams: HashSet[string], procs: Table[string, CirProc]): string =
  for s in mitems(ss):
    case s.kind
    of sVar, sConst:
      let err = rewriteVarCallArgsExpr(fn, s.init, varParams, procs)
      if err.len > 0:
        return err
    of sAsgn:
      let lerr = rewriteVarCallArgsExpr(fn, s.lhs, varParams, procs)
      if lerr.len > 0:
        return lerr
      let rerr = rewriteVarCallArgsExpr(fn, s.rhs, varParams, procs)
      if rerr.len > 0:
        return rerr
    of sIte:
      let cerr = rewriteVarCallArgsExpr(fn, s.cond, varParams, procs)
      if cerr.len > 0:
        return cerr
      let terr = rewriteVarCallArgsStmts(fn, s.thenBody, varParams, procs)
      if terr.len > 0:
        return terr
      let eerr = rewriteVarCallArgsStmts(fn, s.elseBody, varParams, procs)
      if eerr.len > 0:
        return eerr
    of sWhile:
      let cerr = rewriteVarCallArgsExpr(fn, s.whileCond, varParams, procs)
      if cerr.len > 0:
        return cerr
      let werr = rewriteVarCallArgsStmts(fn, s.whileBody, varParams, procs)
      if werr.len > 0:
        return werr
    of sRet:
      let rerr = rewriteVarCallArgsExpr(fn, s.retVal, varParams, procs)
      if rerr.len > 0:
        return rerr
    of sDiscard:
      let derr = rewriteVarCallArgsExpr(fn, s.discardVal, varParams, procs)
      if derr.len > 0:
        return derr
    of sCall:
      let tmp = CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
      let err = rewriteVarCallArgsExpr(fn, tmp, varParams, procs)
      s.callCallee = tmp.callee
      s.callArgs = tmp.args
      if err.len > 0:
        return err
    of sAsm, sEmit:
      discard
  return ""

proc qualifyVarParams(m: CirModule): string =
  ## Marked parameters print with the `inout` qualifier.
  var procs = initTable[string, CirProc]()
  for fn in m.procs:
    procs[fn.name] = fn
  # every parameter retypes before any body walks: a call site in a proc
  # preceding its callee must see the retyped signature
  for fn in m.procs:
    for p in mitems(fn.params):
      if p.varParam:
        if p.typ != nil and p.typ.kind == ctPtr:
          p.typ = p.typ.pElem
        else:
          return refuseAt(fn, "var parameter " & p.name &
            " without a pointer type")
  for fn in m.procs:
    var varParams: HashSet[string]
    for p in fn.params:
      if p.varParam:
        varParams.incl(p.name)
    fn.body.unwrapVarDerefsStmts(varParams)
    let err = rewriteVarCallArgsStmts(fn, fn.body, varParams, procs)
    if err.len > 0:
      return err
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Pointer-local explosion
# ═══════════════════════════════════════════════════════════════════════

proc isBindShape(e: CirExpr): bool =
  ## The pointee's name stands in for the pointer, only a composite local
  ## offers the reads that the alias resolves.
  e.kind == eAddr and e.addrOperand.kind == eSym and
    e.addrOperand.typ != nil and
    e.addrOperand.typ.kind in {ctStruct, ctArray}

proc bindPointee(e: CirExpr): string =
  ## Pointee local name of an address-of binding, `isBindShape` checked by the caller.
  e.addrOperand.name

proc rewritePointerExpr(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string =
  if e.isNil:
    return ""
  case e.kind
  of eSym:
    if e.name in ctx.pendingPtrs:
      return refuseAt(fn, "pointer local " & e.name &
        " read before its address-of binding, GLSL has no function-storage" &
        " addressing")
    if e.name in ctx.aliasOf:
      return refuseAt(fn, "pointer local " & e.name &
        " used as a value, GLSL has no pointer values")
    return ""
  of eDeref:
    let op = e.derefOperand
    if op.kind == eSym and op.name in ctx.aliasOf:
      # pointee stands in for the pointer and the deref unwraps to it
      e.spliceSym(ctx.aliasOf[op.name], op.typ.pElem)
      return ""
    for ch in getExprChildren(e):
      let err = rewritePointerExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    let ot = getExprType(e.derefOperand)
    if ot != nil and ot.kind == ctAptr:
      # a buffer-reference deref for the printer to emit as the indexed access
      return ""
    if e.derefOperand.kind == eSym and e.derefOperand.name in ctx.aliasOf:
      # a child rewrite spliced the operand to an aliased pointer's pointee
      let op2 = e.derefOperand
      e.spliceSym(ctx.aliasOf[op2.name], op2.typ.pElem)
      return ""
    return refuseAt(fn, "deref of a non-buffer-reference operand, no" &
      " pointee: GLSL has no function-storage addressing, the GLSL lowering" &
      " pass rewrites pointer locals to their pointees")
  of eAt, ePat:
    let base = (if e.kind == eAt: e.atBase else: e.patBase)
    if base.kind == eSym:
      if base.name in ctx.pendingPtrs:
        return refuseAt(fn, "pointer local " & base.name &
          " read before its address-of binding, GLSL has no function-storage" &
          " addressing")
      if base.name in ctx.aliasOf:
        # aliased local stands in for the pointer and the index goes to the pointee
        let pointee = sym(ctx.aliasOf[base.name], base.typ.pElem)
        if e.kind == eAt:
          e.atBase = pointee
        else:
          e.patBase = pointee
    for ch in getExprChildren(e):
      let err = rewritePointerExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    return ""
  of eAddr:
    for ch in getExprChildren(e):
      let err = rewritePointerExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    if e.addrOperand.kind == ePat:
      # a buffer element's address is the indexed reference access
      return ""
    return refuseAt(fn, "address-of has no GLSL form outside a buffer" &
      " element, GLSL has no function-storage addressing")
  else:
    for ch in getExprChildren(e):
      let err = rewritePointerExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    return ""

proc explodeVarDecl(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt): string =
  if s.declType.kind == ctPtr:
    if s.init == nil:
      ctx.pendingPtrs.incl(s.name)
    elif isBindShape(s.init):
      ctx.aliasOf[s.name] = s.init.bindPointee
    else:
      return refuseAt(fn, "pointer local " & s.name &
        " must bind the address of a local, GLSL has no function-storage" &
        " addressing")
    return ""
  return rewritePointerExpr(fn, ctx, s.init)

proc explodeAsgn(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt): string =
  if s.lhs.kind == eSym and s.lhs.typ != nil and s.lhs.typ.kind == ctPtr:
    if not isBindShape(s.rhs):
      return refuseAt(fn, "pointer local " & s.lhs.name &
        " must bind the address of a local, GLSL has no function-storage" &
        " addressing")
    if s.lhs.name in ctx.aliasOf:
      return refuseAt(fn, "pointer local " & s.lhs.name &
        " rebound, GLSL has no pointer values")
    ctx.pendingPtrs.excl(s.lhs.name)
    ctx.aliasOf[s.lhs.name] = s.rhs.bindPointee
    return ""
  let lerr = rewritePointerExpr(fn, ctx, s.lhs)
  if lerr.len > 0:
    return lerr
  return rewritePointerExpr(fn, ctx, s.rhs)

proc explodeDropsVar(s: CirStmt, ctx: ExplodeCtx): bool =
  s.declType.kind == ctPtr

proc explodeDropsAsgn(s: CirStmt, ctx: ExplodeCtx): bool =
  s.lhs.kind == eSym and s.lhs.typ != nil and s.lhs.typ.kind == ctPtr

proc newExplodeCtx(m: CirModule, fn: CirProc): ExplodeCtx =
  # GLSL's pointer model needs only the pending and alias tables, no by-reference parameters, no callee argument checks
  ExplodeCtx()


# ═══════════════════════════════════════════════════════════════════════
# Buffer-reference block registration
# ═══════════════════════════════════════════════════════════════════════

proc registerRefBlock(blocks: var OrderedTable[string, CirType], pointee: CirType) =
  ## Registers the buffer-reference block of one runtime-array pointee,
  ## deduped by block name.
  ##
  ## The block's member is the runtime array `x[]`: a fixed-array element nests
  ## after the runtime dimension (`uint8_t x[][32]`), as the unsized dimension is
  ## only legal in the outermost position. The printer composes the member text
  ## and refuses a void or proc pointee at the declaration site.
  let name = refTypeName(pointee)
  if name in blocks:
    return
  blocks[name] = pointee

proc registerTypeRefBlocks(blocks: var OrderedTable[string, CirType], t: CirType) =
  var found = blocks
  walkTypeTree(t, proc(tt: CirType) =
    if tt.kind == ctAptr:
      registerRefBlock(found, tt.pElem))
  blocks = found

proc registerExprRefBlocks(blocks: var OrderedTable[string, CirType], e: CirExpr) =
  ## Every buffer pointer subexpression's pointee needs its block declared
  ## before the module body prints, a cast target included.
  ## Node visits run pre-order, in the printer's declaration-sequence order.
  var found = blocks
  e.walkExprs(proc(e: CirExpr) =
    let t = getExprType(e)
    if t != nil and t.kind == ctAptr:
      registerRefBlock(found, t.pElem))
  blocks = found

proc registerStmtRefBlocks(blocks: var OrderedTable[string, CirType], ss: seq[CirStmt]) =
  var found = blocks
  proc onDecl(s: CirStmt): bool =
    if s.declType.kind == ctAptr:
      registerRefBlock(found, s.declType.pElem)
    true
  proc onExpr(e: CirExpr) =
    registerExprRefBlocks(found, e)
  ss.scanStmtRefs(onDecl, onExpr)
  blocks = found

proc glslRegisterRefBlocks(m: CirModule): string =
  ## Runs after the last transform: the walked declarations and bodies are
  ## final and the registry never invalidates the record.
  for t in m.types:
    registerTypeRefBlocks(m.refBlocks, t)
  for fn in m.procs:
    for p in fn.params:
      if p.typ.kind in {ctAptr, ctPtr}:
        registerRefBlock(m.refBlocks, p.typ.pElem)
    registerStmtRefBlocks(m.refBlocks, fn.body)
  return ""

proc registerGlslRefBlockPass*(reg: var PassRegistry) =
  ## - the pass fills `CirModule.refBlocks` with one entry per block name,
  ##   in registration order
  ## - the printer reads the stored entries and registers nothing itself
  reg.register("glslRegisterRefBlocks", pkAnalysis,
    "Registers the buffer-reference blocks that the printer declares, one" &
    " per distinct block name in registration order",
    run = glslRegisterRefBlocks
  )
