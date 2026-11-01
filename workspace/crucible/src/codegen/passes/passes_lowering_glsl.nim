# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## GLSL lowering: per-target transforms for the GLSL storage model,
## registered by the GLSL pipeline and run before printing.
##
##   kernel body → glslQualifyVarParams → glslExplodePointerLocals → glslHoistProgramScopeLocals
##   → glslRegisterRefBlocks → printed GLSL source
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
## - a pointer read before its address-of binding, a value read of a pointer,
##   or a rebinding refuses in the pointer-local explosion pass
## - a device-space local and a nested-scope or device-function threadgroup local
##   refuse at the printer, GLSL locals are private
## - a const_mem local without an initializer refuses in the hoist pass and again
##   at the printer as a shape guard

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ../backends/glsl_lang
import ./pass_datatypes
import ./pass_helpers
import ./passes_lowering_cfamily
import ./passes_low_address_spaces
import ./passes_low_pointers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerGlslLoweringPasses*(reg: var PassRegistry) =
  ## Registers the GLSL per-target lowering passes:
  ## - `glslExplodePointerLocals` then `glslHoistProgramScopeLocals`, in that order, before printing
  ##
  ## Registration contract: the passes sit after the reserved-word rename,
  ## and the relocated declarations use the renamed names that their body already references.
  reg.register("glslQualifyVarParams", pkTransform,
    "Retypes var parameters to their pointee, unwraps their body derefs and" &
    " the call-site address arguments",
    run = qualifyVarParams
  )
  glslRegisterExplodePointerLocals(reg)
  registerCfamilyHoistProgramScopeLocals(reg, "glslHoistProgramScopeLocals",
    "Relocates smem and const_mem kernel locals into the module's" &
    " program scope", "")
  reg.registerInstantiateAddressSpaceGenerics()

# ═══════════════════════════════════════════════════════════════════════
# var-parameter qualification
# ═══════════════════════════════════════════════════════════════════════

proc splicePlainSym(e: CirExpr, name: string, typ: CirType) =
  ## Splices one expression node into a plain symbol reference, in place.
  ## Expressions are refs, the rewrite applies to the node every sharer sees.
  e.kind = eSym
  e.name = name
  e.typ = typ
  e.builtin = false
  e.coordBuiltin = cbkNone
  e.synchroBuiltin = sbkNone
  e.reductionBuiltin = rbkNone
  e.shuffleBuiltin = shkNone
  e.atomicsBuiltin = abkNone

proc unwrapVarDerefsExpr(e: CirExpr, varParams: HashSet[string]) =
  ## Rewrites one expression tree. Every deref of a `var` parameter
  ## splices to the parameter itself, typed at the pointee the deref read.
  ## Other derefs survive. Nested bodies recurse through the statement overload.
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

proc unwrapVarDerefsStmts(ss: var seq[CirStmt], varParams: HashSet[string]) =
  ## Statement-walk overload of `unwrapVarDerefsExpr`, nested bodies included.
  for s in mitems(ss):
    case s.kind
    of sVar, sConst:
      s.init.unwrapVarDerefsExpr(varParams)
    of sAsgn:
      s.lhs.unwrapVarDerefsExpr(varParams)
      s.rhs.unwrapVarDerefsExpr(varParams)
    of sIte:
      s.cond.unwrapVarDerefsExpr(varParams)
      s.thenBody.unwrapVarDerefsStmts(varParams)
      s.elseBody.unwrapVarDerefsStmts(varParams)
    of sLoop:
      s.loopCond.unwrapVarDerefsExpr(varParams)
      s.loopBody.unwrapVarDerefsStmts(varParams)
    of sRet:
      s.retVal.unwrapVarDerefsExpr(varParams)
    of sDiscard:
      s.discardVal.unwrapVarDerefsExpr(varParams)
    of sCall:
      s.callCallee.unwrapVarDerefsExpr(varParams)
      for a in mitems(s.callArgs):
        a.unwrapVarDerefsExpr(varParams)
    of sAsm, sEmit:
      discard

proc rewriteVarCallArgsExpr(fn: CirProc, e: CirExpr, varParams: HashSet[string], procs: Table[string, CirProc]): string =
  ## Rewrites the call arguments of one expression tree:
  ##
  ## - an argument at a `var` parameter position drops its address-of wrapper
  ## - an argument naming the enclosing proc's own `var` parameter re-types to the pointee
  ## - anything else at a `var` position must be an lvalue
  ##
  ## Returns "" when clean.
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
  ## Statement-walk overload of `rewriteVarCallArgsExpr`, nested bodies included.
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
    of sLoop:
      let cerr = rewriteVarCallArgsExpr(fn, s.loopCond, varParams, procs)
      if cerr.len > 0:
        return cerr
      let lerr = rewriteVarCallArgsStmts(fn, s.loopBody, varParams, procs)
      if lerr.len > 0:
        return lerr
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
  ## Lowers the Nim `var T` parameters of every proc to GLSL inout bindings:
  ##
  ## - each marked parameter retypes to its pointee
  ## - its body derefs unwrap to the parameter
  ## - its call sites pass the plain lvalue
  ## - the printer emits the `inout` qualifier on the marked parameters
  ##
  ## Returns "" when clean, else the first refusal.
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
  ## Address-of binding shape: the address of a struct or array local.
  ## The pointee's name stands in for the pointer, only a composite local
  ## offers the reads that the alias resolves.
  e.kind == eAddr and e.addrOperand.kind == eSym and
    e.addrOperand.typ != nil and
    e.addrOperand.typ.kind in {ctStruct, ctArray}

proc bindPointee(e: CirExpr): string =
  ## Pointee local name of an address-of binding, `isBindShape` checked by the caller.
  e.addrOperand.name

proc rewritePointerExpr(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string =
  ## Rewrites one expression in place for the GLSL storage model.
  ## Returns "" when clean, else the first refusal.
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
      # pointee stands in for the pointer, the deref unwraps to it
      e.spliceSym(ctx.aliasOf[op.name], op.typ.pElem)
      return ""
    for ch in getExprChildren(e):
      let err = rewritePointerExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    let ot = getExprType(e.derefOperand)
    if ot != nil and ot.kind == ctAptr:
      # a buffer-reference deref, the printer emits the indexed access
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
        # aliased local stands in for the pointer, the index goes to the pointee
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
  ## `sVar` case: a pointer local pends for its address-of binding or binds its pointee,
  ## anything else rewrites the initializer.
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
  ## `sAsgn` case: a pointer local binds exactly once, anything else rewrites both operands.
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
  ## A bound or pending pointer local drops from its list with its binding.
  s.declType.kind == ctPtr

proc explodeDropsAsgn(s: CirStmt, ctx: ExplodeCtx): bool =
  ## A pointer-local binding drops from its list.
  s.lhs.kind == eSym and s.lhs.typ != nil and s.lhs.typ.kind == ctPtr

proc newExplodeCtx(m: CirModule, fn: CirProc): ExplodeCtx =
  ExplodeCtx()

proc glslRegisterExplodePointerLocals(reg: var PassRegistry) =
  ## Registers the GLSL pointer-local explosion pass over the shared walk.
  registerExplodePointerLocals(reg, "glslExplodePointerLocals",
    "Rewrites pointer locals to their pointee locals, reads through the" &
    " alias to the pointee symbol",
    newExplodeCtx, explodeVarDecl, explodeAsgn,
    explodeDropsVar, explodeDropsAsgn, rewritePointerExpr)

# ═══════════════════════════════════════════════════════════════════════
# Buffer-reference block registration
# ═══════════════════════════════════════════════════════════════════════

proc registerRefBlock(blocks: var OrderedTable[string, CirType], pointee: CirType) =
  ## Registers the buffer-reference block of one runtime-array pointee,
  ## deduped by block name.
  ##
  ## The block's member is the runtime array `x[]`: a fixed-array element nests
  ## after the runtime dimension (`uint8_t x[][32]`), the unsized dimension is
  ## only legal in the outermost position. The printer composes the member text
  ## and refuses a void or proc pointee at the declaration site.
  let name = refTypeName(pointee)
  if name in blocks:
    return
  blocks[name] = pointee

proc registerTypeRefBlocks(blocks: var OrderedTable[string, CirType], t: CirType) =
  ## Registers the buffer-reference blocks of every device pointer
  ## reachable from the type tree, through pointer targets and array
  ## elements and object fields, dedup by reference-block name.
  var found = blocks
  proc registerAptr(tt: CirType) =
    if tt.kind == ctAptr:
      registerRefBlock(found, tt.pElem)
  walkTypeTree(t, registerAptr)
  blocks = found

proc registerExprRefBlocks(blocks: var OrderedTable[string, CirType], e: CirExpr) =
  ## Registers the buffer-reference blocks reachable from one expression:
  ## every buffer pointer subexpression's pointee needs its block declared
  ## before the module body prints, a cast target included.
  if e.isNil:
    return
  let t = getExprType(e)
  if t != nil and t.kind == ctAptr:
    registerRefBlock(blocks, t.pElem)
  case e.kind
  of eCall:
    registerExprRefBlocks(blocks, e.callee)
    for a in e.args:
      registerExprRefBlocks(blocks, a)
  of eAt:
    registerExprRefBlocks(blocks, e.atBase)
    registerExprRefBlocks(blocks, e.atIndex)
  of ePat:
    registerExprRefBlocks(blocks, e.patBase)
    registerExprRefBlocks(blocks, e.patIndex)
  of eDot:
    registerExprRefBlocks(blocks, e.dotBase)
  of eDeref:
    registerExprRefBlocks(blocks, e.derefOperand)
  of eAddr:
    registerExprRefBlocks(blocks, e.addrOperand)
  of eCast, eConv:
    registerExprRefBlocks(blocks, e.castOperand)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    registerExprRefBlocks(blocks, e.left)
    registerExprRefBlocks(blocks, e.right)
  of eAnd, eOr:
    registerExprRefBlocks(blocks, e.andLeft)
    registerExprRefBlocks(blocks, e.andRight)
  of eTernary:
    registerExprRefBlocks(blocks, e.cond)
    registerExprRefBlocks(blocks, e.thenE)
    registerExprRefBlocks(blocks, e.elseE)
  of eEq, eNeq, eLt, eLe:
    registerExprRefBlocks(blocks, e.a)
    registerExprRefBlocks(blocks, e.b)
  of eNeg:
    registerExprRefBlocks(blocks, e.unOperand)
  of eNot:
    registerExprRefBlocks(blocks, e.notOperand)
  of eOconstr:
    for kvf in e.kvFields:
      registerExprRefBlocks(blocks, kvf.value)
  of eAconstr:
    for el in e.elems:
      registerExprRefBlocks(blocks, el)
  of eNil, eIntLit, eFloatLit, eTrue, eFalse, eSizeof, eSym:
    discard

proc registerStmtRefBlocks(blocks: var OrderedTable[string, CirType], ss: seq[CirStmt]) =
  ## Registers the buffer-reference blocks reachable from one statement list.
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
  ## Registers every buffer-reference block that the printer will declare,
  ## one entry per distinct block name in registration order:
  ##
  ## - every device pointer reachable from the module's type declarations,
  ##   through pointer targets, array elements and object fields
  ## - every proc parameter of pointer type
  ## - every buffer pointer reachable from a proc body, a cast target included
  ##
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
  ## Registers the GLSL ref-block analysis pass:
  ##
  ## - the pass fills `CirModule.refBlocks` with one entry per block name,
  ##   in registration order
  ## - the printer reads the stored entries and registers nothing itself
  reg.register("glslRegisterRefBlocks", pkAnalysis,
    "Registers the buffer-reference blocks that the printer declares, one" &
    " per distinct block name in registration order",
    run = glslRegisterRefBlocks
  )
