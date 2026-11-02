# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ../backends/lang_glsl
import ./pass_datatypes
import ./pass_helpers
import ./passes_lowering
import ./passes_low_reserved
import ./passes_low_address_spaces
import ./analysis_module_requirements
import ./passes_lowering_cfamily
import ./passes_low_pointers

# ############################################################
#
#            GLSL lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerGlslQualifyVarParams*(reg: var PassRegistry) =
  reg.register("glslQualifyVarParams", pkTransform,
    "Retypes var parameters to their pointee, unwraps their body derefs and" &
    " the call-site address arguments",
    run = qualifyVarParams
  )

proc registerGlslRefBlockPass*(reg: var PassRegistry) =
  ## - the pass fills `CirModule.refBlocks` with one entry per block name,
  ##   in registration order
  ## - the printer reads the stored entries and registers nothing itself
  reg.register("glslRegisterRefBlocks", pkAnalysis,
    "Registers the buffer-reference blocks that the printer declares, one" &
    " per distinct block name in registration order",
    run = glslRegisterRefBlocks
  )

func registerGlslHoistProgramScopeLocals*(reg: var PassRegistry) =
  ## Relocated declarations use the renamed names that their body already references.
  registerCfamilyHoistProgramScopeLocals(reg, "glslHoistProgramScopeLocals",
    "Relocates smem and const_mem kernel locals into the module's" &
    " program scope")

func registerGlslLoweringPasses*(reg: var PassRegistry) =
  ## Every pass a GLSL codegen run runs between its normalization stage and its validation passes.
  ##
  ## Contract:
  ## - the `glsl:` macro and the `bwd_diff` artifact pipeline share this pass sequence in registration order
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  registerGlslQualifyVarParams(reg)
  registerSmemInitRewrite(reg)
  registerGlslHoistProgramScopeLocals(reg)
  registerInstantiateAddressSpaceGenerics(reg)
  registerGlslRefBlockPass(reg)
  registerGlslFeatureScan(reg)

# ═══════════════════════════════════════════════════════════════════════
#  var-parameter qualification
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
          return refuseAt(fn, "var parameter call argument must be an lvalue, got " & $arg.kind)
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
          return refuseAt(fn, "var parameter " & p.name & " without a pointer type")
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

proc registerRefBlock(blocks: var OrderedTable[string, CirType], pointee: CirType) =
  ## Registers the buffer-reference block of one runtime-array pointee,
  ## deduped by block name.
  ##
  ## The block's member is the runtime array `x[]`: a fixed-array element nests
  ## after the runtime dimension (`uint8_t x[][32]`), as the unsized dimension is
  ## only legal in the outermost position. The printer builds the member text
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
