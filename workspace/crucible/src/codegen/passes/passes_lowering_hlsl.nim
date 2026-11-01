# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## HLSL lowering: per-target transforms for the HLSL storage model,
## registered by the HLSL pipeline and run before printing.
##
## HLSL has no pointer values, every pointer use outside the dissolve pass refuses loudly,
## the printer never sees a pointer.
##
## - `hlslMaterializeNumWorkgroups`: reads of `threadgroups_per_grid`
##   rewrite to a trailing kernel parameter named `crucible_numworkgroups`,
##   the engine stages the launch's dispatch counts into the uniform block member
##   (HLSL has no shader-model semantic for the grid size).
## - `hlslDissolvePointerLocals`: pointer locals dissolve into their pointee,
##   pointer-holding struct locals into per-field symbol rewrites,
##   `var T` parameters read and write through their `inout` binding.
## - `hlslHoistProgramScopeConstants`: `{.const_mem.}` locals relocate into
##   `CirModule.globals` in encounter order, every proc and scope depth covered,
##   the printer emits the relocated declarations as file-scope `static const`
##   variables
## - dxc folds the relocated declarations at compile time, they take no register
##   in the `$Globals`/cbuffer layout and they are visible to all functions

import std/[sets, strutils, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers
import ./passes_lowering_cfamily
import ./passes_low_address_spaces
import ./passes_low_reserved

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerHlslLoweringPasses*(reg: var PassRegistry) =
  ## Registers the HLSL per-target lowering passes in the order
  ## materialization -> dissolution -> reserved-word renaming -> printing.
  reg.register("hlslMaterializeNumWorkgroups", pkTransform, phaseMain,
    "Rewrites threadgroups_per_grid reads to the materialized trailing" &
    " kernel parameter the engine stages",
    run = materializeNumWorkgroups
  )
  hlslRegisterDissolvePointerLocals(reg)
  reg.registerRenameReservedIdentifiers("hlslReservedIdentifiers", HlslReserved)
  reg.register("hlslHoistProgramScopeConstants", pkTransform, phaseMain,
    "Relocates const_mem locals into the module's program scope",
    run = hlslHoistProgramScopeConstants
  )
  reg.registerInstantiateAddressSpaceGenerics()

# ═══════════════════════════════════════════════════════════════════════
# The materialized dispatch-grid parameter
# ═══════════════════════════════════════════════════════════════════════

const NumWorkgroupsParam* = "crucible_numworkgroups"
  ## Trailing kernel parameter the dispatch grid size materializes into:
  ## the engine stages the launch's dispatch counts into the uniform block
  ## member of the same name.

proc findNumWorkgroupsExpr(e: CirExpr, coordTy: var CirType): bool =
  ## True when the expression reads `threadgroups_per_grid`, with the coordinate struct type of the read in `coordTy`.
  if e.isNil:
    return false
  if e.kind == eSym and e.coordBuiltin == cbkThreadgroupsPerGrid:
    coordTy = e.typ
    return true
  for ch in getExprChildren(e):
    if findNumWorkgroupsExpr(ch, coordTy):
      return true
  return false

proc findNumWorkgroupsStmts(ss: seq[CirStmt], coordTy: var CirType): bool =
  ## Statement-walk overload of `findNumWorkgroupsExpr`.
  for s in ss:
    case s.kind
    of sVar, sConst:
      if s.init != nil and findNumWorkgroupsExpr(s.init, coordTy):
        return true
    of sAsgn:
      if findNumWorkgroupsExpr(s.lhs, coordTy) or
          findNumWorkgroupsExpr(s.rhs, coordTy):
        return true
    of sIte:
      if findNumWorkgroupsExpr(s.cond, coordTy) or
          findNumWorkgroupsStmts(s.thenBody, coordTy) or
          findNumWorkgroupsStmts(s.elseBody, coordTy):
        return true
    of sLoop:
      if findNumWorkgroupsExpr(s.loopCond, coordTy) or
          findNumWorkgroupsStmts(s.loopBody, coordTy):
        return true
    of sRet:
      if s.retVal != nil and findNumWorkgroupsExpr(s.retVal, coordTy):
        return true
    of sDiscard:
      if findNumWorkgroupsExpr(s.discardVal, coordTy):
        return true
    of sCall:
      if findNumWorkgroupsExpr(s.callCallee, coordTy):
        return true
      for a in s.callArgs:
        if findNumWorkgroupsExpr(a, coordTy):
          return true
    of sAsm, sEmit:
      discard
  return false

proc rewriteNumWorkgroupsExpr(e: CirExpr, coordTy: CirType) =
  ## Replaces every `threadgroups_per_grid` read in place with a read of the materialized trailing parameter.
  ## Expressions are refs, the rewrite applies to the node every sharer sees.
  if e.isNil:
    return
  if e.kind == eSym and e.coordBuiltin == cbkThreadgroupsPerGrid:
    e.name = NumWorkgroupsParam
    e.typ = coordTy
    e.builtin = false
    e.coordBuiltin = cbkNone
    return
  for ch in getExprChildren(e):
    rewriteNumWorkgroupsExpr(ch, coordTy)

proc rewriteNumWorkgroupsStmts(ss: seq[CirStmt], coordTy: CirType) =
  ## Statement-walk overload of `rewriteNumWorkgroupsExpr`.
  for s in ss:
    case s.kind
    of sVar, sConst:
      rewriteNumWorkgroupsExpr(s.init, coordTy)
    of sAsgn:
      rewriteNumWorkgroupsExpr(s.lhs, coordTy)
      rewriteNumWorkgroupsExpr(s.rhs, coordTy)
    of sIte:
      rewriteNumWorkgroupsExpr(s.cond, coordTy)
      rewriteNumWorkgroupsStmts(s.thenBody, coordTy)
      rewriteNumWorkgroupsStmts(s.elseBody, coordTy)
    of sLoop:
      rewriteNumWorkgroupsExpr(s.loopCond, coordTy)
      rewriteNumWorkgroupsStmts(s.loopBody, coordTy)
    of sRet:
      rewriteNumWorkgroupsExpr(s.retVal, coordTy)
    of sDiscard:
      rewriteNumWorkgroupsExpr(s.discardVal, coordTy)
    of sCall:
      rewriteNumWorkgroupsExpr(s.callCallee, coordTy)
      for a in s.callArgs:
        rewriteNumWorkgroupsExpr(a, coordTy)
    of sAsm, sEmit:
      discard

proc materializeNumWorkgroups(m: CirModule): string =
  ## Rewrites every `threadgroups_per_grid` read in each kernel to a read of the trailing parameter.
  ## Kernels without such reads stay untouched, device functions never see one,
  ## a coordinate builtin in a device function refuses when printed.
  for fn in m.procs:
    if not fn.isKernel:
      continue
    var coordTy: CirType = nil
    if not findNumWorkgroupsStmts(fn.body, coordTy):
      continue
    if coordTy.isNil:
      return "kernel " & fn.name & " reads threadgroups_per_grid through an" &
        " untyped expression, the materialized parameter has no type"
    var taken: seq[string]
    for p in fn.params:
      taken.add p.name
    var locals: seq[tuple[name: string, typ: CirType]]
    fn.body.collectDeclaredLocals(locals)
    for l in locals:
      taken.add l.name
    if NumWorkgroupsParam in taken:
      return "kernel " & fn.name & ": the materialized parameter name '" &
        NumWorkgroupsParam & "' is taken"
    fn.params.add CirParam(name: NumWorkgroupsParam, typ: coordTy)
    rewriteNumWorkgroupsStmts(fn.body, coordTy)
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Pointer dissolution
# ═══════════════════════════════════════════════════════════════════════

type
  DissolveLeaf = tuple[sym: string, typ: CirType]
    ## One dissolved field read: the leaf symbol standing in for it plus
    ## the leaf type.

  DissolveCtx = object
    procs: Table[string, CirProc]
    ## Module procs by name, by-reference call arguments read callee
    ## parameter types here.
    byRefParams: HashSet[string]
    ## `var T` parameters of the proc in flight, reads and writes go through
    ## the `inout` binding, their derefs unwrap to the bare parameter.
    pendingPtrs: HashSet[string]
    ## Pointer locals awaiting their address-of binding, a read refuses.
    aliasOf: Table[string, string]
    ## Pointer local -> pointee local, the binding dissolved the pair.
    structLocals: HashSet[string]
    ## Pointer-holding struct locals dissolved into field rewrites, use of the whole value refuses.
    fieldOf: Table[string, DissolveLeaf]
    ## Dotted field path of a dissolved struct local -> the leaf symbol bound
    ## by the object construction: `v.data` -> the symbol bound to `data`,
    ## `v.inner.f` -> the symbol bound to the nested `f`.

proc dissolveFail(fn: CirProc, detail: string): string =
  "dissolvePointerLocals: at " & fn.name & ": " & detail

proc registerDissolvedStruct(fn: CirProc, ctx: var DissolveCtx, path: string, t: CirType, c: CirExpr): string =
  ## Records the field rewrites of one pointer-holding struct local bound by a construction of symbols.
  ## Nested constructions recurse. Returns "" when clean, else the refusal.
  if c.kind != eOconstr:
    return dissolveFail(fn, "struct local " & path &
      " initializer is not a construction, HLSL has no pointer struct members")
  if c.ocType.name != t.name:
    return dissolveFail(fn, "struct local " & path & " initializer has type " &
      c.ocType.name & ", expected " & t.name)
  var bound: seq[string]
  for kvv in c.kvFields:
    bound.add kvv.name
    var ft: CirType = nil
    for f in t.fields:
      if f.name == kvv.name:
        ft = f.typ
    if ft.isNil:
      return dissolveFail(fn, "field " & kvv.name & " is not on " & t.name)
    if kvv.value.kind == eOconstr and ft.kind == ctStruct and ft.containsPtrDeep:
      let err = registerDissolvedStruct(fn, ctx, path & "." & kvv.name, ft, kvv.value)
      if err.len > 0:
        return err
    elif kvv.value.kind == eSym:
      ctx.fieldOf[path & "." & kvv.name] = (kvv.value.name, ft)
    else:
      return dissolveFail(fn, "field " & path & "." & kvv.name &
        " initializer is not a symbol, the dissolved local cannot rewrite it")
  for f in t.fields:
    if f.name notin bound:
      return dissolveFail(fn, "field " & f.name & " of " & path &
        " is not bound by the construction")
  return ""

proc structRoot(e: CirExpr): CirExpr =
  ## Symbol at the root of a dotted chain, nil when the expression is
  ## neither a symbol nor a dotted chain over one.
  result = e
  while result != nil and result.kind == eDot:
    result = result.dotBase
  if result != nil and result.kind != eSym:
    result = nil

proc dottedPath(e: CirExpr): string =
  ## Dotted field path of a chain rooted at a symbol, "" otherwise:
  ## `y` -> `y`, `y.shape.F0` -> `y.shape.F0`.
  if e.kind == eSym:
    return e.name
  if e.kind == eDot:
    let base = dottedPath(e.dotBase)
    if base.len > 0:
      return base & "." & e.fieldName
  return ""

proc hasDeeperField(ctx: DissolveCtx, path: string): bool =
  ## True when a dissolved leaf binds strictly below `path`.
  let prefix = path & "."
  for k in ctx.fieldOf.keys:
    if k.startsWith(prefix):
      return true
  return false

proc hasLeafPrefix(ctx: DissolveCtx, path: string): bool =
  ## True when a dissolved leaf binds a strict prefix of `path`, the read continuing into the leaf's own value.
  ## A bound `y.layout` leaf splices the read `y.layout.shape.F0` to the leaf symbol `y_layout`.
  for k in ctx.fieldOf.keys:
    if path.startsWith(k & "."):
      return true
  return false

proc rewriteDissolveExpr(fn: CirProc, ctx: var DissolveCtx, e: CirExpr): string
  ## Forward declaration, expressions recurse into themselves.

proc rewriteDissolveChildren(fn: CirProc, ctx: var DissolveCtx, e: CirExpr): string =
  for ch in getExprChildren(e):
    let err = rewriteDissolveExpr(fn, ctx, ch)
    if err.len > 0:
      return err
  return ""

proc rewriteDissolveExpr(fn: CirProc, ctx: var DissolveCtx, e: CirExpr): string =
  ## Rewrites one expression in place for the HLSL pointer model.
  ## Returns "" when clean, else the first refusal.
  if e.isNil:
    return ""
  case e.kind
  of eSym:
    if e.name in ctx.structLocals:
      return dissolveFail(fn, "struct local " & e.name &
        " used whole, HLSL has no pointer struct members, read the fields")
    if e.name in ctx.pendingPtrs:
      return dissolveFail(fn, "pointer local " & e.name &
        " read before its address-of binding")
    if e.name in ctx.aliasOf:
      return dissolveFail(fn, "pointer local " & e.name &
        " used as a value, HLSL has no pointer values")
    return ""
  of eDot:
    proc spliceDissolvedRead(fn: CirProc, ctx: var DissolveCtx, e: CirExpr, root: CirExpr): string =
      ## One read through a dissolved struct local: a bound leaf splices here,
      ## a read continuing into a bound leaf's own value rewrites the inner chain node,
      ## a deeper bound leaf defers to its own node, anything else refuses.
      let path = dottedPath(e)
      if ctx.fieldOf.hasKey(path):
        let leaf = ctx.fieldOf[path]
        e.kind = eSym
        e.name = leaf.sym
        e.typ = leaf.typ
        e.builtin = false
        e.coordBuiltin = cbkNone
        e.synchroBuiltin = sbkNone
        e.reductionBuiltin = rbkNone
        e.atomicsBuiltin = abkNone
        return ""
      if ctx.hasLeafPrefix(path):
        return rewriteDissolveExpr(fn, ctx, e.dotBase)
      if not ctx.hasDeeperField(path):
        return dissolveFail(fn, "field path " & path &
          " of dissolved struct local " & root.name &
          " has no leaf binding, HLSL has no pointer struct members")
      return ""
    let root = structRoot(e)
    if root != nil and root.name in ctx.structLocals:
      return spliceDissolvedRead(fn, ctx, e, root)
    let err = rewriteDissolveExpr(fn, ctx, e.dotBase)
    if err.len > 0:
      return err
    # child rewrite may have spliced the base chain to a dissolved struct
    # local or its leaf, re-splice this node against the rewritten base
    let root2 = structRoot(e)
    if root2 != nil and root2.name in ctx.structLocals:
      return spliceDissolvedRead(fn, ctx, e, root2)
    return ""
  of eDeref:
    let op = e.derefOperand
    if op.kind != eSym:
      let err = rewriteDissolveExpr(fn, ctx, op)
      if err.len > 0:
        return err
      return dissolveFail(fn, "deref of a non-symbol, HLSL has no pointer values")
    if op.name in ctx.byRefParams or op.name in ctx.aliasOf:
      # an `inout` binding of a parameter, or the pointee of a dissolved
      # pointer local: the symbol stands for its pointee, a dissolved alias
      # takes the pointee local's own name
      e.kind = eSym
      if op.name in ctx.aliasOf:
        e.name = ctx.aliasOf[op.name]
      else:
        e.name = op.name
      e.typ = op.typ.pElem
      e.builtin = false
      e.coordBuiltin = cbkNone
      e.synchroBuiltin = sbkNone
      e.reductionBuiltin = rbkNone
      e.atomicsBuiltin = abkNone
      return ""
    return dissolveFail(fn, "deref of " & op.name &
      ", no pointee: HLSL has no pointer values")
  of eAddr:
    let op = e.addrOperand
    if op.kind == eSym and op.name in ctx.aliasOf:
      return dissolveFail(fn, "address of pointer local " & op.name &
        ", HLSL has no pointer values")
    return dissolveFail(fn, "address-of has no HLSL form, buffers are" &
      " index-only resources and locals dissolve to their pointee")
  of eCall:
    let cerr = rewriteDissolveExpr(fn, ctx, e.callee)
    if cerr.len > 0:
      return cerr
    if e.callee.kind == eSym and ctx.procs.hasKey(e.callee.name):
      let callee = ctx.procs[e.callee.name]
      for i, a in mpairs(e.args):
        if i < callee.params.len and
            callee.params[i].typ.kind == ctPtr and
            callee.params[i].typ.space == caThread:
          # `inout` binding: the call site passes the variable itself
          if a.kind == eAddr and a.addrOperand.kind == eSym:
            # address-of over a variable argument: the `inout` binding
            # takes the variable itself, the printer emits the variable
            # name at the call site
            if a.addrOperand.name in ctx.aliasOf or a.addrOperand.name in ctx.pendingPtrs:
              return dissolveFail(fn, "pointer local " & a.addrOperand.name &
                " passed to the by-reference parameter of " & e.callee.name &
                ", HLSL has no pointer values")
          elif a.kind == eSym and a.typ != nil and a.typ.kind in {ctPtr, ctAptr}:
            return dissolveFail(fn, "pointer local " & a.name &
              " passed to the by-reference parameter of " & e.callee.name &
              ", HLSL has no pointer values")
          else:
            return dissolveFail(fn, "call argument " & $i & " to the" &
              " by-reference parameter of " & e.callee.name &
              " is not a variable, HLSL inout binds lvalues")
          continue
        let aerr = rewriteDissolveExpr(fn, ctx, a)
        if aerr.len > 0:
          return aerr
      return ""
    return rewriteDissolveChildren(fn, ctx, e)
  else:
    return rewriteDissolveChildren(fn, ctx, e)

proc dissolveVarDecl(fn: CirProc, ctx: var DissolveCtx, s: CirStmt): string =
  ## `sVar` case: a pointer local binds its pointee or pends for a later
  ## address-of binding, a pointer-holding struct local registers its field rewrites.
  ## Anything else rewrites the initializer.
  if s.declType.kind == ctPtr and s.declType.space == caThread:
    if s.init != nil:
      if s.init.kind != eAddr or s.init.addrOperand.kind != eSym:
        return dissolveFail(fn, "pointer local " & s.name &
          " without an address-of binding, HLSL has no pointer locals")
      ctx.aliasOf[s.name] = s.init.addrOperand.name
    else:
      ctx.pendingPtrs.incl(s.name)
    return ""
  if s.declType.kind == ctStruct and s.declType.containsPtrDeep:
    if s.init == nil:
      return dissolveFail(fn, "struct local " & s.name &
        " without a construction, HLSL has no pointer struct members")
    let err = registerDissolvedStruct(fn, ctx, s.name, s.declType, s.init)
    if err.len > 0:
      return err
    ctx.structLocals.incl(s.name)
    return ""
  return rewriteDissolveExpr(fn, ctx, s.init)

proc dissolveAsgn(fn: CirProc, ctx: var DissolveCtx, s: CirStmt): string =
  ## `sAsgn` case: a pointer local binds exactly once, a field of a dissolved
  ## struct local never assigns, anything else rewrites both operands.
  if s.lhs.kind == eSym and s.lhs.name in ctx.pendingPtrs:
    if s.rhs.kind != eAddr or s.rhs.addrOperand.kind != eSym:
      return dissolveFail(fn, "pointer local " & s.lhs.name &
        " rebound to a non-address value, HLSL has no pointer values")
    ctx.pendingPtrs.excl(s.lhs.name)
    ctx.aliasOf[s.lhs.name] = s.rhs.addrOperand.name
    return ""
  if s.lhs.kind == eSym and s.lhs.name in ctx.aliasOf:
    return dissolveFail(fn, "pointer local " & s.lhs.name &
      " rebound, HLSL has no pointer values")
  if s.lhs.kind == eDot and s.lhs.dotBase.kind == eSym and
      s.lhs.dotBase.name in ctx.structLocals:
    return dissolveFail(fn, "field of dissolved struct local " &
      s.lhs.dotBase.name & " assigned, HLSL has no pointer struct members")
  let lerr = rewriteDissolveExpr(fn, ctx, s.lhs)
  if lerr.len > 0:
    return lerr
  return rewriteDissolveExpr(fn, ctx, s.rhs)

proc dissolveDropsVar(s: CirStmt, ctx: DissolveCtx): bool =
  ## A pointer local and a dissolved struct local drop from their list.
  (s.declType.kind == ctPtr and s.declType.space == caThread) or
    (s.declType.kind == ctStruct and s.declType.containsPtrDeep)

proc dissolveDropsAsgn(s: CirStmt, ctx: DissolveCtx): bool =
  ## A pointer-local binding drops from its list.
  s.lhs.kind == eSym and
    (s.lhs.name in ctx.pendingPtrs or s.lhs.name in ctx.aliasOf)

proc newDissolveCtx(m: CirModule, fn: CirProc): DissolveCtx =
  ## Module procs go into the context for the by-reference call argument checks,
  ## the `var T` parameters mark their derefs.
  var procs: Table[string, CirProc]
  for f in m.procs:
    procs[f.name] = f
  result = DissolveCtx(procs: procs)
  for p in fn.params:
    if p.typ.kind == ctPtr and p.typ.space == caThread:
      result.byRefParams.incl(p.name)

# ═══════════════════════════════════════════════════════════════════════
# Const_mem locals to program-scope constants
# ═══════════════════════════════════════════════════════════════════════

proc hlslHoistProgramScopeConstants(m: CirModule): string =
  ## Relocates the `{.const_mem.}` locals of every proc into `m.globals` in encounter order,
  ## the declarations dropped from their bodies.
  ##
  ## Contract:
  ## - every proc and scope depth covered, a device-function or nested-scope
  ##   local relocates like a kernel-top-level one
  ## - the initializer is required, a const_mem local without one refuses
  ## - runs after the reserved-word rename, relocated declaration names match
  ##   the renamed body references
  hoistConstMemLocals(m)

proc hlslRegisterDissolvePointerLocals(reg: var PassRegistry) =
  ## Expands the shared dissolve template with the HLSL module-local procs.
  registerCfamilyDissolvePointerLocals(reg, "hlslDissolvePointerLocals",
    "Dissolves pointer locals, pointer-holding struct locals and var" &
    " parameters into the HLSL storage model",
    DissolveCtx, newDissolveCtx, dissolveVarDecl, dissolveAsgn,
    dissolveDropsVar, dissolveDropsAsgn, rewriteDissolveExpr)
