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
import ./pass_datatypes
import ./pass_registry
import ./pass_helpers

# ############################################################
#
#            Pointer lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerExplodeStructWithPtrFields*(reg: var PassRegistry) =
  reg.register(
    name = "explodeGlobalStructWithPtrFields",
    kind = pkTransform,
    description = "Expands kernel parameters of struct type with pointer fields into flat leaves",
    run = explodeGlobalStructWithPtrFields
  )

# ═══════════════════════════════════════════════════════════════════════
#  Kernel struct-parameter explosion
# ═══════════════════════════════════════════════════════════════════════

proc defaultConstr(t: CirType): CirExpr =
  var fields: seq[CirKv]
  for f in t.fields:
    if f.typ.kind == ctVoid:
      continue
    fields.add kv(f.name, defaultConstr(f.typ))
  result = oconstrE(t, fields)

proc buildLeafConstr(t: CirType, name: string): CirExpr =
  ## For `AxpyView{p, n}` exploded under `y`:
  ##   oconstr AxpyView{p: sym y_p, n: sym y_n}
  ## For `AxpyView{p, n}` exploded under `y`:
  ##   oconstr AxpyView{p: sym y_p, n: sym y_n}
  ## - the leaf symbol of field `f` names `name_f`
  ## - nested structs recurse one level deeper
  ## - a static struct field has no leaf and constructs empty
  ##   (no kernel parameter exists for it)
  var fields: seq[CirKv]
  for f in t.fields:
    if f.typ.kind == ctVoid:
      continue
    if f.typ.isStaticStruct:
      fields.add kv(f.name, defaultConstr(f.typ))
    elif f.typ.kind == ctStruct and f.typ.containsPtrDeep:
      fields.add kv(f.name, buildLeafConstr(f.typ, name & "_" & f.name))
    else:
      fields.add kv(f.name, sym(name & "_" & f.name, f.typ))
  result = oconstrE(t, fields)

proc explodeGlobalStructWithPtrFields(m: CirModule) =
  ## Contract:
  ## - buffer leaves name the device address space, value leaves the thread space
  ## - the kernel body opens with the rebuilt struct local
  ## - the pass refuses a leaf name colliding with an existing kernel parameter or body local
  for fn in m.procs:
    if not fn.isKernel:
      continue
    var taken: seq[string]
    for p in fn.params:
      taken.add p.name
    var locals: seq[tuple[name: string, typ: CirType]]
    collectDeclaredLocals(locals, fn.body)
    for l in locals:
      taken.add l.name
    var params: seq[CirParam]
    var preamble: seq[CirStmt]
    for p in fn.params:
      if p.typ.kind == ctStruct and p.typ.containsPtrDeep:
        let leaves = p.typ.kernelParamLeaves(p.name)
        for leaf in leaves:
          if leaf.name in taken:
            raiseAssert("leaf '" & leaf.name &
              "' collides with an existing kernel parameter or local")
          taken.add leaf.name
          let typ =
            if leaf.isBuffer:
              case leaf.typ.kind
              of ctPtr: tPtr(caDevice, leaf.typ.pElem)
              of ctAptr: tAptr(caDevice, leaf.typ.pElem)
              else: leaf.typ
            else:
              leaf.typ
          params.add CirParam(name: leaf.name, typ: typ)
        preamble.add varD(p.name, p.typ, buildLeafConstr(p.typ, p.name))
      elif p.typ.isStaticStruct:
        # the pass drops the parameter from the signature and declares a same-named default local
        preamble.add varD(p.name, p.typ, defaultConstr(p.typ))
      else:
        params.add p
    if preamble.len > 0:
      fn.params = params
      fn.body = preamble & fn.body
# ═══════════════════════════════════════════════════════════════════════
#  Pointer-local explosion
# ═══════════════════════════════════════════════════════════════════════

proc defaultRewritePointerExpr(ctx: var ExplodeCtx, fn: CirProc, e: CirExpr, target: string)

proc defaultPointerArm(ctx: var ExplodeCtx, fn: CirProc, e: CirExpr, target: string) =
  case e.kind
  of eDot:
    ctx.defaultRewritePointerExpr(fn, e.dotBase, target)
  of eDeref:
    let op = e.derefOperand
    if op.kind == eSym:
      if op.name in ctx.pendingPtrs:
        raiseAssert("pointer local " & op.name & " reads ahead of its address-of binding")
      if op.name in ctx.structLocals:
        raiseAssert("deref of exploded struct local " & op.name & ", " & target & " has no pointer struct members")
    ctx.defaultRewritePointerExpr(fn, op, target)
  else:
    for ch in getExprChildren(e):
      ctx.defaultRewritePointerExpr(fn, ch, target)

proc defaultRewritePointerExpr(ctx: var ExplodeCtx, fn: CirProc, e: CirExpr, target: string) =
  ctx.rewritePointerExprShared(fn, e, target, proc(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr) =
    ctx.defaultPointerArm(fn, e, target))

proc defaultExplodeVarDecl(ctx: var ExplodeCtx, fn: CirProc, s: CirStmt, target: string) =
  if s.declType.kind in {ctPtr, ctAptr}:
    if s.init == nil:
      ctx.pendingPtrs.incl(s.name)
    elif s.init.kind == eAddr and s.init.addrOperand.kind == eSym:
      ctx.aliasOf[s.name] = s.init.addrOperand.name
    else:
      raiseAssert("pointer local " & s.name & " binds no address, " & target & " locals are values")
    return
  if s.declType.kind == ctStruct and s.declType.containsPtrDeep:
    ctx.explodeStructVarDecl(fn, s, target)
    return
  ctx.defaultRewritePointerExpr(fn, s.init, target)

proc defaultExplodeAsgn(ctx: var ExplodeCtx, fn: CirProc, s: CirStmt, target: string) =
  if s.lhs.kind == eSym and s.lhs.typ != nil and
      s.lhs.typ.kind in {ctPtr, ctAptr}:
    if s.lhs.name notin ctx.pendingPtrs and s.lhs.name notin ctx.aliasOf:
      # a pointer parameter of the proc in flight, never touched by the pass
      ctx.defaultRewritePointerExpr(fn, s.rhs, target)
      return
    if s.lhs.name in ctx.aliasOf:
      raiseAssert("pointer local " & s.lhs.name & " rebinds, " & target & " keeps one binding")
    if s.rhs.kind != eAddr or s.rhs.addrOperand.kind != eSym:
      raiseAssert("pointer local " & s.lhs.name & " rebinds to a non-address value, " & target & " keeps one binding")
    ctx.pendingPtrs.excl(s.lhs.name)
    ctx.aliasOf[s.lhs.name] = s.rhs.addrOperand.name
    return
  ctx.defaultRewritePointerExpr(fn, s.lhs, target)
  ctx.defaultRewritePointerExpr(fn, s.rhs, target)

proc defaultExplodeDropsVar(s: CirStmt, ctx: ExplodeCtx): bool =
  s.declType.kind in {ctPtr, ctAptr} or
    (s.declType.kind == ctStruct and s.declType.containsPtrDeep)

proc defaultExplodeDropsAsgn(s: CirStmt, ctx: ExplodeCtx): bool =
  s.lhs.kind == eSym and s.lhs.typ != nil and
    s.lhs.typ.kind in {ctPtr, ctAptr} and
    (s.lhs.name in ctx.pendingPtrs or s.lhs.name in ctx.aliasOf)


template registerExplodePointerLocalsImpl(reg: var PassRegistry, passName,
    description: string, newExplodeCtx, explodeVarDecl, explodeAsgn,
    explodeDropsVar, explodeDropsAsgn, rewritePointerExpr: untyped) =
  ## Registers the pointer-local explosion pass of one target over the shared
  ## statement walk. Refusal procs from the instantiating module raise every refusal,
  ## the walk itself only recurses, rewrites and drops.
  ##
  ## Args:
  ## - `passName`, `description`: the registry entry
  ## - `newExplodeCtx(m, fn)`: the explosion context of one proc with the target's
  ##   callee-table and by-reference parameter setup inside
  ## - `explodeVarDecl`, `explodeAsgn`, `explodeDropsVar`, `explodeDropsAsgn`:
  ##   the `sVar` and `sAsgn` variants and their drop predicates
  ## - `rewritePointerExpr(fn, ctx, e)`: the expression rewrite
  ##
  ## Pass order: reserved-word rename → pointer-local explosion → printing
  ## - the pass sits after the reserved-word rename
  ## - the rewritten reads use the renamed names that their body already references
  # Each target module expands the template through a file-private forwarding
  # proc placed after its machinery, one that passes the module-local procs
  # as explicit template arguments.
  # Nim's codeReordering keeps the registration call at the module top
  proc explodeWalk(ctx: var ExplodeCtx, fn: CirProc, ss: var seq[CirStmt]) =
    var i = 0
    while i < ss.len:
      let s = ss[i]
      var drops = false
      case s.kind
      of sVar:
        explodeVarDecl(fn, ctx, s)
        drops = explodeDropsVar(s, ctx)
      of sAsgn:
        explodeAsgn(fn, ctx, s)
        drops = explodeDropsAsgn(s, ctx)
      of sIte:
        rewritePointerExpr(fn, ctx, s.cond)
        ctx.explodeWalk(fn, s.thenBody)
        ctx.explodeWalk(fn, s.elseBody)
      of sWhile:
        rewritePointerExpr(fn, ctx, s.whileCond)
        ctx.explodeWalk(fn, s.whileBody)
      of sRet:
        rewritePointerExpr(fn, ctx, s.retVal)
      of sDiscard:
        rewritePointerExpr(fn, ctx, s.discardVal)
      of sCall:
        let tmp = CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
        rewritePointerExpr(fn, ctx, tmp)
        s.callCallee = tmp.callee
        s.callArgs = tmp.args
      of sAsm, sEmit:
        discard
      if drops:
        ss.delete(i)
        continue
      i.inc
  proc explodePointerLocals(m: CirModule) {.nimcall.} =
    for fn in m.procs:
      var ctx = newExplodeCtx(m, fn)
      ctx.explodeWalk(fn, fn.body)
  reg.register(passName, pkTransform, description,
    run = explodePointerLocals)

template registerExplodePointerLocals*(reg: var PassRegistry, passName, description: string, target: string) =
  registerExplodePointerLocalsImpl(reg, passName, description,
    proc(m: CirModule, fn: CirProc): ExplodeCtx = ExplodeCtx(),
    proc(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt) = ctx.defaultExplodeVarDecl(fn, s, target),
    proc(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt) = ctx.defaultExplodeAsgn(fn, s, target),
    defaultExplodeDropsVar, defaultExplodeDropsAsgn,
    proc(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr) = ctx.defaultRewritePointerExpr(fn, e, target))
