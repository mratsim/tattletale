# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Pointer machinery shared across targets: struct kernel-parameter explosion and pointer-local explosion.
##
## - kernel parameters of struct type carrying (nested) pointer fields expand into flat ptr/scalar leaves on the flat resource targets
## - pointer locals rewrite to their pointee, pointer-holding struct locals to their field leaves, before the GLSL, WGSL and HLSL printers run
##
## Emission contract: one buffer slot per leaf in field order matching the launch packing.
## Pointer leaves name the device buffers, the kernel body opens with a local rebuilt from the leaves.
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

proc buildLeafConstr(t: CirType, name: string): CirExpr =
  ## For `AxpyView{p, n}` exploded under `y`:
  ##   oconstr AxpyView{p: sym y_p, n: sym y_n}
  ## The leaf symbol of field `f` names `name_f`, nested structs recurse one level deeper.
  var fields: seq[CirKv]
  for f in t.fields:
    if f.typ.kind == ctVoid:
      continue
    if f.typ.kind == ctStruct and f.typ.containsPtrDeep:
      fields.add kv(f.name, buildLeafConstr(f.typ, name & "_" & f.name))
    else:
      fields.add kv(f.name, sym(name & "_" & f.name, f.typ))
  result = oconstrE(t, fields)

proc explodeGlobalStructWithPtrFields(m: CirModule): string =
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
    fn.body.collectDeclaredLocals(locals)
    for l in locals:
      taken.add l.name
    var params: seq[CirParam]
    var preamble: seq[CirStmt]
    for p in fn.params:
      if p.typ.kind == ctStruct and p.typ.containsPtrDeep:
        let leaves = p.typ.kernelParamLeaves(p.name)
        for leaf in leaves:
          if leaf.name in taken:
            return "kernel " & fn.name & ": leaf '" & leaf.name &
              "' collides with an existing kernel parameter or local"
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
      else:
        params.add p
    if preamble.len > 0:
      fn.params = params
      fn.body = preamble & fn.body
  return ""

# ═══════════════════════════════════════════════════════════════════════
#  Pointer-local explosion
# ═══════════════════════════════════════════════════════════════════════

template registerExplodePointerLocals*(reg: var PassRegistry, passName,
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
  proc explodeWalk(fn: CirProc, ctx: var ExplodeCtx, ss: var seq[CirStmt]): string =
    var i = 0
    while i < ss.len:
      let s = ss[i]
      var drops = false
      case s.kind
      of sVar:
        let err = explodeVarDecl(fn, ctx, s)
        if err.len > 0:
          return err
        drops = explodeDropsVar(s, ctx)
      of sAsgn:
        let err = explodeAsgn(fn, ctx, s)
        if err.len > 0:
          return err
        drops = explodeDropsAsgn(s, ctx)
      of sConst:
        let err = rewritePointerExpr(fn, ctx, s.init)
        if err.len > 0:
          return err
      of sIte:
        let cerr = rewritePointerExpr(fn, ctx, s.cond)
        if cerr.len > 0:
          return cerr
        let terr = explodeWalk(fn, ctx, s.thenBody)
        if terr.len > 0:
          return terr
        let eerr = explodeWalk(fn, ctx, s.elseBody)
        if eerr.len > 0:
          return eerr
      of sWhile:
        let cerr = rewritePointerExpr(fn, ctx, s.whileCond)
        if cerr.len > 0:
          return cerr
        let werr = explodeWalk(fn, ctx, s.whileBody)
        if werr.len > 0:
          return werr
      of sRet:
        let rerr = rewritePointerExpr(fn, ctx, s.retVal)
        if rerr.len > 0:
          return rerr
      of sDiscard:
        let derr = rewritePointerExpr(fn, ctx, s.discardVal)
        if derr.len > 0:
          return derr
      of sCall:
        let tmp = CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
        let err = rewritePointerExpr(fn, ctx, tmp)
        s.callCallee = tmp.callee
        s.callArgs = tmp.args
        if err.len > 0:
          return err
      of sAsm, sEmit:
        discard
      if drops:
        ss.delete(i)
        continue
      i.inc
    return ""
  proc explodePointerLocals(m: CirModule): string {.nimcall.} =
    for fn in m.procs:
      var ctx = newExplodeCtx(m, fn)
      let err = explodeWalk(fn, ctx, fn.body)
      if err.len > 0:
        return err
    return ""
  reg.register(passName, pkTransform, description,
    run = explodePointerLocals)
