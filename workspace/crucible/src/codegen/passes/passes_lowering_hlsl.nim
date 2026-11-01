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
## HLSL has no pointer values, every pointer use outside the pointer-local explosion pass refuses loudly,
## the printer never sees a pointer.
##
## - `hlslMaterializeNumWorkgroups`: reads of `threadgroups_per_grid`
##   rewrite to a trailing kernel parameter named `crucible_numworkgroups`,
##   the engine stages the launch's dispatch counts into the uniform block member
##   (HLSL has no shader-model semantic for the grid size).
## - `hlslExplodePointerLocals`: pointer locals rewrite to their pointee,
##   pointer-holding struct locals into per-field symbol rewrites,
##   `var T` parameters read and write through their `inout` binding.
## - `hlslHoistProgramScopeConstants`: `{.const_mem.}` locals relocate into
##   `CirModule.globals` in encounter order, every proc and scope depth covered,
##   the printer emits the relocated declarations as file-scope `static const`
##   variables
##
## `hlslClassifyParams` fills `CirModule.paramClasses` with one classification
## record per proc, one registration per flavor.
##
## `hlslScanStructRefs` fills `CirModule.structRefs` with the struct types
## the printer declares, every struct reachable through fields included.
## - dxc folds the relocated declarations at compile time, they take no register
##   in the `$Globals`/cbuffer layout and they are visible to all functions

import std/[sets, strutils, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ../backends/hlsl_lang
import ./pass_datatypes
import ./pass_helpers
import ./passes_low_pointers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerHlslMaterializeNumWorkgroups*(reg: var PassRegistry) =
  reg.register("hlslMaterializeNumWorkgroups", pkTransform,
    "Rewrites threadgroups_per_grid reads to the materialized trailing" &
    " kernel parameter the engine stages",
    run = materializeNumWorkgroups
  )

func registerHlslExplodePointerLocals*(reg: var PassRegistry) =
  registerExplodePointerLocals(reg, "hlslExplodePointerLocals",
    "Rewrites pointer locals, pointer-holding struct locals and var" &
    " parameters to the HLSL storage model",
    newExplodeCtx, explodeVarDecl, explodeAsgn,
    explodeDropsVar, explodeDropsAsgn, rewritePointerExpr)

func registerHlslHoistProgramScopeConstants*(reg: var PassRegistry) =
  reg.register("hlslHoistProgramScopeConstants", pkTransform,
    "Relocates const_mem locals into the module's program scope",
    run = hlslHoistProgramScopeConstants
  )

# ═══════════════════════════════════════════════════════════════════════
# The materialized dispatch-grid parameter
# ═══════════════════════════════════════════════════════════════════════

const NumWorkgroupsParam* = "crucible_numworkgroups"
  ## Trailing kernel parameter the dispatch grid size materializes into:
  ## the engine stages the launch's dispatch counts into the uniform block
  ## member of the same name.

proc findNumWorkgroupsExpr(e: CirExpr, coordTy: var CirType): bool =
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
    of sWhile:
      if findNumWorkgroupsExpr(s.whileCond, coordTy) or
          findNumWorkgroupsStmts(s.whileBody, coordTy):
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
    of sWhile:
      rewriteNumWorkgroupsExpr(s.whileCond, coordTy)
      rewriteNumWorkgroupsStmts(s.whileBody, coordTy)
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
# Pointer-local explosion
# ═══════════════════════════════════════════════════════════════════════

proc hasDeeperField(ctx: ExplodeCtx, path: string): bool =
  let prefix = path & "."
  for k in ctx.leafOf.keys:
    if k.startsWith(prefix):
      return true
  return false

proc hasLeafPrefix(ctx: ExplodeCtx, path: string): bool =
  ## A bound `y.layout` leaf splices the read `y.layout.shape.F0` to the leaf
  ## symbol `y_layout`, the read continues into the leaf's own value.
  for k in ctx.leafOf.keys:
    if path.startsWith(k & "."):
      return true
  return false

proc rewritePointerExpr(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string
  ## Forward declaration, expressions recurse into themselves.

proc rewritePointerChildren(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string =
  for ch in getExprChildren(e):
    let err = rewritePointerExpr(fn, ctx, ch)
    if err.len > 0:
      return err
  return ""

proc hlslPointerArm(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string =
  case e.kind
  of eDot:
    proc spliceExplodedRead(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr, root: CirExpr): string =
      let path = dottedPath(e)
      if ctx.hasLeafPrefix(path):
        return rewritePointerExpr(fn, ctx, e.dotBase)
      if not ctx.hasDeeperField(path):
        return refuseAt(fn, "field path " & path &
          " of exploded struct local " & root.name &
          " has no leaf binding, HLSL has no pointer struct members")
      return ""
    let root = structRoot(e)
    if root != nil and root.name in ctx.structLocals:
      return spliceExplodedRead(fn, ctx, e, root)
    let err = rewritePointerExpr(fn, ctx, e.dotBase)
    if err.len > 0:
      return err
    # child rewrite may have spliced the base chain to an exploded struct
    # local or its leaf, re-splice this node against the rewritten base
    let root2 = structRoot(e)
    if root2 != nil and root2.name in ctx.structLocals:
      return spliceExplodedRead(fn, ctx, e, root2)
    return ""
  of eDeref:
    let op = e.derefOperand
    if op.kind != eSym:
      let err = rewritePointerExpr(fn, ctx, op)
      if err.len > 0:
        return err
      return refuseAt(fn, "deref of a non-symbol, HLSL has no pointer values")
    return refuseAt(fn, "deref of " & op.name &
      ", no pointee: HLSL has no pointer values")
  of eAddr:
    let op = e.addrOperand
    if op.kind == eSym and op.name in ctx.aliasOf:
      return refuseAt(fn, "address of pointer local " & op.name &
        ", HLSL has no pointer values")
    return refuseAt(fn, "address-of has no HLSL form, buffers are" &
      " index-only resources and locals rewrite to their pointee")
  of eCall:
    let cerr = rewritePointerExpr(fn, ctx, e.callee)
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
              return refuseAt(fn, "pointer local " & a.addrOperand.name &
                " passed to the by-reference parameter of " & e.callee.name &
                ", HLSL has no pointer values")
          elif a.kind == eSym and a.typ != nil and a.typ.kind in {ctPtr, ctAptr}:
            return refuseAt(fn, "pointer local " & a.name &
              " passed to the by-reference parameter of " & e.callee.name &
              ", HLSL has no pointer values")
          else:
            return refuseAt(fn, "call argument " & $i & " to the" &
              " by-reference parameter of " & e.callee.name &
              " is not a variable, HLSL inout binds lvalues")
          continue
        let aerr = rewritePointerExpr(fn, ctx, a)
        if aerr.len > 0:
          return aerr
      return ""
    return rewritePointerChildren(fn, ctx, e)
  else:
    return rewritePointerChildren(fn, ctx, e)

proc rewritePointerExpr(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string =
  rewritePointerExprShared(fn, ctx, e, "HLSL", hlslPointerArm)

proc explodeVarDecl(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt): string =
  if s.declType.kind == ctPtr and s.declType.space == caThread:
    if s.init != nil:
      if s.init.kind != eAddr or s.init.addrOperand.kind != eSym:
        return refuseAt(fn, "pointer local " & s.name &
          " without an address-of binding, HLSL has no pointer locals")
      ctx.aliasOf[s.name] = s.init.addrOperand.name
    else:
      ctx.pendingPtrs.incl(s.name)
    return ""
  if s.declType.kind == ctStruct and s.declType.containsPtrDeep:
    return explodeStructVarDecl(fn, ctx, s, "HLSL")
  return rewritePointerExpr(fn, ctx, s.init)

proc explodeAsgn(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt): string =
  if s.lhs.kind == eSym and s.lhs.name in ctx.pendingPtrs:
    if s.rhs.kind != eAddr or s.rhs.addrOperand.kind != eSym:
      return refuseAt(fn, "pointer local " & s.lhs.name &
        " rebound to a non-address value, HLSL has no pointer values")
    ctx.pendingPtrs.excl(s.lhs.name)
    ctx.aliasOf[s.lhs.name] = s.rhs.addrOperand.name
    return ""
  if s.lhs.kind == eSym and s.lhs.name in ctx.aliasOf:
    return refuseAt(fn, "pointer local " & s.lhs.name &
      " rebound, HLSL has no pointer values")
  if s.lhs.kind == eDot and s.lhs.dotBase.kind == eSym and
      s.lhs.dotBase.name in ctx.structLocals:
    return refuseAt(fn, "field of exploded struct local " &
      s.lhs.dotBase.name & " assigned, HLSL has no pointer struct members")
  let lerr = rewritePointerExpr(fn, ctx, s.lhs)
  if lerr.len > 0:
    return lerr
  return rewritePointerExpr(fn, ctx, s.rhs)

proc explodeDropsVar(s: CirStmt, ctx: ExplodeCtx): bool =
  (s.declType.kind == ctPtr and s.declType.space == caThread) or
    (s.declType.kind == ctStruct and s.declType.containsPtrDeep)

proc explodeDropsAsgn(s: CirStmt, ctx: ExplodeCtx): bool =
  s.lhs.kind == eSym and
    (s.lhs.name in ctx.pendingPtrs or s.lhs.name in ctx.aliasOf)

proc newExplodeCtx(m: CirModule, fn: CirProc): ExplodeCtx =
  ## Module procs go into the context for the by-reference call argument checks,
  ## the `var T` parameters mark their derefs.
  var procs: Table[string, CirProc]
  for f in m.procs:
    procs[f.name] = f
  result = ExplodeCtx(procs: procs)
  for p in fn.params:
    if p.typ.kind == ctPtr and p.typ.space == caThread:
      result.byRefParams.incl(p.name)


# ═══════════════════════════════════════════════════════════════════════
# Const_mem locals to program-scope constants
# ═══════════════════════════════════════════════════════════════════════

proc hlslHoistProgramScopeConstants(m: CirModule): string =
  hoistConstMemLocals(m)

# ═══════════════════════════════════════════════════════════════════════
# Parameter classification
# ═══════════════════════════════════════════════════════════════════════

proc classifyParams(fn: CirProc, flavor: HlslFlavor, pushParams: var HashSet[string], buffers: var seq[CirParam]): string =
  ## - kernels: values join the Push block, buffers take register ordinals
  ## - device functions: values pass by value (a `var T` binds `inout`),
  ##   buffers are unbound resource params
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      if p.typ.space != caDevice:
        return "buffer parameter " & p.name & " in " & $p.typ.space &
          " space, buffers address device memory"
      if flavor == fvVulkan:
        # element types are checked up front, the access emitter and the buffer-pointer typedef both read them
        let (_, err) = bpElementStrideCheck(p.typ.pElem)
        if err.len > 0:
          return err
      buffers.add p
    of ctPtr:
      if p.typ.space == caThread:
        if fn.isKernel:
          return "by-reference parameter " & p.name & " in kernel `" &
            fn.name & "` has no binding, kernels take values and buffers"
        # device function: the `inout` binding, reads and writes are direct
        discard
      else:
        return "pointer parameter " & p.name & " in " & $p.typ.space &
          " space has no binding"
    of ctArray:
      return "array parameter " & p.name & " in `" & fn.name &
        "` has no binding, buffers pass as `ptr UncheckedArray[T]`"
    of ctSpan:
      return "span parameter " & p.name & " in `" & fn.name &
        "` has no HLSL binding, openArray spans split to pointer + length" &
        " before printing"
    of ctInt, ctUint:
      if p.typ.bits != 32:
        return "64-bit integer parameter " & p.name & " in `" & fn.name &
          "` is rejected: int64_t is a parse error in the HLSL frontend and" &
          " uint64_t pulls the SPIR-V Int64 capability"
      if fn.isKernel:
        pushParams.incl(p.name)
    of ctFloat:
      if p.typ.bits == 16:
        return "float16 parameter " & p.name & " in `" & fn.name &
          "`, float16 types are outside the HLSL subset"
      if fn.isKernel:
        pushParams.incl(p.name)
    of ctBool:
      if fn.isKernel:
        if flavor == fvVulkan:
          return "bool parameter " & p.name & ", the fvVulkan push block has" &
            " no bool member"
        pushParams.incl(p.name)
    of ctVoid, ctVector, ctStruct:
      if fn.isKernel:
        pushParams.incl(p.name)
    of ctProc:
      return "procedure parameter " & p.name & " in `" & fn.name & "` has no binding"
  return ""

proc classifyHlslModuleParams(m: CirModule, flavor: HlslFlavor): string =
  ## Fills `CirModule.paramClasses`, one record per proc.
  ##
  ## Contract:
  ## - the printer reads the stored records and never reclassifies
  ## - runs after the last transform, the record never invalidates
  for fn in m.procs:
    var pc = ParamClass()
    let err = classifyParams(fn, flavor, pc.push, pc.buffers)
    if err.len > 0:
      return "at " & fn.name & ": " & err
    m.paramClasses[fn.name] = pc
  return ""

proc classifyFvD3D12(m: CirModule): string =
  classifyHlslModuleParams(m, fvD3D12)

proc classifyFvVulkan(m: CirModule): string =
  classifyHlslModuleParams(m, fvVulkan)

proc registerHlslParamClassPass*(reg: var PassRegistry, flavor: HlslFlavor) =
  ## Flavor selection: the pipeline registers the flavor that the macro compiles, which selects the refusal set.
  let description = "Splits every proc's parameters into buffer resources and" &
    " push-block values, refusing the shapes the resource model cannot bind"
  case flavor
  of fvD3D12:
    reg.register("hlslClassifyParams", pkAnalysis, description,
      run = classifyFvD3D12)
  of fvVulkan:
    reg.register("hlslClassifyParams", pkAnalysis, description,
      run = classifyFvVulkan)


# ═══════════════════════════════════════════════════════════════════════
# Struct-reference registration
# ═══════════════════════════════════════════════════════════════════════

proc registerStructRefs(refs: var HashSet[string], t: CirType) =
  ## Records the struct type names reachable from `t`, through fields, array elements and vector elements.
  ## Pointer targets stay unvisited (`descendPtrs = false`): a pointer target
  ## prints no HLSL type name and the printer refuses a struct with a pointer field.
  var found = refs
  proc inclStruct(tt: CirType) =
    if tt.kind == ctStruct:
      found.incl(tt.name)
  walkTypeTree(t, inclStruct, descendPtrs = false)
  refs = found

proc scanStmtStructRefs(refs: var HashSet[string], ss: seq[CirStmt]) =
  var found = refs
  proc onDecl(s: CirStmt): bool =
    registerStructRefs(found, s.declType)
    true
  proc onExpr(e: CirExpr) =
    if e.kind == eOconstr:
      registerStructRefs(found, e.ocType)
  ss.scanStmtRefs(onDecl, onExpr)
  refs = found

proc hlslScanStructRefs(m: CirModule): string =
  ## Records the struct types the printer declares into `m.structRefs`:
  ##
  ## The registration walk recurses through fields, array elements and vector elements,
  ## so every struct the printer's declaration recursion reaches is already in the record.
  ## The pass runs after the last transform, the record it stores never invalidates.
  let kernel = m.singleKernel()
  if kernel != nil:
    for p in kernel.params:
      if p.name != NumWorkgroupsParam:
        registerStructRefs(m.structRefs, p.typ)
  for fn in m.procs:
    if fn.isKernel:
      continue
    registerStructRefs(m.structRefs, fn.ret)
    for p in fn.params:
      registerStructRefs(m.structRefs, p.typ)
    scanStmtStructRefs(m.structRefs, fn.body)
  if kernel != nil:
    scanStmtStructRefs(m.structRefs, kernel.body)
  # relocated const_mem declarations, struct-typed ones included
  scanStmtStructRefs(m.structRefs, m.globals)
  return ""

proc registerHlslStructRefPass*(reg: var PassRegistry) =
  ## Printer contract: it declares exactly the stored names and scans for nothing itself.
  reg.register("hlslScanStructRefs", pkAnalysis,
    "Records the struct types the printer declares, every struct reachable" &
    " through fields and element types included",
    run = hlslScanStructRefs
  )
