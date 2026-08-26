## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan IR legalization pass 3: per-call-site device-fn ptr binding.
##
## Device fns with `ptr` params are cloned per agreeing call-site arg tuple,
## and the ptr args are substituted ident→expression into the body
## (`buf +% baseOff` shapes). Ptr-arg indexing over pointer-arithmetic
## chains then folds to SSBO element indexing. Runs after pass 2.

import std/[algorithm, tables]
import ../ir/gpu_types
import ./passes_legalization_vulkan_helpers

# ═════════════════════════════════════════════════════════════════════════
#  Pass 3: vulkanBindDeviceFnPtrParams
# ═════════════════════════════════════════════════════════════════════════

proc structuralKey(n: GpuAst): string =
  ## Structural pretty-print with ident display names replaced by iSyms, so
  ## two expressions over different symbols never produce the same grouping
  ## key (BUG-B-002: name-keyed grouping merged distinct buffers).
  case n.kind
  of gpuIdent:
    if n.symbol != nil:
      result = "id(" & n.symbol.iSym & ")"
    else:
      result = "id(?)"
  of gpuCast: result = "cast[" & $n.cTo.kind & "](" & structuralKey(n.cExpr) & ")"
  of gpuConv: result = "conv[" & $n.convTo.kind & "](" & structuralKey(n.convExpr) & ")"
  of gpuBinOp:
    result = "(" & structuralKey(n.bLeft) & " " & structuralKey(n.bOp) &
             " " & structuralKey(n.bRight) & ")"
  of gpuIndex: result = structuralKey(n.iArr) & "[" & structuralKey(n.iIndex) & "]"
  of gpuDot: result = structuralKey(n.dParent) & "." & structuralKey(n.dField)
  of gpuDeref: result = "*" & structuralKey(n.dOf)
  of gpuAddr: result = "&" & structuralKey(n.aOf)
  of gpuLit: result = "lit(" & n.lValue & ")"
  of gpuPrefix: result = "prefix(" & n.pOp & "," & structuralKey(n.pVal) & ")"
  of gpuCall:
    result = "call(" & structuralKey(n.cName)
    for a in n.cArgs:
      result.add "," & structuralKey(a)
    result.add ")"
  of gpuArrayLit:
    result = "arr("
    for v in n.aValues:
      result.add structuralKey(v) & ","
    result.add ")"
  of gpuObjConstr:
    result = "constr("
    for f in n.ocFields:
      result.add f.name & ":" & structuralKey(f.value) & ","
    result.add ")"
  else: result = $n.kind

proc callArgKey(a: GpuAst): string =
  ## Canonical grouping key for a ptr call-site arg: idents by symbol
  ## identity (iSym), others by their structure with base-ident iSyms
  ## substituted for display names.
  if a.kind == gpuIdent and a.symbol != nil:
    result = "ident:" & a.symbol.iSym
  else:
    result = "expr:" & structuralKey(a)

proc bindDeviceFnPtrParams*(ctx: var GpuContext) =
  ## Pass 3 (vulkanBindDeviceFnPtrParams): per-call-site device-fn ptr
  ## binding (see module header).

  let ctxL = ctx  # closures cannot capture the var param, so alias it
  let reachable = reachableFns(ctx)
  var byISym = initTable[string, GpuAst]()
  for fn in reachable:
    byISym[fn.pName.symbol.iSym] = fn

  # call depth: kernels 0, device fns = 1 + max caller depth
  var depthMemo = initTable[string, int]()
  proc fnDepth(iSym: string): int =
    if iSym in depthMemo: return depthMemo[iSym]
    let fn = byISym[iSym]
    if fn.isGlobalFn(): return 0
    var maxCaller = 0
    for host in reachable:
      if host.pBody.isNil: continue
      var calls: seq[GpuAst]
      collectCalls(host.pBody, calls)
      for c in calls:
        if c.cName.symbol != nil and c.cName.symbol.iSym == iSym:
          maxCaller = max(maxCaller, fnDepth(host.pName.symbol.iSym))
    result = 1 + maxCaller
    depthMemo[iSym] = result

  var deviceFns: seq[tuple[depth: int, fn: GpuAst]]
  for fn in reachable:
    if not fn.isGlobalFn():
      var hasPtr = false
      for p in fn.pParams:
        if p.typ.kind == gtPtr:
          hasPtr = true
          break
      if hasPtr:
        deviceFns.add (fnDepth(fn.pName.symbol.iSym), fn)
  deviceFns.sort(proc(a, b: tuple[depth: int, fn: GpuAst]): int = cmp(a.depth, b.depth))

  for (_, fn) in deviceFns:
    let fnISym = fn.pName.symbol.iSym
    var ptrPos: seq[int]
    for i, p in fn.pParams:
      if p.typ.kind == gtPtr:
        ptrPos.add i
    if ptrPos.len == 0:
      continue
    # collect call sites across reachable fns
    var sites: seq[tuple[host: GpuAst, call: GpuAst]]
    for host in reachable:
      if host.pBody.isNil: continue
      var calls: seq[GpuAst]
      collectCalls(host.pBody, calls)
      for c in calls:
        if c.cName.symbol != nil and c.cName.symbol.iSym == fnISym:
          sites.add (host, c)
    if sites.len == 0:
      # unreachable from a kernel (or dead), so leave untouched (never emitted)
      continue
    for (host, call) in sites:
      if call.cArgs.len != fn.pParams.len:
        raiseAssert "Vulkan: arity mismatch calling device fn '" & fn.pName.ident() &
          "' from '" & host.pName.ident() & "' (" & $call.cArgs.len & " args vs " &
          $fn.pParams.len & " params)"
    # group by ptr-arg tuple
    var groups = initTable[string, seq[tuple[host: GpuAst, call: GpuAst]]]()
    var groupOrder: seq[string]
    for (host, call) in sites:
      var key = ""
      for pos in ptrPos:
        key.add callArgKey(call.cArgs[pos]) & "|"
      if key notin groups:
        groupOrder.add key
      groups.mgetOrPut(key, @[]).add (host, call)
    if groups.len == 1:
      # single agreement: bind in place
      let sites2 = groups[groupOrder[0]]
      var renames = initTable[string, GpuAst]()
      for pos in ptrPos:
        let arg = sites2[0].call.cArgs[pos]
        renames[fn.pParams[pos].ident.symbol.iSym] = arg
      if not fn.pBody.isNil:
        fn.pBody = substIdents(fn.pBody, renames)
      # drop the ptr params (highest position first) and the matching args
      for i in countdown(ptrPos.len - 1, 0):
        fn.pParams.delete(ptrPos[i])
        for (host, call) in sites2:
          call.cArgs.delete(ptrPos[i])
    else:
      # clone per group
      var clones: seq[tuple[key: string, fn: GpuAst]]
      for gi, key in groupOrder:
        let gs = groups[key]
        let cl = fn.clone()
        # fresh pName symbol
        let newName = fn.pName.ident() & "_vk" & $gi
        cl.pName = newGpuIdent(newName)
        cl.pName.symbol.iSym = newName
        cl.pName.symbol.symKind = gsProc
        var renames = initTable[string, GpuAst]()
        for pos in ptrPos:
          let arg = gs[0].call.cArgs[pos]
          renames[cl.pParams[pos].ident.symbol.iSym] = arg
        if not cl.pBody.isNil:
          cl.pBody = substIdents(cl.pBody, renames)
        for i in countdown(ptrPos.len - 1, 0):
          cl.pParams.delete(ptrPos[i])
        clones.add (key, cl)
      # point each group's call sites at its clone
      for gi, key in groupOrder:
        let cl = clones[gi].fn
        addFn(ctx, cl)
        for (host, call) in groups[key]:
          call.cName = cl.pName.clone()
          for i in countdown(ptrPos.len - 1, 0):
            call.cArgs.delete(ptrPos[i])
      # original is superseded: no group keeps using it
      removeFn(ctx, fnISym)

  # ── post: fold ptr-index bases introduced by the substitutions ─────────
  # (Index over `cast[ptr](uint64(base) + off*sizeof)` → base[off + idx])
  # Fresh iSym → fn table (includes the clones added above) for exprType.
  var foldFns = initTable[string, GpuAst]()
  for fnIdent, fn in ctx.fnTab:
    foldFns[fnIdent.symbol.iSym] = fn
  proc foldPtrIndexes(n: var GpuAst) =
    case n.kind
    of gpuIndex:
      foldPtrIndexes(n.iArr)
      foldPtrIndexes(n.iIndex)
      if n.iArr.kind == gpuCast and n.iArr.cTo.kind == gtPtr:
        # fold ptr-arith index bases via the shared helper
        let folded = foldPtrIndexToElement(n.iArr, n.iIndex, ctxL, foldFns)
        if not folded.isNil:
          n = folded
    else:
      for ch in n.mitems:
        foldPtrIndexes(ch)
  # Iterate ctx.fnTab (not the stale pre-clone `reachable` snapshot): the
  # per-call-site clones were added mid-pass, and their bodies are exactly
  # the ones that received substituted ptr-arith expressions (BUG-A-004).
  for fnIdent, fn in ctx.fnTab.mpairs:
    if not fn.pBody.isNil:
      foldPtrIndexes(fn.pBody)
