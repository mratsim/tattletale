## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan IR legalization pass 1: device-fn `var T` params to value params.
##
## Array-typed `var` params inline the fn at each call site, each copy
## wrapped in its own block (GLSL has no array returns). Struct and scalar
## `var` params become value params, and a written param returns by value,
## so call sites become `x = f(x, …)`. Runs before pass 2.

import std/[sets, tables]
import ../ir/gpu_types
import ./passes_legalization_vulkan_helpers
import ./passes_utils

# ═════════════════════════════════════════════════════════════════════════
#  Pass 1: vulkanVarParamsToValue
# ═════════════════════════════════════════════════════════════════════════

proc varParamWritten(fn: GpuAst; pISym: string): bool =
  ## True when the var param is written somewhere in the body (assign target
  ## or address-taken as a var arg). Struct var params are written through
  ## field/index chains (`tmp.frags[n][m].frag[v] = …`), so the assign
  ## target's BASE (after unwrapping Index/Dot/Deref) is checked for the
  ## param ident.
  var found = false
  proc assignBase(l0: GpuAst): GpuAst =
    var l = l0
    while true:
      case l.kind
      of gpuIndex: l = l.iArr
      of gpuDot: l = l.dParent
      of gpuDeref: l = l.dOf
      else: break
    l
  proc walk(n: GpuAst) =
    if found: return
    case n.kind
    of gpuAssign:
      let b = assignBase(n.aLeft)
      if b.kind == gpuIdent and b.symbol != nil and b.symbol.iSym == pISym:
        found = true
        return
    of gpuAddr:
      let b = assignBase(n.aOf)
      if b.kind == gpuIdent and b.symbol != nil and b.symbol.iSym == pISym:
        found = true
        return
    else:
      discard
    for ch in n:
      if not found: walk(ch)
  if not fn.pBody.isNil:
    walk(fn.pBody)
  result = found

proc rewriteBodyDerefs(n: var GpuAst; paramISyms: HashSet[string]) =
  ## `Deref(Ident p)` → `Ident p` for converted value params.
  case n.kind
  of gpuDeref:
    if n.dOf.kind == gpuIdent and n.dOf.symbol != nil and
       n.dOf.symbol.iSym in paramISyms:
      n = n.dOf
    else:
      for ch in n.mitems:
        rewriteBodyDerefs(ch, paramISyms)
  else:
    for ch in n.mitems:
      rewriteBodyDerefs(ch, paramISyms)

proc convertVarParams*(ctx: var GpuContext) =
  ## Device-fn `var T` params → GLSL-legal form:
  ## - array-typed `var` params: inline the fn at each call site (GLSL has
  ##   no array returns), each copy wrapped in its own block.
  ## - struct/scalar `var` params: value param + return-by-value when written
  ##   (call sites become `x = f(x, …)`). `Deref(p)` in bodies collapses.
  let reachable = reachableFns(ctx)
  var byISym = initTable[string, GpuAst]()
  for fn in reachable:
    byISym[fn.pName.symbol.iSym] = fn

  # ── Pass 1a: inline array-typed var-param fns ──────────────────────────
  proc paramIndexedAsArray(fn: GpuAst; pISym: string): bool =
    ## True when the body indexes the param ident directly (array-style,
    ## `d[0]`). The IR models `var array[N, T]` params as `var T` (element
    ## ptr) with array indexing on the ident. The array-ness only shows up
    ## in the body, not in the param type.
    if fn.pBody.isNil: return false
    var found = false
    proc scan(n: GpuAst) =
      if found: return
      case n.kind
      of gpuIndex:
        if n.iArr.kind == gpuIdent and n.iArr.symbol != nil and
           n.iArr.symbol.iSym == pISym:
          found = true
          return
        scan(n.iArr)
        scan(n.iIndex)
      else:
        for ch in n:
          if not found: scan(ch)
    scan(fn.pBody)
    result = found

  var arrayVarFns: seq[GpuAst]
  for fn in reachable:
    if fn.isGlobalFn(): continue
    for p in fn.pParams:
      if p.typ.kind == gtPtr and p.typ.implicit:
        if (not p.typ.to.isNil and p.typ.to.kind == gtArray) or
           paramIndexedAsArray(fn, p.ident.symbol.iSym):
          arrayVarFns.add fn
          break

  for fnItem in arrayVarFns:
    let fn = fnItem  # ref copy, since lent loop vars cannot be captured
    let fnISym = fn.pName.symbol.iSym
    # locate hosts that call it
    var hosts: seq[GpuAst]
    for host in reachable:
      if host.pBody.isNil: continue
      var calls: seq[GpuAst]
      collectCalls(host.pBody, calls)
      for c in calls:
        if c.cName.symbol != nil and c.cName.symbol.iSym == fnISym:
          hosts.add host
          break
    if hosts.len == 0:
      # dead: pass 2 removes it when tainted-returning, otherwise it stays
      continue
    for host in hosts:
      # inline every call to this fn in one walk, matching by callee iSym
      # (GpuAst `==` only supports idents. Ref identity is unavailable in
      # the compile-time VM). Only STATEMENT-position calls are inlined
      # (GLSL has no expression blocks). An expression-position call to an
      # array-var-param fn is rejected loudly. Any `return` in the callee
      # body is also rejected: an inlined `return` would return from the
      # HOST kernel, silently truncating it (BUG-A-003).
      proc checkNoReturn(n: GpuAst) =
        case n.kind
        of gpuReturn:
          raiseAssert "Vulkan: cannot inline array-var-param device fn '" &
            fn.pName.ident() & "': body contains a `return` (an inlined " &
            "return would return from the host kernel)"
        else:
          for ch in n:
            checkNoReturn(ch)
      proc inlineWalk(n: var GpuAst)   # forward, called from inlineBlock
      proc inlineBlock(stmts: var seq[GpuAst]) =
        var outS: seq[GpuAst]
        for st in stmts.mitems:
          var s = st
          if s.kind == gpuCall and s.cName.symbol != nil and
             s.cName.symbol.iSym == fnISym:
            if s.cArgs.len != fn.pParams.len:
              raiseAssert "Vulkan: arity mismatch inlining device fn '" &
                fn.pName.ident() & "' (" & $s.cArgs.len & " args vs " &
                $fn.pParams.len & " params)"
            var subst = initTable[string, GpuAst]()
            for i, p in fn.pParams:
              subst[p.ident.symbol.iSym] = s.cArgs[i]
            var inlined = fn.pBody.clone()
            inlined = substIdents(inlined, subst)
            checkNoReturn(inlined)
            # replace the call statement with the inlined body in a fresh block
            outS.add GpuAst(kind: gpuBlock, statements: @[inlined])
          else:
            inlineWalk(s)
            outS.add s
        stmts = outS
      proc inlineWalk(n: var GpuAst) =
        case n.kind
        of gpuBlock:
          inlineBlock(n.statements)
        of gpuCall:
          if n.cName.symbol != nil and n.cName.symbol.iSym == fnISym:
            raiseAssert "Vulkan: cannot inline array-var-param device fn '" &
              fn.pName.ident() &
              "' in expression position (GLSL has no expression blocks)"
          for a in n.cArgs.mitems:
            inlineWalk(a)
        else:
          for ch in n.mitems:
            inlineWalk(ch)
      inlineWalk(host.pBody)
    removeFn(ctx, fnISym)

  # ── Pass 1b: struct/scalar var params → value + return ────────────────
  for fnItem in reachable:
    let fn = fnItem  # ref copy, since lent loop vars cannot be captured
    if fn.isGlobalFn(): continue
    var varPos = newSeq[int]()
    for i, p in fn.pParams:
      if p.typ.kind == gtPtr and p.typ.implicit and
         (p.typ.to.isNil or p.typ.to.kind != gtArray):
        if i notin varPos:  # dedup, genericInsts may share param objects
          varPos.add i
    if varPos.len == 0:
      continue
    let fnISym = fn.pName.symbol.iSym
    # map param iSym → leaf param name (same name, value type)
    var valueParamISyms = initHashSet[string]()
    var writtenISyms = newSeq[string]()
    for pos in varPos:
      let p = fn.pParams[pos]
      if p.typ.kind != gtPtr:
        # already a value param (dup varPos or a shared param object from a
        # generic-instance clone), so nothing left to convert
        continue
      valueParamISyms.incl p.ident.symbol.iSym
      if varParamWritten(fn, p.ident.symbol.iSym):
        writtenISyms.add p.ident.symbol.iSym
      # change the param type: var T → T (value)
      fn.pParams[pos].typ = p.typ.to
      if fn.pParams[pos].typ.isNil:
        raiseAssert "Vulkan: var param '" & p.ident.ident() & "' of fn '" &
          fn.pName.ident() & "' has nil pointee type"
      fn.pParams[pos].passByRef = false
    # collapse Deref(p) in the body
    if not fn.pBody.isNil:
      rewriteBodyDerefs(fn.pBody, valueParamISyms)
    # if any written: return the mutated value(s). GLSL returns one value, so
    # a fn with >1 written var param cannot be lowered this way.
    if writtenISyms.len > 1:
      raiseAssert "Vulkan: device fn '" & fn.pName.ident() &
        "' has " & $writtenISyms.len &
        " written var params (GLSL fns return one value)"
    if writtenISyms.len == 1:
      let retISym = writtenISyms[0]
      var retIdent: GpuAst
      var retPos = -1
      for i, p in fn.pParams:
        if p.ident.symbol.iSym == retISym:
          retPos = i
          retIdent = p.ident
          break
      if not fn.pRetType.isNil and fn.pRetType.kind != gtVoid:
        raiseAssert "Vulkan: device fn '" & fn.pName.ident() &
          "' already returns a value and mutates a var param, so it cannot be lowered"
      # return the param's final value. Use the CONVERTED param type (the
      # pointee), not the param symbol's type. The symbol still carries the
      # original `var T` (gtPtr) type and would leak a raw pointer into the
      # fn signature (codegen rejects gtPtr return types).
      if retPos < 0 or fn.pParams[retPos].typ.isNil:
        raiseAssert "Vulkan: written var param '" & retISym & "' of fn '" &
          fn.pName.ident() & "' has no converted value type"
      let newRet = fn.pParams[retPos].typ
      fn.pRetType = newRet
      if not fn.pBody.isNil and fn.pBody.kind == gpuBlock:
        # Every `return` in the body must carry the mutated value: the old
        # code only converted the trailing return, leaving bare `return;` in
        # early-exit branches, which is invalid GLSL in a non-void fn
        # (BUG-A-002).
        var body = fn.pBody
        proc fixReturn(n: var GpuAst) =
          case n.kind
          of gpuReturn:
            n.rValue = retIdent.clone()
          of gpuProc:
            # nested device fn. Its returns belong to its own scope and the
            # outer fn's retIdent is not in scope there (the old
            # full-tree recursion rewrote them, emitting `return x;` inside
            # the nested fn where x is undeclared)
            discard
          else:
            for ch in n.mitems:
              fixReturn(ch)
        fixReturn(body)
        # GLSL requires every path of a non-void fn to return a value, so
        # append a trailing `return x` for the fall-through path (dead code
        # when the body already ends with a return).
        body.statements.add GpuAst(kind: gpuReturn, rValue: retIdent.clone())
    # rewrite call sites as `x = f(x, …)`, one walk per host, matching calls
    # by callee iSym (GpuAst `==` only supports idents. Ref identity is
    # unavailable in the compile-time VM). Each call node computes its own
    # lvalue from its own args.
    var hosts: seq[GpuAst]
    for host in reachable:
      if host.pBody.isNil: continue
      var calls: seq[GpuAst]
      collectCalls(host.pBody, calls)
      for c in calls:
        if c.cName.symbol != nil and c.cName.symbol.iSym == fnISym:
          hosts.add host
          break
    for host in hosts:
      proc rewriteCalls(n: var GpuAst) =
        case n.kind
        of gpuCall:
          if n.cName.symbol != nil and n.cName.symbol.iSym == fnISym:
            if n.cArgs.len != fn.pParams.len:
              raiseAssert "Vulkan: arity mismatch calling device fn '" & fn.pName.ident() &
                "' (" & $n.cArgs.len & " args vs " & $fn.pParams.len & " params)"
            # find the var arg(s): the position(s) that were var params.
            # Only WRITTEN var args conflict with GLSL's one-return-value
            # rule. Unwritten ones become plain value params (BUG-A-005).
            # The old guard counted ALL var args and rejected valid calls
            # with a factually wrong message.
            var writtenArgPos: seq[int]
            for pos in varPos:
              if pos < n.cArgs.len and writtenISyms.len == 1 and
                 fn.pParams[pos].ident.symbol.iSym == writtenISyms[0]:
                writtenArgPos.add pos
            if writtenArgPos.len > 1:
              raiseAssert "Vulkan: call to '" & fn.pName.ident() &
                "' passes " & $writtenArgPos.len &
                " written var args (GLSL fns return one value)"
            # the callee now takes VALUE params, so strip addr/deref from every
            # var arg (written ones additionally come back as the return value)
            var newArgs = n.cArgs
            for pos in varPos:
              if pos >= newArgs.len:
                continue
              var a = newArgs[pos]
              if a.kind == gpuAddr:
                a = a.aOf
              if a.kind == gpuDeref:
                a = a.dOf
              newArgs[pos] = a
            var newCall = GpuAst(kind: gpuCall, cIsExpr: true, cName: n.cName)
            for a in newArgs:
              newCall.cArgs.add a
            if writtenArgPos.len == 1:
              # the callee returns the mutated value → `lvalue = f(…lvalue…)`
              let lvalue = newArgs[writtenArgPos[0]]
              let assign = GpuAst(kind: gpuAssign, aLeft: lvalue.clone(), aRight: newCall)
              n = assign
            else:
              # no written var param: plain value call
              n = newCall
        of gpuBlock:
          for i, st in n.statements.mpairs:
            rewriteCalls(st)
        else:
          for ch in n.mitems:
            rewriteCalls(ch)
      rewriteCalls(host.pBody)
