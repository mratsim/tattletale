## Phase 6: materializeIndexBuiltinParams pass test
##
## MSL device functions have no implicit thread index, so a body that
## references a canonical coordinate builtin must bind it to a param the caller
## forwards. The pass computes each function's transitive builtin needs once
## and appends one param per need into `pParams`, for kernels AND device
## functions, after the declared params. Each appended param's symbol carries
## the builtin kind (`ident.symbol.coordBuiltin`), so the Metal printer
## discriminates the attribute form (kernels) from the plain form (device
## functions) by that symbol field alone. This test asserts the rewrite on
## hand-built IR:
## - builtin params land in `pParams` (kernel + device fn) with
##   `ident.symbol.coordBuiltin` carrying the kind
## - declared params keep their positions and types (the buffer-binding order
##   the printer relies on)
## - a call site forwards the args in the callee's needs order
## - a local shadowing a canonical name (gbkNone) is NOT collected
## - a device fn calling itself terminates and rewrites consistently
## - a fn registered in both tables is rewritten once
## - a barrier call is never treated as a callee
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_materializeIndexBuiltinParams.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ── IR builders ─────────────────────────────────────────────────────────────

proc builtinIdent(name: string; kind: GpuCoordBuiltinKind): GpuAst =
  ## gpuIdent for a canonical coordinate builtin, with its IR kind marked.
  let sym = newSymbol(name, iSym = name & "_sym", symKind = gsBuiltin)
  sym.coordBuiltin = kind
  GpuAst(kind: gpuIdent, symbol: sym)

proc plainIdent(name: string): GpuAst =
  ## gpuIdent for a plain (non-builtin) name, e.g. a local shadowing a
  ## canonical builtin name: `coordBuiltin` stays `gbkNone`.
  GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = name & "_local"))

proc makeCall(callee: GpuAst): GpuAst =
  ## gpuCall to `callee`, sharing its pName symbol so the pass's table lookup
  ## (hashed on the symbol) resolves it.
  GpuAst(kind: gpuCall, cName: callee.pName.clone(), cArgs: @[])

proc makeProc(name: string; body: GpuAst; isKernel = false): GpuAst =
  ## gpuProc with a fresh symbol; the body is a gpuBlock of statements.
  var fn = GpuAst(kind: gpuProc)
  fn.pName = GpuAst(kind: gpuIdent,
                    symbol: newSymbol(name, iSym = name & "_isym", symKind = gsProc))
  fn.pRetType = GpuType(kind: gtVoid)
  fn.pBody = body
  if isKernel:
    fn.pAttributes = {attGlobal}
  else:
    fn.pAttributes = {attDevice}
  fn

proc makeParam(name: string; typ: GpuType; space: AddressSpace): GpuParam =
  ## Declared param (gbkNone symbol), as a kernel or device fn would carry.
  GpuParam(ident: GpuAst(kind: gpuIdent,
                         symbol: newSymbol(name, iSym = name & "_isym",
                                           symKind = gsGlobalKernelParam)),
           typ: typ, addressSpace: space, passByRef: false)

proc blockOf(stmts: varargs[GpuAst]): GpuAst =
  ## gpuBlock of statements, the shape every pass body walk expects.
  GpuAst(kind: gpuBlock, statements: @stmts)

proc callArgNames(fn: GpuAst; stmtIdx: int): seq[string] =
  ## Ident names of the call-site args of `fn`'s `stmtIdx`-th statement.
  let call = fn.pBody.statements[stmtIdx]
  doAssert call.kind == gpuCall, "statement " & $stmtIdx & " is not a call"
  for a in call.cArgs:
    result.add a.ident()

proc paramNames(fn: GpuAst): seq[string] =
  ## Ident names of `fn`'s params, in order.
  for p in fn.pParams:
    result.add p.ident.ident()

proc paramKinds(fn: GpuAst): seq[GpuCoordBuiltinKind] =
  ## `coordBuiltin` kinds of `fn`'s params, in order.
  for p in fn.pParams:
    result.add p.ident.symbol.coordBuiltin

# ═══════════════════════════════════════════════════════════════════════════
# 1. Device fn gains the builtin params: canonical name + coordBuiltin kind + type
# ═══════════════════════════════════════════════════════════════════════════
block:
  let body = blockOf(
    builtinIdent("thread_position_in_threadgroup", gbkThreadPositionInThreadgroup),
    builtinIdent("thread_index_in_threadgroup", gbkThreadIndexInThreadgroup)
  )
  var fn = makeProc("need", body)
  var ctx = GpuContext()
  ctx.allFnTab[fn.pName] = fn

  materializeIndexBuiltinParamsImpl(ctx)

  doAssert fn.pParams.len == 2,
    "device fn must gain one builtin param per need, got " & $fn.pParams.len
  doAssert fn.pParams[0].ident.ident() == "thread_position_in_threadgroup",
    "builtin param must use the canonical name"
  doAssert fn.pParams[0].ident.symbol.coordBuiltin == gbkThreadPositionInThreadgroup,
    "builtin param symbol must carry the builtin kind"
  doAssert fn.pParams[0].typ.kind == gtGenericInst,
    "vector builtin param must carry the uint3 generic spelling"
  doAssert fn.pParams[0].typ.gName == "uint3"
  doAssert fn.pParams[1].ident.ident() == "thread_index_in_threadgroup"
  doAssert fn.pParams[1].ident.symbol.coordBuiltin == gbkThreadIndexInThreadgroup,
    "flat thread index param symbol must carry the builtin kind"
  doAssert fn.pParams[1].typ.kind == gtUint32,
    "flat thread index param must be scalar uint32"
  echo "  OK — device fn gains builtin params (canonical name, coordBuiltin kind, uint3/uint type)"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Kernel gains the builtin params: own ref AND transitive callee need,
#    declared params keep positions and types (binding-order invariant)
# ═══════════════════════════════════════════════════════════════════════════
block:
  var callee = makeProc("callee",
    blockOf(builtinIdent("thread_position_in_grid", gbkThreadPositionInGrid)))
  var kernel = makeProc("kern",
    blockOf(
      builtinIdent("threadgroups_per_grid", gbkThreadgroupsPerGrid),
      makeCall(callee)
    ), isKernel = true)
  # Declared params, as a real kernel carries: a buffer and a scalar.
  kernel.pParams = @[
    makeParam("data", GpuType(kind: gtPtr, to: GpuType(kind: gtUint32)), asDevice),
    makeParam("n", GpuType(kind: gtUint32), asConstant)
  ]
  var ctx = GpuContext()
  ctx.allFnTab[callee.pName] = callee
  ctx.allFnTab[kernel.pName] = kernel

  materializeIndexBuiltinParamsImpl(ctx)

  # Declared params keep their positions, types, and unmarked symbols.
  doAssert kernel.pParams.len == 4,
    "kernel pParams = 2 declared + 2 builtin params, got " & $kernel.pParams.len
  doAssert paramNames(kernel) == @["data", "n",
    "threadgroups_per_grid", "thread_position_in_grid"],
    "declared params first, builtin params appended after"
  doAssert kernel.pParams[0].typ.kind == gtPtr,
    "declared buffer param keeps its pointer type"
  doAssert kernel.pParams[0].typ.to.kind == gtUint32
  doAssert kernel.pParams[1].typ.kind == gtUint32,
    "declared scalar param keeps its type"
  doAssert paramKinds(kernel) == @[GpuCoordBuiltinKind.gbkNone, GpuCoordBuiltinKind.gbkNone,
    gbkThreadgroupsPerGrid, gbkThreadPositionInGrid],
    "declared params stay gbkNone; builtin params carry the kinds"
  doAssert callee.pParams.len == 1,
    "callee device fn must gain its own builtin param"
  doAssert callee.pParams[0].ident.ident() == "thread_position_in_grid"
  doAssert callee.pParams[0].ident.symbol.coordBuiltin == gbkThreadPositionInGrid
  echo "  OK — kernel pParams: declared params keep positions/types, builtin params appended with kinds"

# ═══════════════════════════════════════════════════════════════════════════
# 3. Call site forwards the args in the callee's needs order
# ═══════════════════════════════════════════════════════════════════════════
block:
  var d1 = makeProc("d1",
    blockOf(builtinIdent("thread_position_in_grid", gbkThreadPositionInGrid)))
  var d2 = makeProc("d2",
    blockOf(
      builtinIdent("thread_position_in_threadgroup", gbkThreadPositionInThreadgroup),
      builtinIdent("thread_index_in_threadgroup", gbkThreadIndexInThreadgroup)
    ))
  var kernel = makeProc("kern2",
    blockOf(makeCall(d1), makeCall(d2)), isKernel = true)
  var ctx = GpuContext()
  ctx.allFnTab[d1.pName] = d1
  ctx.allFnTab[d2.pName] = d2
  ctx.allFnTab[kernel.pName] = kernel

  materializeIndexBuiltinParamsImpl(ctx)

  # Kernel params: first-seen over own refs (none) then callee subtrees.
  doAssert paramNames(kernel) == @[
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "thread_index_in_threadgroup"
  ], "kernel builtin params must follow callee order, deduped"
  doAssert paramKinds(kernel) == @[
    gbkThreadPositionInGrid,
    gbkThreadPositionInThreadgroup,
    gbkThreadIndexInThreadgroup
  ]
  # Call 1 forwards d1's single need; call 2 forwards d2's needs in d2's order.
  doAssert callArgNames(kernel, 0) == @["thread_position_in_grid"],
    "call to d1 must forward exactly d1's need"
  doAssert callArgNames(kernel, 1) == @[
    "thread_position_in_threadgroup", "thread_index_in_threadgroup"
  ], "call to d2 must forward d2's needs in d2's order"
  doAssert paramNames(d2) == @[
    "thread_position_in_threadgroup", "thread_index_in_threadgroup"
  ]
  doAssert paramKinds(d2) == @[gbkThreadPositionInThreadgroup, gbkThreadIndexInThreadgroup]
  echo "  OK — call sites forward the callee's needs in callee order"

# ═══════════════════════════════════════════════════════════════════════════
# 4. Shadowing local (coordBuiltin == gbkNone) is NOT collected
# ═══════════════════════════════════════════════════════════════════════════
block:
  let shadow = plainIdent("thread_position_in_grid")
  doAssert shadow.symbol.coordBuiltin == gbkNone,
    "a plain local must carry no coordinate kind"
  var fn = makeProc("shadow", blockOf(shadow))
  var ctx = GpuContext()
  ctx.allFnTab[fn.pName] = fn

  materializeIndexBuiltinParamsImpl(ctx)

  doAssert fn.pParams.len == 0,
    "a local shadowing a canonical name must not produce a builtin param"
  echo "  OK — shadowed builtin local (gbkNone) not collected"

# ═══════════════════════════════════════════════════════════════════════════
# 5. Recursion (a device fn calling itself) terminates
# ═══════════════════════════════════════════════════════════════════════════
block:
  var fn = makeProc("rec", GpuAst(kind: gpuBlock))
  fn.pBody = blockOf(
    builtinIdent("thread_index_in_threadgroup", gbkThreadIndexInThreadgroup),
    makeCall(fn)
  )
  var ctx = GpuContext()
  ctx.allFnTab[fn.pName] = fn

  materializeIndexBuiltinParamsImpl(ctx)

  doAssert fn.pParams.len == 1,
    "recursive fn must gain only its own body's need"
  doAssert fn.pParams[0].ident.ident() == "thread_index_in_threadgroup"
  doAssert fn.pParams[0].ident.symbol.coordBuiltin == gbkThreadIndexInThreadgroup
  doAssert callArgNames(fn, 1) == @["thread_index_in_threadgroup"],
    "the recursive call must forward the fn's own need"
  echo "  OK — recursive device fn terminates and rewrites consistently"

# ═══════════════════════════════════════════════════════════════════════════
# 6. A fn registered in both tables is rewritten once
# ═══════════════════════════════════════════════════════════════════════════
block:
  # Pulled-in module-level device fns land in `allFnTab` AND `genericInsts`
  # under the same symbol; the rewrite must not double-append.
  var fn = makeProc("both",
    blockOf(builtinIdent("thread_position_in_grid", gbkThreadPositionInGrid)))
  var ctx = GpuContext()
  ctx.allFnTab[fn.pName] = fn
  ctx.genericInsts[fn.pName] = fn

  materializeIndexBuiltinParamsImpl(ctx)

  doAssert fn.pParams.len == 1,
    "builtin param must be appended once despite two table entries"
  echo "  OK — fn in both tables rewritten once"

# ═══════════════════════════════════════════════════════════════════════════
# 7. Barrier call is not a callee
# ═══════════════════════════════════════════════════════════════════════════
block:
  let barrierSym = newSymbol("threadgroup_barrier", iSym = "barrier_isym",
                             symKind = gsBuiltin)
  barrierSym.synchroBuiltin = gbkThreadgroupBarrier
  let barrierCall = GpuAst(kind: gpuCall,
                           cName: GpuAst(kind: gpuIdent, symbol: barrierSym),
                           cArgs: @[])
  var fn = makeProc("bar",
    blockOf(barrierCall,
            builtinIdent("thread_position_in_grid", gbkThreadPositionInGrid)))
  var ctx = GpuContext()
  ctx.allFnTab[fn.pName] = fn

  materializeIndexBuiltinParamsImpl(ctx)

  doAssert fn.pParams.len == 1,
    "the barrier must not count as a callee need"
  doAssert fn.pParams[0].ident.ident() == "thread_position_in_grid"
  echo "  OK — barrier call skipped as a callee"

echo ""
echo "  All materializeIndexBuiltinParams tests passed."
