## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_patchBoolToI32.nim

import std / [sets, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering_wgsl
import workspace/crucible/src/codegen/passes/passes_low_bools
import workspace/crucible/tests/ir/ir_testutils

proc registerWgslChain(reg: var PassRegistry) {.nimcall.} =
  reg.registerWgslLoweringPasses()
  reg.donePasses.incl("ensureBlock")
  reg.donePasses.incl("pruneCompileTimeParams")

proc body(stmts: varargs[GpuAst]): GpuAst =
  GpuAst(kind: gpuBlock, statements: @stmts)


proc findKind(n: GpuAst, kind: GpuNodeKind): GpuAst =
  ## First node of `kind` in the tree, nil when absent.
  ## Iterates the known child fields directly, the items iterator segfaults
  ## when it resumes on a node whose iterImpl variant is empty under recursion.
  if n.kind == kind:
    return n
  template walk(ch: GpuAst): untyped =
    if not ch.isNil:
      let hit = ch.findKind(kind)
      if hit != nil:
        return hit
  case n.kind
  of gpuProc:
    walk(n.pBody)
  of gpuBlock:
    for st in n.statements:
      walk(st)
  of gpuAssign:
    walk(n.aLeft)
    walk(n.aRight)
  of gpuConv:
    walk(n.convExpr)
  of gpuDeref:
    walk(n.dOf)
  of gpuIndex:
    walk(n.iArr)
    walk(n.iIndex)
  else: discard

proc ident(sym: Symbol): GpuAst = GpuAst(kind: gpuIdent, symbol: sym)

proc boolTyp(): GpuType = GpuType(kind: gtBool)

proc boolGlobal(ctx: var GpuContext, name: string): Symbol =
  ## A bool storage global, the pipeline state after param lifting
  result = newSymbol(name, iSym = name & "_h0", typ = boolTyp())
  ctx.globals[name & "_h0"] = GpuParam(ident: ident(result), typ: boolTyp(),
                                       addressSpace: asDevice)

proc lit(t: GpuType, v: string): GpuAst = GpuAst(kind: gpuLit, lType: t, lValue: v)

proc kernelFn(stmts: varargs[GpuAst]): GpuAst =
  ## A device fn, processed by the WGSL registry without
  ## the storage-address rewriting reserved for `{.global.}` kernels
  GpuAst(kind: gpuProc,
         pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k_h0")),
         pRetType: GpuType(kind: gtVoid),
         pBody: body(stmts))

proc globalKernelFn(params: seq[GpuParam], stmts: varargs[GpuAst]): GpuAst =
  ## A `{.global.}` kernel carrying its params, the SPIR-V shape where params
  ## stay on the proc and become push-constant members
  GpuAst(kind: gpuProc,
         pName: GpuAst(kind: gpuIdent, symbol: newSymbol("gk", iSym = "gk_h0")),
         pRetType: GpuType(kind: gtVoid),
         pParams: params,
         pAttributes: {attGlobal},
         pBody: body(stmts))

proc boolParam(name: string): GpuParam =
  GpuParam(ident: ident(newSymbol(name, iSym = name & "_h0", typ = boolTyp())),
           typ: boolTyp())

# ═══════════════════════════════════════════════════════════════════════
# 1. bool global: load use wraps in conv(gtBool), storage becomes i32
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let sym = ctx.boolGlobal("flag")
  let k = kernelFn(ident(sym))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerWgslChain.runPassPrepared(ctx)
  let use = k.pBody.findKind(gpuConv)
  doAssert use != nil, "load use should wrap in conv, none found"
  doAssert use.convTo.kind == gtBool
  doAssert use.convExpr.kind == gpuIdent
  doAssert ctx.globals["flag_h0"].typ.kind == gtInt32, "storage type should be i32"
  echo "  OK — bool global load wraps in conv(gtBool), storage becomes i32"

# ═══════════════════════════════════════════════════════════════════════
# 2. assignment to a bool global: RHS wraps in conv(gtInt32), target stays bare
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let sym = ctx.boolGlobal("flag")
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(sym), aRight: lit(boolTyp(), "true")))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerWgslChain.runPassPrepared(ctx)
  let assign = k.pBody.findKind(gpuAssign)
  doAssert assign != nil
  doAssert assign.aLeft.kind == gpuIdent, "assignment target keeps its storage form"
  doAssert assign.aRight.kind == gpuConv and assign.aRight.convTo.kind == gtInt32,
    "RHS should wrap in conv(gtInt32)"
  echo "  OK — store site wraps the RHS in conv(gtInt32)"

# ═══════════════════════════════════════════════════════════════════════
# 3. ptr bool global: deref load wraps in conv(gtBool), storage becomes i32
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let ptrBool = initGpuPtrType(boolTyp(), implicitPtr = false)
  let sym = newSymbol("pflag", iSym = "pflag_h0", typ = ptrBool)
  ctx.globals["pflag_h0"] = GpuParam(ident: ident(sym), typ: ptrBool, addressSpace: asDevice)
  let k = kernelFn(GpuAst(kind: gpuDeref, dOf: ident(sym)))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerWgslChain.runPassPrepared(ctx)
  let use = k.pBody.findKind(gpuConv)
  doAssert use != nil and use.convTo.kind == gtBool,
    "deref load should wrap in conv(gtBool)"
  doAssert use.convExpr.kind == gpuDeref, "the deref itself stays"
  doAssert ctx.globals["pflag_h0"].typ.kind == gtInt32,
    "storage type is final after the bundle, the pointer is stripped"
  echo "  OK — ptr bool deref wraps in conv(gtBool), storage becomes i32"

# ═══════════════════════════════════════════════════════════════════════
# 4. i32 global untouched
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let sym = newSymbol("val", iSym = "val_h0", typ = initGpuType(gtInt32))
  ctx.globals["val_h0"] = GpuParam(ident: ident(sym), typ: initGpuType(gtInt32),
                                   addressSpace: asDevice)
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(sym),
                          aRight: lit(initGpuType(gtInt32), "1")))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerWgslChain.runPassPrepared(ctx)
  doAssert k.pBody.findKind(gpuConv).isNil, "no conv nodes expected"
  doAssert ctx.globals["val_h0"].typ.kind == gtInt32
  echo "  OK — i32 global untouched"

static:
  # C8: bare bool RHS assigned to a bool local
  var ctx = GpuContext()
  let flag = ctx.boolGlobal("flag")
  let ySym = newSymbol("y", iSym = "y_h0", typ = boolTyp())
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(ySym), aRight: ident(flag)))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerWgslChain.runPassPrepared(ctx)
  let assign = k.pBody.statements[0]
  doAssert assign.kind == gpuAssign
  let loadWrap = assign.aRight
  doAssert loadWrap.kind == gpuConv and loadWrap.convTo.kind == gtBool,
    "bare bool RHS should wrap in conv(gtBool)"
  doAssert loadWrap.convExpr.kind == gpuIdent
  doAssert assign.aLeft.kind == gpuIdent, "target stays bare"
  echo "  OK - bare bool RHS wraps in conv(gtBool)"

static:
  # M8: the storage type strips to i32, the symbol keeps mutable and implicit
  var ctx = GpuContext()
  let ptrBool = initGpuPtrType(boolTyp(), implicitPtr = true)
  ptrBool.mutable = true
  let sym = newSymbol("pflag", iSym = "pflag_h0", typ = ptrBool)
  ctx.globals["pflag_h0"] = GpuParam(ident: ident(sym), typ: ptrBool, addressSpace: asDevice)
  let k = kernelFn(GpuAst(kind: gpuDeref, dOf: ident(sym)))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerWgslChain.runPassPrepared(ctx)
  doAssert ctx.globals["pflag_h0"].typ.kind == gtInt32,
    "storage type is final after the bundle, the pointer is stripped"
  doAssert sym.typ.mutable, "mutable stays on the symbol type"
  doAssert sym.typ.implicit, "implicit stays on the symbol type"
  echo "  OK - storage strips to i32, the symbol keeps mutable and implicit"

static:
  # 5. bool kernel param: param declares i32, use wraps in conv(gtBool)
  proc boolOnly(reg: var PassRegistry) {.nimcall.} =
    reg.registerPatchBoolToI32()
    reg.donePasses.incl("makeCodeValid")
  var ctx = GpuContext()
  let p = boolParam("flag")
  let k = globalKernelFn(@[p], ident(p.ident.symbol))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  boolOnly.runPassPrepared(ctx)
  let param = k.pParams[0]
  doAssert param.typ.kind == gtInt32, "param storage type should be i32"
  doAssert param.ident.symbol.typ.kind == gtInt32, "param symbol type should be i32"
  let use = k.pBody.findKind(gpuConv)
  doAssert use != nil and use.convTo.kind == gtBool, "param load should wrap in conv(gtBool)"
  doAssert use.convExpr.kind == gpuIdent and use.convExpr.symbol == param.ident.symbol
  echo "  OK - bool kernel param declares i32, its use wraps in conv(gtBool)"

static:
  # 6. hazard negative: an i32 kernel param passes through untouched
  proc boolOnly2(reg: var PassRegistry) {.nimcall.} =
    reg.registerPatchBoolToI32()
    reg.donePasses.incl("makeCodeValid")
  var ctx = GpuContext()
  let p = GpuParam(ident: ident(newSymbol("n", iSym = "n_h0", typ = initGpuType(gtInt32))),
                   typ: initGpuType(gtInt32))
  let k = globalKernelFn(@[p], ident(p.ident.symbol))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  boolOnly2.runPassPrepared(ctx)
  doAssert k.pParams[0].typ.kind == gtInt32, "i32 param storage stays i32"
  doAssert k.pBody.findKind(gpuConv).isNil, "no conv nodes expected"
  echo "  OK - i32 kernel param untouched"

echo "  OK - t_ir_patchBoolToI32"
