## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_patchBoolToI32.nim

import std / [sets, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering_wgsl

proc boolTyp(): GpuType = GpuType(kind: gtBool)

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

proc lit(t: GpuType, v: string): GpuAst = GpuAst(kind: gpuLit, lType: t, lValue: v)

proc kernelFn(stmts: varargs[GpuAst]): GpuAst =
  ## A device fn, processed by the WGSL registry without
  ## the storage-address rewriting reserved for `{.global.}` kernels
  GpuAst(kind: gpuProc,
         pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k_h0")),
         pRetType: GpuType(kind: gtVoid),
         pBody: body(stmts))

proc runWgslChain(ctx: var GpuContext) =
  ## Runs the WGSL registry, the public pipeline entry
  var reg = PassRegistry.new()
  reg.registerWgslLoweringPasses()
  # the fixture satisfies the common-pipeline preconditions by construction:
  # its body is a block, its params are runtime values
  reg.donePasses.incl("ensureBlock")
  reg.donePasses.incl("pruneCompileTimeParams")
  ctx.runPasses(reg)

proc boolGlobal(ctx: var GpuContext, name: string): Symbol =
  ## A bool storage global, the pipeline state after param lifting
  result = newSymbol(name, iSym = name & "_h0", typ = boolTyp())
  ctx.globals[name & "_h0"] = GpuParam(ident: ident(result), typ: boolTyp(),
                                       addressSpace: asDevice)

# ═══════════════════════════════════════════════════════════════════════════
# 1. bool global: load use wraps in conv(gtBool), storage becomes i32
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let sym = ctx.boolGlobal("flag")
  let k = kernelFn(ident(sym))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  ctx.runWgslChain()
  let use = k.pBody.findKind(gpuConv)
  doAssert use != nil, "load use should wrap in conv, none found"
  doAssert use.convTo.kind == gtBool
  doAssert use.convExpr.kind == gpuIdent
  doAssert ctx.globals["flag_h0"].typ.kind == gtInt32, "storage type should be i32"
  echo "  OK — bool global load wraps in conv(gtBool), storage becomes i32"

# ═══════════════════════════════════════════════════════════════════════════
# 2. assignment to a bool global: RHS wraps in conv(gtInt32), target stays bare
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let sym = ctx.boolGlobal("flag")
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(sym), aRight: lit(boolTyp(), "true")))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  ctx.runWgslChain()
  let assign = k.pBody.findKind(gpuAssign)
  doAssert assign != nil
  doAssert assign.aLeft.kind == gpuIdent, "assignment target keeps its storage form"
  doAssert assign.aRight.kind == gpuConv and assign.aRight.convTo.kind == gtInt32,
    "RHS should wrap in conv(gtInt32)"
  echo "  OK — store site wraps the RHS in conv(gtInt32)"

# ═══════════════════════════════════════════════════════════════════════════
# 3. ptr bool global: deref load wraps in conv(gtBool), storage becomes ptr i32
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let ptrBool = initGpuPtrType(boolTyp(), implicitPtr = false)
  let sym = newSymbol("pflag", iSym = "pflag_h0", typ = ptrBool)
  ctx.globals["pflag_h0"] = GpuParam(ident: ident(sym), typ: ptrBool, addressSpace: asDevice)
  let k = kernelFn(GpuAst(kind: gpuDeref, dOf: ident(sym)))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  ctx.runWgslChain()
  let use = k.pBody.findKind(gpuConv)
  doAssert use != nil and use.convTo.kind == gtBool,
    "deref load should wrap in conv(gtBool)"
  doAssert use.convExpr.kind == gpuDeref, "the deref itself stays"
  doAssert ctx.globals["pflag_h0"].typ.kind == gtPtr
  doAssert ctx.globals["pflag_h0"].typ.to.kind == gtInt32
  echo "  OK — ptr bool deref wraps in conv(gtBool), storage becomes ptr i32"

# ═══════════════════════════════════════════════════════════════════════════
# 4. i32 global untouched
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let sym = newSymbol("val", iSym = "val_h0", typ = initGpuType(gtInt32))
  ctx.globals["val_h0"] = GpuParam(ident: ident(sym), typ: initGpuType(gtInt32),
                                   addressSpace: asDevice)
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(sym),
                          aRight: lit(initGpuType(gtInt32), "1")))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  ctx.runWgslChain()
  doAssert k.pBody.findKind(gpuConv).isNil, "no conv nodes expected"
  doAssert ctx.globals["val_h0"].typ.kind == gtInt32
  echo "  OK — i32 global untouched"

block:
  # C8: bare bool RHS assigned to a bool local
  var ctx = GpuContext()
  let flag = ctx.boolGlobal("flag")
  let ySym = newSymbol("y", iSym = "y_h0", typ = boolTyp())
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(ySym), aRight: ident(flag)))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  ctx.runWgslChain()
  let assign = k.pBody.statements[0]
  doAssert assign.kind == gpuAssign
  let loadWrap = assign.aRight
  doAssert loadWrap.kind == gpuConv and loadWrap.convTo.kind == gtBool,
    "bare bool RHS should wrap in conv(gtBool)"
  doAssert loadWrap.convExpr.kind == gpuIdent
  doAssert assign.aLeft.kind == gpuIdent, "target stays bare"
  echo "  OK - bare bool RHS wraps in conv(gtBool)"

block:
  # M8: mutable and implicit survive the ptr bool to ptr i32 conversion
  var ctx = GpuContext()
  let ptrBool = initGpuPtrType(boolTyp(), implicitPtr = true)
  ptrBool.mutable = true
  let sym = newSymbol("pflag", iSym = "pflag_h0", typ = ptrBool)
  ctx.globals["pflag_h0"] = GpuParam(ident: ident(sym), typ: ptrBool, addressSpace: asDevice)
  let k = kernelFn(GpuAst(kind: gpuDeref, dOf: ident(sym)))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  ctx.runWgslChain()
  doAssert ctx.globals["pflag_h0"].typ.to.kind == gtInt32
  doAssert ctx.globals["pflag_h0"].typ.mutable, "mutable must survive the conversion"
  doAssert ctx.globals["pflag_h0"].typ.implicit, "implicit must survive the conversion"
  echo "  OK - ptr bool to ptr i32 keeps mutable and implicit"

echo "  OK - t_ir_patchBoolToI32"
