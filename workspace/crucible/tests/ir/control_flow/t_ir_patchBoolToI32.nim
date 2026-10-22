## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_patchBoolToI32.nim

import std / [sets, strutils, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/pass_registry

proc body(stmts: varargs[GpuAst]): GpuAst =
  GpuAst(kind: gpuBlock, statements: @stmts)

proc fnKey(name: string): GpuAst =
  GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = name))

proc findKind(n: GpuAst, kind: GpuNodeKind): GpuAst =
  ## First node of `kind` in the tree, nil when absent
  if n.kind == kind: return n
  for ch in n.items:
    if ch == nil: continue
    let hit = ch.findKind(kind)
    if hit != nil: return hit
proc kernelFn(stmts: varargs[GpuAst]): GpuAst =
  ## A kernel proc wrapping the statements, the shape the pipeline expects
  GpuAst(kind: gpuProc,
         pName: fnKey("k"),
         pRetType: GpuType(kind: gtVoid),
         pBody: body(stmts))

proc boolGlobal(iSym: string): (Symbol, GpuParam) =
  let typ = GpuType(kind: gtBool)
  let sym = newSymbol(iSym.split('_')[0], iSym = iSym, typ = typ)
  result = (sym, GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: typ,
                          addressSpace: asDevice))

# ═══════════════════════════════════════════════════════════════════════════
# 1. Bool global: load use gets conv(gtBool), storage type becomes i32
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let (sym, g) = boolGlobal("flag_h1")
  ctx.globals["flag_h1"] = g
  var n = GpuAst(kind: gpuIdent, symbol: sym)
  var fn = body(n)
  let k = kernelFn(fn.statements)
  ctx.fnTab[fnKey("k")] = k
  ctx.allFnTab[fnKey("k")] = k
  var reg = PassRegistry.new()
  reg.registerWgslLoweringPasses()
  # the fixture satisfies the common-pipeline preconditions by construction:
  # its body is a block, its params are runtime values
  reg.donePasses.incl("ensureBlock")
  reg.donePasses.incl("pruneCompileTimeParams")
  # the fixture satisfies the common-pipeline preconditions by construction:
  # its body is a block, its params are runtime values
  reg.donePasses.incl("ensureBlock")
  reg.donePasses.incl("pruneCompileTimeParams")
  ctx.runPasses(reg)
  let use = k.findKind(gpuConv)
  doAssert use != nil, "load use should wrap in conv, none found"
  doAssert use.convTo.kind == gtBool
  doAssert use.convExpr.kind == gpuIdent and use.convExpr.symbol == sym
  doAssert ctx.globals["flag_h1"].typ.kind == gtInt32, "storage type should be i32"
  echo "  OK — bool global load wraps in conv(gtBool), storage becomes i32"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Assignment to a bool global: RHS gets conv(gtInt32), LHS stays storage
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let (sym, g) = boolGlobal("flag_h2")
  ctx.globals["flag_h2"] = g
  let lhs = GpuAst(kind: gpuIdent, symbol: sym)
  let rhs = GpuAst(kind: gpuLit, lType: GpuType(kind: gtBool), lValue: "true")
  var fn = body(GpuAst(kind: gpuAssign, aLeft: lhs, aRight: rhs))
  let k = kernelFn(fn.statements)
  ctx.fnTab[fnKey("k")] = k
  ctx.allFnTab[fnKey("k")] = k
  var reg = PassRegistry.new()
  reg.registerPatchBoolToI32()
  reg.donePasses.incl("makeCodeValid")  # unit test: the pass runs alone
  ctx.runPasses(reg)
  let assign = k.findKind(gpuAssign)
  doAssert assign != nil
  doAssert assign.aLeft.kind == gpuIdent, "assignment target keeps its storage form"
  doAssert assign.aRight.kind == gpuConv and assign.aRight.convTo.kind == gtInt32,
    "RHS should wrap in conv(gtInt32)"
  echo "  OK — store site wraps the RHS in conv(gtInt32)"

# ═══════════════════════════════════════════════════════════════════════════
# 3. ptr bool global: deref load gets conv(gtBool), pointer storage becomes ptr i32
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let ptrBool = GpuType(kind: gtPtr, to: GpuType(kind: gtBool))
  let sym = newSymbol("pflag", iSym = "pflag_h3", typ = ptrBool)
  ctx.globals["pflag_h3"] = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym),
                                     typ: ptrBool, addressSpace: asDevice)
  let deref = GpuAst(kind: gpuDeref, dOf: GpuAst(kind: gpuIdent, symbol: sym))
  var fn = body(deref)
  let k = kernelFn(fn.statements)
  ctx.fnTab[fnKey("k")] = k
  ctx.allFnTab[fnKey("k")] = k
  var reg = PassRegistry.new()
  reg.registerPatchBoolToI32()
  reg.donePasses.incl("makeCodeValid")  # unit test: the pass runs alone
  ctx.runPasses(reg)
  let use = k.findKind(gpuConv)
  doAssert use != nil and use.convTo.kind == gtBool,
    "deref load should wrap in conv(gtBool)"
  doAssert use.convExpr.kind == gpuDeref, "the deref itself stays"
  doAssert ctx.globals["pflag_h3"].typ.kind == gtPtr
  doAssert ctx.globals["pflag_h3"].typ.to.kind == gtInt32
  echo "  OK — ptr bool deref wraps in conv(gtBool), storage becomes ptr i32"

# ═══════════════════════════════════════════════════════════════════════════
# 4. Non-global and i32 globals stay unchanged
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let localSym = newSymbol("localFlag", iSym = "local_flag_h4", typ = GpuType(kind: gtBool))
  let intSym = newSymbol("val", iSym = "val_h5", typ = GpuType(kind: gtInt32))
  ctx.globals["val_h5"] = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: intSym),
                                   typ: GpuType(kind: gtInt32), addressSpace: asDevice)
  # negative controls as legal statements, neither may gain conv nodes:
  # a bool local assignment and an i32 global assignment
  var fn = body(GpuAst(kind: gpuAssign,
                       aLeft: GpuAst(kind: gpuIdent, symbol: localSym),
                       aRight: GpuAst(kind: gpuLit, lType: GpuType(kind: gtBool), lValue: "true")),
                GpuAst(kind: gpuAssign,
                       aLeft: GpuAst(kind: gpuIdent, symbol: intSym),
                       aRight: GpuAst(kind: gpuLit, lType: GpuType(kind: gtInt32), lValue: "1")))
  let k = kernelFn(fn.statements)
  ctx.fnTab[fnKey("k")] = k
  ctx.allFnTab[fnKey("k")] = k
  var reg = PassRegistry.new()
  reg.registerPatchBoolToI32()
  reg.donePasses.incl("makeCodeValid")  # unit test: the pass runs alone
  ctx.runPasses(reg)
  for stmt in k.pBody.statements:
    doAssert stmt.kind == gpuAssign, "statement shape kept, got: " & $stmt.kind
    doAssert stmt.aLeft.kind == gpuIdent, "assignment target stays a plain ident"
    doAssert stmt.aRight.kind == gpuLit, "RHS stays a plain literal"
  doAssert ctx.globals["val_h5"].typ.kind == gtInt32
  echo "  OK — non-global bool and i32 globals unchanged"

echo ""
echo "  All patchBoolToI32 tests passed."
