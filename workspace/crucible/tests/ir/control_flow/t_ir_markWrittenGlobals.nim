## markWrittenGlobals marks only written globals kVar
##
## Test that the write analysis upgrades exactly the assigned globals,
## read-only globals stay kIn:
##
## - `output[indices[0]] = 1` writes `output`, only READS `indices`
##   (the index expression is a read, not a write)
## - `scratch[output[0]] = 2` writes `scratch`, only READS `output`
## - the WGSL printer derives `read_write` from kVar, so a wrongly upgraded
##   read-only global would over-restrict the binding
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_markWrittenGlobals.nim

import std / [sets, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering_wgsl {.all.}
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

proc registerMarkWritten(reg: var PassRegistry) {.nimcall.} =
  reg.registerMarkWrittenGlobals()
  # the analysis runs late in the WGSL bundle, its deps are fixture-free here
  reg.donePasses.incl "makeCodeValid"

proc ptrGlobal(ctx: var GpuContext, name: string, elem: GpuType): Symbol =
  let typ = initGpuPtrType(elem, implicitPtr = false)
  result = newSymbol(name, iSym = name & "_h0", typ = typ)
  ctx.globals[name & "_h0"] = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: result),
                                       typ: typ, addressSpace: asDevice)

proc ident(sym: Symbol): GpuAst = GpuAst(kind: gpuIdent, symbol: sym)

proc lit(t: GpuType, v: string): GpuAst = GpuAst(kind: gpuLit, lType: t, lValue: v)

proc index(base: GpuAst, i: GpuAst): GpuAst =
  GpuAst(kind: gpuIndex, iArr: base, iIndex: i)

proc kernel(assign: GpuAst): GpuAst =
  GpuAst(kind: gpuProc,
         pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k")),
         pRetType: GpuType(kind: gtVoid),
         pBody: GpuAst(kind: gpuBlock, statements: @[assign]))

proc i32(): GpuType = GpuType(kind: gtInt32)

static:
  # ── output[indices[0]] = 1: output written, indices read-only ──
  var ctx = GpuContext()
  let outSym = ctx.ptrGlobal("output", initGpuType(gtFloat32))
  let idxSym = ctx.ptrGlobal("indices", initGpuType(gtInt32))
  let k = kernel(GpuAst(kind: gpuAssign,
                        aLeft: index(ident(outSym), index(ident(idxSym), lit(i32(), "0"))),
                        aRight: lit(initGpuType(gtFloat32), "1")))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerMarkWritten.runPassPrepared(ctx)
  doAssert ctx.globals["output_h0"].flavor == kVar,
    "the assigned base must be kVar, got " & $ctx.globals["output_h0"].flavor
  doAssert ctx.globals["indices_h0"].flavor == kIn,
    "the index expression is a read, the global must stay kIn, got " &
    $ctx.globals["indices_h0"].flavor
  echo "  OK — output kVar, indices stays kIn"

static:
  # ── scratch[output[0]] = 2: scratch written, output read-only here ──
  var ctx = GpuContext()
  let outSym = ctx.ptrGlobal("output", initGpuType(gtFloat32))
  let scrSym = ctx.ptrGlobal("scratch", initGpuType(gtFloat32))
  let k = kernel(GpuAst(kind: gpuAssign,
                        aLeft: index(ident(scrSym), index(ident(outSym), lit(i32(), "0"))),
                        aRight: lit(initGpuType(gtFloat32), "2")))
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerMarkWritten.runPassPrepared(ctx)
  doAssert ctx.globals["scratch_h0"].flavor == kVar,
    "the assigned base must be kVar, got " & $ctx.globals["scratch_h0"].flavor
  doAssert ctx.globals["output_h0"].flavor == kIn,
    "a global read as an index must stay kIn, got " & $ctx.globals["output_h0"].flavor
  echo "  OK — scratch kVar, output stays kIn"

echo "t_ir_markWrittenGlobals: all static assertions passed"
