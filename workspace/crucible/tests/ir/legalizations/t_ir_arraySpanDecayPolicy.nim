## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/legalizations/t_ir_arraySpanDecayPolicy.nim

import std / [tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/pass_registry

proc boolParamFn(iSym: string): GpuAst =
  ## A kernel fn with one `var array[4, int32]` param, indexed once in the body
  let arrTyp = initGpuPtrType(GpuType(kind: gtArray, aTyp: initGpuType(gtInt32), aLen: 4),
                              implicitPtr = false)
  let sym = newSymbol("buf", iSym = iSym, typ = arrTyp)
  var fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k_" & iSym)),
                  pRetType: GpuType(kind: gtVoid),
                  pBody: GpuAst(kind: gpuBlock))
  fn.pParams.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: arrTyp,
                          addressSpace: asDevice)
  fn.pBody.statements.add GpuAst(kind: gpuIndex, iArr: GpuAst(kind: gpuIdent, symbol: sym),
                                 iIndex: GpuAst(kind: gpuLit, lValue: "0", lType: initGpuType(gtInt32)))
  result = fn

proc runPolicy(decay: static bool): GpuAst =
  var ctx = GpuContext()
  let fn = boolParamFn("f" & $int(decay))
  ctx.allFnTab[fn.pName] = fn
  var reg = PassRegistry.new()
  reg.registerNormalizeArraySpanParams(decayArrays = decay)
  ctx.runPasses(reg)
  ctx.allFnTab[fn.pName]

# ═══════════════════════════════════════════════════════════════════════════
# decay = true (C model): the param becomes an element pointer
# ═══════════════════════════════════════════════════════════════════════════
block:
  let fn = runPolicy(true)
  let p = fn.pParams[0]
  doAssert p.typ.kind == gtPtr and p.typ.to.kind == gtInt32,
    "decaying policy: gtPtr(gtArray) becomes gtPtr(T), got " & $p.typ.kind
  echo "  OK — decay=true: param is gtPtr(gtInt32)"

# ═══════════════════════════════════════════════════════════════════════════
# decay = false (array first-class): the param keeps its array-ness
# ═══════════════════════════════════════════════════════════════════════════
block:
  let fn = runPolicy(false)
  let p = fn.pParams[0]
  doAssert p.typ.kind == gtPtr and p.typ.to.kind == gtArray,
    "non-decaying policy: gtPtr(gtArray) is preserved, got " & $p.typ.kind
  doAssert p.typ.to.aTyp.kind == gtInt32
  echo "  OK — decay=false: param stays gtPtr(gtArray)"

echo "  OK - t_ir_arraySpanDecayPolicy"
