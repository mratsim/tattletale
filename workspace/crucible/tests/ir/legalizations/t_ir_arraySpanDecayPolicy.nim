## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/legalizations/t_ir_arraySpanDecayPolicy.nim

import std / [tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

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

proc registerDecayTrue(reg: var PassRegistry) {.nimcall.} =
  reg.registerNormalizeArraySpanParams(decayArrays = true)

proc registerDecayFalse(reg: var PassRegistry) {.nimcall.} =
  reg.registerNormalizeArraySpanParams(decayArrays = false)

proc runPolicy(decay: static bool): GpuAst =
  let fn = boolParamFn("f" & $int(decay))
  when decay:
    discard registerDecayTrue.runPass(fn)
  else:
    discard registerDecayFalse.runPass(fn)
  fn

# ═══════════════════════════════════════════════════════════════════════════
# decay = true (C model): the param becomes an element pointer
# ═══════════════════════════════════════════════════════════════════════════
static:
  let fn = runPolicy(true)
  let p = fn.pParams[0]
  doAssert p.typ.kind == gtPtr and p.typ.to.kind == gtInt32,
    "decaying policy: gtPtr(gtArray) becomes gtPtr(T), got " & $p.typ.kind
  echo "  OK — decay=true: param is gtPtr(gtInt32)"

# ═══════════════════════════════════════════════════════════════════════════
# decay = false (array first-class): the param keeps its array-ness
# ═══════════════════════════════════════════════════════════════════════════
static:
  let fn = runPolicy(false)
  let p = fn.pParams[0]
  doAssert p.typ.kind == gtPtr and p.typ.to.kind == gtArray,
    "non-decaying policy: gtPtr(gtArray) is preserved, got " & $p.typ.kind
  doAssert p.typ.to.aTyp.kind == gtInt32
  echo "  OK — decay=false: param stays gtPtr(gtArray)"

echo "  OK - t_ir_arraySpanDecayPolicy"
