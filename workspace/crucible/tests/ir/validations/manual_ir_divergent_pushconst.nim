## Divergent multi-kernel push-constant layouts are rejected
##
## Manual test: two global kernels with different ordered params must fail
## compilation with the shared push-constant block validation error.
##
## Expected: compile fails at the runPassPrepared call, carrying
##   "Multi-kernel GLSL sources share one push-constant block"
##
## Run (must FAIL):
##   nim c --hints:off --warnings:off ##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir ##     workspace/crucible/tests/ir/validations/manual_ir_divergent_pushconst.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_validations_glsl {.all.}
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

proc kernel(name: string, params: seq[(string, GpuType, AddressSpace)]): GpuAst =
  let fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = "fn_" & name)),
                  pAttributes: {attGlobal},
                  pBody: GpuAst(kind: gpuBlock))
  for (pname, ptyp, space) in params:
    let sym = newSymbol(pname, iSym = name & "_" & pname, typ = ptyp)
    fn.pParams.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym),
                            typ: ptyp, addressSpace: space, passByRef: false)
  result = fn

static:
  let u32t = GpuType(kind: gtUint32)
  var ctx = GpuContext()
  let ka = kernel("ka", @[("out", u32t, asRMEM), ("val", u32t, asRMEM)])
  let kb = kernel("kb", @[("out", u32t, asRMEM), ("a", u32t, asRMEM), ("b", u32t, asRMEM)])
  ctx.allFnTab[ka.pName] = ka
  ctx.allFnTab[kb.pName] = kb
  registerGlslValidationPasses.runPassPrepared(ctx)
