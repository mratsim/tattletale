## Run:
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/ir/validations/t_ir_validateSharedPushConstLayout.nim

import std/[tables, strutils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_validations_glsl
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

proc kernel(name: string, params: seq[(string, GpuType, AddressSpace)]): GpuAst =
  ## A `{.global.}` kernel proc with the given ordered params
  let fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = "fn_" & name)),
                  pAttributes: {attGlobal},
                  pBody: GpuAst(kind: gpuBlock))
  for (pname, ptyp, space) in params:
    let sym = newSymbol(pname, iSym = name & "_" & pname, typ = ptyp)
    fn.pParams.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym),
                            typ: ptyp, addressSpace: space, passByRef: false)
  result = fn

proc kernels(ctx: var GpuContext, fns: varargs[GpuAst]) =
  for fn in fns:
    ctx.allFnTab[fn.pName] = fn


# ═══════════════════════════════════════════════════════════════════════════
# 1. two kernels with agreeing ordered params pass
# ═══════════════════════════════════════════════════════════════════════════
static:
  let u32t = GpuType(kind: gtUint32)
  var ctx = GpuContext()
  ctx.kernels(
    kernel("ka", @[("out", u32t, asRMEM), ("val", u32t, asRMEM)]),
    kernel("kb", @[("out", u32t, asRMEM), ("val", u32t, asRMEM)]))
  registerGlslValidationPasses.runPassPrepared(ctx)
  echo "  OK: agreeing multi-kernel layouts pass"


# 3. smem params are excluded from the layout
# ═══════════════════════════════════════════════════════════════════════════
static:
  let u32t = GpuType(kind: gtUint32)
  var ctx = GpuContext()
  ctx.kernels(
    kernel("ka", @[("out", u32t, asRMEM), ("scratch", u32t, asSMEM)]),
    kernel("kb", @[("out", u32t, asRMEM)]))
  registerGlslValidationPasses.runPassPrepared(ctx)
  echo "  OK: smem params excluded from the layout"

# ═══════════════════════════════════════════════════════════════════════════
# 4. non-global fns are ignored
# ═══════════════════════════════════════════════════════════════════════════
static:
  let u32t = GpuType(kind: gtUint32)
  var ctx = GpuContext()
  ctx.kernels(kernel("ka", @[("out", u32t, asRMEM)]))
  let device = GpuAst(kind: gpuProc,
                      pName: GpuAst(kind: gpuIdent, symbol: newSymbol("dev", iSym = "fn_dev")),
                      pBody: GpuAst(kind: gpuBlock))
  device.pParams.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: newSymbol("x", iSym = "dev_x", typ = u32t)),
                              typ: u32t, addressSpace: asRMEM, passByRef: false)
  ctx.allFnTab[device.pName] = device
  registerGlslValidationPasses.runPassPrepared(ctx)
  echo "  OK: non-global fns ignored"

