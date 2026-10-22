## Run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/ir/validations/t_ir_validateSharedPushConstLayout.nim (from tattletale)

import std/[tables, strutils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_validations_glsl
import workspace/crucible/src/codegen/passes/pass_registry

proc runOne(ctx: var GpuContext) =
  ## Runs the single pass under test through the public registry.
  var reg = PassRegistry.new()
  reg.registerGlslValidationPasses()
  ctx.runPasses(reg)

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

let u32 = GpuType(kind: gtUint32)

# ═══════════════════════════════════════════════════════════════════════════
# 1. two kernels with agreeing ordered params pass
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  ctx.kernels(
    kernel("ka", @[("out", u32, asRMEM), ("val", u32, asRMEM)]),
    kernel("kb", @[("out", u32, asRMEM), ("val", u32, asRMEM)]))
  runOne(ctx)
  echo "  OK: agreeing multi-kernel layouts pass"

# ═══════════════════════════════════════════════════════════════════════════
# 2. divergent ordered params are rejected
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  ctx.kernels(
    kernel("ka", @[("out", u32, asRMEM), ("val", u32, asRMEM)]),
    kernel("kb", @[("out", u32, asRMEM), ("a", u32, asRMEM), ("b", u32, asRMEM)]))
  try:
    runOne(ctx)
    doAssert false, "divergent layouts must be rejected"
  except AssertionDefect as e:
    doAssert "push-constant block" in e.msg, "unexpected error: " & e.msg
  echo "  OK: divergent layouts rejected"

# ═══════════════════════════════════════════════════════════════════════════
# 3. smem params are excluded from the layout
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  ctx.kernels(
    kernel("ka", @[("out", u32, asRMEM), ("scratch", u32, asSMEM)]),
    kernel("kb", @[("out", u32, asRMEM)]))
  runOne(ctx)
  echo "  OK: smem params excluded from the layout"

# ═══════════════════════════════════════════════════════════════════════════
# 4. non-global fns are ignored
# ═══════════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  ctx.kernels(kernel("ka", @[("out", u32, asRMEM)]))
  let device = GpuAst(kind: gpuProc,
                      pName: GpuAst(kind: gpuIdent, symbol: newSymbol("dev", iSym = "fn_dev")),
                      pBody: GpuAst(kind: gpuBlock))
  device.pParams.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: newSymbol("x", iSym = "dev_x", typ = u32)),
                              typ: u32, addressSpace: asRMEM, passByRef: false)
  ctx.allFnTab[device.pName] = device
  runOne(ctx)
  echo "  OK: non-global fns ignored"

echo "  OK - t_ir_validateSharedPushConstLayout"
