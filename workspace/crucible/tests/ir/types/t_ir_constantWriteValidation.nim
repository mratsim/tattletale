## Writes to constant storage are rejected
##
## Test that an assignment whose target designates constant storage is rejected:
## - a top-level global, directly or through a pointer
## - writes to thread and device storage pass
##
## Run:
##   nim c --hints:off --warnings:off --out:build/wip/t_ir_constantWriteValidation \
##     --nimcache:nimcache/wipConstantWriteValidation \
##     -r workspace/crucible/tests/ir/types/t_ir_constantWriteValidation.nim

import std/tables
import workspace/crucible/src/codegen/passes/passes_validations {.all.}
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/tests/ir/ir_testutils

proc fnIdent(name: string): GpuAst =
  ## Map key and proc name of a fixture fn
  GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = "fn_" & name))

proc typedIdent(name: string, typ: GpuType): GpuAst =
  ## An ident carrying a fresh symbol of the given type
  GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = "sym_" & name, typ = typ))

proc fixtureFn(body: GpuAst): GpuAst =
  ## A device fn wrapping the given body, the runPass fixture shape
  GpuAst(kind: gpuProc,
         pName: GpuAst(kind: gpuIdent, symbol: newSymbol("f", iSym = "fn_f")),
         pRetType: GpuType(kind: gtVoid),
         pBody: body)

proc fixtureBlock(stmts: varargs[GpuAst]): GpuAst =
  GpuAst(kind: gpuBlock, statements: @stmts)

proc assign(target, value: GpuAst): GpuAst =
  GpuAst(kind: gpuAssign, aLeft: target, aOp: "=", aRight: value)

proc registerConstantWriteValidation(reg: var PassRegistry) {.nimcall.} =
  reg.registerValidateConstantWrites()

proc rejects(ctx: var GpuContext): bool =
  try:
    registerConstantWriteValidation.runPassPrepared(ctx)
  except ValueError:
    return true
  return false

# ── 1. A write to a global is rejected ──
static:
  let global = typedIdent("counter", GpuType(kind: gtInt32, addressSpace: asConstant))
  var ctx = GpuContext()
  ctx.allFnTab[fnIdent("f")] = fixtureFn(
    fixtureBlock(assign(global, GpuAst(kind: gpuLit, lValue: "1", lType: GpuType(kind: gtInt32)))))
  doAssert ctx.rejects(), "a write to a global must be rejected"
  echo "  OK - a write to a global is rejected"

# ── 2. A write through a pointer to constant storage is rejected ──
static:
  let f32 = GpuType(kind: gtFloat32)
  let constPtr = initGpuPtrType(f32, implicitPtr = false, addressSpace = asConstant)
  let ptrLocal = typedIdent("p", constPtr)
  var ctx = GpuContext()
  ctx.allFnTab[fnIdent("f")] = fixtureFn(
    fixtureBlock(assign(GpuAst(kind: gpuDeref, dOf: ptrLocal),
                        GpuAst(kind: gpuLit, lValue: "1.0", lType: f32))))
  doAssert ctx.rejects(), "a write through a constant pointer must be rejected"
  echo "  OK - a write through a constant pointer is rejected"

# ── 3. Writes to thread and device storage pass ──
static:
  let f32 = GpuType(kind: gtFloat32)
  let devPtr = initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice)
  var ctx = GpuContext()
  ctx.allFnTab[fnIdent("f")] = fixtureFn(
    fixtureBlock(
      assign(typedIdent("local", f32),
             GpuAst(kind: gpuLit, lValue: "1.0", lType: f32)),
      assign(GpuAst(kind: gpuDeref, dOf: typedIdent("outp", devPtr)),
             GpuAst(kind: gpuLit, lValue: "2.0", lType: f32))))
  doAssert not ctx.rejects(), "thread and device writes stay legal"
  echo "  OK - writes to thread and device storage pass"

echo "t_ir_constantWriteValidation: all static assertions passed"
