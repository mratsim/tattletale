## Implicit cross-space pointer casts are rejected
##
## Manual test: a cast with a device-space target over thread-space
## storage must fail compilation with the cross-space cast validation error.
##
## Expected: compile fails at the runPass call, carrying
##   "cross-space cast: the operand accesses asRMEM storage, the target space is asDevice"
##
## Run (must FAIL):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/validations/manual_ir_spaceCastCrossSpace.nim

import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/tests/ir/ir_testutils

proc fixtureBlock(stmts: varargs[GpuAst]): GpuAst =
  GpuAst(kind: gpuBlock, statements: @stmts)

static:
  let f32 = GpuType(kind: gtFloat32)
  # the C-style shape: a thread-space operand behind a device-space target,
  # the k_gemm failure class the Nim surface cannot express
  let castNode = GpuAst(kind: gpuCast,
                        cTo: initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice),
                        cExpr: GpuAst(kind: gpuIdent, symbol: newSymbol("p",
                          iSym = "sym_p",
                          typ = initGpuPtrType(f32, implicitPtr = false, addressSpace = asRMEM))))
  let fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol("f", iSym = "fn_f")),
                  pRetType: GpuType(kind: gtVoid),
                  pBody: fixtureBlock(castNode))
  discard registerValidateAddressSpaceCasts.runPass(fn)
