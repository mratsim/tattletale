## Cross-space pointer-cast validation
##
## Test that a cast whose pointer target space differs from the operand's
## accessed space is rejected, same-space and provenance casts pass
##
## Run:
##   nim c --hints:off --warnings:off --out:build/wip/t_ir_spaceCastValidation \
##     --nimcache:nimcache/wipSpaceCastValidation \
##     -r workspace/crucible/tests/ir/types/t_ir_spaceCastValidation.nim

import workspace/crucible/src/codegen/passes/passes_validations {.all.}
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/tests/ir/ir_testutils

proc findFirst(n: GpuAst, kind: GpuNodeKind): GpuAst =
  ## First node of `kind` in depth-first pre-order, nil when absent
  if n.isNil:
    return nil
  if n.kind == kind:
    return n
  for ch in n.items:
    let found = findFirst(ch, kind)
    if not found.isNil:
      return found

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

# ── 1. Same-space cast passes: device operand, device cast target ──
static:
  let f32 = GpuType(kind: gtFloat32)
  let devPtr = initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice)
  let castNode = GpuAst(kind: gpuCast,
                        cTo: initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice),
                        cExpr: typedIdent("p", devPtr))
  discard registerValidateAddressSpaceCasts.runPass(fixtureFn(fixtureBlock(castNode)))
  echo "  OK - a cast whose operand and target are device pointers passes"

# ── 2. Provenance round-trip passes: cast[uint64](p) + off re-derives p's space ──
static:
  let f32 = GpuType(kind: gtFloat32)
  let devPtr = initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice)
  let toU64 = GpuAst(kind: gpuCast,
                     cTo: GpuType(kind: gtUint64),
                     cExpr: typedIdent("p", devPtr))
  let offset = GpuAst(kind: gpuBinOp,
                      bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+", iSym = "op_add")),
                      bLeft: toU64,
                      bRight: GpuAst(kind: gpuLit, lValue: "4", lType: GpuType(kind: gtUint64)),
                      bType: GpuType(kind: gtUint64))
  let roundTrip = GpuAst(kind: gpuCast,
                         cTo: initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice),
                         cExpr: offset)
  discard registerValidateAddressSpaceCasts.runPass(fixtureFn(fixtureBlock(roundTrip)))
  echo "  OK - an int-to-ptr cast keeps its operand's pointer provenance"

# ── 3. Thread-space cast passes: unprovenanced int, thread-space target ──
static:
  let f32 = GpuType(kind: gtFloat32)
  let thrPtr = initGpuPtrType(f32, implicitPtr = false, addressSpace = asRMEM)
  let fromInt = GpuAst(kind: gpuCast,
                       cTo: thrPtr,
                       cExpr: typedIdent("raw", GpuType(kind: gtUint64)))
  discard registerValidateAddressSpaceCasts.runPass(fixtureFn(fixtureBlock(fromInt)))
  echo "  OK - an unprovenanced int casts to a thread-space pointer"

# ── 4. Same-space conversion passes ──
static:
  let f32 = GpuType(kind: gtFloat32)
  let devPtr = initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice)
  let convNode = GpuAst(kind: gpuConv,
                        convTo: initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice),
                        convExpr: typedIdent("p", devPtr))
  discard registerValidateAddressSpaceCasts.runPass(fixtureFn(fixtureBlock(convNode)))
  echo "  OK - a conversion whose operand and target are device pointers passes"

# ── 5. Frontend kernel casts pass validation ──
static:
  let ir = toCrucibleAst:
    proc castKernel(outp: ptr UncheckedArray[float32]) {.global.} =
      var scratch {.smem.}: array[4, uint32]
      let d = cast[ptr float32](outp)
      let s = cast[ptr uint32](addr scratch[0])
      outp[0] = d[] + float32(s[])

  let fn = ir.findFirst(gpuProc)
  doAssert not fn.isNil, "expected the kernel proc"
  # the frontend derives every target from its operand, so the validation passes
  discard registerValidateAddressSpaceCasts.runPass(fn)
  echo "  OK - frontend casts carry their operand's space and pass validation"

# ── 6. Top-level statements: a cross-space cast outside any proc is rejected ──
static:
  let f32 = GpuType(kind: gtFloat32)
  let devPtr = initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice)
  let rmemPtr = initGpuPtrType(f32, implicitPtr = false, addressSpace = asRMEM)
  let castNode = GpuAst(kind: gpuCast,
                        cTo: devPtr,
                        cExpr: typedIdent("seed", GpuType(kind: gtUint64)))
  # global var blocks hold top-level statements, the validator sweeps them
  # like function bodies
  var ctx = GpuContext()
  ctx.globalBlocks.add fixtureBlock(castNode)
  proc registerSpaceCastValidation(reg: var PassRegistry) {.nimcall.} =
    reg.registerValidateAddressSpaceCasts()
  var rejected = false
  try:
    registerSpaceCastValidation.runPassPrepared(ctx)
  except ValueError:
    rejected = true
  doAssert rejected,
    "a cross-space cast at global scope must be rejected like any other"
  echo "  OK - a cross-space cast at global scope is rejected"

echo "t_ir_spaceCastValidation: all static assertions passed"
