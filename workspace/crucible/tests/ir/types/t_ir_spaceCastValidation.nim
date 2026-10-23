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

# ── 5. Frontend casts validate: device operand and smem operand ──
static:
  let ir = toCrucibleAst:
    proc castKernel(outp: ptr UncheckedArray[float32]) {.global.} =
      var scratch {.smem.}: array[4, uint32]
      let d = cast[ptr float32](outp)
      let s = cast[ptr uint32](addr scratch[0])
      outp[0] = d[] + float32(s[])

  let fn = ir.findFirst(gpuProc)
  doAssert not fn.isNil, "expected the kernel proc"
  var casts: seq[GpuAst]
  var n = fn
  proc collectCasts(n: GpuAst) =
    if n.kind == gpuCast:
      casts.add n
    for ch in n.items:
      collectCasts(ch)
  n.collectCasts()
  doAssert casts.len == 2, "expected the device and smem casts, got " & $casts.len
  doAssert casts[0].cTo.addressSpace == asDevice,
    "the frontend stamps the cast target from the device operand, got " &
    $casts[0].cTo.addressSpace
  doAssert casts[1].cTo.addressSpace == asSMEM,
    "the frontend stamps the cast target from the smem operand, got " &
    $casts[1].cTo.addressSpace
  # the frontend stamps every target from its operand, so the validation passes
  discard registerValidateAddressSpaceCasts.runPass(fn)
  echo "  OK - frontend casts carry their operand's space and pass validation"

# ── 6. The rejection decision, driven on hand-built nodes both directions ──
static:
  # the error path cannot raise inside a static block, so the decision
  # boundary is asserted directly: mismatch rejects, same space passes,
  # for cast and conversion targets alike
  var ctx = GpuContext()
  let rmemPtr = GpuAst(kind: gpuIdent,
    symbol: newSymbol("p", typ = initGpuPtrType(GpuType(kind: gtFloat32), implicitPtr = false)))
  let devTarget = initGpuPtrType(GpuType(kind: gtFloat32), implicitPtr = false,
                                 addressSpace = asDevice)
  let crossCast = GpuAst(kind: gpuCast, cTo: devTarget, cExpr: rmemPtr)
  doAssert ctx.spaceCastRejection(crossCast, "k").len > 0,
    "a thread-operand cast to a device ptr is rejected"
  let sameCast = GpuAst(kind: gpuCast,
    cTo: initGpuPtrType(GpuType(kind: gtFloat32), implicitPtr = false),
    cExpr: rmemPtr)
  doAssert ctx.spaceCastRejection(sameCast, "k").len == 0,
    "a same-space cast passes"
  let crossConv = GpuAst(kind: gpuConv, convTo: devTarget, convExpr: rmemPtr)
  doAssert ctx.spaceCastRejection(crossConv, "k").len > 0,
    "a thread-operand conversion to a device ptr is rejected"
  let smemTarget = initGpuPtrType(GpuType(kind: gtFloat32), implicitPtr = false,
                                  addressSpace = asSMEM)
  let smemCast = GpuAst(kind: gpuCast, cTo: smemTarget, cExpr: rmemPtr)
  doAssert ctx.spaceCastRejection(smemCast, "k").len > 0,
    "a thread-operand cast to a threadgroup ptr is rejected"
  echo "  OK - the rejection decision fires on cross-space, cast and conv"

echo "t_ir_spaceCastValidation: all static assertions passed"
