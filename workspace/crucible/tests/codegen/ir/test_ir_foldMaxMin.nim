## IR test: foldMaxMinToBuiltins pass (passes_optimizations.nim)
##
## Verifies the ternary max/min idiom is folded to the backend-native
## max/min builtin (plain name — the ambiguous-builtin name every backend
## supports for basic types) so it lowers to the hardware instruction
## (PTX max.f32 / max.u32 / max.s32) instead of a compare+select sequence.
##
## Matched shapes (operands A = tThen, B = tElse):
##   max: (B <= A) ? A : B                  — generic [T: not SomeFloat] body
##        ((B <= A) || !(B == B)) ? A : B    — float32/64 body (NaN guard)
##   min: (A <= B) ? A : B                  — generic [T: not SomeFloat] body
##        ((A <= B) || !(B == B)) ? A : B    — float32/64 body (NaN guard)
##
## NOT folded (guards):
##   - unguarded form on floats — NaN propagates to tElse there, so it is
##     NOT fmax; only the NaN-guarded form is fmax-equivalent on floats
##   - non-basic operand types (gtObject Int[N] structs, gtBool, ...) —
##     ceramic's genBinOp handles those
##   - then/else not matching the comparison operands
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_foldMaxMin.nim

import std/[tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_optimizations

# ─── IR construction helpers (mirror the shapes nim_to_gpu produces) ───

proc ident(name: string, t: GpuType): GpuAst =
  var sym = newSymbol(name)
  sym.typ = t
  GpuAst(kind: gpuIdent, symbol: sym)

proc binOp(op: string, a, b: GpuAst): GpuAst =
  var opSym = GpuAst(kind: gpuIdent, symbol: newSymbol(op))
  GpuAst(kind: gpuBinOp, bOp: opSym, bLeft: a, bRight: b,
         bIsOverloaded: false, bType: GpuType(kind: gtBool))

proc nanGuard(x: GpuAst): GpuAst =
  ## `!(x == x)` — the NaN guard in Nim's float32/64 min/max bodies.
  GpuAst(kind: gpuPrefix, pOp: "!", pVal: binOp("==", x, x))

proc foldInCtx(body: GpuAst; retType: GpuType): GpuAst =
  ## Wraps `body` as `result = body` in a device fn, runs the pass, and
  ## returns the fn body (a gpuBlock with one gpuAssign statement).
  var ctx = GpuContext()
  let fnName = ident("max___hash", retType)
  var fn = GpuAst(kind: gpuProc, pName: fnName, pRetType: retType,
                  pBody: GpuAst(kind: gpuBlock, statements: @[
                    GpuAst(kind: gpuAssign,
                           aLeft: ident("result", retType),
                           aRight: body)]))
  ctx.allFnTab[fnName] = fn
  ctx.foldMaxMinToBuiltins()
  ctx.allFnTab[fnName].pBody

proc assignRhs(body: GpuAst): GpuAst =
  body.statements[0].aRight

proc rhsKind(body: GpuAst): string =
  $body.statements[0].aRight.kind

proc rhsCallName(body: GpuAst): string =
  doAssert body.statements[0].aRight.kind == gpuCall,
    "expected gpuCall, got " & $body.statements[0].aRight.kind
  body.statements[0].aRight.cName.symbol.name

# ─── Tests ─────────────────────────────────────────────────────────────

let f32 = GpuType(kind: gtFloat32)
let i32 = GpuType(kind: gtInt32)
let u32 = GpuType(kind: gtUint32)

block: # 1. float32 NaN-guarded max folds to max
  let x = ident("x", f32)
  let y = ident("y", f32)
  let guardForm = GpuAst(kind: gpuTernary,
                         tCond: binOp("||", binOp("<=", y, x), nanGuard(y)),
                         tThen: x, tElse: y)
  let body = foldInCtx(guardForm, f32)
  doAssert rhsKind(body) == "gpuCall", "guard-form float32 max: got " & rhsKind(body)
  doAssert rhsCallName(body) == "max", "guard-form float32 max: got " & rhsCallName(body)
  echo "  OK — float32 NaN-guarded max folds to max call"

block: # 2. float32 NaN-guarded min folds to min
  let x = ident("x", f32)
  let y = ident("y", f32)
  let guardForm = GpuAst(kind: gpuTernary,
                         tCond: binOp("||", binOp("<=", x, y), nanGuard(y)),
                         tThen: x, tElse: y)
  let body = foldInCtx(guardForm, f32)
  doAssert rhsKind(body) == "gpuCall", "guard-form float32 min: got " & rhsKind(body)
  doAssert rhsCallName(body) == "min", "guard-form float32 min: got " & rhsCallName(body)
  echo "  OK — float32 NaN-guarded min folds to min call"

block: # 3. unguarded int max folds to max
  let a = ident("a", i32)
  let b = ident("b", i32)
  let plainForm = GpuAst(kind: gpuTernary, tCond: binOp("<=", b, a),
                         tThen: a, tElse: b)
  let body = foldInCtx(plainForm, i32)
  doAssert rhsKind(body) == "gpuCall", "plain int32 max: got " & rhsKind(body)
  doAssert rhsCallName(body) == "max", "plain int32 max: got " & rhsCallName(body)
  echo "  OK — unguarded int32 max folds to max call"

block: # 4. unguarded uint32 min folds to min
  let a = ident("a", u32)
  let b = ident("b", u32)
  let plainForm = GpuAst(kind: gpuTernary, tCond: binOp("<=", a, b),
                         tThen: a, tElse: b)
  let body = foldInCtx(plainForm, u32)
  doAssert rhsKind(body) == "gpuCall", "plain uint32 min: got " & rhsKind(body)
  doAssert rhsCallName(body) == "min", "plain uint32 min: got " & rhsCallName(body)
  echo "  OK — unguarded uint32 min folds to min call"

block: # 5. unguarded float max does NOT fold (NaN propagates to tElse)
  let x = ident("x", f32)
  let y = ident("y", f32)
  let plainForm = GpuAst(kind: gpuTernary, tCond: binOp("<=", y, x),
                         tThen: x, tElse: y)
  let body = foldInCtx(plainForm, f32)
  doAssert rhsKind(body) == "gpuTernary",
    "unguarded float max must stay a ternary (NaN semantics), got " & rhsKind(body)
  echo "  OK — unguarded float max stays a ternary (NaN semantics)"

block: # 6. then/else not matching the comparison operands does NOT fold
  let x = ident("x", f32)
  let y = ident("y", f32)
  let z = ident("z", f32)
  let badForm = GpuAst(kind: gpuTernary, tCond: binOp("<=", y, x),
                       tThen: x, tElse: z)
  let body = foldInCtx(badForm, f32)
  doAssert rhsKind(body) == "gpuTernary",
    "mismatched operands must stay a ternary, got " & rhsKind(body)
  echo "  OK — mismatched then/else stays a ternary"

block: # 7. object types (Int[N] structs) do NOT fold
  let objT = GpuType(kind: gtObject, name: "Int1")
  let a = ident("a", objT)
  let b = ident("b", objT)
  let objForm = GpuAst(kind: gpuTernary, tCond: binOp("<=", b, a),
                       tThen: a, tElse: b)
  let body = foldInCtx(objForm, objT)
  doAssert rhsKind(body) == "gpuTernary",
    "Int[N] struct ternary must stay (ceramic genBinOp handles it), got " & rhsKind(body)
  echo "  OK — Int[N] struct ternary stays (genBinOp territory)"

block: # 8. bool operands do NOT fold
  let bt = GpuType(kind: gtBool)
  let a = ident("a", bt)
  let b = ident("b", bt)
  let boolForm = GpuAst(kind: gpuTernary, tCond: binOp("<=", b, a),
                        tThen: a, tElse: b)
  let body = foldInCtx(boolForm, bt)
  doAssert rhsKind(body) == "gpuTernary",
    "bool ternary must stay, got " & rhsKind(body)
  echo "  OK — bool ternary stays"

echo "  All foldMaxMinToBuiltins IR checks passed"
