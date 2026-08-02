## Phase 0: IR roundtrip test
##
## Verifies that `toGpuAst` produces expected top-level GpuNodeKind values
## for various Nim constructs. These are lightweight structural assertions
## that do NOT depend on any pass pipeline.
##
## NOTE: `toGpuAst:` wraps the top-level body in a `gpuBlock` (stmt list).
## The defined proc lives in `ir.statements[0]`.
## A proc body is wrapped in gpuBlock only when there are multiple statements.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_roundtrip.nim

import std/macros
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types

# Helper: count nodes matching a predicate (avoid sequtils generics)
proc countPred(n: GpuAst; pred: proc(n: GpuAst): bool): int =
  if n == nil: return 0
  result = if pred(n): 1 else: 0
  for child in n.items:
    result += countPred(child, pred)

# Helper: extract proc from toGpuAst block
proc getProc(ir: GpuAst): GpuAst =
  doAssert ir.kind == gpuBlock,
    "Expected gpuBlock at top level, got " & $ir.kind
  doAssert ir.statements.len >= 1,
    "Expected at least one statement in top-level block"
  result = ir.statements[0]
  doAssert result.kind == gpuProc,
    "Expected gpuProc as first statement, got " & $result.kind

# ── Test: Top-level is gpuBlock ──
block:
  let ir = toGpuAst:
    proc emptyKernel() {.device.} =
      discard
  doAssert ir.kind == gpuBlock,
    "Expected gpuBlock at top level, got " & $ir.kind
echo "  OK — top-level is gpuBlock"

# ── Test: Empty proc produces gpuProc / gpuIdent / gpuDiscard ──
block:
  let ir = toGpuAst:
    proc emptyKernel() {.device.} =
      discard
  let fn = ir.getProc()
  doAssert fn.pName.kind == gpuIdent,
    "Expected gpuIdent for proc name, got " & $fn.pName.kind
  doAssert fn.pBody.kind == gpuDiscard,
    "Expected gpuDiscard for discard body, got " & $fn.pBody.kind
echo "  OK — empty proc produces gpuProc with gpuDiscard body"

# ── Test: Var declaration ──
block:
  let ir = toGpuAst:
    proc varKernel() {.device.} =
      let x = 5
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  doAssert body.statements.len >= 1
  doAssert body.statements[0].kind == gpuVar,
    "Expected gpuVar for let declaration, got " & $body.statements[0].kind
echo "  OK — var declaration produces gpuVar"

# ── Test: Binary operation (use param to avoid constant folding) ──
block:
  let ir = toGpuAst:
    proc binOpKernel(x: int32) {.device.} =
      let y = x + 1
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  doAssert body.statements[0].vInit.kind == gpuBinOp,
    "Expected gpuBinOp for x+1, got " & $body.statements[0].vInit.kind
echo "  OK — binary operation produces gpuBinOp"

# ── Test: If statement (single-stmt body, not wrapped in gpuBlock) ──
block:
  let ir = toGpuAst:
    proc ifKernel(x: int32) {.device.} =
      if x == 0:
        var y = x
      else:
        var z = x
  let fn = ir.getProc()
  # Single-statement body is NOT wrapped in gpuBlock
  doAssert fn.pBody.kind == gpuIf,
    "Expected gpuIf as body, got " & $fn.pBody.kind
  doAssert fn.pBody.ifElse.kind != gpuDiscard,
    "ifElse should not be gpuDiscard with var statements in else"
  let ifCount = countPred(fn.pBody, proc(n: GpuAst): bool = n.kind == gpuIf)
  doAssert ifCount >= 1, "Expected gpuIf in tree, found " & $ifCount
echo "  OK — if statement produces gpuIf (single-stmt body)"

# ── Test: For loop ──
block:
  let ir = toGpuAst:
    proc forKernel(x: int32) {.device.} =
      for i in 0 .. x:
        var y = x
  let fn = ir.getProc()
  let forCount = countPred(fn.pBody, proc(n: GpuAst): bool = n.kind == gpuFor)
  doAssert forCount >= 1,
    "Expected at least one gpuFor in tree, found " & $forCount
echo "  OK — for loop produces gpuFor"

# ── Test: Object construction ──
block:
  type
    MyStruct = object
      a: int32
      b: float32
  let ir = toGpuAst:
    proc objKernel() {.device.} =
      let s = MyStruct(a: 1, b: 2.0)
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  let varStmt = body.statements[0]
  doAssert varStmt.kind == gpuVar
  doAssert varStmt.vInit.kind in {gpuObjConstr, gpuBlock},
    "Expected gpuObjConstr or gpuBlock for object construction, got " & $varStmt.vInit.kind
echo "  OK — object construction produces gpuObjConstr or block"

# ── Test: Block expression ──
block:
  let ir = toGpuAst:
    proc blockKernel() {.device.} =
      discard block:
        let tmp = int32(42)
        tmp
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  let blkCount = countPred(body, proc(n: GpuAst): bool = n.kind == gpuBlock)
  doAssert blkCount >= 1,
    "Expected at least one gpuBlock in body"
echo "  OK — block expression produces gpuBlock"

# ── Test: Two statements in body produces gpuBlock ──
block:
  let ir = toGpuAst:
    proc multiStmtKernel(x: int32) {.device.} =
      let y = x
      discard y
  let fn = ir.getProc()
  doAssert fn.pBody.kind == gpuBlock,
    "Expected gpuBlock for multi-statement body, got " & $fn.pBody.kind
echo "  OK — multi-statement body produces gpuBlock"

echo ""
echo "  All roundtrip tests passed."
