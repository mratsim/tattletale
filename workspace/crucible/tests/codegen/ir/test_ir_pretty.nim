## Phase 0: IR pretty-printing test
##
## Exercises `toGpuAst` and verifies the `pretty(GpuAst)` output format
## for known IR structures. Covers at least 5 distinct GpuNodeKind variants
## with 10+ assertions.
##
## NOTE: `toGpuAst:` wraps the top-level body in a `gpuBlock`.
## The defined proc lives in `ir.statements[0]`.
## A proc body is wrapped in gpuBlock only when there are multiple statements.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_pretty.nim

import std/[macros, strutils]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types

# ── Helper: count nodes matching a predicate ──
proc countPred(n: GpuAst; pred: proc(n: GpuAst): bool): int =
  if n == nil: return 0
  result = if pred(n): 1 else: 0
  for child in n.items:
    result += countPred(child, pred)

# ── Helper: get proc from toGpuAst block ──
proc getProc(ir: GpuAst): GpuAst =
  doAssert ir.kind == gpuBlock
  doAssert ir.statements.len >= 1
  result = ir.statements[0]
  doAssert result.kind == gpuProc

# ═══════════════════════════════════════════════════════════════════════
# 1. Empty proc
# ═══════════════════════════════════════════════════════════════════════
block:
  let ir = toGpuAst:
    proc emptyProc() {.device.} =
      discard
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  doAssert fn.kind == gpuProc
  let pretty = fn.pretty()
  doAssert pretty.contains("Proc"), "Expected 'Proc' in pretty output, got:\n" & pretty
  doAssert pretty.contains("Discard"), "Expected 'Discard' in pretty output, got:\n" & pretty
  echo "  OK — empty proc: renders Proc/Discard"

# ═══════════════════════════════════════════════════════════════════════
# 2. Var declaration
# ═══════════════════════════════════════════════════════════════════════
block:
  let ir = toGpuAst:
    proc varDecl() {.device.} =
      let x = 42
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  let varCount = countPred(body, proc(n: GpuAst): bool = n.kind == gpuVar)
  doAssert varCount >= 1, "Expected at least 1 gpuVar, found " & $varCount
  let litCount = countPred(body, proc(n: GpuAst): bool = n.kind == gpuLit)
  doAssert litCount >= 1, "Expected at least 1 gpuLit, found " & $litCount
  let pretty = fn.pretty()
  doAssert pretty.contains("Var"), "Expected 'Var' in pretty output, got:\n" & pretty
  echo "  OK — var declaration: ", varCount, " gpuVar, ", litCount, " gpuLit"

# ═══════════════════════════════════════════════════════════════════════
# 3. Binary operation (use param to avoid constant folding)
# ═══════════════════════════════════════════════════════════════════════
block:
  let ir = toGpuAst:
    proc binOp(x: int32) {.device.} =
      let y = x + 1
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  let binOpCount = countPred(body, proc(n: GpuAst): bool = n.kind == gpuBinOp)
  doAssert binOpCount >= 1, "Expected at least 1 gpuBinOp, found " & $binOpCount
  let pretty = fn.pretty()
  doAssert pretty.contains("BinOp"), "Expected 'BinOp' in pretty output, got:\n" & pretty
  echo "  OK — binary operation: ", binOpCount, " gpuBinOp"

# ═══════════════════════════════════════════════════════════════════════
# 4. If-else statement
# ═══════════════════════════════════════════════════════════════════════
block:
  let ir = toGpuAst:
    proc ifElseTest(x: int32) {.device.} =
      if x == 0:
        var y = x
      else:
        var z = x
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  let gpuIfNode = fn.pBody
  doAssert gpuIfNode.kind == gpuIf,
    "Expected gpuIf as body, got " & $gpuIfNode.kind
  let pretty = fn.pretty()
  doAssert pretty.contains("IfCond"), "Expected 'IfCond' in pretty output, got:\n" & pretty
  doAssert pretty.contains("IfThen"), "Expected 'IfThen' in pretty output, got:\n" & pretty
  doAssert pretty.contains("IfElse"), "Expected 'IfElse' in pretty output, got:\n" & pretty
  echo "  OK — if-else: IfCond/IfThen/IfElse present in pretty output"

# ═══════════════════════════════════════════════════════════════════════
# 5. For loop
# ═══════════════════════════════════════════════════════════════════════
block:
  let ir = toGpuAst:
    proc forLoopTest(x: int32) {.device.} =
      for i in 0 .. x:
        var y = x
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  let forCount = countPred(fn, proc(n: GpuAst): bool = n.kind == gpuFor)
  doAssert forCount == 1, "Expected 1 gpuFor, found " & $forCount
  let pretty = fn.pretty()
  doAssert pretty.contains("For"), "Expected 'For' in pretty output, got:\n" & pretty
  echo "  OK — for loop: ", forCount, " gpuFor"

# ═══════════════════════════════════════════════════════════════════════
# 5b. For loop with exclusive range
# ═══════════════════════════════════════════════════════════════════════
block:
  let ir = toGpuAst:
    proc forRangeExc(x: int32) {.device.} =
      for i in 0 ..< x:
        var y = x
  doAssert ir.kind == gpuBlock
  let fnExc = ir.statements[0]
  doAssert fnExc.kind == gpuProc
  # Find gpuFor (body may be gpuBlock or directly gpuFor)
  var foundExclusive = false
  if fnExc.pBody.kind == gpuFor:
    foundExclusive = true
    doAssert fnExc.pBody.fRangeKind == rkExclusive, "Expected rkExclusive"
    let pretty = fnExc.pretty()
    doAssert pretty.contains("RangeKind"), "Expected RangeKind in pretty output"
  else:
    for stmt in fnExc.pBody.statements:
      if stmt.kind == gpuFor:
        foundExclusive = true
        doAssert stmt.fRangeKind == rkExclusive, "Expected rkExclusive"
        let pretty = fnExc.pretty()
        doAssert pretty.contains("RangeKind"), "Expected RangeKind in pretty output"
  doAssert foundExclusive, "Should find gpuFor in exclusive range"
  echo "  OK — for loop exclusive range: rkExclusive in pretty output"

# ═══════════════════════════════════════════════════════════════════════
# 6. Object construction
# ═══════════════════════════════════════════════════════════════════════
block:
  type
    Point = object
      x: int32
      y: int32
  let ir = toGpuAst:
    proc objConstrTest() {.device.} =
      let p = Point(x: 10, y: 20)
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  let objCount = countPred(body, proc(n: GpuAst): bool = n.kind == gpuObjConstr)
  doAssert objCount >= 1, "Expected at least 1 gpuObjConstr, found " & $objCount
  let pretty = fn.pretty()
  doAssert pretty.contains("ObjConstr"), "Expected 'ObjConstr' in pretty output, got:\n" & pretty
  echo "  OK — object construction: ", objCount, " gpuObjConstr"

# ═══════════════════════════════════════════════════════════════════════
# 7. Block expression
# ═══════════════════════════════════════════════════════════════════════
block:
  let ir = toGpuAst:
    proc blockExprTest() {.device.} =
      discard block:
        let val = int32(99)
        val
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  let blkCount = countPred(body, proc(n: GpuAst): bool = n.kind == gpuBlock)
  doAssert blkCount >= 1, "Expected at least one gpuBlock in body"
  let pretty = fn.pretty()
  doAssert pretty.contains("Block"), "Expected 'Block' in pretty output, got:\n" & pretty
  echo "  OK — block expression renders in pretty output"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  GpuNodeKind variants exercised: Proc, Block, Discard,"
echo "    Var, Lit, BinOp, If, For, ObjConstr"
echo "  All IR pretty-printing tests passed."
