## Phase 0: IR roundtrip + pretty-printing test.
##
## Verifies `toGpuAst` output for various Nim constructs. Structural assertions only, no pass pipeline.
## Also verifies `pretty(GpuAst)` rendering: Proc, Block, Discard, Var, Lit, BinOp, If, For, ObjConstr, RangeKind.
##
## `toGpuAst:` wraps the top-level body in a `gpuBlock` (stmt list), the proc is `ir.statements[0]`.
## A proc body is wrapped in gpuBlock only when there are multiple statements.
##
## Run (from tattletale):
##   nim c -r --hints:off --warnings:off --outdir:build --nimcache:nimcache workspace/crucible/tests/ir/mangling/t_ir_roundtrip.nim

import std/[macros, strutils]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types

import workspace/crucible/src/codegen/ir/gpu_type_constructors
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

# Helper: find the first gpuBinOp with a given operator name in a subtree
proc findBinOpOp(n: GpuAst; op: string): GpuAst =
  if n == nil: return nil
  if n.kind == gpuBinOp and n.bOp.ident() == op:
    return n
  for ch in n.items:
    let r = findBinOpOp(ch, op)
    if not r.isNil: return r
  return nil

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

# ── Test: primitive nnkInfix gpuBinOp carries its numeric bType ──
block:
  let ir = toGpuAst:
    proc binOpTypeKernel(x: int32) {.device.} =
      let y = x + 1
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.statements[0].vInit.kind == gpuBinOp,
    "Expected gpuBinOp for x+1, got " & $body.statements[0].vInit.kind
  let binop = body.statements[0].vInit
  doAssert not binop.bType.isNil, "crash-path nnkInfix gpuBinOp must carry non-nil bType"
  doAssert binop.bType == initGpuType(gtInt32),
    "x + 1 must carry gtInt32 bType, got " & $binop.bType
echo "  OK — primitive nnkInfix gpuBinOp carries numeric bType"

# ── Test: comparison gpuBinOp carries its gtBool bType ──
block:
  let ir = toGpuAst:
    proc cmpTypeKernel(x: int32) {.device.} =
      let b = x == 1
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.statements[0].vInit.kind == gpuBinOp,
    "Expected gpuBinOp for x==1, got " & $body.statements[0].vInit.kind
  let binop = body.statements[0].vInit
  doAssert not binop.bType.isNil, "comparison gpuBinOp must carry non-nil bType"
  doAssert binop.bType.kind == gtBool,
    "comparison must carry gtBool bType, got " & $binop.bType.kind
echo "  OK — comparison gpuBinOp carries gtBool bType"

# ── Test: compound-assign (+=) derives its bType from the LHS operand ──
block:
  let ir = toGpuAst:
    proc compoundKernel(x: int32) {.device.} =
      var y = x
      y += 1
  let fn = ir.getProc()
  let binop = fn.pBody.findBinOpOp("+=")
  doAssert not binop.isNil, "Expected a += gpuBinOp in the body"
  doAssert not binop.bType.isNil, "compound-assign gpuBinOp must carry non-nil bType"
  doAssert binop.bType == initGpuType(gtInt32),
    "y += 1 must carry gtInt32 bType (derived from the LHS operand), got " & $binop.bType
echo "  OK — compound-assign gpuBinOp derives bType from the LHS operand"

# ── Test: compound-assign (+=) on float derives gtFloat32 from the LHS ──
block:
  let ir = toGpuAst:
    proc compoundFloatKernel(w: float32) {.device.} =
      var z = w
      z += 2.0
  let fn = ir.getProc()
  let binop = fn.pBody.findBinOpOp("+=")
  doAssert not binop.isNil, "Expected a += gpuBinOp in the body"
  doAssert not binop.bType.isNil, "compound-assign gpuBinOp must carry non-nil bType"
  doAssert binop.bType == initGpuType(gtFloat32),
    "z += 2.0 must carry gtFloat32 bType (derived from the LHS operand), got " & $binop.bType
echo "  OK — float compound-assign gpuBinOp derives gtFloat32 from the LHS"

# ═════════════════════════════════════════════════════════════════
#  Pretty-printing tests (absorbed from t_ir_pretty.nim)
# ═════════════════════════════════════════════════════════════════

# ── Pretty: empty proc ──
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
  echo "  OK — pretty: empty proc renders Proc/Discard"

# ── Pretty: var declaration ──
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
  echo "  OK — pretty: var declaration renders Var (", varCount, " gpuVar, ", litCount, " gpuLit)"

# ── Pretty: binary operation (use param to avoid constant folding) ──
block:
  let ir = toGpuAst:
    proc binOpDecl(x: int32) {.device.} =
      let y = x + 1
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  let binOpCount = countPred(body, proc(n: GpuAst): bool = n.kind == gpuBinOp)
  doAssert binOpCount >= 1, "Expected at least 1 gpuBinOp, found " & $binOpCount
  let pretty = fn.pretty()
  doAssert pretty.contains("BinOp"), "Expected 'BinOp' in pretty output, got:\n" & pretty
  echo "  OK — pretty: binary operation renders BinOp (", binOpCount, " gpuBinOp)"

# ── Pretty: if-else statement ──
block:
  let ir = toGpuAst:
    proc ifElsePretty(x: int32) {.device.} =
      if x == 0:
        var y = x
      else:
        var z = x
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  doAssert fn.pBody.kind == gpuIf,
    "Expected gpuIf as body, got " & $fn.pBody.kind
  let pretty = fn.pretty()
  doAssert pretty.contains("IfCond"), "Expected 'IfCond' in pretty output, got:\n" & pretty
  doAssert pretty.contains("IfThen"), "Expected 'IfThen' in pretty output, got:\n" & pretty
  doAssert pretty.contains("IfElse"), "Expected 'IfElse' in pretty output, got:\n" & pretty
  echo "  OK — pretty: if-else renders IfCond/IfThen/IfElse"

# ── Pretty: for loop ──
block:
  let ir = toGpuAst:
    proc forLoopPretty(x: int32) {.device.} =
      for i in 0 .. x:
        var y = x
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  let forCount = countPred(fn, proc(n: GpuAst): bool = n.kind == gpuFor)
  doAssert forCount == 1, "Expected 1 gpuFor, found " & $forCount
  let pretty = fn.pretty()
  doAssert pretty.contains("For"), "Expected 'For' in pretty output, got:\n" & pretty
  echo "  OK — pretty: for loop renders For (", forCount, " gpuFor)"

# ── Pretty: for loop with exclusive range ──
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
  echo "  OK — pretty: for loop exclusive range renders RangeKind"

# ── Pretty: object construction ──
block:
  type
    PrettyPoint = object
      x: int32
      y: int32
  let ir = toGpuAst:
    proc objConstrPretty() {.device.} =
      let p = PrettyPoint(x: 10, y: 20)
  doAssert ir.kind == gpuBlock
  let fn = ir.getProc()
  let body = fn.pBody
  doAssert body.kind == gpuBlock
  let objCount = countPred(body, proc(n: GpuAst): bool = n.kind == gpuObjConstr)
  doAssert objCount >= 1, "Expected at least 1 gpuObjConstr, found " & $objCount
  let pretty = fn.pretty()
  doAssert pretty.contains("ObjConstr"), "Expected 'ObjConstr' in pretty output, got:\n" & pretty
  echo "  OK — pretty: object construction renders ObjConstr (", objCount, " gpuObjConstr)"

# ── Pretty: block expression ──
block:
  let ir = toGpuAst:
    proc blockExprPretty() {.device.} =
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
  echo "  OK — pretty: block expression renders Block"

# ── Test: gpuBinOp clone preserves bType ──
block:
  # Fixture with a known result type — clone must carry it over.
  let f = GpuAst(kind: gpuBinOp,
                 bType: initGpuType(gtInt32),
                 bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("*", iSym = "*")),
                 bLeft: GpuAst(kind: gpuLit, lValue: "1", lType: initGpuType(gtInt32)),
                 bRight: GpuAst(kind: gpuLit, lValue: "2", lType: initGpuType(gtInt32)))
  let c = f.clone()
  doAssert c.kind == gpuBinOp, "clone of gpuBinOp must stay gpuBinOp"
  doAssert not c.bType.isNil, "clone must copy bType (got nil)"
  doAssert c.bType == f.bType, "clone must preserve bType value"
echo "  OK — gpuBinOp clone preserves bType"

echo ""
echo "  All roundtrip + pretty-printing tests passed."
