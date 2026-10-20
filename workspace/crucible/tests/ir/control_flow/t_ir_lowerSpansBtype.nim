## lowerSpans bType + lType remediation test
##
## Test that the lowerSpans rewrites carry the span index type into
## the constructed `+ 1` literal, unknown-shape indexes fall back loudly
##
## - a hand-built gtInt64-typed index defeats any default typing, VALUE-asserted
## - getLoweringExprType never returns nil, a nil lType crashes genLit at emission
## - the fallback warns via the `warning` builtin, suppressible with --warnings:off
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_lowerSpansBtype.nim


import std/tables
import workspace/crucible/src/codegen/passes/passes_lowering
import workspace/crucible/src/codegen/ir/gpu_ast_datatypes
import workspace/crucible/src/codegen/ir/gpu_type_constructors

proc assertLenExprShape(n: GpuAst; expectTyp: GpuType) =
  ## Assert a lowered lenExpr `(last - first) + 1`: the `+` binop, the inner
  ## `-` binop and the `+ 1` literal all carry `expectTyp`.
  doAssert n.kind == gpuBinOp, "expected a `+` gpuBinOp, got " & $n.kind
  doAssert n.bType == expectTyp,
    "lenExpr bType must equal the index type, got " & $n.bType
  doAssert n.bLeft.kind == gpuBinOp, "expected `-` lenSub under the `+`, got " & $n.bLeft.kind
  doAssert n.bLeft.bType == expectTyp,
    "lenSub bType must equal the index type, got " & $n.bLeft.bType
  let lit = n.bRight
  doAssert lit.kind == gpuLit and lit.lValue == "1", "expected literal 1, got " & lit.lValue
  doAssert lit.lType == expectTyp,
    "the `+ 1` literal must carry the index type, got " & $lit.lType

proc buildSpanProc(bodyStmts: seq[GpuAst]): GpuAst =
  ## Build a `proc(s: openArray[int32])` whose body is `bodyStmts`.
  let spanTyp = initGpuSpanType(initGpuType(gtInt32))
  result = GpuAst(
    kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("spanLower", iSym = "spanLower", symKind = gsProc)),
    pRetType: GpuType(kind: gtVoid),
    pParams: @[GpuParam(
      ident: GpuAst(kind: gpuIdent, symbol: newSymbol("s", iSym = "s", typ = spanTyp)),
      typ: spanTyp,
      addressSpace: asRMEM,
      passByRef: false)],
    pBody: GpuAst(kind: gpuBlock, isExpr: false, statements: bodyStmts))

proc toOpenArrayCall(firstIdx: GpuAst): GpuAst =
  ## `toOpenArray(data, first, last)` with data: ptr int32, first/last: gtInt64.
  let dataSym = newSymbol("data", iSym = "data",
                          typ = initGpuPtrType(initGpuType(gtInt32), implicitPtr = false))
  let lastSym = newSymbol("last", iSym = "last", typ = initGpuType(gtInt64))
  result = GpuAst(
    kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol("toOpenArray", iSym = "toOpenArray")),
    cArgs: @[
      GpuAst(kind: gpuIdent, symbol: dataSym),
      firstIdx,
      GpuAst(kind: gpuIdent, symbol: lastSym),
    ])

proc runLower(n: GpuAst): GpuAst =
  ## Run the lowerSpans pass on a proc and return the rewritten proc.
  var ctx = GpuContext()
  ctx.allFnTab[n.pName] = n
  lowerSpans(ctx)
  result = ctx.allFnTab[n.pName]

# ── Part (a): len(toOpenArray(...)) — ident index typed gtInt64 ──
static:
  let idxTyp = initGpuType(gtInt64)
  let firstIdx = GpuAst(kind: gpuIdent, symbol: newSymbol("first", iSym = "first", typ = idxTyp))
  let lenCall = GpuAst(
    kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol("len", iSym = "len")),
    cArgs: @[toOpenArrayCall(firstIdx)])
  let lowered = runLower(buildSpanProc(@[lenCall]))
  doAssert lowered.pBody.statements.len == 1,
    "len(toOpenArray) must lower to exactly one statement, got " & $lowered.pBody.statements.len
  # the len(toOpenArray) call is rewritten in place to the `+` binop
  assertLenExprShape(lowered.pBody.statements[0], idxTyp)

# ── Part (a2): bare toOpenArray(...) inline — the second `+ 1` literal site ──
static:
  let idxTyp = initGpuType(gtInt64)
  let firstIdx = GpuAst(kind: gpuIdent, symbol: newSymbol("first", iSym = "first", typ = idxTyp))
  let lowered = runLower(buildSpanProc(@[toOpenArrayCall(firstIdx)]))
  doAssert lowered.pBody.statements.len == 1,
    "bare toOpenArray must lower to exactly one statement, got " & $lowered.pBody.statements.len
  let blk = lowered.pBody.statements[0]
  doAssert blk.kind == gpuBlock and blk.isExpr,
    "bare toOpenArray must lower to a 2-value expr block, got " & $blk.kind
  doAssert blk.statements.len == 2, "expected { ptr + first, last - first + 1 }, got " & $blk.statements.len
  doAssert not blk.statements[0].bType.isNil, "ptrPlus bType must be non-nil"
  assertLenExprShape(blk.statements[1], idxTyp)

# ── Part (b): unknown-shape index (gpuDot) — loud gtInt32 fallback ──
static:
  let dotIdx = GpuAst(
    kind: gpuDot,
    dParent: GpuAst(kind: gpuIdent, symbol: newSymbol("pair", iSym = "pair", typ = GpuType(kind: gtInt32))),
    dField: GpuAst(kind: gpuIdent, symbol: newSymbol("i", iSym = "i")))
  let lenCall = GpuAst(
    kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol("len", iSym = "len")),
    cArgs: @[toOpenArrayCall(dotIdx)])
  let lowered = runLower(buildSpanProc(@[lenCall]))
  assertLenExprShape(lowered.pBody.statements[0], initGpuType(gtInt32))

# ── Part (c): nil-typed ident index — the other fallback arm ──
static:
  let nilIdx = GpuAst(kind: gpuIdent,
                      symbol: newSymbol("first", iSym = "first", typ = nil))
  let lenCall = GpuAst(
    kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol("len", iSym = "len")),
    cArgs: @[toOpenArrayCall(nilIdx)])
  let lowered = runLower(buildSpanProc(@[lenCall]))
  assertLenExprShape(lowered.pBody.statements[0], initGpuType(gtInt32))

echo "t_ir_lowerSpansBtype: all static assertions passed"
