## lowerSpans bType + lType remediation test (TEST-001 / SLOP-002 / SLOP-010)
##
## The SLOP-002 fix made the `+ 1` literal constructed by the two lowerSpans
## rewrites carry the span index type (lType = idxTyp) instead of the
## hardcoded gtInt32. This test closes the C3 gap by VALUE-asserting that
## remediation: it drives lowerSpans on hand-built IR whose span index
## expressions are gpuIdents typed gtInt64 — a NON-default type — so any
## default/hardcoded typing fails the `lType == idxTyp` assertions.
##
## It also locks SLOP-010: exprGpuType must never hand back nil (a nil lType
## on the literal crashes genLit with a FieldDefect at emission). Both
## rewrites (len(toOpenArray) and bare toOpenArray inline) are covered.
##
## Part (b) — loudness contract: an index of an unknown shape (gpuDot) hits
## the loud gtInt32 fallback, which uses the `warning` builtin (not a raw
## echo — it is suppressible with --warnings:off). Compile WITHOUT
## --warnings:off and grep the output: exactly TWO "[lowerSpans] WARNING:"
## lines must be present (part (b) gpuDot + part (c) nil-typed ident — the
## normal path in parts (a)/(a2) must emit none). The fallback always
## produces a non-nil, usable gtInt32 type (no nil lType, no FieldDefect).
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_lowerSpansBtype.nim
## Loudness check (exactly 2 warnings, parts (b) + (c)):
##   nim c -r --hints:off --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_lowerSpansBtype.nim \
##     2>&1 | grep -c "\[lowerSpans\] WARNING:"

import std/[strutils, tables]
import workspace/crucible/src/codegen/passes/passes_lowering
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/ir/gpu_type_constructors

proc countExprBlocks(n: GpuAst): int =
  ## Count gpuBlock(isExpr: true) nodes in a subtree.
  if n == nil: return 0
  result = if n.kind == gpuBlock and n.isExpr: 1 else: 0
  for ch in n.items:
    result += countExprBlocks(ch)

proc assertLenExprShape(n: GpuAst; expectTyp: GpuType) =
  ## Assert a lowered lenExpr `(last - first) + 1`: the `+` binop, the inner
  ## `-` binop and the `+ 1` literal all carry `expectTyp` (non-nil).
  doAssert n.kind == gpuBinOp, "expected a `+` gpuBinOp, got " & $n.kind
  doAssert not n.bType.isNil, "lenExpr bType must be non-nil (SLOP-010)"
  doAssert n.bType == expectTyp,
    "lenExpr bType must equal the index type, got " & $n.bType
  doAssert n.bLeft.kind == gpuBinOp, "expected `-` lenSub under the `+`, got " & $n.bLeft.kind
  doAssert not n.bLeft.bType.isNil, "lenSub bType must be non-nil"
  doAssert n.bLeft.bType == expectTyp,
    "lenSub bType must equal the index type, got " & $n.bLeft.bType
  let lit = n.bRight
  doAssert lit.kind == gpuLit, "expected literal 1, got " & $lit.kind
  doAssert lit.lValue == "1", "expected literal 1, got " & lit.lValue
  doAssert not lit.lType.isNil,
    "the `+ 1` literal lType must be non-nil (SLOP-010 — nil crashes genLit with a FieldDefect)"
  doAssert lit.lType == expectTyp,
    "the `+ 1` literal must carry the index type (SLOP-002), got " & $lit.lType

proc buildSpanProc(firstIdx: GpuAst; bodyStmts: seq[GpuAst]): GpuAst =
  ## Build a `proc(s: openArray[int32])` whose body is `bodyStmts`.
  let spanTyp = initGpuSpanType(kOpenArray, initGpuType(gtInt32))
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
  let fn = buildSpanProc(firstIdx, @[lenCall])
  let lowered = runLower(fn)
  doAssert lowered.pBody.statements.len == 1,
    "len(toOpenArray) must lower to exactly one statement, got " & $lowered.pBody.statements.len
  # The len(toOpenArray) call is rewritten in place to the `+` binop.
  assertLenExprShape(lowered.pBody.statements[0], idxTyp)
  doAssert countExprBlocks(lowered.pBody) == 0,
    "len(toOpenArray) lowering must not leave expr blocks, leftover: " & $countExprBlocks(lowered.pBody)
  # Every gpuBinOp in the lowered body must carry a non-nil bType.
  var nilBinops = 0
  proc checkBinops(n: GpuAst) =
    if n == nil: return
    if n.kind == gpuBinOp and n.bType.isNil:
      inc nilBinops
    for ch in n.items:
      checkBinops(ch)
  checkBinops(lowered.pBody)
  doAssert nilBinops == 0, "all lowered binops must carry non-nil bType, found " & $nilBinops

# ── Part (a2): bare toOpenArray(...) inline — the second `+ 1` literal site ──
static:
  let idxTyp = initGpuType(gtInt64)
  let firstIdx = GpuAst(kind: gpuIdent, symbol: newSymbol("first", iSym = "first", typ = idxTyp))
  let fn = buildSpanProc(firstIdx, @[toOpenArrayCall(firstIdx)])
  let lowered = runLower(fn)
  doAssert lowered.pBody.statements.len == 1,
    "bare toOpenArray must lower to exactly one statement, got " & $lowered.pBody.statements.len
  let blk = lowered.pBody.statements[0]
  doAssert blk.kind == gpuBlock and blk.isExpr,
    "bare toOpenArray must lower to a 2-value expr block, got " & $blk.kind
  doAssert blk.statements.len == 2, "expected { ptr + first, last - first + 1 }, got " & $blk.statements.len
  # ptrPlus carries the data pointer type (DEV-004), never nil.
  doAssert not blk.statements[0].bType.isNil, "ptrPlus bType must be non-nil"
  # lenExpr is the second block value — its `+ 1` literal must carry idxTyp.
  assertLenExprShape(blk.statements[1], idxTyp)

# ── Part (b): unknown-shape index (gpuDot) — loud gtInt32 fallback, never nil ──
static:
  let dotIdx = GpuAst(
    kind: gpuDot,
    dParent: GpuAst(kind: gpuIdent, symbol: newSymbol("pair", iSym = "pair", typ = GpuType(kind: gtInt32))),
    dField: GpuAst(kind: gpuIdent, symbol: newSymbol("i", iSym = "i")))
  let lenCall = GpuAst(
    kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol("len", iSym = "len")),
    cArgs: @[toOpenArrayCall(dotIdx)])
  let fn = buildSpanProc(dotIdx, @[lenCall])
  # This drive emits the suppressible "[lowerSpans] WARNING:" (the loudness
  # contract — assert the fallback still yields a usable non-nil gtInt32 type).
  let lowered = runLower(fn)
  assertLenExprShape(lowered.pBody.statements[0], initGpuType(gtInt32))

# ── Part (c): nil-typed ident index — the exact pre-fix SLOP-010 FieldDefect shape ──
static:
  # Pre-fix, exprGpuType(gpuIdent) returned the nil symbol.typ, so the
  # constructed `+ 1` literal carried lType = nil and genLit crashed with a
  # FieldDefect (`ast.lType.kind` deref at emission). The loud gtInt32
  # fallback must now absorb nil/void derivations: the literal gets a non-nil
  # gtInt32 lType and the FieldDefect path is structurally unreachable.
  let nilIdx = GpuAst(kind: gpuIdent,
                      symbol: newSymbol("first", iSym = "first", typ = nil))
  let lenCall = GpuAst(
    kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol("len", iSym = "len")),
    cArgs: @[toOpenArrayCall(nilIdx)])
  let fn = buildSpanProc(nilIdx, @[lenCall])
  # This drive emits the second suppressible "[lowerSpans] WARNING:".
  let lowered = runLower(fn)
  assertLenExprShape(lowered.pBody.statements[0], initGpuType(gtInt32))

block:
  echo "test_ir_lowerSpansBtype: all static assertions passed (parts a/a2/b/c)"
