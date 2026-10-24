## lowerTernaryToIf rewrites ternaries and lazy bools into if-statements over hoisted temps
##
## Test that ternaries and lazy short-circuit bools lower to if-statements over hoisted temps
##
## - `x = c ? a : b` splices `var _tn_N: T` + `if c: _tn_N = a else: _tn_N = b`, `a && b` / `a || b` normalize likewise
## - a ternary nested in a condition, an assignment RHS, a return value, a call argument, or a branch body eliminates at its own position
## - an elif chain whose elif condition carries a rewrite nests into `if c1: B1 else: (if c2: B2 else: B3)`, the elif temp joins the else
## - a ternary-free fn passes through structurally identical, globalBlocks entries lower like fn bodies
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_lower_ternary.nim

import std/[strutils, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering_spirv {.all.}
import workspace/crucible/tests/ir/ir_testutils

proc findKind(n: GpuAst, kind: GpuNodeKind): GpuAst =
  ## First node of `kind` in the tree, nil when absent.
  ## The nil guard covers the child fields iterImpl yields unchecked.
  if n.isNil:
    return nil
  if n.kind == kind:
    return n
  for ch in n.items:
    let hit = ch.findKind(kind)
    if hit != nil:
      return hit

proc ident(sym: Symbol): GpuAst = GpuAst(kind: gpuIdent, symbol: sym)

proc ident(name: string): GpuAst = GpuAst(kind: gpuIdent, symbol: newSymbol(name))

proc lit(t: GpuType, v: string): GpuAst = GpuAst(kind: gpuLit, lType: t, lValue: v)

proc op(name: string): GpuAst =
  GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = name))

proc local(name: string, t: GpuType): Symbol =
  newSymbol(name, iSym = name, typ = t, symKind = gsLocal)

proc kernelFn(stmts: varargs[GpuAst]): GpuAst =
  GpuAst(kind: gpuProc,
         pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k_h0")),
         pRetType: GpuType(kind: gtVoid),
         pBody: GpuAst(kind: gpuBlock, statements: @stmts))

proc runLower(ctx: var GpuContext, k: GpuAst) =
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerLowerTernaryToIf.runPassPrepared(ctx)

proc sameNode(a, b: GpuAst): bool =
  ## Structural equality over the node kinds this test builds.
  ## GpuAst `==` supports idents only, this covers the whole fixture grammar.
  if a.isNil or b.isNil:
    return a.isNil and b.isNil
  if a.kind != b.kind:
    return false
  case a.kind
  of gpuIdent:
    return a.symbol == b.symbol
  of gpuLit:
    return a.lValue == b.lValue and a.lType == b.lType
  of gpuProc:
    return a.pName == b.pName and a.pRetType == b.pRetType and
      sameNode(a.pBody, b.pBody)
  of gpuBlock:
    if a.statements.len != b.statements.len:
      return false
    for i in 0 ..< a.statements.len:
      if not sameNode(a.statements[i], b.statements[i]):
        return false
    return true
  of gpuVar:
    return a.vType == b.vType and a.vMutable == b.vMutable and
      sameNode(a.vName, b.vName) and sameNode(a.vInit, b.vInit)
  of gpuAssign:
    return a.aOp == b.aOp and sameNode(a.aLeft, b.aLeft) and
      sameNode(a.aRight, b.aRight)
  of gpuIf:
    if a.ifIsExpr != b.ifIsExpr or a.ifElifs.len != b.ifElifs.len:
      return false
    if not sameNode(a.ifCond, b.ifCond) or not sameNode(a.ifThen, b.ifThen) or
        not sameNode(a.ifElse, b.ifElse):
      return false
    for i in 0 ..< a.ifElifs.len:
      if not sameNode(a.ifElifs[i].cond, b.ifElifs[i].cond) or
          not sameNode(a.ifElifs[i].body, b.ifElifs[i].body):
        return false
    return true
  of gpuFor:
    return a.fRangeKind == b.fRangeKind and sameNode(a.fVar, b.fVar) and
      sameNode(a.fStart, b.fStart) and sameNode(a.fEnd, b.fEnd) and
      sameNode(a.fStep, b.fStep) and sameNode(a.fBody, b.fBody)
  of gpuBinOp:
    return a.bType == b.bType and a.bIsOverloaded == b.bIsOverloaded and
      sameNode(a.bOp, b.bOp) and sameNode(a.bLeft, b.bLeft) and
      sameNode(a.bRight, b.bRight)
  of gpuTernary:
    return sameNode(a.tCond, b.tCond) and sameNode(a.tThen, b.tThen) and
      sameNode(a.tElse, b.tElse)
  of gpuDiscard:
    return true
  else:
    return false

# ═══════════════════════════════════════════════════════════════════════
# 1. `x = c ? 1 : 2` rewrites to [var, if] and the expression reads the temp
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let xSym = local("x", i32t)
  let ternary = GpuAst(kind: gpuTernary,
                       tCond: ident(cSym),
                       tThen: lit(i32t, "1"),
                       tElse: lit(i32t, "2"))
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(xSym), aOp: "=", aRight: ternary))
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 3,
    "the rewrite splices [var, if] ahead of the assignment"
  let decl = k.pBody.statements[0]
  doAssert decl.kind == gpuVar, "the temp declares first"
  doAssert decl.vType.kind == gtInt32, "the temp carries the ternary result type"
  doAssert decl.vInit.kind == gpuDiscard,
    "the temp declares uninitialized, both branches store it"
  doAssert decl.vMutable, "the temp is mutable, both branches store it"
  doAssert decl.vName.symbol.name.startsWith("_tn_"),
    "the temp carries the pass's fresh-name prefix"
  let branchIf = k.pBody.statements[1]
  doAssert branchIf.kind == gpuIf and not branchIf.ifIsExpr,
    "the selection is a statement if"
  doAssert branchIf.ifElifs.len == 0, "the rewrite builds a plain two-way if"
  doAssert branchIf.ifCond.kind == gpuIdent and branchIf.ifCond.symbol.iSym == "c",
    "the condition is the original condition"
  doAssert branchIf.ifThen.kind == gpuAssign and branchIf.ifThen.aOp == "=",
    "the then branch is a plain assignment"
  doAssert branchIf.ifThen.aLeft.kind == gpuIdent and
    branchIf.ifThen.aLeft.symbol.name == decl.vName.symbol.name,
    "the then branch stores the temp"
  doAssert branchIf.ifThen.aRight.kind == gpuLit and
    branchIf.ifThen.aRight.lValue == "1",
    "the then branch stores the original then-value"
  doAssert branchIf.ifElse.kind == gpuAssign and
    branchIf.ifElse.aLeft.symbol.name == decl.vName.symbol.name,
    "the else branch stores the same temp"
  doAssert branchIf.ifElse.aRight.kind == gpuLit and
    branchIf.ifElse.aRight.lValue == "2",
    "the else branch stores the original else-value"
  let assign = k.pBody.statements[2]
  doAssert assign.kind == gpuAssign and assign.aLeft.symbol.iSym == "x",
    "the original assignment survives"
  doAssert assign.aRight.kind == gpuIdent and
    assign.aRight.symbol.name == decl.vName.symbol.name,
    "the expression reads the temp"
  echo "  OK — a ternary rewrites to [var, if] and the expression reads the temp"

# ═══════════════════════════════════════════════════════════════════════
# 2. `r = a && b` keeps short-circuit: b assigns only in the taken branch
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let boolt = GpuType(kind: gtBool)
  let aSym = local("a", boolt)
  let bSym = local("b", boolt)
  let rSym = local("r", boolt)
  let lazyAnd = GpuAst(kind: gpuBinOp, bOp: op("&&"),
                       bLeft: ident(aSym), bRight: ident(bSym),
                       bIsOverloaded: false, bType: boolt)
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(rSym), aOp: "=", aRight: lazyAnd))
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 3,
    "the lazy bool rewrites to [var, if] ahead of the assignment"
  let decl = k.pBody.statements[0]
  doAssert decl.kind == gpuVar and decl.vType.kind == gtBool,
    "the temp carries gtBool"
  doAssert decl.vInit.kind == gpuDiscard,
    "the temp declares uninitialized, both branches store it"
  let branchIf = k.pBody.statements[1]
  doAssert branchIf.kind == gpuIf
  doAssert branchIf.ifCond.kind == gpuIdent and branchIf.ifCond.symbol.iSym == "a",
    "the condition is the left operand"
  doAssert branchIf.ifThen.aRight.kind == gpuIdent and
    branchIf.ifThen.aRight.symbol.iSym == "b",
    "the right operand evaluates only when the left operand holds"
  doAssert branchIf.ifElse.aRight.kind == gpuIdent and
    branchIf.ifElse.aRight.symbol.name == "false",
    "the untaken branch stores the neutral false"
  let assign = k.pBody.statements[2]
  doAssert assign.aRight.kind == gpuIdent and
    assign.aRight.symbol.name == decl.vName.symbol.name,
    "the expression reads the temp"
  echo "  OK — a && b rewrites to [var, if], b evaluates only when a holds"

# ═══════════════════════════════════════════════════════════════════════
# 3. `r = a || b` keeps short-circuit: the neutral true rides the then branch
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let boolt = GpuType(kind: gtBool)
  let aSym = local("a", boolt)
  let bSym = local("b", boolt)
  let rSym = local("r", boolt)
  let lazyOr = GpuAst(kind: gpuBinOp, bOp: op("||"),
                      bLeft: ident(aSym), bRight: ident(bSym),
                      bIsOverloaded: false, bType: boolt)
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(rSym), aOp: "=", aRight: lazyOr))
  ctx.runLower(k)
  let branchIf = k.pBody.statements[1]
  doAssert branchIf.kind == gpuIf
  doAssert branchIf.ifThen.aRight.kind == gpuIdent and
    branchIf.ifThen.aRight.symbol.name == "true",
    "a true left operand short-circuits to true"
  doAssert branchIf.ifElse.aRight.kind == gpuIdent and
    branchIf.ifElse.aRight.symbol.iSym == "b",
    "a false left operand evaluates the right operand"
  echo "  OK — a || b rewrites to [var, if], the neutral true rides the then branch"

# ═══════════════════════════════════════════════════════════════════════
# 4. a ternary nested in the assignment RHS eliminates completely
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let xSym = local("x", i32t)
  let nested = GpuAst(kind: gpuBinOp, bOp: op("+"),
                      bLeft: lit(i32t, "1"),
                      bRight: GpuAst(kind: gpuTernary,
                                     tCond: ident(cSym),
                                     tThen: lit(i32t, "2"),
                                     tElse: lit(i32t, "3")),
                      bIsOverloaded: false, bType: i32t)
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(xSym), aOp: "=", aRight: nested))
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 3,
    "the nested rewrite splices [var, if] ahead of the assignment"
  doAssert k.pBody.findKind(gpuTernary).isNil, "no ternary survives the pass"
  let assign = k.pBody.statements[2]
  doAssert assign.kind == gpuAssign
  doAssert assign.aRight.kind == gpuBinOp and assign.aRight.bOp.symbol.name == "+",
    "the enclosing binop survives"
  doAssert assign.aRight.bLeft.kind == gpuLit and assign.aRight.bLeft.lValue == "1",
    "the binop's other operand is untouched"
  doAssert assign.aRight.bRight.kind == gpuIdent and
    assign.aRight.bRight.symbol.name.startsWith("_tn_"),
    "the nested ternary position reads a temp"
  echo "  OK — a ternary nested in the assignment RHS eliminates completely"

# ═══════════════════════════════════════════════════════════════════════
# 5. a lazy bool nested in an if-condition hoists ahead of the if
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let aSym = local("a", boolt)
  let bSym = local("b", boolt)
  let xSym = local("x", i32t)
  let cond = GpuAst(kind: gpuBinOp, bOp: op("&&"),
                    bLeft: ident(aSym), bRight: ident(bSym),
                    bIsOverloaded: false, bType: boolt)
  let stmtIf = GpuAst(kind: gpuIf, ifIsExpr: false,
                      ifCond: cond,
                      ifThen: GpuAst(kind: gpuAssign, aLeft: ident(xSym),
                                     aOp: "=", aRight: lit(i32t, "1")),
                      ifElse: GpuAst(kind: gpuDiscard),
                      ifElifs: @[])
  let k = kernelFn(stmtIf)
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 3,
    "the hoisted [var, if] precedes the original statement if"
  let decl = k.pBody.statements[0]
  doAssert decl.kind == gpuVar and decl.vType.kind == gtBool,
    "the condition temp declares first"
  let condIf = k.pBody.statements[2]
  doAssert condIf.kind == gpuIf and condIf.ifIsExpr == false,
    "the original if survives as a statement"
  doAssert condIf.ifCond.kind == gpuIdent and
    condIf.ifCond.symbol.name == decl.vName.symbol.name,
    "the condition reads the temp"
  doAssert condIf.ifThen.kind == gpuAssign and condIf.ifThen.aRight.lValue == "1",
    "the original branch body is untouched"
  doAssert condIf.ifElse.kind == gpuDiscard, "the absent else stays absent"
  echo "  OK — a lazy bool in an if-condition hoists ahead of the if"

# ═══════════════════════════════════════════════════════════════════════
# 6. a ternary in the loop bound hoists ahead of the loop
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let iSym = local("i", i32t)
  let bound = GpuAst(kind: gpuTernary,
                     tCond: ident(cSym),
                     tThen: lit(i32t, "4"),
                     tElse: lit(i32t, "8"))
  let loop = GpuAst(kind: gpuFor, fVar: ident(iSym),
                    fStart: lit(i32t, "0"), fEnd: bound, fStep: lit(i32t, "1"),
                    fRangeKind: rkExclusive,
                    fBody: GpuAst(kind: gpuBlock, statements: @[]))
  let k = kernelFn(loop)
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 3,
    "the hoisted [var, if] precedes the loop"
  let decl = k.pBody.statements[0]
  doAssert decl.kind == gpuVar and decl.vType.kind == gtInt32,
    "the bound temp declares first"
  let loopOut = k.pBody.statements[2]
  doAssert loopOut.kind == gpuFor, "the loop survives the pass"
  doAssert loopOut.fEnd.kind == gpuIdent and
    loopOut.fEnd.symbol.name == decl.vName.symbol.name,
    "the loop bound reads the temp"
  doAssert loopOut.fStart.kind == gpuLit and loopOut.fStart.lValue == "0",
    "the loop start is untouched"
  doAssert loopOut.fStep.kind == gpuLit and loopOut.fStep.lValue == "1",
    "the loop step is untouched"
  echo "  OK — a ternary in the loop bound hoists ahead of the loop"

# ═══════════════════════════════════════════════════════════════════════
# 7. bool-valued branches: `true`/`false` idents read gtBool, no literal refuse
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let rSym = local("r", boolt)
  let ternary = GpuAst(kind: gpuTernary,
                       tCond: ident(cSym),
                       tThen: ident("true"),
                       tElse: ident("false"))
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(rSym), aOp: "=", aRight: ternary))
  ctx.runLower(k)
  let decl = k.pBody.statements[0]
  doAssert decl.kind == gpuVar and decl.vType.kind == gtBool,
    "true/false branches type the temp gtBool"
  doAssert decl.vInit.kind == gpuDiscard,
    "the temp declares uninitialized, both branches store it"
  let branchIf = k.pBody.statements[1]
  doAssert branchIf.ifThen.aRight.symbol.name == "true" and
    branchIf.ifElse.aRight.symbol.name == "false",
    "the branches store the original true/false idents"
  echo "  OK — true/false branches type the temp gtBool and store the idents"

# ═══════════════════════════════════════════════════════════════════════
# 8. hazard negative: a ternary-free fn passes through structurally identical
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let xSym = local("x", i32t)
  let iSym = local("i", i32t)
  let assign = GpuAst(kind: gpuAssign, aLeft: ident(xSym), aOp: "=",
                      aRight: GpuAst(kind: gpuBinOp, bOp: op("+"),
                                     bLeft: ident(xSym), bRight: lit(i32t, "1"),
                                     bIsOverloaded: false, bType: i32t))
  let stmtIf = GpuAst(kind: gpuIf, ifIsExpr: false,
                      ifCond: ident(cSym),
                      ifThen: GpuAst(kind: gpuBlock, statements: @[assign.clone()]),
                      ifElse: GpuAst(kind: gpuDiscard),
                      ifElifs: @[])
  let loop = GpuAst(kind: gpuFor, fVar: ident(iSym),
                    fStart: lit(i32t, "0"), fEnd: lit(i32t, "4"),
                    fStep: lit(i32t, "1"), fRangeKind: rkExclusive,
                    fBody: GpuAst(kind: gpuBlock, statements: @[assign.clone()]))
  let k = kernelFn(assign, stmtIf, loop)
  let snapshot = k.clone()
  ctx.runLower(k)
  doAssert sameNode(k.pBody, snapshot.pBody),
    "a ternary-free fn passes through structurally identical"
  echo "  OK — a ternary-free fn passes through structurally identical"


# ═══════════════════════════════════════════════════════════════════════
# 9. an elif cond carrying a rewrite nests the chain so its temp joins the else branch
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let ySym = local("y", boolt)
  let zSym = local("z", boolt)
  let rSym = local("r", i32t)
  let elifCond = GpuAst(kind: gpuBinOp, bOp: op("||"),
                        bLeft: ident(ySym), bRight: ident(zSym),
                        bIsOverloaded: false, bType: boolt)
  let stmtIf = GpuAst(kind: gpuIf, ifIsExpr: false,
                      ifCond: ident(cSym),
                      ifThen: GpuAst(kind: gpuAssign, aLeft: ident(ySym),
                                     aOp: "=", aRight: lit(boolt, "false")),
                      ifElifs: @[(cond: elifCond,
                                  body: GpuAst(kind: gpuAssign,
                                               aLeft: ident(rSym), aOp: "=",
                                               aRight: lit(i32t, "1")))],
                      ifElse: GpuAst(kind: gpuDiscard))
  let k = kernelFn(stmtIf)
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 1,
    "nothing hoists ahead of the whole if, the elif cond evaluates later"
  let outer = k.pBody.statements[0]
  doAssert outer.kind == gpuIf and outer.ifElifs.len == 0,
    "the flat elif chain rewrites into nested two-way ifs"
  doAssert outer.ifThen.kind == gpuAssign and outer.ifThen.aLeft.symbol.iSym == "y",
    "the then-branch keeps writing y first"
  doAssert outer.ifThen.aRight.kind == gpuLit and outer.ifThen.aRight.lValue == "false",
    "the then-branch stores the original then-value"
  doAssert outer.ifElse.kind == gpuBlock,
    "the elif cond's temps splice into the else branch"
  doAssert outer.ifElse.statements.len == 3,
    "the else branch carries [var, lazy-bool if, elif if]"
  let decl = outer.ifElse.statements[0]
  doAssert decl.kind == gpuVar and decl.vType.kind == gtBool,
    "the elif cond temp declares inside the else branch"
  doAssert decl.vName.symbol.name.startsWith("_tn_"),
    "the elif cond temp carries the pass's fresh-name prefix"
  let lazyIf = outer.ifElse.statements[1]
  doAssert lazyIf.kind == gpuIf and lazyIf.ifCond.kind == gpuIdent and
    lazyIf.ifCond.symbol.iSym == "y",
    "the lazy bool's left operand evaluates after the then-branch wrote y"
  doAssert lazyIf.ifThen.aRight.kind == gpuIdent and
    lazyIf.ifThen.aRight.symbol.name == "true",
    "the taken branch stores the neutral true"
  doAssert lazyIf.ifElse.aRight.kind == gpuIdent and
    lazyIf.ifElse.aRight.symbol.iSym == "z",
    "the untaken branch evaluates the right operand"
  let elifIf = outer.ifElse.statements[2]
  doAssert elifIf.kind == gpuIf and elifIf.ifCond.kind == gpuIdent and
    elifIf.ifCond.symbol.name == decl.vName.symbol.name,
    "the elif condition reads its temp"
  doAssert elifIf.ifThen.kind == gpuAssign and elifIf.ifThen.aLeft.symbol.iSym == "r",
    "the elif body assigns in its own branch"
  echo "  OK — an elif cond carrying a rewrite nests the chain, its temp joins the else branch"

static:
  # two rewriting elif conds: the chain nests two deep, each temp sits
  # inside its own else branch at its own evaluation position
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let aSym = local("a", boolt)
  let bSym = local("b", boolt)
  let rSym = local("r", i32t)
  let cond2 = GpuAst(kind: gpuBinOp, bOp: op("&&"),
                     bLeft: ident(aSym), bRight: ident(bSym),
                     bIsOverloaded: false, bType: boolt)
  let cond3 = GpuAst(kind: gpuTernary,
                     tCond: ident(aSym),
                     tThen: lit(boolt, "true"),
                     tElse: ident(bSym))
  let branch = GpuAst(kind: gpuAssign, aLeft: ident(rSym), aOp: "=",
                      aRight: lit(i32t, "1"))
  let stmtIf = GpuAst(kind: gpuIf, ifIsExpr: false,
                      ifCond: ident(cSym),
                      ifThen: GpuAst(kind: gpuAssign, aLeft: ident(rSym),
                                     aOp: "=", aRight: lit(i32t, "2")),
                      ifElifs: @[(cond: cond2, body: branch.clone()),
                                 (cond: cond3, body: branch.clone())],
                      ifElse: GpuAst(kind: gpuDiscard))
  let k = kernelFn(stmtIf)
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 1,
    "nothing hoists ahead of the whole if with two rewriting elifs"
  let outer = k.pBody.statements[0]
  doAssert outer.kind == gpuIf and outer.ifElifs.len == 0,
    "the outer chain flattens to a two-way if"
  doAssert outer.ifElse.kind == gpuBlock and outer.ifElse.statements.len == 3,
    "the first elif's temps splice into the first else branch"
  let midIf = outer.ifElse.statements[2]
  doAssert midIf.kind == gpuIf and midIf.ifElifs.len == 0,
    "the second elif nests one level deeper, the chain terminates"
  doAssert midIf.ifElse.kind == gpuBlock and midIf.ifElse.statements.len == 3,
    "the second elif's temps splice into the second else branch"
  let innerIf = midIf.ifElse.statements[2]
  doAssert innerIf.kind == gpuIf and innerIf.ifElifs.len == 0,
    "the nesting bottoms out at a two-way if"
  doAssert innerIf.ifThen.kind == gpuAssign and
    innerIf.ifThen.aRight.kind == gpuLit and innerIf.ifThen.aRight.lValue == "1",
    "the last elif body keeps its original assignment"
  echo "  OK — two rewriting elifs nest two deep, each temp at its own position"

# ═══════════════════════════════════════════════════════════════════════
# 10. a ternary in a return value hoists ahead of the return
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let ternary = GpuAst(kind: gpuTernary,
                       tCond: ident(cSym),
                       tThen: lit(i32t, "1"),
                       tElse: lit(i32t, "2"))
  let k = kernelFn(GpuAst(kind: gpuReturn, rValue: ternary))
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 3,
    "the rewrite splices [var, if] ahead of the return"
  doAssert k.pBody.statements[0].kind == gpuVar,
    "the temp declares first"
  let ret = k.pBody.statements[2]
  doAssert ret.kind == gpuReturn, "the return survives the pass"
  doAssert ret.rValue.kind == gpuIdent and
    ret.rValue.symbol.name == k.pBody.statements[0].vName.symbol.name,
    "the return reads the temp"
  echo "  OK — a ternary in a return value hoists ahead of the return"

# ═══════════════════════════════════════════════════════════════════════
# 11. a ternary in a call argument hoists ahead of the call
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let ternary = GpuAst(kind: gpuTernary,
                       tCond: ident(cSym),
                       tThen: lit(i32t, "1"),
                       tElse: lit(i32t, "2"))
  let call = GpuAst(kind: gpuCall,
                    cName: GpuAst(kind: gpuIdent,
                                  symbol: newSymbol("foo", iSym = "foo")),
                    cArgs: @[ternary])
  let k = kernelFn(call)
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 3,
    "the rewrite splices [var, if] ahead of the call"
  let decl = k.pBody.statements[0]
  doAssert decl.kind == gpuVar, "the temp declares first"
  let outCall = k.pBody.statements[2]
  doAssert outCall.kind == gpuCall and outCall.cName.symbol.iSym == "foo",
    "the call survives the pass"
  doAssert outCall.cArgs.len == 1 and outCall.cArgs[0].kind == gpuIdent and
    outCall.cArgs[0].symbol.name == decl.vName.symbol.name,
    "the call argument reads the temp"
  echo "  OK — a ternary in a call argument hoists ahead of the call"

# ═══════════════════════════════════════════════════════════════════════
# 12. a ternary inside a branch body rewraps the branch into a block
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let dSym = local("d", boolt)
  let xSym = local("x", i32t)
  let ternary = GpuAst(kind: gpuTernary,
                       tCond: ident(dSym),
                       tThen: lit(i32t, "1"),
                       tElse: lit(i32t, "2"))
  let stmtIf = GpuAst(kind: gpuIf, ifIsExpr: false,
                      ifCond: ident(cSym),
                      ifThen: GpuAst(kind: gpuAssign, aLeft: ident(xSym),
                                     aOp: "=", aRight: ternary),
                      ifElifs: @[],
                      ifElse: GpuAst(kind: gpuDiscard))
  let k = kernelFn(stmtIf)
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 1,
    "the branch-internal rewrite stays inside the branch"
  let stmtIfOut = k.pBody.statements[0]
  doAssert stmtIfOut.kind == gpuIf and stmtIfOut.ifElifs.len == 0,
    "the statement if survives the pass"
  doAssert stmtIfOut.ifThen.kind == gpuBlock,
    "the bare then-branch rewraps into a block"
  doAssert stmtIfOut.ifThen.blockLabel == "",
    "no splice-block label leaks into the branch body"
  doAssert stmtIfOut.ifThen.statements.len == 3,
    "the branch body carries [var, if, assign]"
  let decl = stmtIfOut.ifThen.statements[0]
  doAssert decl.kind == gpuVar and decl.vType.kind == gtInt32,
    "the temp declares inside the branch"
  let assign = stmtIfOut.ifThen.statements[2]
  doAssert assign.kind == gpuAssign and assign.aLeft.symbol.iSym == "x",
    "the original assignment survives in the branch"
  doAssert assign.aRight.kind == gpuIdent and
    assign.aRight.symbol.name == decl.vName.symbol.name,
    "the assignment reads the branch-local temp"
  echo "  OK — a ternary inside a branch body rewraps the branch into a block"

# ═══════════════════════════════════════════════════════════════════════
# 13. the globalBlocks entry path lowers like a fn body
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let boolt = GpuType(kind: gtBool)
  let cSym = local("c", boolt)
  let xSym = local("x", i32t)
  let ternary = GpuAst(kind: gpuTernary,
                       tCond: ident(cSym),
                       tThen: lit(i32t, "1"),
                       tElse: lit(i32t, "2"))
  ctx.globalBlocks.add GpuAst(kind: gpuBlock, statements: @[
    GpuAst(kind: gpuAssign, aLeft: ident(xSym), aOp: "=", aRight: ternary)])
  registerLowerTernaryToIf.runPassPrepared(ctx)
  let blk = ctx.globalBlocks[0]
  doAssert blk.kind == gpuBlock and blk.statements.len == 3,
    "the global block splices [var, if] ahead of the assignment"
  doAssert blk.statements[0].kind == gpuVar and blk.statements[0].vType.kind == gtInt32,
    "the temp declares first in the global block"
  doAssert blk.statements[1].kind == gpuIf,
    "the selection is a statement if in the global block"
  let assign = blk.statements[2]
  doAssert assign.kind == gpuAssign and assign.aLeft.symbol.iSym == "x",
    "the original assignment survives in the global block"
  doAssert assign.aRight.kind == gpuIdent and
    assign.aRight.symbol.name == blk.statements[0].vName.symbol.name,
    "the global assignment reads the temp"
  echo "  OK — the globalBlocks entry path lowers like a fn body"

echo "  OK - t_ir_lower_ternary"
