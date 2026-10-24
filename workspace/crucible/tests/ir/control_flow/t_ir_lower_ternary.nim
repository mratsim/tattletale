## lowerTernaryToIf rewrites ternaries and lazy bools into if-statements over hoisted temps
##
## Test that ternaries and lazy short-circuit bools lower to if-statements over hoisted temps
##
## - `x = c ? a : b` splices `var _tn_N: T = <zero>` + `if c: _tn_N = a else: _tn_N = b`
##   ahead of the statement, the expression reads the temp
## - `a && b` and `a || b` normalize to ternaries and rewrite likewise
## - a ternary nested in a condition or an assignment RHS eliminates completely
## - a ternary-free fn passes through structurally identical
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
  ## Iterates the known child fields directly, the items iterator segfaults
  ## when it resumes on a node whose iterImpl variant is empty under recursion.
  if n.kind == kind:
    return n
  template walk(ch: GpuAst): untyped =
    if not ch.isNil:
      let hit = ch.findKind(kind)
      if hit != nil:
        return hit
  case n.kind
  of gpuProc:
    walk(n.pBody)
  of gpuBlock:
    for st in n.statements:
      walk(st)
  of gpuVar:
    walk(n.vName)
    walk(n.vInit)
  of gpuAssign:
    walk(n.aLeft)
    walk(n.aRight)
  of gpuIf:
    walk(n.ifCond)
    walk(n.ifThen)
    for el in n.ifElifs:
      walk(el.cond)
      walk(el.body)
    walk(n.ifElse)
  of gpuFor:
    walk(n.fVar)
    walk(n.fStart)
    walk(n.fEnd)
    walk(n.fStep)
    walk(n.fBody)
  of gpuTernary:
    walk(n.tCond)
    walk(n.tThen)
    walk(n.tElse)
  of gpuBinOp:
    walk(n.bOp)
    walk(n.bLeft)
    walk(n.bRight)
  else: discard

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
  doAssert decl.vInit.kind == gpuLit and decl.vInit.lValue == "0",
    "the temp initializes to the zero literal"
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
  doAssert decl.vInit.kind == gpuIdent and decl.vInit.symbol.name == "false",
    "the temp initializes to false, bools literalize as idents"
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
  doAssert decl.vInit.kind == gpuIdent and decl.vInit.symbol.name == "false",
    "the temp initializes to the false ident"
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

echo "  OK - t_ir_lower_ternary"
