## inlineKernelCallees pass
##
## Test that non-intrinsic callees splice into the kernel body:
## - by-value parameters bind once to a fresh copy, the callee's result binds to the target
## - callee-owned names rename with a fresh suffix per call site, the same callee
##   spliced twice keeps distinct locals
## - the spliced callee drops from fnTab, an intrinsic call stays a call
##
## Run:
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/ir/functions/t_ir_inline_kernel_callees.nim

import std/[sets, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering_hlsl {.all.}
import workspace/crucible/tests/ir/ir_testutils
import workspace/crucible/src/codegen/passes/pass_registry

proc lit(v: string): GpuAst =
  GpuAst(kind: gpuLit, lValue: v, lType: GpuType(kind: gtUint32))

proc ident(name: string): GpuAst =
  GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = name,
    typ = GpuType(kind: gtUint32), symKind = gsLocal))

proc binop(op: string, l, r: GpuAst): GpuAst =
  GpuAst(kind: gpuBinOp, bOp: GpuAst(kind: gpuIdent, symbol: newSymbol(op, iSym = op)),
         bLeft: l, bRight: r, bType: GpuType(kind: gtUint32))

proc makeCallee(): GpuAst =
  ## `proc doubleIt(x: uint32): uint32` with one local and the implicit result
  let x = ident("x")
  result = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("doubleIt", iSym = "doubleIt")),
    pRetType: GpuType(kind: gtUint32),
    pParams: @[GpuParam(ident: x.clone(), typ: GpuType(kind: gtUint32))],
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuVar, vName: ident("result"), vType: GpuType(kind: gtUint32),
             vInit: GpuAst(kind: gpuDiscard)),
      GpuAst(kind: gpuVar, vName: ident("doubled"), vType: GpuType(kind: gtUint32),
             vInit: binop("*", x.clone(), lit("2"))),
      GpuAst(kind: gpuAssign, aLeft: ident("result"),
             aRight: binop("+", ident("doubled").clone(), lit("1"))),
      GpuAst(kind: gpuReturn, rValue: ident("result"))]))

proc makeKernel(calleeName: string): GpuAst =
  ## `output[0] = doubleIt(output[0])` + `let y = doubleIt(7)`
  let output = GpuAst(kind: gpuIdent, symbol: newSymbol("output", iSym = "output",
    typ = initGpuPtrType(GpuType(kind: gtUncheckedArray, uaTo: GpuType(kind: gtUint32)), implicitPtr = false), symKind = gsGlobalKernelParam))
  let idx0 = GpuAst(kind: gpuIndex, iArr: output.clone(), iIndex: lit("0"))
  let call1 = GpuAst(kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol(calleeName, iSym = calleeName)),
    cArgs: @[idx0.clone()])
  let call2 = GpuAst(kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol(calleeName, iSym = calleeName)),
    cArgs: @[lit("7")])
  result = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k")),
    pRetType: GpuType(kind: gtVoid),
    pAttributes: {attGlobal},
    pParams: @[GpuParam(ident: output.clone(), typ: output.symbol.typ)],
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuAssign, aLeft: idx0.clone(), aRight: call1),
      GpuAst(kind: gpuVar, vName: ident("y"), vType: GpuType(kind: gtUint32),
             vInit: call2)]))

static:
  var ctx = GpuContext()
  let callee = makeCallee()
  let kernel = makeKernel("doubleIt")
  for fn in @[callee, kernel]:
    ctx.allFnTab[fn.pName] = fn
    ctx.fnTab[fn.pName] = fn
  registerInlineKernelCallees.runPassPrepared(ctx)

  let stmts = kernel.pBody.statements
  doAssert stmts.len == 7, "both calls splice, got " & $stmts.len & " statements"
  # call site 1: the by-value argument binds once to a fresh copy, the callee's local renames to a fresh suffix
  doAssert stmts[0].kind == gpuVar and stmts[0].vName.symbol.name == "x_0", stmts[0].vName.symbol.name
  doAssert stmts[0].vInit.kind == gpuIndex, $stmts[0].vInit.kind
  doAssert stmts[0].vInit.iArr.symbol.name == "output"
  doAssert stmts[1].vName.symbol.name == "doubled_1", stmts[1].vName.symbol.name
  doAssert stmts[1].vInit.bLeft.symbol.name == "x_0",
    "the copy's uses follow the substitution, got " & stmts[1].vInit.bLeft.symbol.name
  doAssert stmts[2].aLeft.kind == gpuIndex, "the result binds to the assignment target"
  doAssert stmts[2].aRight.bLeft.symbol.name == "doubled_1"
  # call site 2: the same callee spliced again renames to a distinct suffix,
  # the non-void result binds to the var initializer
  doAssert stmts[3].kind == gpuVar and stmts[3].vInit.kind == gpuDiscard
  doAssert stmts[4].kind == gpuVar and stmts[4].vName.symbol.name == "x_2", stmts[4].vName.symbol.name
  doAssert stmts[4].vInit.lValue == "7"
  doAssert stmts[5].vName.symbol.name == "doubled_3", stmts[5].vName.symbol.name
  doAssert stmts[6].aLeft.symbol.name == "y", stmts[6].aLeft.symbol.name
  doAssert stmts[6].aRight.bLeft.symbol.name == "doubled_3"
  # the fully spliced callee drops from fnTab, the kernel stays
  doAssert ctx.fnTab.len == 1, "spliced callee must drop from fnTab"

static:
  # hazard negative: an intrinsic call is not a device call, it stays a call
  var ctx = GpuContext()
  let kernel = makeKernel("max")
  ctx.fnTab[kernel.pName] = kernel
  registerInlineKernelCallees.runPassPrepared(ctx)
  let call = kernel.pBody.statements[0].aRight
  doAssert call.kind == gpuCall, "an intrinsic call must pass through untouched"
  doAssert call.cName.symbol.name == "max"

proc makeBumpCallee(): GpuAst =
  ## `proc bump(a: uint32): uint32` that writes its parameter twice before reading it
  let a = ident("a")
  result = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("bump", iSym = "bump")),
    pRetType: GpuType(kind: gtUint32),
    pParams: @[GpuParam(ident: a.clone(), typ: GpuType(kind: gtUint32))],
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuVar, vName: ident("result"), vType: GpuType(kind: gtUint32),
             vInit: GpuAst(kind: gpuDiscard)),
      GpuAst(kind: gpuAssign, aLeft: a.clone(), aRight: binop("+", a.clone(), lit("1"))),
      GpuAst(kind: gpuAssign, aLeft: a.clone(), aRight: binop("+", a.clone(), lit("1"))),
      GpuAst(kind: gpuAssign, aLeft: ident("result"), aRight: a.clone()),
      GpuAst(kind: gpuReturn, rValue: ident("result"))]))

proc makeBumpKernel(): GpuAst =
  ## `var x = 5; let y = bump(x); output[0] = x + y`, the callee must not touch `x`
  let output = GpuAst(kind: gpuIdent, symbol: newSymbol("output", iSym = "output",
    typ = initGpuPtrType(GpuType(kind: gtUncheckedArray, uaTo: GpuType(kind: gtUint32)), implicitPtr = false), symKind = gsGlobalKernelParam))
  let call = GpuAst(kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol("bump", iSym = "bump")),
    cArgs: @[ident("x")])
  result = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k")),
    pRetType: GpuType(kind: gtVoid),
    pAttributes: {attGlobal},
    pParams: @[GpuParam(ident: output.clone(), typ: output.symbol.typ)],
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuVar, vName: ident("x"), vType: GpuType(kind: gtUint32),
             vInit: lit("5")),
      GpuAst(kind: gpuVar, vName: ident("y"), vType: GpuType(kind: gtUint32),
             vInit: call),
      GpuAst(kind: gpuAssign, aLeft: GpuAst(kind: gpuIndex,
              iArr: output.clone(), iIndex: lit("0")),
             aRight: binop("+", ident("x").clone(), ident("y").clone()))]))

static:
  # a callee that writes its by-value parameter mutates a fresh copy,
  # the caller's variable keeps its value
  var ctx = GpuContext()
  let bumpCallee = makeBumpCallee()
  let bumpKernel = makeBumpKernel()
  for fn in @[bumpCallee, bumpKernel]:
    ctx.allFnTab[fn.pName] = fn
    ctx.fnTab[fn.pName] = fn
  registerInlineKernelCallees.runPassPrepared(ctx)

  let stmts = bumpKernel.pBody.statements
  doAssert stmts[0].kind == gpuVar and stmts[0].vName.symbol.name == "x" and
    stmts[0].vInit.lValue == "5", "the caller's variable declaration stays untouched"
  doAssert stmts[2].kind == gpuVar and stmts[2].vName.symbol.name == "a_0" and
    stmts[2].vInit.symbol.name == "x", "the by-value argument binds once to a fresh copy"
  doAssert stmts[3].aLeft.symbol.name == "a_0", "the callee's writes hit the copy"
  doAssert stmts[5].aLeft.symbol.name == "y" and stmts[5].aRight.symbol.name == "a_0",
    "the callee's result reads the copy"
  for s in stmts:
    if s.kind == gpuAssign and s.aLeft.kind == gpuIdent:
      doAssert s.aLeft.symbol.name != "x", "the caller's variable is never an assignment target"
  doAssert stmts[6].aRight.bLeft.symbol.name == "x", "the caller's variable keeps its value"

proc makeWrapper(): GpuAst =
  ## `proc wrapper(): uint32 = result = doubleIt(3)`, no kernel calls it
  result = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("wrapper", iSym = "wrapper")),
    pRetType: GpuType(kind: gtUint32),
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuVar, vName: ident("result"), vType: GpuType(kind: gtUint32),
             vInit: GpuAst(kind: gpuCall,
               cName: GpuAst(kind: gpuIdent, symbol: newSymbol("doubleIt", iSym = "doubleIt")),
               cArgs: @[lit("3")]))]))

static:
  # a fn only an unreachable fn calls is unreachable too, the deletion sweep
  # iterates until the fnTab stops shrinking
  var ctx = GpuContext()
  let idle = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("idle", iSym = "idle")),
    pRetType: GpuType(kind: gtVoid),
    pAttributes: {attGlobal},
    pBody: GpuAst(kind: gpuBlock))
  for fn in @[idle, makeWrapper(), makeCallee()]:
    ctx.allFnTab[fn.pName] = fn
    ctx.fnTab[fn.pName] = fn
  registerInlineKernelCallees.runPassPrepared(ctx)
  doAssert ctx.fnTab.len == 1, "the unreachable callees drop transitively, got " & $ctx.fnTab.len

proc makeLimitCallee(): GpuAst =
  ## `proc limit(): uint32` with no parameters
  GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("limit", iSym = "limit")),
    pRetType: GpuType(kind: gtUint32),
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuReturn, rValue: lit("4"))]))

proc makeGuardedKernel(calleeName: string): GpuAst =
  ## `if limit() > 0: output[0] = 0`, the if condition evaluates once, its call hoists legally
  let output = GpuAst(kind: gpuIdent, symbol: newSymbol("output", iSym = "output",
    typ = initGpuPtrType(GpuType(kind: gtUncheckedArray, uaTo: GpuType(kind: gtUint32)), implicitPtr = false), symKind = gsGlobalKernelParam))
  result = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k")),
    pRetType: GpuType(kind: gtVoid),
    pAttributes: {attGlobal},
    pParams: @[GpuParam(ident: output.clone(), typ: output.symbol.typ)],
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuIf,
        ifCond: binop(">",
          GpuAst(kind: gpuCall,
            cName: GpuAst(kind: gpuIdent, symbol: newSymbol(calleeName, iSym = calleeName)),
            cArgs: @[]),
          lit("0")),
        ifThen: GpuAst(kind: gpuBlock, statements: @[
          GpuAst(kind: gpuAssign, aLeft: GpuAst(kind: gpuIndex,
                  iArr: output.clone(), iIndex: lit("0")),
                 aRight: lit("0"))]),
        ifElse: GpuAst(kind: gpuDiscard))]))

static:
  # legal boundary: a device call in an if condition evaluates once, it hoists to a temporary before the if, the branch body splices around it
  var ctx = GpuContext()
  let callee = makeLimitCallee()
  let kernel = makeGuardedKernel("limit")
  for fn in @[callee, kernel]:
    ctx.allFnTab[fn.pName] = fn
    ctx.fnTab[fn.pName] = fn
  registerInlineKernelCallees.runPassPrepared(ctx)

  let stmts = kernel.pBody.statements
  doAssert stmts.len == 3, "the call hoists to one temp plus its splice, got " & $stmts.len
  doAssert stmts[0].kind == gpuVar and stmts[0].vName.symbol.name == "call_0",
    stmts[0].vName.symbol.name
  doAssert stmts[1].aLeft.symbol.name == "call_0", "the splice targets the temp"
  doAssert stmts[2].kind == gpuIf, $stmts[2].kind
  doAssert stmts[2].ifCond.kind == gpuBinOp and stmts[2].ifCond.bLeft.symbol.name == "call_0",
    "the condition reads the temp, it did not move into a branch"
  doAssert ctx.fnTab.len == 1, "the spliced callee drops from fnTab"

static:
  # hazard negative: a short-circuit operand without a device call passes through untouched
  var ctx = GpuContext()
  let kernel = makeGuardedKernel("max")
  ctx.fnTab[kernel.pName] = kernel
  registerInlineKernelCallees.runPassPrepared(ctx)
  let ifNode = kernel.pBody.statements[0]
  doAssert ifNode.kind == gpuIf, $ifNode.kind
  doAssert ifNode.ifCond.kind == gpuBinOp and ifNode.ifCond.bLeft.kind == gpuCall and
    ifNode.ifCond.bLeft.cName.symbol.name == "max",
    "an intrinsic call is not a device call, the condition keeps the call"
  doAssert ifNode.ifCond.bRight.lValue == "0", "the non-call operand stays untouched"
