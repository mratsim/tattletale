## lowerCompoundAssign rewrites compound assignments into binop assignments
##
## Test that compound assignments lower to plain binop assignments
##
## - `acc += y` over a float32 local rewrites to `acc = acc + y`, `-=` and `/=` map likewise
## - `a[i] *= y` over an array element keeps the indexed target as the store target
## - an address-path call hoists into a `_ca_N` temp both the read and store use
## - a plain `=` assignment passes through untouched
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_lower_compound_assign.nim

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
  of gpuAssign:
    walk(n.aLeft)
    walk(n.aRight)
  of gpuBinOp:
    walk(n.bLeft)
    walk(n.bRight)
  of gpuVar:
    walk(n.vInit)
  of gpuIndex:
    walk(n.iArr)
    walk(n.iIndex)
  else: discard

proc ident(sym: Symbol): GpuAst = GpuAst(kind: gpuIdent, symbol: sym)

proc lit(t: GpuType, v: string): GpuAst = GpuAst(kind: gpuLit, lType: t, lValue: v)

proc kernelFn(stmts: varargs[GpuAst]): GpuAst =
  GpuAst(kind: gpuProc,
         pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k_h0")),
         pRetType: GpuType(kind: gtVoid),
         pBody: GpuAst(kind: gpuBlock, statements: @stmts))

proc runLower(ctx: var GpuContext, k: GpuAst) =
  ctx.fnTab[k.pName] = k
  ctx.allFnTab[k.pName] = k
  registerLowerCompoundAssign.runPassPrepared(ctx)

# ═══════════════════════════════════════════════════════════════════════
# 1. `acc += y` over a float32 local rewrites to `acc = acc + y`
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let f32t = initGpuType(gtFloat32)
  let accSym = newSymbol("acc", iSym = "acc_h0", typ = f32t, symKind = gsLocal)
  let rhs = lit(f32t, "1.5")
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(accSym), aOp: "+=", aRight: rhs))
  ctx.runLower(k)
  let assign = k.pBody.statements[0]
  doAssert assign.kind == gpuAssign and assign.aOp == "=",
    "the compound token must be gone"
  doAssert assign.aLeft.kind == gpuIdent and assign.aLeft.symbol.iSym == "acc_h0",
    "the target stays the bare ident"
  let binop = assign.aRight
  doAssert binop.kind == gpuBinOp, "the RHS is the binop"
  doAssert binop.bOp.kind == gpuIdent and binop.bOp.symbol.name == "+",
    "the binop token is the applied operator"
  doAssert not binop.bIsOverloaded, "scalar operands stay non-overloaded"
  doAssert binop.bLeft.kind == gpuIdent and binop.bLeft.symbol.iSym == "acc_h0",
    "the binop reads the target"
  doAssert binop.bRight.kind == gpuLit and binop.bRight.lValue == "1.5",
    "the original RHS rides on the binop"
  doAssert binop.bType.kind == gtFloat32, "the binop result is the element type"
  echo "  OK — acc += y rewrites to acc = acc + y"

# ═══════════════════════════════════════════════════════════════════════
# 2. `a[i] *= y` over an array element keeps the indexed target
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let f32t = initGpuType(gtFloat32)
  let arrT = GpuType(kind: gtArray, aTyp: f32t, aLen: 4)
  let arrSym = newSymbol("a", iSym = "a_h1", typ = arrT, symKind = gsLocal)
  let idxSym = newSymbol("i", iSym = "i_h1", typ = initGpuType(gtInt32), symKind = gsLocal)
  let idx = GpuAst(kind: gpuIndex, iArr: ident(arrSym), iIndex: ident(idxSym))
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: idx, aOp: "*=", aRight: lit(f32t, "2.0")))
  ctx.runLower(k)
  let assign = k.pBody.statements[0]
  doAssert assign.kind == gpuAssign and assign.aOp == "="
  doAssert assign.aLeft.kind == gpuIndex and assign.aLeft.iArr.kind == gpuIdent,
    "the indexed target stays the store target"
  doAssert assign.aLeft.iArr.symbol.iSym == "a_h1"
  let binop = assign.aRight
  doAssert binop.kind == gpuBinOp and binop.bOp.symbol.name == "*"
  doAssert binop.bLeft.kind == gpuIndex and binop.bLeft.iArr.symbol.iSym == "a_h1",
    "the binop reads a clone of the indexed target"
  doAssert binop.bType.kind == gtFloat32
  echo "  OK — a[i] *= y keeps the indexed target, the binop reads its clone"

# ═══════════════════════════════════════════════════════════════════════
# 3. a call inside the address path hoists into a `_ca_N` temp
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let f32t = initGpuType(gtFloat32)
  # a device fn the address-path call resolves through
  let fooFn = GpuAst(kind: gpuProc,
                     pName: GpuAst(kind: gpuIdent, symbol: newSymbol("foo", iSym = "foo_h2")),
                     pRetType: f32t,
                     pBody: GpuAst(kind: gpuBlock, statements: @[]))
  ctx.allFnTab[fooFn.pName] = fooFn
  let arrT = GpuType(kind: gtArray, aTyp: f32t, aLen: 8)
  let arrSym = newSymbol("a", iSym = "a_h2", typ = arrT, symKind = gsLocal)
  let callIdx = GpuAst(kind: gpuIndex, iArr: ident(arrSym),
                       iIndex: GpuAst(kind: gpuCall, cName: fooFn.pName, cArgs: @[]))
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: callIdx, aOp: "+=", aRight: lit(f32t, "1.0")))
  ctx.allFnTab[k.pName] = k
  ctx.fnTab[k.pName] = k
  registerLowerCompoundAssign.runPassPrepared(ctx)
  doAssert k.pBody.statements.len == 2,
    "the hoisted temp precedes the rewritten assignment"
  let decl = k.pBody.statements[0]
  doAssert decl.kind == gpuVar and decl.vInit.kind == gpuCall,
    "the temp declares with the call as its initializer"
  doAssert decl.vName.symbol.name.startsWith("_ca_"),
    "the temp carries the pass's fresh-name prefix"
  doAssert decl.vType.kind == gtFloat32
  let assign = k.pBody.statements[1]
  doAssert assign.kind == gpuAssign and assign.aOp == "="
  doAssert assign.aLeft.kind == gpuIndex
  doAssert assign.aLeft.iIndex.kind == gpuIdent and
    assign.aLeft.iIndex.symbol.name == decl.vName.symbol.name,
    "the store address reads the temp"
  let binop = assign.aRight
  doAssert binop.kind == gpuBinOp and binop.bOp.symbol.name == "+"
  doAssert binop.bLeft.kind == gpuIndex and
    binop.bLeft.iIndex.symbol.name == decl.vName.symbol.name,
    "the binop read evaluates the same temp, the address runs once"
  echo "  OK — the address-path call hoists into a _ca_N temp both sites read"

# ═══════════════════════════════════════════════════════════════════════
# 4. `-=` and `/=` map to the `-` and `/` binops
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let sSym = newSymbol("s", iSym = "s_h3", typ = i32t, symKind = gsLocal)
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(sSym), aOp: "-=", aRight: lit(i32t, "2")))
  ctx.runLower(k)
  doAssert k.pBody.statements[0].aRight.bOp.symbol.name == "-",
    "-= maps to the - binop"

  var ctx2 = GpuContext()
  let uSym = newSymbol("u", iSym = "u_h3", typ = initGpuType(gtUint32), symKind = gsLocal)
  let k2 = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(uSym), aOp: "/=", aRight: lit(initGpuType(gtUint32), "2")))
  ctx2.runLower(k2)
  doAssert k2.pBody.statements[0].aRight.bOp.symbol.name == "/",
    "/= maps to the / binop"
  echo "  OK — -= and /= map to the - and / binops"

# ═══════════════════════════════════════════════════════════════════════
# 5. hazard negative: a plain `=` assignment passes through untouched
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let i32t = initGpuType(gtInt32)
  let xSym = newSymbol("x", iSym = "x_h4", typ = i32t, symKind = gsLocal)
  let rhs = lit(i32t, "7")
  let k = kernelFn(GpuAst(kind: gpuAssign, aLeft: ident(xSym), aOp: "=", aRight: rhs))
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 1, "no statements added"
  let assign = k.pBody.statements[0]
  doAssert assign.kind == gpuAssign and assign.aOp == "="
  doAssert assign.aRight.kind == gpuLit and assign.aRight.lValue == "7",
    "the RHS stays the bare literal, no binop wrapped around it"
  doAssert k.pBody.findKind(gpuBinOp).isNil, "no binop introduced"
  echo "  OK — a plain = assignment passes through untouched"

# ═══════════════════════════════════════════════════════════════════════
# 6. an arithmetic index is a pure expression, no temp hoists
# ═══════════════════════════════════════════════════════════════════════
static:
  var ctx = GpuContext()
  let f32t = initGpuType(gtFloat32)
  let i32t = initGpuType(gtInt32)
  let arrT = GpuType(kind: gtArray, aTyp: f32t, aLen: 8)
  let arrSym = newSymbol("a", iSym = "a_h5", typ = arrT, symKind = gsLocal)
  let iSym = newSymbol("i", iSym = "i_h5", typ = i32t, symKind = gsLocal)
  let nSym = newSymbol("n", iSym = "n_h5", typ = i32t, symKind = gsLocal)
  let jSym = newSymbol("j", iSym = "j_h5", typ = i32t, symKind = gsLocal)
  proc arithBinop(op: string, l, r: GpuAst): GpuAst =
    GpuAst(kind: gpuBinOp, bOp: GpuAst(kind: gpuIdent, symbol: newSymbol(op)),
           bLeft: l, bRight: r, bIsOverloaded: false, bType: i32t)
  let arithIdx = arithBinop("+", arithBinop("*", ident(iSym), ident(nSym)), ident(jSym))
  let k = kernelFn(GpuAst(kind: gpuAssign,
                          aLeft: GpuAst(kind: gpuIndex, iArr: ident(arrSym),
                                       iIndex: arithIdx),
                          aOp: "+=", aRight: lit(f32t, "1.0")))
  ctx.runLower(k)
  doAssert k.pBody.statements.len == 1,
    "a pure-expression index hoists no temp"
  let assign = k.pBody.statements[0]
  doAssert assign.kind == gpuAssign and assign.aOp == "="
  doAssert assign.aLeft.kind == gpuIndex and assign.aLeft.iIndex.kind == gpuBinOp,
    "the store target keeps the arithmetic index"
  doAssert assign.aLeft.iIndex.bOp.symbol.name == "+"
  let binop = assign.aRight
  doAssert binop.kind == gpuBinOp and binop.bOp.symbol.name == "+"
  doAssert binop.bLeft.kind == gpuIndex and binop.bLeft.iIndex.kind == gpuBinOp,
    "the binop read clones the same arithmetic index"
  doAssert binop.bLeft.iIndex.bLeft.bOp.symbol.name == "*",
    "the nested i*n product survives the clone"
  echo "  OK — a[i*n + j] += y rewrites in place, the arithmetic index needs no temp"

echo "  OK - t_ir_lower_compound_assign"
