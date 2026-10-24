## flattenObjectParams decomposes object kernel params into leaf params
##
## Test that a kernel param of object type flattens into one scalar/pointer param per field leaf
##
## - kernel object params decompose to `<param>_<field>` leaf params in field order, dot chains read the leaf params
## - a bool leaf stays gtBool (patchBoolToI32 retypes it next), a ptr leaf becomes a gtPtr param, nested fields join their path
## - a written scalar leaf gets a Function-storage shadow, a scalar/ptr-only kernel passes through identical, whole-object uses refuse
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_flatten_object_params.nim

import std/[strutils, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering_spirv {.all.}
import workspace/crucible/tests/ir/ir_testutils

proc ident(sym: Symbol): GpuAst =
  GpuAst(kind: gpuIdent, symbol: sym)

proc ident(name: string): GpuAst =
  GpuAst(kind: gpuIdent, symbol: newSymbol(name))

proc lit(t: GpuType, v: string): GpuAst =
  GpuAst(kind: gpuLit, lType: t, lValue: v)

proc dot(parent: GpuAst, field: string): GpuAst =
  GpuAst(kind: gpuDot, dParent: parent, dField: ident(field))

proc binop(op: string, l, r: GpuAst, t: GpuType): GpuAst =
  GpuAst(kind: gpuBinOp,
         bOp: GpuAst(kind: gpuIdent, symbol: newSymbol(op, iSym = op)),
         bLeft: l, bRight: r, bIsOverloaded: false, bType: t)

proc kparam(name: string, t: GpuType): GpuParam =
  GpuParam(ident: ident(newSymbol(name, iSym = name, typ = t,
                                  symKind = gsGlobalKernelParam)),
           typ: t, addressSpace: asRMEM, passByRef: false)

proc local(name: string, t: GpuType): Symbol =
  newSymbol(name, iSym = name, typ = t, symKind = gsLocal)

proc kernelFn(params: seq[GpuParam], stmts: varargs[GpuAst]): GpuAst =
  GpuAst(kind: gpuProc,
         pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k_h0")),
         pRetType: GpuType(kind: gtVoid),
         pParams: params,
         pAttributes: {attGlobal},
         pBody: GpuAst(kind: gpuBlock, statements: @stmts))

proc runFlatten(k: GpuAst) =
  var ctx = GpuContext()
  ctx.allFnTab[k.pName] = k
  registerFlattenObjectParams.runPassPrepared(ctx)

proc sameNode(a, b: GpuAst): bool =
  ## Structural equality over the node kinds this test builds.
  ## GpuAst `==` supports idents only, this covers the fixture grammar.
  if a.isNil or b.isNil:
    return a.isNil and b.isNil
  if a.kind != b.kind:
    return false
  case a.kind
  of gpuIdent:
    return a.symbol == b.symbol
  of gpuLit:
    return a.lValue == b.lValue and a.lType == b.lType
  of gpuBlock:
    if a.statements.len != b.statements.len:
      return false
    for i in 0 ..< a.statements.len:
      if not sameNode(a.statements[i], b.statements[i]):
        return false
    return true
  of gpuAssign:
    return a.aOp == b.aOp and sameNode(a.aLeft, b.aLeft) and
      sameNode(a.aRight, b.aRight)
  of gpuBinOp:
    return a.bType == b.bType and sameNode(a.bOp, b.bOp) and
      sameNode(a.bLeft, b.bLeft) and sameNode(a.bRight, b.bRight)
  of gpuDot:
    return sameNode(a.dParent, b.dParent) and sameNode(a.dField, b.dField)
  of gpuDeref:
    return sameNode(a.dOf, b.dOf)
  of gpuIndex:
    return sameNode(a.iArr, b.iArr) and sameNode(a.iIndex, b.iIndex)
  of gpuVar:
    return a.vType == b.vType and a.vMutable == b.vMutable and
      sameNode(a.vName, b.vName) and sameNode(a.vInit, b.vInit)
  of gpuDiscard:
    return true
  else:
    return false

# ═══════════════════════════════════════════════════════════════════════
# 1. Two scalar fields decompose to p_x/p_y params in field order, chains read the leaf params
# ═══════════════════════════════════════════════════════════════════════
static:
  let i32t = GpuType(kind: gtInt32)
  let vec2 = initGpuObjectType("Vec2", @[
    GpuTypeField(name: "x", typ: i32t),
    GpuTypeField(name: "y", typ: i32t)])
  let outSym = local("out0", i32t)
  let p = kparam("p", vec2)
  let k = kernelFn(@[p],
    GpuAst(kind: gpuAssign, aLeft: ident(outSym), aOp: "=",
           aRight: binop("+", dot(ident(p.ident.symbol), "x"),
                         dot(ident(p.ident.symbol), "y"), i32t)))
  k.runFlatten()
  doAssert k.pParams.len == 2,
    "the object param decomposes into one param per scalar leaf"
  doAssert k.pParams[0].ident.symbol.name == "p_x" and
    k.pParams[1].ident.symbol.name == "p_y",
    "leaf params carry the param_field names in field order"
  doAssert k.pParams[0].typ.kind == gtInt32 and k.pParams[1].typ.kind == gtInt32,
    "scalar leaves keep their scalar type"
  doAssert k.pParams[0].ident.symbol.symKind == gsGlobalKernelParam,
    "leaf params declare as global kernel params"
  doAssert not k.pParams[0].passByRef and not k.pParams[1].passByRef,
    "leaf params pass by value"
  doAssert k.pParams[0].addressSpace == asRMEM,
    "leaf params carry the original param's address space"
  let rhs = k.pBody.statements[0].aRight
  doAssert rhs.kind == gpuBinOp, "the addition survives the rewrite"
  doAssert rhs.bLeft.kind == gpuIdent and rhs.bLeft.symbol.name == "p_x",
    "the x chain reads the leaf param"
  doAssert rhs.bRight.kind == gpuIdent and rhs.bRight.symbol.name == "p_y",
    "the y chain reads the leaf param"
  doAssert rhs.bLeft.symbol.typ.kind == gtInt32,
    "the leaf param ident carries the leaf type"

# ═══════════════════════════════════════════════════════════════════════
# 2. A bool leaf stays gtBool at pass time, patchBoolToI32 retypes it later
# ═══════════════════════════════════════════════════════════════════════
static:
  let boolt = GpuType(kind: gtBool)
  let i32t = GpuType(kind: gtInt32)
  let flagObj = initGpuObjectType("Flag", @[
    GpuTypeField(name: "f", typ: boolt)])
  let outSym = local("out0", i32t)
  let p = kparam("p", flagObj)
  let k = kernelFn(@[p],
    GpuAst(kind: gpuAssign, aLeft: ident(outSym), aOp: "=",
           aRight: dot(ident(p.ident.symbol), "f")))
  k.runFlatten()
  doAssert k.pParams.len == 1, "the bool leaf is the only param"
  doAssert k.pParams[0].typ.kind == gtBool,
    "the bool leaf stays gtBool at pass time, patchBoolToI32 retypes it next"
  doAssert k.pBody.statements[0].aRight.symbol.name == "p_f",
    "the field chain reads the bool leaf param"

# ═══════════════════════════════════════════════════════════════════════
# 3. A ptr leaf becomes a gtPtr param, the deref/index chain roots at the leaf param ident
# ═══════════════════════════════════════════════════════════════════════
static:
  let u32t = GpuType(kind: gtUint32)
  let bufTy = initGpuPtrType(initGpuUncheckedArrayType(u32t), implicitPtr = false)
  let holder = initGpuObjectType("Holder", @[
    GpuTypeField(name: "buf", typ: bufTy)])
  let outSym = local("out0", u32t)
  let p = kparam("p", holder)
  let k = kernelFn(@[p],
    GpuAst(kind: gpuAssign, aLeft: ident(outSym), aOp: "=",
           aRight: GpuAst(kind: gpuIndex,
                          iArr: GpuAst(kind: gpuDeref,
                                       dOf: dot(ident(p.ident.symbol), "buf")),
                          iIndex: lit(u32t, "0"))))
  k.runFlatten()
  doAssert k.pParams.len == 1, "the ptr leaf is the only param"
  doAssert k.pParams[0].typ.kind == gtPtr and
    k.pParams[0].typ.to.kind == gtUncheckedArray,
    "the ptr leaf param keeps the pointer-to-unchecked-array type"
  let iArr = k.pBody.statements[0].aRight.iArr
  doAssert iArr.kind == gpuDeref and iArr.dOf.kind == gpuIdent and
    iArr.dOf.symbol.name == "p_buf",
    "the index chain roots at the leaf param ident, the deref/index shape survives"

# ═══════════════════════════════════════════════════════════════════════
# 4. A nested field joins its path into the leaf name
# ═══════════════════════════════════════════════════════════════════════
static:
  let i32t = GpuType(kind: gtInt32)
  let inner = initGpuObjectType("Inner", @[
    GpuTypeField(name: "v", typ: i32t)])
  let outer = initGpuObjectType("Outer", @[
    GpuTypeField(name: "inner", typ: inner)])
  let outSym = local("out0", i32t)
  let p = kparam("p", outer)
  let k = kernelFn(@[p],
    GpuAst(kind: gpuAssign, aLeft: ident(outSym), aOp: "=",
           aRight: dot(dot(ident(p.ident.symbol), "inner"), "v")))
  k.runFlatten()
  doAssert k.pParams.len == 1 and k.pParams[0].ident.symbol.name == "p_inner_v",
    "the nested leaf joins its field path into the name"
  doAssert k.pBody.statements[0].aRight.symbol.name == "p_inner_v",
    "the two-link chain rewrites to the nested leaf param"

# ═══════════════════════════════════════════════════════════════════════
# 5. A written scalar leaf gets a Function-storage shadow, the assignment targets the shadow
# ═══════════════════════════════════════════════════════════════════════
static:
  let i32t = GpuType(kind: gtInt32)
  let vec2 = initGpuObjectType("Vec2", @[
    GpuTypeField(name: "x", typ: i32t),
    GpuTypeField(name: "y", typ: i32t)])
  let outSym = local("out0", i32t)
  let p = kparam("p", vec2)
  let k = kernelFn(@[p],
    GpuAst(kind: gpuAssign, aLeft: dot(ident(p.ident.symbol), "x"), aOp: "=",
           aRight: lit(i32t, "5")),
    GpuAst(kind: gpuAssign, aLeft: ident(outSym), aOp: "=",
           aRight: dot(ident(p.ident.symbol), "x")))
  k.runFlatten()
  doAssert k.pBody.statements.len == 3,
    "the shadow declaration joins the two statements"
  let shadow = k.pBody.statements[0]
  doAssert shadow.kind == gpuVar and shadow.vName.symbol.name == "p_x_shadow",
    "the written scalar leaf declares a shadow variable first"
  doAssert shadow.vType.kind == gtInt32 and shadow.vMutable,
    "the shadow carries the leaf type in Function storage"
  doAssert shadow.vInit.kind == gpuIdent and shadow.vInit.symbol.name == "p_x",
    "the shadow initializes from the leaf param's push load"
  let store = k.pBody.statements[1]
  doAssert store.aLeft.kind == gpuIdent and store.aLeft.symbol.name == "p_x_shadow",
    "the field store targets the shadow"
  let load = k.pBody.statements[2].aRight
  doAssert load.symbol.name == "p_x_shadow",
    "a read after the store reads the shadow, not the push load"

# ═══════════════════════════════════════════════════════════════════════
# 6. Hazard negative: a kernel with only scalar/ptr params passes through identical
# ═══════════════════════════════════════════════════════════════════════
static:
  let u32t = GpuType(kind: gtUint32)
  let bufTy = initGpuPtrType(initGpuUncheckedArrayType(u32t), implicitPtr = false)
  let outSym = local("out0", u32t)
  let z = kparam("z", u32t)
  let buf = kparam("buf", bufTy)
  let k = kernelFn(@[z, buf],
    GpuAst(kind: gpuAssign, aLeft: ident(outSym), aOp: "=",
           aRight: binop("+", ident(z.ident.symbol),
                         GpuAst(kind: gpuIndex,
                                iArr: GpuAst(kind: gpuDeref,
                                             dOf: ident(buf.ident.symbol)),
                                iIndex: lit(u32t, "0")), u32t)))
  let before = k.clone()
  k.runFlatten()
  doAssert k.pParams.len == 2, "no object param, no param rewrite"
  doAssert sameNode(before.pBody, k.pBody),
    "an object-free kernel body passes through structurally identical"

# ═══════════════════════════════════════════════════════════════════════
# 7. A whole-object use of a flattened param refuses loudly
# ═══════════════════════════════════════════════════════════════════════
static:
  doAssert not compiles(block:
    static:
      let i32t = GpuType(kind: gtInt32)
      let vec2 = initGpuObjectType("Vec2", @[
        GpuTypeField(name: "x", typ: i32t),
        GpuTypeField(name: "y", typ: i32t)])
      let p = kparam("p", vec2)
      let k = kernelFn(@[p],
        GpuAst(kind: gpuVar, vName: ident(local("l", vec2)), vType: vec2,
               vInit: ident(p.ident.symbol), vMutable: true,
               addressSpace: asRMEM))
      k.runFlatten()
  ), "a whole-object use of a flattened param must refuse at compile time"
