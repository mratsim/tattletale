## explodeKernelParams (flat route) expands object kernel params into leaf params
##
## Test that the SPIR-V explode route decomposes object kernel params into leaf params with a reconstruction local
##
## - object params decompose to `<param>_<field>` leaf params in field order, field chains read them through the local
## - the local reuses the param's symbol, a static field contributes no leaf, whole-object uses survive
## - scalar or pointer leaf writes refuse, leaves narrower than 32 bits refuse, leaf-name collisions refuse
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_explode_kernel_params.nim

import std/[sets, strutils, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_low_pointers {.all.}
import workspace/crucible/src/codegen/passes/pass_helpers
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

proc runFlatExplode(k: GpuAst) =
  var ctx = GpuContext()
  ctx.allFnTab[k.pName] = k
  registerExplodeKernelParams.runPassPrepared(ctx)

proc reconLocal(k: GpuAst, name: string): GpuAst =
  ## Finds the reconstruction local an exploded param leaves at the body top
  for st in k.pBody.statements:
    if st.kind == gpuVar and st.vName.symbol.name == name:
      return st
  doAssert false, "no reconstruction local named " & name

# ═══════════════════════════════════════════════════════════════════════
# 1. Pure-value object: two scalar fields decompose to p_x/p_y leaf params with a local
# ═══════════════════════════════════════════════════════════════════════
static:
  let i32t = GpuType(kind: gtInt32)
  let vec2 = initGpuObjectType("Vec2", @[
    GpuTypeField(name: "x", typ: i32t),
    GpuTypeField(name: "y", typ: i32t)])
  let outSym = local("out0", i32t)
  let p = kparam("p", vec2)
  let pIdent = p.ident.symbol
  let k = kernelFn(@[p],
    GpuAst(kind: gpuAssign, aLeft: ident(outSym), aOp: "=",
           aRight: binop("+", dot(ident(pIdent), "x"),
                         dot(ident(pIdent), "y"), i32t)))
  k.runFlatExplode()
  doAssert k.pParams.len == 2,
    "the object param decomposes into one param per scalar leaf"
  doAssert k.pParams[0].ident.symbol.name == "p_x" and
    k.pParams[1].ident.symbol.name == "p_y",
    "leaf params carry the param_field names in field order"
  doAssert k.pParams[0].typ.kind == gtInt32 and k.pParams[1].typ.kind == gtInt32,
    "scalar leaves keep their scalar type"
  doAssert k.pParams[0].addressSpace == asRMEM,
    "a value leaf crosses as register memory"
  let recon = k.reconLocal("p")
  doAssert recon.vName.symbol == pIdent,
    "the reconstruction local reuses the param's symbol, body idents resolve to it"
  doAssert recon.vType == vec2 and not recon.vMutable,
    "the reconstruction local holds the whole object, immutable"
  doAssert recon.vInit.kind == gpuObjConstr,
    "the local initializes from an object construction over the leaf params"
  doAssert recon.vInit.ocFields[0].value.kind == gpuIdent and
    recon.vInit.ocFields[0].value.symbol.name == "p_x",
    "the first field reads the first leaf param"
  doAssert k.pBody.statements.len == 2,
    "the reconstruction local joins the body statements"
  doAssert k.pBody.statements[1].aRight.bLeft.kind == gpuDot,
    "the body's field chains keep their shape, they read the local"

# ═══════════════════════════════════════════════════════════════════════
# 2. A static field contributes no leaf, the reconstruction inits it from its compile-time value
# ═══════════════════════════════════════════════════════════════════════
static:
  let i32t = GpuType(kind: gtInt32)
  let stat = GpuType(kind: gtStatic, sValue: 42)
  let sized = initGpuObjectType("Sized", @[
    GpuTypeField(name: "v", typ: i32t),
    GpuTypeField(name: "c", typ: stat)])
  let outSym = local("out0", i32t)
  let p = kparam("p", sized)
  let k = kernelFn(@[p],
    GpuAst(kind: gpuAssign, aLeft: ident(outSym), aOp: "=",
           aRight: dot(ident(p.ident.symbol), "v")))
  k.runFlatExplode()
  doAssert k.pParams.len == 1 and k.pParams[0].ident.symbol.name == "p_v",
    "the static field contributes no leaf param"
  let recon = k.reconLocal("p")
  doAssert recon.vInit.ocFields.len == 2,
    "the reconstruction keeps every field of the object type"
  doAssert recon.vInit.ocFields[1].value.kind == gpuDiscard or
    recon.vInit.ocFields[1].typ.kind == gtStatic,
    "the static field initializes without a runtime read"

# ═══════════════════════════════════════════════════════════════════════
# 3. A ptr leaf becomes a gtPtr param, the body's index chain roots at the local
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
  k.runFlatExplode()
  doAssert k.pParams.len == 1 and k.pParams[0].typ.kind == gtPtr and
    k.pParams[0].typ.to.kind == gtUncheckedArray,
    "the ptr leaf param keeps the pointer-to-unchecked-array type"
  doAssert k.pParams[0].addressSpace == asDevice,
    "a pointer leaf crosses as a device buffer"
  let iArr = k.pBody.statements[1].aRight.iArr
  doAssert iArr.kind == gpuDeref and iArr.dOf.kind == gpuDot and
    iArr.dOf.dParent.symbol.name == "p",
    "the index chain roots at the reconstruction local"

# ═══════════════════════════════════════════════════════════════════════
# 4. A whole-object use of the parameter survives through the local
# ═══════════════════════════════════════════════════════════════════════
static:
  let i32t = GpuType(kind: gtInt32)
  let vec2 = initGpuObjectType("Vec2", @[
    GpuTypeField(name: "x", typ: i32t),
    GpuTypeField(name: "y", typ: i32t)])
  let outSym = local("out0", i32t)
  let p = kparam("p", vec2)
  let k = kernelFn(@[p],
    GpuAst(kind: gpuVar, vName: ident(local("l", vec2)), vType: vec2,
           vInit: ident(p.ident.symbol), vMutable: true,
           addressSpace: asRMEM))
  k.runFlatExplode()
  doAssert k.pBody.statements.len == 2,
    "the whole-object read survives, the local initializes from the reconstruction"

# ═══════════════════════════════════════════════════════════════════════
# 5. A write to a scalar leaf field refuses loudly, naming the field
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
        GpuAst(kind: gpuAssign, aLeft: dot(ident(p.ident.symbol), "x"), aOp: "=",
               aRight: lit(i32t, "5")))
      k.runFlatExplode()
  ), "a write to a scalar leaf of an exploded param must refuse at compile time"

# ═══════════════════════════════════════════════════════════════════════
# 6. A write to a pointer leaf field refuses loudly, device-address parameters are read-only
# ═══════════════════════════════════════════════════════════════════════
static:
  doAssert not compiles(block:
    static:
      let u32t = GpuType(kind: gtUint32)
      let bufTy = initGpuPtrType(initGpuUncheckedArrayType(u32t), implicitPtr = false)
      let holder = initGpuObjectType("Holder", @[
        GpuTypeField(name: "buf", typ: bufTy)])
      let p = kparam("p", holder)
      let q = kparam("q", bufTy)
      let k = kernelFn(@[p, q],
        GpuAst(kind: gpuAssign, aOp: "=",
               aLeft: dot(ident(p.ident.symbol), "buf"),
               aRight: ident(q.ident.symbol)))
      k.runFlatExplode()
  ), "a write to a pointer leaf field must refuse at compile time"

# ═══════════════════════════════════════════════════════════════════════
# 7. A leaf narrower than 32 bits refuses loudly at pass time
# ═══════════════════════════════════════════════════════════════════════
static:
  doAssert not compiles(block:
    static:
      let u32t = GpuType(kind: gtUint32)
      let u8t = GpuType(kind: gtUint8)
      let stats = initGpuObjectType("Stats", @[
        GpuTypeField(name: "count", typ: u32t),
        GpuTypeField(name: "flag", typ: u8t)])
      let p = kparam("p", stats)
      let k = kernelFn(@[p],
        GpuAst(kind: gpuDiscard))
      k.runFlatExplode()
  ), "a leaf narrower than 32 bits must refuse at pass time"

# ═══════════════════════════════════════════════════════════════════════
# 8. A body local named like a leaf collides: declaration positions count
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
        GpuAst(kind: gpuVar, vName: ident(local("p_x", i32t)), vType: i32t,
               vInit: lit(i32t, "1"), vMutable: true, addressSpace: asRMEM))
      k.runFlatExplode()
  ), "a body local named like a leaf param must refuse at pass time"

# ═══════════════════════════════════════════════════════════════════════
# 9. Hazard negative: a kernel without object params passes through identical
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
  k.runFlatExplode()
  doAssert k.pParams.len == 2, "no object param, no param rewrite"
  doAssert k.pBody.statements.len == before.pBody.statements.len,
    "an object-free kernel body gains no reconstruction local"
