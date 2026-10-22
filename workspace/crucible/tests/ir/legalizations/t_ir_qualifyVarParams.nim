## qualifyVarParams pass
##
## Test that kVar params lower to GLSL inout bindings
##
## - params retyped to their pointee
## - body derefs unwrapped
## - call-site gpuAddr args unwrapped
##
## - in-flavored params and their derefs are left untouched
## - kernel functions (attGlobal) have no kVar params by construction
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/ir/legalizations/t_ir_qualifyVarParams.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering_glsl
import workspace/crucible/src/codegen/passes/pass_registry

proc runOne(ctx: var GpuContext) =
  ## Runs the single pass under test through the public registry.
  var reg = PassRegistry.new()
  reg.registerQualifyVarParams()
  ctx.runPasses(reg)

# ═══════════════════════════════════════════════════════════════════════════
# 1. kVar param retyped to its pointee
# ═══════════════════════════════════════════════════════════════════════════
block:
  let pointee = GpuType(kind: gtInt32)
  let ptrTy = initGpuPtrType(pointee, implicitPtr = true)
  let sym = newSymbol("out", iSym = "out_var", typ = ptrTy)
  let param = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: ptrTy,
                       addressSpace: asRMEM, passByRef: false, flavor: kVar)
  var procNode = GpuAst(kind: gpuProc,
                        pName: GpuAst(kind: gpuIdent, symbol: newSymbol("fn")),
                        pParams: @[param],
                        pBody: GpuAst(kind: gpuBlock))
  var ctx = GpuContext()
  ctx.allFnTab[procNode.pName] = procNode

  runOne(ctx)

  doAssert procNode.pParams[0].typ.kind == gtInt32,
    "kVar param should be retyped to its pointee, got: " & $procNode.pParams[0].typ.kind
  doAssert procNode.pParams[0].flavor == kVar
  echo "  OK: kVar param retyped to pointee"

# ═══════════════════════════════════════════════════════════════════════════
# 2. body derefs of kVar params unwrap to the plain ident
# ═══════════════════════════════════════════════════════════════════════════
block:
  let pointee = GpuType(kind: gtInt32)
  let ptrTy = initGpuPtrType(pointee, implicitPtr = true)
  let sym = newSymbol("out", iSym = "out_var2", typ = ptrTy)
  let param = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: ptrTy,
                       addressSpace: asRMEM, passByRef: false, flavor: kVar)
  # body: `out = 3` arrives as gpuAssign(gpuDeref(gpuIdent(out)), 3)
  let derefed = GpuAst(kind: gpuDeref, dOf: GpuAst(kind: gpuIdent, symbol: sym))
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuAssign, aLeft: derefed,
                             aRight: GpuAst(kind: gpuLit, lValue: "3", lType: pointee))
  var procNode = GpuAst(kind: gpuProc,
                        pName: GpuAst(kind: gpuIdent, symbol: newSymbol("fn")),
                        pParams: @[param], pBody: body)
  var ctx = GpuContext()
  ctx.allFnTab[procNode.pName] = procNode

  runOne(ctx)

  let stmt = procNode.pBody.statements[0]
  doAssert stmt.aLeft.kind == gpuIdent,
    "deref of kVar param should unwrap to the plain ident, got: " & $stmt.aLeft.kind
  echo "  OK: kVar param deref unwrapped"

# ═══════════════════════════════════════════════════════════════════════════
# 3. call-site gpuAddr args to kVar params unwrap to the plain lvalue
# ═══════════════════════════════════════════════════════════════════════════
block:
  let pointee = GpuType(kind: gtInt32)
  let ptrTy = initGpuPtrType(pointee, implicitPtr = true)
  let calleeSym = newSymbol("bump", iSym = "bump_fn", typ = nil)
  let calleeParam = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: newSymbol("x", iSym = "x_var")),
                             typ: ptrTy, addressSpace: asRMEM, passByRef: false, flavor: kVar)
  var callee = GpuAst(kind: gpuProc,
                      pName: GpuAst(kind: gpuIdent, symbol: calleeSym),
                      pParams: @[calleeParam],
                      pBody: GpuAst(kind: gpuBlock))

  # caller: local `v`, calls `bump(v)` -- the frontend wrapped the arg in gpuAddr
  let localSym = newSymbol("v", iSym = "v_local", typ = pointee)
  let addrArg = GpuAst(kind: gpuAddr, aOf: GpuAst(kind: gpuIdent, symbol: localSym))
  var call = GpuAst(kind: gpuCall, cName: GpuAst(kind: gpuIdent, symbol: calleeSym),
                    cArgs: @[addrArg])
  var callerBody = GpuAst(kind: gpuBlock)
  callerBody.statements.add call
  var caller = GpuAst(kind: gpuProc,
                      pName: GpuAst(kind: gpuIdent, symbol: newSymbol("caller")),
                      pParams: @[],
                      pBody: callerBody)

  var ctx = GpuContext()
  ctx.allFnTab[callee.pName] = callee
  ctx.allFnTab[caller.pName] = caller

  runOne(ctx)

  doAssert caller.pBody.statements[0].cArgs[0].kind == gpuIdent,
    "gpuAddr arg to a kVar param should unwrap to the plain lvalue, got: " &
      $caller.pBody.statements[0].cArgs[0].kind
  echo "  OK: call-site gpuAddr arg unwrapped"

# ═══════════════════════════════════════════════════════════════════════════
# 4. in-flavored params and derefs of non-kVar idents stay untouched
# ═══════════════════════════════════════════════════════════════════════════
block:
  let devicePtr = initGpuPtrType(GpuType(kind: gtFloat32), implicitPtr = false)
  let bufSym = newSymbol("buf", iSym = "buf_dev", typ = devicePtr)
  let param = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: bufSym), typ: devicePtr,
                       addressSpace: asDevice, passByRef: false, flavor: kIn)
  # body: deref of a DEVICE pointer (buffer read) must survive
  let derefed = GpuAst(kind: gpuDeref, dOf: GpuAst(kind: gpuIdent, symbol: bufSym))
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuAssign,
                             aLeft: GpuAst(kind: gpuIdent, symbol: newSymbol("r", iSym = "r_local", typ = GpuType(kind: gtFloat32))),
                             aRight: derefed)
  var procNode = GpuAst(kind: gpuProc,
                        pName: GpuAst(kind: gpuIdent, symbol: newSymbol("fn")),
                        pParams: @[param], pBody: body)
  var ctx = GpuContext()
  ctx.allFnTab[procNode.pName] = procNode

  runOne(ctx)

  doAssert procNode.pParams[0].typ.kind == gtPtr,
    "kIn param type must stay a pointer"
  doAssert procNode.pBody.statements[0].aRight.kind == gpuDeref,
    "deref of a non-kVar ident must survive"
  echo "  OK: kIn params and device-pointer derefs untouched"

echo "\n  OK - t_ir_qualifyVarParams"
