## Address spaces survive the lowering passes
##
## Test that address spaces survive the lowering passes:
## span-to-pointer splitting, generic instantiation, smem arena hoisting,
## ptr-array decay and view decomposition
##
## Run:
##   nim c --hints:off --warnings:off --out:build/wip/t_ir_spaceStamps \
##     --nimcache:nimcache/wipSpaceStamps \
##     -r workspace/crucible/tests/ir/types/t_ir_spaceStamps.nim

import std/tables
import workspace/crucible/src/codegen/ir/ast_type_constructors
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/tests/ir/ir_testutils

proc findVar(n: GpuAst, name: string): GpuAst =
  ## First gpuVar named `name` among the statements, recursing into blocks,
  ## nil when absent
  for stmt in n.statements:
    if stmt.kind == gpuVar and stmt.vName.symbol.name == name:
      return stmt
    if stmt.kind == gpuBlock:
      let found = stmt.findVar(name)
      if not found.isNil:
        return found

# ── 1. A span built over device memory keeps device through lowerArraysAndViews ──
proc registerLowerDecay(reg: var PassRegistry) {.nimcall.} =
  reg.registerLowerArraysAndViews(decayArrays = true)

static:
  let ir = toCrucibleAst:
    proc scaleData(outp: ptr UncheckedArray[float32], data: openArray[float32]) {.device.} =
      outp[0] = data[0] * 2.0'f32
      outp[1] = data[1] * 2.0'f32
    proc spanKernel(outp: ptr UncheckedArray[float32], inp: ptr UncheckedArray[float32], n: int32) {.global.} =
      scaleData(outp, toOpenArray(inp, 0, n - 1))
      outp[2] = outp[0]

  # both fns share the context: the split consults the callee's span signature
  var ctx = GpuContext()
  var kern: GpuAst = nil
  for stmt in ir.statements:
    if stmt.kind == gpuProc:
      ctx.allFnTab[stmt.pName] = stmt
      if stmt.pName.symbol.name == "spanKernel":
        kern = stmt
        # the farmed kernel is the one called subset entry, as ingestTopLevel does
        ctx.fnTab[stmt.pName] = stmt
  registerLowerDecay.runPassPrepared(ctx)

  # the view argument splits into ptr + len, the pointer side derives its type
  # from the base pointer operand and keeps its device space
  proc findSplitCall(n: GpuAst, callee: string): GpuAst =
    if n.isNil:
      return nil
    if n.kind == gpuCall and n.cName.symbol.name == callee:
      return n
    for ch in n.items:
      let found = findSplitCall(ch, callee)
      if not found.isNil:
        return found

  let call = findSplitCall(kern.pBody, "scaleData")
  doAssert call.kind == gpuCall and call.cArgs.len == 3,
    "the span argument splits into ptr + len, got " & $call.cArgs.len

  # the split pointer side is a base pointer ident and a derived `inp + 0` binop
  # both nodes carry the base pointer's device space
  let ptrIdent = call.cArgs[0]
  doAssert ptrIdent.kind == gpuIdent and ptrIdent.symbol.typ.kind == gtPtr and
    ptrIdent.symbol.typ.addressSpace == asDevice,
    "the split view pointer ident keeps the device space, got " &
    $ptrIdent.symbol.typ.addressSpace
  let ptrDerived = call.cArgs[1]
  doAssert ptrDerived.kind == gpuBinOp and ptrDerived.bType.kind == gtPtr and
    ptrDerived.bType.addressSpace == asDevice,
    "the derived span pointer keeps the device space, got " &
    $ptrDerived.bType.addressSpace

  # the callee's openArray param splits to ptr + len in its rewritten signature
  # the split ptr keeps the DECLARED span type's space (asRMEM at birth)
  var scale: GpuAst = nil
  for stmt in ir.statements:
    if stmt.kind == gpuProc and stmt.pName.symbol.name == "scaleData":
      scale = stmt
  doAssert not scale.isNil, "expected the device fn in the context"
  doAssert scale.pParams[1].typ.kind == gtPtr,
    "the callee's openArray param splits to a ptr in its rewritten signature"

  # a callee span param is implicitly generic over its space: the split ptr is
  # born asRMEM (the declared openArray carries no written space) and the call
  # site's device argument specializes the instantiation's split ptr to device
  proc registerInstantiate(reg: var PassRegistry) {.nimcall.} =
    reg.registerInstantiateImplicitAddressSpaceGenerics()
  registerInstantiate.runPassPrepared(ctx)
  # `call` is the same node the pass retitled in place
  doAssert call.cName.symbol.iSym != scale.pName.symbol.iSym,
    "the device span call site references its own instantiation, got: " &
    call.cName.symbol.iSym
  let inst = ctx.allFnTab[call.cName]
  doAssert inst.pParams[1].typ.kind == gtPtr and
    inst.pParams[1].typ.addressSpace == asDevice,
    "the instantiated callee's split ptr param carries the call site's device space, got: " &
    $inst.pParams[1].typ.addressSpace
  echo "  OK - device span survives span-to-pointer normalization"

# ── 2. Smem arena: the hoisted threadgroup pointer carries asSMEM ──
proc registerMetalLowering(reg: var PassRegistry) {.nimcall.} =
  # the bundle's shared lowering passes depend on the early legalization phase
  reg.registerLegalizationPasses()
  reg.registerMetalLoweringPasses()

static:
  let ir = toCrucibleAst:
    proc foldScratch(inp: ptr UncheckedArray[float32], outp: ptr UncheckedArray[float32]) {.inline.} =
      var scratch {.smem.}: array[64, float32]
      let t = thread_position_in_threadgroup.x
      scratch[t] = inp[t]
      outp[t] = scratch[t]
    proc smemKernel(outp: ptr UncheckedArray[float32], inp: ptr UncheckedArray[float32]) {.global.} =
      var scratch {.smem.}: array[64, float32]
      foldScratch(inp, outp)

  var ctx = GpuContext()
  var kern: GpuAst = nil
  for stmt in ir.statements:
    if stmt.kind == gpuProc:
      ctx.allFnTab[stmt.pName] = stmt
      if stmt.pName.symbol.name == "smemKernel":
        kern = stmt
  registerMetalLowering.runPassPrepared(ctx)

  # the device fn's {.smem.} var hoists to a threadgroup pointer param
  var fold: GpuAst = nil
  for _, fn in ctx.allFnTab:
    if fn.kind == gpuProc and fn.pName.symbol.name == "foldScratch":
      fold = fn
  doAssert not fold.isNil, "expected the device fn in the context"
  var smemParams = 0
  for p in fold.pParams:
    if p.addressSpace == asSMEM:
      smemParams += 1
      doAssert p.typ.kind == gtPtr and p.typ.addressSpace == asSMEM,
        "the hoisted {.smem.} arena is a threadgroup pointer, got " &
        $p.typ.addressSpace
  doAssert smemParams == 1,
    "the device fn's arena hoists to one threadgroup param, got " & $smemParams

  # the kernel owns its arena: the var side keeps asSMEM with the bare array type
  let arena = kern.pBody.findVar("scratch")
  doAssert not arena.isNil and arena.kind == gpuVar and
    arena.addressSpace == asSMEM,
    "the kernel's {.smem.} arena keeps asSMEM on the var side, got " &
    $arena.addressSpace
  doAssert arena.vType.kind == gtArray,
    "the {.smem.} arena's type stays the bare array, got " & $arena.vType.kind

  # the call site forwards the arena, the bound arg carries the threadgroup
  # pointer type the callee expects
  var forward: GpuAst = nil
  for stmt in kern.pBody.statements:
    if stmt.kind == gpuCall and stmt.cName.symbol.name == "foldScratch":
      forward = stmt
  doAssert not forward.isNil, "expected the forwarded call in the kernel body"
  # the arena arg carries the threadgroup pointer type the callee expects
  var arenaArgs = 0
  for arg in forward.cArgs:
    if arg.kind == gpuIdent and arg.symbol.typ.kind == gtPtr and
        arg.symbol.typ.addressSpace == asSMEM:
      arenaArgs += 1
  doAssert arenaArgs == 1,
    "exactly one call arg forwards the threadgroup arena, got " & $arenaArgs
  echo "  OK - smem arena carries asSMEM on the hoisted type"

# ── 3. A ptr-to-array param decays keeping its address space ──
static:
  let ir = toCrucibleAst:
    proc reduce(dst: ptr array[16, float32], src: ptr array[16, float32]) {.device.} =
      dst[0] = src[0]
      dst[1] = src[1]
    proc arrKernel(outp: ptr UncheckedArray[float32]) {.global.} =
      var buf: array[16, float32]
      reduce(addr buf, addr buf)
      outp[0] = buf[0]

  var ctx = GpuContext()
  var reduceFn: GpuAst = nil
  for stmt in ir.statements:
    if stmt.kind == gpuProc:
      ctx.allFnTab[stmt.pName] = stmt
      if stmt.pName.symbol.name == "reduce":
        reduceFn = stmt
  registerLowerDecay.runPassPrepared(ctx)

  # the decay rebuilds gtPtr(gtArray) as gtPtr(gtUncheckedArray)
  # a fresh rebuild would reset the space, the birth space must survive
  for i in 0 ..< 2:
    doAssert reduceFn.pParams[i].typ.kind == gtPtr and
      reduceFn.pParams[i].typ.addressSpace == asDevice,
      "the decayed ptr-array param keeps asDevice, got " &
      $reduceFn.pParams[i].typ.addressSpace
  echo "  OK - ptr-array decay keeps the birth address space"

# ── 4. A span view binding decomposes keeping the span's space ──
static:
  # Nim cannot bind openArray to a local, the binding shape only exists as IR
  # after span lowering, so the pass runs on hand-built IR per the standard
  # pass-test pattern used across tests/ir
  let f32 = GpuType(kind: gtFloat32)
  let spanTyp = initGpuSpanType(f32, addressSpace = asDevice)
  let devPtr = GpuAst(kind: gpuIdent,
    symbol: newSymbol("inp", typ = initGpuPtrType(f32, implicitPtr = false, addressSpace = asDevice)))
  let view = GpuAst(kind: gpuVar,
    vName: GpuAst(kind: gpuIdent, symbol: newSymbol("view", typ = spanTyp)),
    vType: spanTyp,
    vInit: GpuAst(kind: gpuCall,
      cName: GpuAst(kind: gpuIdent, symbol: newSymbol("toOpenArray")),
      cArgs: @[devPtr,
               GpuAst(kind: gpuLit, lValue: "0", lType: initGpuType(gtInt32)),
               GpuAst(kind: gpuLit, lValue: "7", lType: initGpuType(gtInt32))]))
  let fn = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("viewFn")),
    pParams: @[],
    pBody: GpuAst(kind: gpuBlock, statements: @[view]))

  let lowered = registerLowerDecay.runPass(fn)
  let viewVar = lowered.pBody.findVar("view")
  doAssert not viewVar.isNil, "expected the view binding after decomposition"
  doAssert viewVar.vType.kind == gtPtr and viewVar.vType.addressSpace == asDevice,
    "decomposePtrLen rebuilds the view binding as a device ptr, got " &
    $viewVar.vType.addressSpace
  echo "  OK - span view binding decomposes to a device ptr"

echo "t_ir_spaceStamps: all static assertions passed"
