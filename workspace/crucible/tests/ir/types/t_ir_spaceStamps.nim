## Address spaces stamped on GpuType at birth
##
## Test that pointer and span types carry their address space from birth through lowering
##
## Run:
##   nim c --hints:off --warnings:off --out:build/wip/t_ir_spaceStamps \
##     --nimcache:nimcache/wipSpaceStamps \
##     -r workspace/crucible/tests/ir/types/t_ir_spaceStamps.nim

import std/tables
import workspace/crucible/src/codegen/ir/ast_type_constructors
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/tests/ir/ir_testutils

type Holder = object
  p: ptr float32

proc findProc(n: GpuAst): GpuAst =
  ## First gpuProc statement of a top-level block, nil when absent
  for stmt in n.statements:
    if stmt.kind == gpuProc:
      return stmt

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

proc findFirst(n: GpuAst, kind: GpuNodeKind): GpuAst =
  ## First node of `kind` in depth-first pre-order, nil when absent
  if n.isNil:
    return nil
  if n.kind == kind:
    return n
  for ch in n.items:
    let found = findFirst(ch, kind)
    if not found.isNil:
      return found

proc collectObjConstrs(n: GpuAst, acc: var seq[GpuAst]) =
  ## Every gpuObjConstr in depth-first pre-order
  if n.isNil:
    return
  if n.kind == gpuObjConstr:
    acc.add n
  for ch in n.items:
    collectObjConstrs(ch, acc)

# ── 1. Birth stamps: explicit `ptr T` is a device pointer, `var T` stays thread ──
static:
  let ir = toCrucibleAst:
    var gbuf: ptr float32

    proc stampKernel(outp: ptr UncheckedArray[float32], v: var float32, p: ptr float32) {.global.} =
      v = p[] + outp[0]
      let c = cast[ptr float32](outp)
      outp[3] = c[]
      gbuf[] = v

  let fn = ir.findProc
  doAssert fn.kind == gpuProc, "expected the kernel proc, got " & $fn.kind

  let outpTyp = fn.pParams[0].typ
  doAssert outpTyp.kind == gtPtr and not outpTyp.implicit and
    outpTyp.addressSpace == asDevice,
    "explicit `ptr UncheckedArray[float32]` param stamps asDevice, got " &
    $outpTyp.addressSpace

  let varTyp = fn.pParams[1].typ
  doAssert varTyp.kind == gtPtr and varTyp.implicit and
    varTyp.addressSpace == asRMEM,
    "`var float32` param stays asRMEM (implicit pointer, thread storage), got " &
    $varTyp.addressSpace

  let ptrTyp = fn.pParams[2].typ
  doAssert ptrTyp.kind == gtPtr and not ptrTyp.implicit and
    ptrTyp.addressSpace == asDevice,
    "explicit `ptr float32` param stamps asDevice, got " & $ptrTyp.addressSpace

  let gbufVar = ir.findVar("gbuf")
  doAssert gbufVar.kind == gpuVar, "expected the global gpuVar gbuf"
  doAssert gbufVar.vType.kind == gtPtr and gbufVar.vType.addressSpace == asDevice,
    "global `ptr float32` stamps asDevice, got " & $gbufVar.vType.addressSpace

  # the cast target type flows through resolveType like any written type
  # a special-cased cast path cannot silently drop the stamp
  let castNode = findFirst(fn.pBody, gpuCast)
  doAssert not castNode.isNil, "expected the cast[ptr float32] node in the body"
  doAssert castNode.cTo.kind == gtPtr and castNode.cTo.addressSpace == asDevice,
    "a cast target `ptr float32` stamps asDevice, got " &
    $castNode.cTo.addressSpace
  echo "  OK - explicit ptr stamps asDevice, var stays asRMEM"

# ── 2. Address-of: the result inherits the space of the addressed storage ──
static:
  let ir = toCrucibleAst:
    proc addrKernel(outp: ptr UncheckedArray[float32]) {.global.} =
      var tmp: float32 = 0.0'f32
      var hDev = Holder(p: addr outp[0])
      var hThr = Holder(p: addr tmp)
      outp[1] = hDev.p[] + hThr.p[]

  let fn = ir.findProc
  var constrs: seq[GpuAst]
  fn.collectObjConstrs(constrs)
  doAssert constrs.len == 2,
    "expected the two Holder constructions, got " & $constrs.len

  doAssert constrs[0].ocFields[0].typ.kind == gtPtr and
    constrs[0].ocFields[0].typ.addressSpace == asDevice,
    "Holder(p: addr outp[0]) field type is a device pointer, got " &
    $constrs[0].ocFields[0].typ.addressSpace
  doAssert constrs[1].ocFields[0].typ.kind == gtPtr and
    constrs[1].ocFields[0].typ.addressSpace == asRMEM,
    "Holder(p: addr tmp) field type is a thread pointer, got " &
    $constrs[1].ocFields[0].typ.addressSpace

  # the same stamp must flow through the operand-type resolver the passes read
  var addrs: seq[GpuAst]
  # depth-first pre-order over the body, collecting gpuAddr nodes
  proc collectAddrs(n: GpuAst, acc: var seq[GpuAst]) =
    if n.isNil:
      return
    if n.kind == gpuAddr:
      acc.add n
    for ch in n.items:
      collectAddrs(ch, acc)
  collectAddrs(fn.pBody, addrs)
  doAssert addrs.len == 2, "expected the two address-of nodes, got " & $addrs.len

  var ctx = GpuContext()
  doAssert ctx.getOperandType(addrs[0]).addressSpace == asDevice,
    "addr outp[0] derives a device pointer, got " &
    $ctx.getOperandType(addrs[0]).addressSpace
  doAssert ctx.getOperandType(addrs[1]).addressSpace == asRMEM,
    "addr tmp derives a thread pointer, got " &
    $ctx.getOperandType(addrs[1]).addressSpace
  echo "  OK - address-of inherits the addressed storage's space"

# ── 3. A span built over device memory keeps device through lowerArraysAndViews ──
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
  # a callee span param is implicitly generic over its space, like a struct's
  # ptr field. The instantiation pass (M3) owns the split ptr's device space
  var scale: GpuAst = nil
  for stmt in ir.statements:
    if stmt.kind == gpuProc and stmt.pName.symbol.name == "scaleData":
      scale = stmt
  doAssert not scale.isNil, "expected the device fn in the context"
  doAssert scale.pParams[1].typ.kind == gtPtr,
    "the callee's openArray param splits to a ptr in its rewritten signature"
  echo "  OK - device span survives span-to-pointer normalization"

# ── 4. Smem arena: the hoisted threadgroup pointer stamps asSMEM ──
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
  echo "  OK - smem arena stamps asSMEM on the hoisted type"

# ── 5. const_mem keeps its storage space on the variable side ──
static:
  let ir = toCrucibleAst:
    proc constKernel(outp: ptr UncheckedArray[float32]) {.global.} =
      var scale {.const_mem.}: float32
      outp[0] = scale

  let fn = ir.findProc
  let scale = fn.pBody.findVar("scale")
  doAssert scale.kind == gpuVar and scale.addressSpace == asConstant,
    "a {.const_mem.} scalar keeps asConstant on the variable side, got " &
    $scale.addressSpace
  doAssert scale.vType.kind == gtFloat32,
    "the {.const_mem.} scalar's type is the plain scalar, got " & $scale.vType.kind
  echo "  OK - const_mem storage space stays on the variable side"

# ── 6. withAddressSpace is copy-on-write and spans clone whole ──
static:
  let elem = GpuType(kind: gtFloat32)
  let span = initGpuSpanType(elem)

  # same space: the returned type is the same ref, a change through it
  # is visible through the original
  let restamped = withAddressSpace(span, asRMEM)
  restamped.sElemTyp = GpuType(kind: gtBool)
  doAssert span.sElemTyp.kind == gtBool,
    "restamping the default space returns the same ref, not a clone"
  span.sElemTyp = elem

  # different space: a clone, the shared type is never mutated
  let deviceSpan = withAddressSpace(span, asDevice)
  deviceSpan.sElemTyp = GpuType(kind: gtBool)
  doAssert span.sElemTyp.kind == gtFloat32,
    "a different space returns a clone, the shared type keeps its element type"
  doAssert deviceSpan.addressSpace == asDevice,
    "the span clone carries the stamped space, got " & $deviceSpan.addressSpace
  echo "  OK - withAddressSpace clones copy-on-write, span clones keep sElemTyp"

# ── 7. A ptr-to-array param decays keeping its stamped space ──
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

  # the decay rebuilds gtPtr(gtArray) as gtPtr(gtUA)
  # a fresh rebuild would reset the space, the stamped birth space must survive
  for i in 0 ..< 2:
    doAssert reduceFn.pParams[i].typ.kind == gtPtr and
      reduceFn.pParams[i].typ.addressSpace == asDevice,
      "the decayed ptr-array param keeps asDevice, got " &
      $reduceFn.pParams[i].typ.addressSpace
  echo "  OK - ptr-array decay keeps the stamped space"

# ── 8. A span view binding decomposes keeping the span's space ──
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
