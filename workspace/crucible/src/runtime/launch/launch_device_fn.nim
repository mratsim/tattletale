# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, hashes, strutils, tables]
import ../../codegen/gpu_compiler
import ../engines_common
import launch_kernel {.all.}
import launch_serialization {.all.}

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════

type
  DeviceFnSugar*[E: HwEngine] = object
    engine: E

  DeviceFnProxy*[E: HwEngine] = object
    engine: E
    grid: Dim3
    blk: Dim3

  BridgeAccumulator = object
    signature: seq[tuple[arg: string, argTy: NimNode]]
    deviceFnCall: NimNode


# ═════════════════════════════════════════════════════════════════════════
# ▸ Bridge construction
# ═════════════════════════════════════════════════════════════════════════

proc makeDeviceFnSugar[E: HwEngine](engine: E): DeviceFnSugar[E] {.inline.} =
  DeviceFnSugar[E](engine: engine)

proc isPhantomType(t: NimNode): bool =
  ## True when the type carries no runtime fields
  case t.kind
  of nnkObjectTy:
    result = true
    for field in t[^1]:
      if field.kind in {nnkEmpty, nnkNilLit}: continue
      if field.kind != nnkIdentDefs: return false
      for i in 0 ..< field.len - 2:
        if not isPhantomType(field[i]): return false
  of nnkTupleTy:
    result = true
    for i in 0 ..< t.len:
      if not isPhantomType(t[i]): return false
  of nnkSym:
    let impl = t.getImpl()
    if impl.isNil or impl.kind != nnkTypeDef: return false
    result = isPhantomType(impl[2])
  of nnkBracketExpr:
    result = isPhantomType(t[0])
  else:
    result = false

proc needTempStorage(n: NimNode): bool =
  case n.kind
  of nnkVarTy:
    error("launchDeviceFn: it is unsafe to capture a `var` parameter '" & repr(n) & "' by value")
  of nnkStaticTy, nnkCharLit..nnkNilLit:
    return false
  else:
    let t = n.getTypeInst()
    if t.typeKind in {ntyTypeDesc, ntyStatic}:
      return false
    # no-runtime-field types bake
    case t.kind
    of nnkObjectTy:
      for field in t[^1]:
        if field.kind in {nnkEmpty, nnkNilLit}: continue
        if field.kind != nnkIdentDefs: return true
        for i in 0 ..< field.len - 2:
          if not field[i].isPhantomType(): return true
      return false
    of nnkTupleTy:
      for i in 0 ..< t.len:
        if not t[i].isPhantomType(): return true
      return false
    else:
      return true

proc base58(h: int): string =
  const alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  var x = uint64(h)
  if x == 0: return "1"
  while x > 0:
    result = alphabet[int(x mod 58)] & result
    x = x div 58

proc declGlobalFn(name: string, acc: BridgeAccumulator): NimNode =
  ## proc <name>(<signature>) {.global.} = <deviceFnCall>
  var params = nnkFormalParams.newTree(newEmptyNode())
  for (pname, ptype) in acc.signature:
    params.add(nnkIdentDefs.newTree(ident(pname), ptype, newEmptyNode()))
  result = nnkProcDef.newTree(
    ident(name),
    newEmptyNode(), newEmptyNode(),
    params,
    nnkPragma.newTree(ident"global"),
    newEmptyNode(),
    nnkStmtList.newTree(acc.deviceFnCall))

macro bridge(engine: HwEngine, node: untyped): untyped =
  let metalCg = bindSym"metal"
  let cudaCg = bindSym"cuda"
  let openclCg = bindSym"opencl"
  let vulkanCg = bindSym"vulkan"
  let webgpuCg = bindSym"webgpu"
  return quote do:
    when typeof(`engine`).backendKind == bkMetal: `metalCg`(`node`)
    elif typeof(`engine`).backendKind == bkCuda: `cudaCg`(`node`)
    elif typeof(`engine`).backendKind == bkOpenCL: `openclCg`(`node`)
    elif typeof(`engine`).backendKind == bkVulkan: `vulkanCg`(`node`)
    else: `webgpuCg`(`node`)

macro buildGlobalFn(engine: HwEngine, kernel: typed, gridN, blkN: untyped, values: varargs[typed]): untyped =
  var acc = BridgeAccumulator(deviceFnCall: nnkCall.newTree(kernel))
  var argsTuple = nnkTupleConstr.newTree()
  var outputNode: NimNode = nil

  for v in values:
    if v.needTempStorage():
      # env member: signature param, forwarded to deviceFnCall
      let pname = "a" & $acc.signature.len
      acc.signature.add((pname, v.getTypeInst()))
      acc.deviceFnCall.add(ident(pname))
      if outputNode.isNil:
        outputNode = v
      else:
        argsTuple.add(v)
    else: # constant-value, embedded in the global function
      acc.deviceFnCall.add(v)

  doAssert not outputNode.isNil, "launchDeviceFn " & $kernel & ": the kernel needs a launch value formal as output"

  var accRepr = ""
  for (pname, ptype) in acc.signature:
    accRepr.add(pname & ":" & repr(ptype) & ";")
  accRepr.add(repr(acc.deviceFnCall))
  let name = $kernel & "_" & base58(hash(accRepr))
  let bridgeNode = nnkStmtList.newTree(declGlobalFn(name, acc))
  let gridD = bindSym"dim3".newCall(gridN)
  let blkD = bindSym"dim3".newCall(blkN)

  let bridgeSource = ident("crucibleBridgeSource" & name)
  let bridgeN = bindSym"bridge"

  return quote do:
    block:
      const `bridgeSource` = `bridgeN`(`engine`, `bridgeNode`)
      if `engine`.getArtifact() != `bridgeSource`:
        `engine`.ingest(`bridgeSource`)
      `engine`.launchKernelImpl(`name`, `outputNode`, `argsTuple`, `gridD`, `blkD`)

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

macro launchDeviceFn*[E](engine: E): DeviceFnSugar[E] =
  ## Chevron form accessor for device-fn launches.
  ## - Chevrons:
  ##   `engine.launchDeviceFn<<(grid, blk)>> fn(args)`
  ## - Do form:
  ##     `engine.launchDeviceFn <kernel-expr> do:`
  ##       `grid = (ceil_div(m, 32), ceil_div(n, 32))`
  ##       `blk = (32, 1)`
  result = newCall(bindSym"makeDeviceFnSugar", engine)

proc `<<`*[E: HwEngine, G, B: int|tuple](r: DeviceFnSugar[E], cfg: tuple[grid: G, blk: B]): DeviceFnProxy[E] {.inline.} =
  DeviceFnProxy[E](engine: r.engine, grid: dim3(cfg.grid), blk: dim3(cfg.blk))

macro launchDeviceFn*(engine: HwEngine, call: untyped): untyped =
  ## Launches a device function with whole values, the schedule is a `do:` block.
  ## - Chevrons:
  ##   `engine.launchDeviceFn<<(grid, blk)>> fn(args)`
  ## - Do form:
  ##     `engine.launchDeviceFn gemm(dD, dA, dB, 1) do:`
  ##       `grid = (ceil_div(m, 32), ceil_div(n, 32))`
  ##       `blk = (32, 1)`
  var schedule: NimNode = nil
  var callNode = call
  if callNode[^1].kind == nnkStmtList:
    schedule = callNode[^1]
    callNode = nnkCall.newTree(call[0 ..< call.len - 1])
  doAssert schedule.isNil or schedule.kind == nnkStmtList, "launchDeviceFn: the schedule must be a `do:` block, got " & schedule.repr
  var gridN = nnkPar.newTree(newLit(1), newLit(1))
  var blkN = nnkPar.newTree(newLit(1), newLit(1))
  if not schedule.isNil:
    for a in schedule:
      doAssert a.kind == nnkAsgn and $a[0] in ["grid", "blk"], "launchDeviceFn: schedule expects grid = ... / blk = ..., got " & a.repr
      if $a[0] == "grid":
        gridN = a[1]
      else:
        blkN = a[1]
  let buildN = bindSym"buildGlobalFn"
  result = nnkCall.newTree(buildN, engine, callNode[0], gridN, blkN)
  for i in 1 ..< callNode.len:
    result.add(callNode[i])

macro `>>`*[E: HwEngine](proxy: DeviceFnProxy[E], call: untyped): untyped =
  var callNode = call
  if callNode.kind == nnkPar and callNode.len == 1:
    callNode = callNode[0]
  doAssert callNode.kind == nnkCall, "launchDeviceFn: expected fn(args...), got " & callNode.repr
  let buildN = bindSym"buildGlobalFn"
  result = nnkCall.newTree(buildN,
    proxy.newDotExpr(ident"engine"), callNode[0],
    proxy.newDotExpr(ident"grid"),
    proxy.newDotExpr(ident"blk"))
  for i in 1 ..< callNode.len:
    result.add(callNode[i])
