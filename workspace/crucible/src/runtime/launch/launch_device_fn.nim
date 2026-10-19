# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, hashes]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines/engines_api
import workspace/crucible/src/runtime/engines/nvrtc
import workspace/crucible/src/runtime/engines/cl
import workspace/crucible/src/runtime/engines/vk
import workspace/crucible/src/runtime/engines/wgpu
import workspace/crucible/src/runtime/engines/metal
import workspace/crucible/src/runtime/launch/launch_kernel {.all.}
import workspace/crucible/src/runtime/launch/launch_serialization {.all.}

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
    sharedMem: int


# ═════════════════════════════════════════════════════════════════════════
# ▸ Bridge construction
# ═════════════════════════════════════════════════════════════════════════

func makeDeviceFnSugar[E: HwEngine](engine: E): DeviceFnSugar[E] {.inline.} =
  DeviceFnSugar[E](engine: engine)

func base58(h: int): string =
  const alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  var x = uint64(h)
  if x == 0: return "1"
  while x > 0:
    result = alphabet[int(x mod 58)] & result
    x = x div 58

macro bridge(engine: HwEngine, node: untyped): untyped =
  ## Defers the generated global function generation
  ## so that it happens in a `device` environment.
  ## This allows functions that depend on `ccGetBackend`
  ## to correctly resolve the target backend and not resolve the CPU host.
  let metalCg = bindSym"metal"
  let cudaCg = bindSym"cuda"
  let openclCg = bindSym"opencl"
  let vulkanCg = bindSym"vulkan"
  let webgpuCg = bindSym"webgpu"
  let t = engine.getTypeInst()
  if sameType(t, bindSym"MetalEngine"):
    result = metalCg.newCall(node)
  elif sameType(t, bindSym"CudaEngine"):
    result = cudaCg.newCall(node)
  elif sameType(t, bindSym"OpenCLEngine"):
    result = openclCg.newCall(node)
  elif sameType(t, bindSym"VulkanEngine"):
    result = vulkanCg.newCall(node)
  elif sameType(t, bindSym"WgpuEngine"):
    result = webgpuCg.newCall(node)
  else:
    error("launchDeviceFn: engine type " & t.repr & " has no backend codegen")

func isCompileTime(ftype: NimNode): bool =
  ftype.kind == nnkStaticTy or
    (ftype.kind in {nnkIdent, nnkSym} and ftype.eqIdent("typedesc")) or
    (ftype.kind in {nnkCommand, nnkBracketExpr, nnkCall} and
     $ftype[0] in ["static", "typedesc"])

func isDeviceMemType(t: NimNode): bool =
  t.kind == nnkBracketExpr and t[0].eqIdent("DeviceMem")

func getElemType(formal: NimNode): NimNode =
  formal.expectKind(nnkPtrTy)
  let p = formal[0]
  if p.kind == nnkBracketExpr and p[0].eqIdent("UncheckedArray"): p[1]
  else: p

func assertValidDeviceMem(kernel, pname, formal, vtype: NimNode) =
  if formal.kind != nnkPtrTy:
    error("launchDeviceFn " & $kernel & ": parameter " & $pname & " is " &
      formal.repr & ", a DeviceMem argument needs a pointer parameter")
  if not sameType(vtype[2], getElemType(formal)):
    error("launchDeviceFn " & $kernel & ": parameter " & $pname & " points at " &
      getElemType(formal).repr & ", the DeviceMem argument holds " & vtype[2].repr)

func getDeviceMemFields(v: NimNode): seq[tuple[name: string, t: NimNode]] =
  ## DeviceMem fields of the value's type: a top-level DeviceMem yields
  ## the value itself with an empty name
  let t = v.getTypeInst()
  if isDeviceMemType(t):
    result.add((name: "", t: t))
    return
  let impl = v.getTypeImpl()
  case impl.kind
  of nnkObjectTy:
    let rec = impl[^1]
    if rec.kind != nnkRecList: return
    for f in rec:
      f.expectKind(nnkIdentDefs)
      if isDeviceMemType(f[^2]):
        for j in 0 ..< f.len - 2:
          result.add((name: $f[j], t: f[^2]))
  of nnkTupleTy:
    for i, f in impl:
      if isDeviceMemType(f[^2]):
        result.add((name: "field" & $i, t: f[^2]))
  else:
    discard

func hashNode(n: NimNode): Hash =
  ## Structural hash of an AST node: kind, leaves by value, children by shape
  result = hash(n.kind)
  case n.kind
  of nnkIdent, nnkSym:
    result = result !& hash(n.strVal)
  of nnkIntLit .. nnkUInt64Lit:
    result = result !& hash(n.intVal)
  of nnkFloatLit .. nnkFloat64Lit:
    result = result !& hash(n.floatVal)
  of nnkEmpty:
    discard
  else:
    for child in n:
      result = result !& hashNode(child)
  result = !$result

func bindFormalGenericArgs(formal, v: NimNode): NimNode =
  let vt = v.getTypeInst()
  if formal.kind != nnkDistinctTy or vt.kind != nnkBracketExpr:
    return formal
  let tsym = formal[0]
  if tsym.kind != nnkSym:
    return formal
  let timpl = tsym.getImpl()
  if timpl.kind != nnkTypeDef or timpl[1].kind != nnkGenericParams:
    return formal
  if vt.len - 1 != timpl[1].len:
    return formal
  var args = nnkBracketExpr.newTree(tsym)
  for j in 1 ..< vt.len:
    let a = vt[j]
    if isDeviceMemType(a):
      args.add(a[^1])
    else:
      args.add(a)
  return args

func getBridgeParamType(kernel: NimNode, formalName, formalType, arg: NimNode): NimNode =
  let argType = arg.getTypeInst()
  if isDeviceMemType(argType):
    assertValidDeviceMem(kernel, formalName, formalType, argType)
    return formalType
  if getDeviceMemFields(arg).len > 0:
    return bindFormalGenericArgs(formalType, arg)
  return argType

func buildGlobalFn(kernel: NimNode, statics, values: NimNode): NimNode =
  ## ```
  ##   kernel  = kern                                        (typed callee symbol)
  ##   statics = [tiledMma(f32, f16, f16, 32, 32, 16)]       (raw, one per compile-time parameter)
  ##   values  = [Int[3](), v.addr]                          (type-checked, one per runtime parameter)
  ##     │
  ##     │  zip against the kernel's parameters, in declaration order:
  ##     │    compile-time parameter → its argument is written into the kernel call as-is
  ##     │    runtime parameter     → a new kernel parameter, typed by its argument
  ##     │      (special-cased for DeviceMem: typed from the kernel's declared parameter)
  ##     ▼
  ##   proc kern_Xy3kP(x: Int[3], v: ptr float32) {.global.} =
  ##     kern(tiledMma(f32, f16, f16, 32, 32, 16), x, v)
  ##
  let impl = kernel.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalNames: seq[NimNode]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      formalNames.add(f[j])
      formalTypes.add(f[^2])
  # identity: the callee, the compile-time values, the runtime parameter types
  var h = hash($kernel)
  var kcall = ident($kernel).newCall()
  var params = nnkFormalParams.newTree(newEmptyNode())
  var si = 0
  var vi = 0
  for i in 0 ..< formalTypes.len:
    if isCompileTime(formalTypes[i]):
      let v = statics[si]
      si.inc
      kcall.add(v)
      h = h !& hashNode(v)
    else:
      let v = values[vi]
      vi.inc
      let pname = ident($formalNames[i])
      let ptype = getBridgeParamType(kernel, formalNames[i], formalTypes[i], v)
      params.add(nnkIdentDefs.newTree(pname, ptype, newEmptyNode()))
      kcall.add(pname)
      h = h !& hashNode(ptype)

  doAssert vi > 0, "launchDeviceFn " & $kernel & ": the kernel needs a runtime parameter as output"

  let procdef = nnkProcDef.newTree(
    ident($kernel & "_" & base58(int(h))),
    newEmptyNode(), newEmptyNode(),
    params,
    nnkPragma.newTree(ident"global"),
    newEmptyNode(),
    nnkStmtList.newTree(kcall))
  return procdef

macro launchGlobalFn(engine: HwEngine, kernel: typed, statics: untyped, gridN, blkN, sharedMemN: untyped, values: varargs[typed]): untyped =
  ## Builds the `{.global.}` proc for the kernel, ingests its lowered source
  ## when it differs and launches it: the first value is the output.
  let procdef = buildGlobalFn(kernel, statics, values)
  let kname = newLit($procdef.name)
  let bridgeN = bindSym"bridge"
  let bridgeSource = ident("crucibleBridgeSource" & $procdef.name)
  let gridD = bindSym"dim3".newCall(gridN)
  let blkD = bindSym"dim3".newCall(blkN)
  let sharedMemD = sharedMemN
  let outputN = values[0]
  var argsN = nnkTupleConstr.newTree()
  for i in 1 ..< values.len:
    argsN.add(values[i])
  result = quote do:
    block:
      const `bridgeSource` = `bridgeN`(`engine`, `procdef`)
      if `engine`.getArtifact() != `bridgeSource`:
        `engine`.ingest(`bridgeSource`)
      var argsVal = `argsN`
      `engine`.launchKernelImpl(`kname`, `outputN`, argsVal,
        LaunchConfig(grid: `gridD`, blk: `blkD`, sharedMem: `sharedMemD`))

macro resolveDeviceCall(engine: HwEngine, kernel: typed, gridN, blkN, sharedMemN: untyped, args: varargs[untyped]): untyped =
  ## Call an autogenerated {.global.} kernel
  ## This defers arguments so they are inlined in the global function and ccGetBackend resolves properly
  let impl = kernel.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      formalTypes.add(f[^2])
  doAssert args.len == formalTypes.len,
    "launchDeviceFn " & $kernel & ": " & $args.len & " args for " &
    $formalTypes.len & " parameters"
  var statics = nnkBracket.newTree()
  result = bindSym"launchGlobalFn".newCall(engine, kernel, statics, gridN, blkN, sharedMemN)
  for i, a in args:
    if isCompileTime(formalTypes[i]):
      statics.add(a)
    else:
      result.add(a)

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

func `<<`*[E: HwEngine, G, B: int|tuple](r: DeviceFnSugar[E], cfg: tuple[grid: G, blk: B]): DeviceFnProxy[E] {.inline.} =
  DeviceFnProxy[E](engine: r.engine, grid: dim3(cfg.grid), blk: dim3(cfg.blk))

func `<<`*[E: HwEngine, G, B: int|tuple, S: int](r: DeviceFnSugar[E],
    cfg: tuple[grid: G, blk: B, sharedMem: S]): DeviceFnProxy[E] {.inline.} =
  DeviceFnProxy[E](engine: r.engine, grid: dim3(cfg.grid), blk: dim3(cfg.blk),
                   sharedMem: cfg.sharedMem)

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
  var gridN = nnkPar.newTree(newLit(1), newLit(1))
  var blkN = nnkPar.newTree(newLit(1), newLit(1))
  var sharedMemN = newLit(0)
  if not schedule.isNil:
    for a in schedule:
      doAssert a.kind == nnkAsgn and $a[0] in ["grid", "blk", "sharedMem"],
        "launchDeviceFn: schedule expects grid = ... / blk = ... / sharedMem = ..., got " & a.repr
      case $a[0]
      of "grid": gridN = a[1]
      of "blk": blkN = a[1]
      else: sharedMemN = a[1]
  result = bindSym"resolveDeviceCall".newCall(engine, callNode[0], gridN, blkN, sharedMemN)
  for i in 1 ..< callNode.len:
    result.add(callNode[i])

macro `>>`*[E: HwEngine](proxy: DeviceFnProxy[E], call: untyped): untyped =
  var callNode = call
  if callNode.kind == nnkPar and callNode.len == 1:
    callNode = callNode[0]
  doAssert callNode.kind == nnkCall, "launchDeviceFn: expected fn(args...), got " & callNode.repr
  result = bindSym"resolveDeviceCall".newCall(
    proxy.newDotExpr(ident"engine"), callNode[0],
    proxy.newDotExpr(ident"grid"),
    proxy.newDotExpr(ident"blk"),
    proxy.newDotExpr(ident"sharedMem"))
  for i in 1 ..< callNode.len:
    result.add(callNode[i])
