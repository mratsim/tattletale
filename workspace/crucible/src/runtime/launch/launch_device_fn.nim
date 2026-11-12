# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, hashes]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines/engines_api
import workspace/crucible/src/runtime/engines/cuda_api
import workspace/crucible/src/runtime/engines/opencl_api
import workspace/crucible/src/runtime/engines/vulkan_api
import workspace/crucible/src/runtime/engines/webgpu_api
import workspace/crucible/src/runtime/engines/metal_api
import workspace/crucible/src/runtime/launch/launch_kernel {.all.}
import workspace/crucible/src/runtime/launch/launch_serialization {.all.}

# ═══════════════════════════════════════════════════════════════════════
# ▸ Types
# ═══════════════════════════════════════════════════════════════════════

type
  DeviceFnSugar*[E: HwEngine] = object
    engine: E

  DeviceFnProxy*[E: HwEngine, L: static string = "spirv"] = object
    engine: E
    grid: Dim3
    blk: Dim3
    sharedMem: int

func parseLangName(n: NimNode): string =
  var name = $n
  if n.kind == nnkStrLit:
    name = n.strVal
  if name notin ["spirv", "glsl", "hlsl"]:
    error("launchDeviceFn: lang expects glsl, hlsl or spirv, got " & name)
  name


# ═══════════════════════════════════════════════════════════════════════
# ▸ Bridge construction
# ═══════════════════════════════════════════════════════════════════════

func makeDeviceFnSugar[E: HwEngine](engine: E): DeviceFnSugar[E] {.inline.} =
  DeviceFnSugar[E](engine: engine)

func base58(h: int): string =
  const alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  var x = uint64(h)
  if x == 0:
    return "1"
  while x > 0:
    result = alphabet[int(x mod 58)] & result
    x = x div 58

macro bridge(engine: HwEngine, lang: static string, node: untyped): untyped =
  let metalCg = bindSym"metal"
  let cudaCg = bindSym"cuda"
  let openclCg = bindSym"opencl"
  let glslCg = bindSym"glsl"
  let spirvCg = bindSym"spirv"
  let hlslCg = bindSym"hlsl"
  let webgpuCg = bindSym"webgpu"
  let t = engine.getTypeInst()
  if sameType(t, bindSym"MetalEngine"):
    if lang != "spirv":
      error("launchDeviceFn: lang = " & lang & " is only meaningful on VulkanEngine, got engine " & t.repr)
    result = metalCg.newCall(node)
  elif sameType(t, bindSym"CudaEngine"):
    if lang != "spirv":
      error("launchDeviceFn: lang = " & lang & " is only meaningful on VulkanEngine, got engine " & t.repr)
    result = cudaCg.newCall(node)
  elif sameType(t, bindSym"OpenCLEngine"):
    if lang != "spirv":
      error("launchDeviceFn: lang = " & lang & " is only meaningful on VulkanEngine, got engine " & t.repr)
    result = openclCg.newCall(node)
  elif sameType(t, bindSym"VulkanEngine"):
    case lang
    of "spirv": result = spirvCg.newCall(node)
    of "glsl": result = glslCg.newCall(node)
    of "hlsl": result = hlslCg.newCall(node)
  elif sameType(t, bindSym"WebgpuEngine"):
    if lang != "spirv":
      error("launchDeviceFn: lang = " & lang & " is only meaningful on VulkanEngine, got engine " & t.repr)
    result = webgpuCg.newCall(node)
  elif sameType(t, bindSym"CpuEngine"):
    error("launchDeviceFn: the CpuEngine launches through launchCpuFn," &
      " launchDeviceFn has no cpu backend")
  else:
    error("launchDeviceFn: engine type " & t.repr & " has no backend codegen")

func isCompileTime(ftype: NimNode): bool =
  ftype.kind == nnkStaticTy or
    (ftype.kind in {nnkIdent, nnkSym} and ftype.eqIdent("typedesc")) or
    (ftype.kind in {nnkCommand, nnkBracketExpr, nnkCall} and
     $ftype[0] in ["static", "typedesc"])

func isDeviceMemType(t: NimNode): bool =
  t.kind == nnkBracketExpr and t[0].eqIdent("DeviceMem")

func unwrapDeviceMemArg(t: NimNode): NimNode =
  if t.kind == nnkRefTy and isDeviceMemType(t[0]):
    return t[0]
  if isDeviceMemType(t):
    return t
  return newEmptyNode()

func getElemType(formal: NimNode): NimNode =
  formal.expectKind(nnkPtrTy)
  let p = formal[0]
  if p.kind == nnkBracketExpr and p[0].eqIdent("UncheckedArray"):
    p[1]
  else:
    p

func assertValidDeviceMem(deviceFn, pname, formal, vtype: NimNode) =
  if formal.kind != nnkPtrTy:
    error("launchDeviceFn " & $deviceFn & ": parameter " & $pname & " is " &
      formal.repr & ", a DeviceMem argument needs a pointer parameter")
  if not sameType(vtype[2], getElemType(formal)):
    error("launchDeviceFn " & $deviceFn & ": parameter " & $pname & " points at " &
      getElemType(formal).repr & ", the DeviceMem argument holds " & vtype[2].repr)

func getDeviceMemFields(v: NimNode): seq[tuple[name: string, t: NimNode]] =
  let t = unwrapDeviceMemArg(v.getTypeInst())
  if t.kind != nnkEmpty:
    result.add((name: "", t: t))
    return
  let impl = v.getTypeImpl()
  case impl.kind
  of nnkObjectTy:
    let rec = impl[^1]
    if rec.kind != nnkRecList:
      return
    for f in rec:
      f.expectKind(nnkIdentDefs)
      let ft = unwrapDeviceMemArg(f[^2])
      if ft.kind != nnkEmpty:
        for j in 0 ..< f.len - 2:
          result.add((name: $f[j], t: ft))
  of nnkTupleTy:
    for i, f in impl:
      let ft = unwrapDeviceMemArg(f[^2])
      if ft.kind != nnkEmpty:
        result.add((name: "field" & $i, t: ft))
  else:
    discard

func hashNode(n: NimNode): Hash =
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
  let nParams = timpl[1].len
  if vt.len - 1 < nParams:
    return formal
  # A formal with fewer parameters than the value's type arguments binds leading arguments first.
  # TensorBasicView[T] takes the element type from the bridge's DeviceMem slot and drops the layout slot.
  var args = nnkBracketExpr.newTree(tsym)
  for j in 1 .. nParams:
    let a = unwrapDeviceMemArg(vt[j])
    if a.kind != nnkEmpty:
      args.add(a[^1])
    else:
      args.add(vt[j])
  return args

func getBridgeParamType(deviceFn: NimNode, formalName, formalType, arg: NimNode): NimNode =
  let dm = unwrapDeviceMemArg(arg.getTypeInst())
  if dm.kind != nnkEmpty:
    assertValidDeviceMem(deviceFn, formalName, formalType, dm)
    return formalType
  if getDeviceMemFields(arg).len > 0:
    return bindFormalGenericArgs(formalType, arg)
  return arg.getTypeInst()

func buildGlobalFn(deviceFn: NimNode, statics, values: NimNode): NimNode =
  ## ```
  ##   deviceFn = kern                                       (typed callee symbol)
  ##   statics = [tiledMma(f32, f16, f16, 32, 32, 16)]       (raw, one per compile-time parameter)
  ##   values  = [Int[3](), v.addr]                          (type-checked, one per runtime parameter)
  ##     │
  ##     │  zip against the device function's parameters, in declaration order:
  ##     │    compile-time parameter → its argument is written into the device function call as-is
  ##     │    runtime parameter     → a new parameter, typed by its argument
  ##     │      (special-cased for DeviceMem: typed from the device function's declared parameter)
  ##     ▼
  ##   proc kern_Xy3kP(x: Int[3], v: ptr float32) {.global.} =
  ##     kern(tiledMma(f32, f16, f16, 32, 32, 16), x, v)
  ##
  let impl = deviceFn.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalNames: seq[NimNode]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      formalNames.add(f[j])
      formalTypes.add(f[^2])
  # identity: the callee, its compile-time values and the runtime parameter types
  var h = hash($deviceFn)
  # Bound callee symbol. The wrapper sem-checks at the materialization site,
  # where the callee identifier is out of scope.
  var deviceFnCall = deviceFn.newCall()
  var params = nnkFormalParams.newTree(newEmptyNode())
  var si = 0
  var vi = 0
  for i in 0 ..< formalTypes.len:
    if si >= statics.len and vi >= values.len:
      # values beyond the given formals are defaults
      break
    if isCompileTime(formalTypes[i]):
      let v = statics[si]
      si.inc
      deviceFnCall.add(v)
      h = h !& hashNode(v)
    else:
      let v = values[vi]
      vi.inc
      let pname = ident($formalNames[i])
      let ptype = getBridgeParamType(deviceFn, formalNames[i], formalTypes[i], v)
      params.add(nnkIdentDefs.newTree(pname, ptype, newEmptyNode()))
      deviceFnCall.add(pname)
      h = h !& hashNode(ptype)

  doAssert vi > 0, "launchDeviceFn " & $deviceFn & ": the device function needs a runtime parameter as output"

  let procdef = nnkProcDef.newTree(
    ident($deviceFn & "_" & base58(int(h))),
    newEmptyNode(), newEmptyNode(),
    params,
    nnkPragma.newTree(bindSym"global"),
    newEmptyNode(),
    nnkStmtList.newTree(deviceFnCall))
  return procdef

func getWorkgroupPragma(blk: Dim3): NimNode =
  ## Workgroup pragma `(X, Y, Z)` from a compile-time block size, nil when unit
  if blk.x == 1 and blk.y == 1 and blk.z == 1:
    return nil
  var size = nnkTupleConstr.newTree()
  for d in [blk.x, blk.y, blk.z]:
    size.add newLit(d)
  nnkExprColonExpr.newTree(bindSym"workgroup", size)

func getWorkgroupPragma(blkN: NimNode): NimNode =
  ## Workgroup pragma `(X, Y, Z)` for a compile-time literal block size, nil otherwise
  var dims: array[3, int] = [1, 1, 1]
  if blkN.kind == nnkIntLit:
    dims[0] = blkN.intVal.int
  elif blkN.kind in {nnkPar, nnkTupleConstr, nnkBracket} and
      blkN.len in {2, 3}:
    for i, e in blkN:
      if e.kind != nnkIntLit:
        return nil
      dims[i] = e.intVal.int
  else:
    return nil
  if dims[0] == 1 and dims[1] == 1 and dims[2] == 1:
    return nil
  var size = nnkTupleConstr.newTree()
  for d in dims:
    size.add newLit(d)
  nnkExprColonExpr.newTree(bindSym"workgroup", size)

macro launchGlobalFn(engine: HwEngine, deviceFn: typed, statics: untyped, gridN, blkN, sharedMemN: untyped, lang: static string, values: varargs[typed]): untyped =
  ## Builds and launches the `{.global.}` wrapper: the first value is the output.
  let procdef = buildGlobalFn(deviceFn, statics, values)
  let wg = getWorkgroupPragma(blkN)
  if not wg.isNil():
    procdef[4].add(wg)
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
  let langLit = newLit(lang)
  let ingestSpirvWords = sameType(engine.getTypeInst(), bindSym"VulkanEngine") and lang == "spirv"
  let ingestN = if ingestSpirvWords:
    quote do:
      if `engine`.getSpirV() != `bridgeSource`:
        `engine`.ingest(`bridgeSource`)
  else:
    quote do:
      if `engine`.getArtifact() != `bridgeSource`:
        `engine`.ingest(`bridgeSource`)
  result = quote do:
    block:
      const `bridgeSource` = `bridgeN`(`engine`, `langLit`, `procdef`)
      `ingestN`
      `engine`.launchKernelImpl(`outputN`, `argsN`,
        LaunchConfig(grid: `gridD`, blk: `blkD`, sharedMem: `sharedMemD`), `kname`)

func countFormals(candFormals: NimNode): tuple[total, required: int] =
  for i in 1 ..< candFormals.len:
    let f = candFormals[i]
    result.total += f.len - 2
    if f[^1].kind == nnkEmpty:
      result.required += f.len - 2

proc resolveDeviceFnOverload*(callNode: NimNode): NimNode =
  let deviceFn = callNode[0]
  let argCount = callNode.len - 1
  if deviceFn.kind notin {nnkOpenSymChoice, nnkClosedSymChoice}:
    return deviceFn
  var match: NimNode = nil
  var count = 0
  for cand in deviceFn:
    let candImpl = cand.getImpl()
    if candImpl.kind != nnkProcDef:
      continue
    let (total, required) = countFormals(candImpl[3])
    if required <= argCount and argCount <= total:
      match = cand
      count.inc
  doAssert count == 1,
    "launchDeviceFn " & $deviceFn & ": " & $count & " overloads take " &
    $argCount & " arguments"
  match

macro resolveDeviceCall(engine: HwEngine, deviceFn: typed, gridN, blkN, sharedMemN: untyped, lang: static string, args: varargs[untyped]): untyped =
  ## This defers arguments so they are inlined in the global function and ccGetBackend resolves properly
  for a in args:
    doAssert a.kind notin {nnkExprEqExpr, nnkExprColonExpr},
      "launchDeviceFn " & $deviceFn & ": named arguments are not supported, pass arguments positionally"
  var callN = nnkCall.newTree(deviceFn)
  for a in args:
    callN.add(a)
  let k = callN.resolveDeviceFnOverload()
  let impl = k.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      formalTypes.add(f[^2])
  doAssert args.len <= formalTypes.len,
    "launchDeviceFn " & $deviceFn & ": " & $args.len & " args for " &
    $formalTypes.len & " parameters"
  var statics = nnkBracket.newTree()
  let langLit = newLit(lang)
  result = bindSym"launchGlobalFn".newCall(engine, k, statics, gridN, blkN, sharedMemN, langLit)
  for i, a in args:
    if isCompileTime(formalTypes[i]):
      statics.add(a)
    else:
      result.add(a)

# ═══════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

macro launchDeviceFn*[E](engine: E): DeviceFnSugar[E] =
  result = newCall(bindSym"makeDeviceFnSugar", engine)

macro `<<`*[E: HwEngine](r: DeviceFnSugar[E], cfg: untyped): untyped =
  ## Chevron schedule: `(grid, blk)` or `(grid, blk, sharedMem)`, positional or named
  ## (`grid: ..., blk: ...`). The `lang = glsl / hlsl / spirv` extra selects
  ## the SPIR-V ingestion language of a VulkanEngine launch.
  var gridN = nnkPar.newTree(newLit(1), newLit(1))
  var blkN = nnkPar.newTree(newLit(1), newLit(1))
  var sharedMemN = newLit(0)
  var lang = "spirv"
  var c = cfg
  if c.kind in {nnkPar, nnkTupleConstr, nnkBracket} and c.len == 1:
    c = c[0]
  if c.kind in {nnkPar, nnkTupleConstr, nnkBracket}:
    var pos = 0
    for f in c:
      if f.kind in {nnkExprColonExpr, nnkExprEqExpr}:
        case $f[0]
        of "grid": gridN = f[1]
        of "blk": blkN = f[1]
        of "sharedMem": sharedMemN = f[1]
        of "lang": lang = parseLangName(f[1])
        else:
          error("launchDeviceFn: chevron schedule expects grid / blk / sharedMem / lang, got " & f.repr)
      else:
        case pos
        of 0: gridN = f
        of 1: blkN = f
        of 2: sharedMemN = f
        else:
          error("launchDeviceFn: chevron schedule takes grid, blk, sharedMem, got " & f.repr)
        pos.inc
  else:
    error("launchDeviceFn: chevron schedule expects (grid, blk), got " & cfg.repr)
  let E = r.getTypeInst()[1]
  let langLit = newLit(lang)
  let gridD = bindSym"dim3".newCall(gridN)
  let blkD = bindSym"dim3".newCall(blkN)
  result = quote do:
    DeviceFnProxy[`E`, `langLit`](engine: `r`.engine, grid: `gridD`, blk: `blkD`, sharedMem: `sharedMemN`)

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
  var lang = "spirv"
  if not schedule.isNil():
    for a in schedule:
      doAssert a.kind == nnkAsgn and $a[0] in ["grid", "blk", "sharedMem", "lang"],
        "launchDeviceFn: schedule expects grid = ... / blk = ... / sharedMem = ... / lang = ..., got " & a.repr
      case $a[0]
      of "grid": gridN = a[1]
      of "blk": blkN = a[1]
      of "sharedMem": sharedMemN = a[1]
      else: lang = parseLangName(a[1])
  let langLit = newLit(lang)
  result = bindSym"resolveDeviceCall".newCall(engine, callNode[0], gridN, blkN, sharedMemN, langLit)
  for i in 1 ..< callNode.len:
    result.add(callNode[i])

macro `>>`*(proxy: DeviceFnProxy, call: untyped): untyped =
  var callNode = call
  if callNode.kind == nnkPar and callNode.len == 1:
    callNode = callNode[0]
  doAssert callNode.kind == nnkCall, "launchDeviceFn: expected fn(args...), got " & callNode.repr
  # lang is a static string on the proxy type, baked in by the `<<` chevron
  let t = proxy.getTypeInst()
  var lang = "spirv"
  if t.len > 2:
    lang = t[2].strVal
  let langLit = newLit(lang)
  result = bindSym"resolveDeviceCall".newCall(
    proxy.newDotExpr(ident"engine"), callNode[0],
    proxy.newDotExpr(ident"grid"),
    proxy.newDotExpr(ident"blk"),
    proxy.newDotExpr(ident"sharedMem"),
    langLit)
  for i in 1 ..< callNode.len:
    result.add(callNode[i])
