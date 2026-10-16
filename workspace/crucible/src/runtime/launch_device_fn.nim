# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Wrap-and-run: launch a device function through the chevron.

import std/[macros, macrocache, hashes, strutils, tables]
import ../codegen/gpu_compiler
import launch_kernel {.all.}
import launch_common {.all.}
import engines

export launch_kernel

const
  BridgeDefs = CacheTable"crucible.launchDeviceFn.bridges"
    # `{.global.}` bridge proc per kernel launch shape, built on first launch

proc substituteIdents(n: NimNode, bindings: Table[string, NimNode]): NimNode =
  # Replaces identifier leaves matched in `bindings`, recursing everywhere
  # except dots' right sides (`a.b` substitutes `a` only)
  case n.kind
  of nnkIdent, nnkSym:
    let key = $n
    (if bindings.hasKey(key): bindings[key] else: n)
  of nnkDotExpr:
    nnkDotExpr.newTree(substituteIdents(n[0], bindings), n[1])
  of nnkEmpty, nnkCharLit..nnkFloat128Lit:
    n
  else:
    var r = copyNimNode(n)
    for child in n:
      r.add(substituteIdents(child, bindings))
    r

proc isStaticFormalType(t: NimNode): bool =
  ## True when a formal's declared type is `static T`,
  ## such formals bake the call-site expression into the bridge body
  t.kind == nnkStaticTy or
    (t.kind == nnkCommand and $t[0] == "static")

proc isBindManyArg(t: NimNode): bool =
  ## True when the formal type is a `distinct` typeclass, such formals bind by value type
  if t.kind == nnkDistinctTy: return true
  if t.kind == nnkSym and t.getTypeInst().kind == nnkDistinctTy: return true
  return false

proc typeIsPhantom(t: NimNode): bool

proc fieldsArePhantom(fields: NimNode): bool =
  ## True when every field of an object recList or tuple body is phantom
  for field in fields:
    # an empty body arrives as a nil placeholder, no fields, phantom
    if field.kind in {nnkEmpty, nnkNilLit}: continue
    # a recCase carries a runtime discriminant, never phantom
    if field.kind != nnkIdentDefs: return false
    for i in 0 ..< field.len - 2:
      if not typeIsPhantom(field[i]): return false
  result = true

proc typeIsPhantom(t: NimNode): bool =
  ## True when the type carries no runtime fields, its value lives in the type
  ## (empty phantom objects like `Phantom[8]` or ceramic's `Int[V]`).
  ## - decision predicate for baking a formal into the bridge body
  ## - launch_common.isStaticArg applies the same predicate to the binder's byte layout
  ## - divergence: zero-length arrays are static for isStaticArg, never phantom here
  ## - instantiated generics arrive as a bracket expression over the generic symbol,
  ##   or as the instantiation's own symbol, both resolve through `getImpl`
  case t.kind
  of nnkBracketExpr:
    result = typeIsPhantom(t[0])
  of nnkSym:
    let impl = t.getImpl()
    if impl.isNil or impl.kind != nnkTypeDef: return false
    let body = impl[2]
    case body.kind
    of nnkObjectTy: result = fieldsArePhantom(body[^1])
    of nnkTupleTy:  result = fieldsArePhantom(body)
    of nnkBracketExpr, nnkSym:
      # a type alias whose body is the generic instantiation or another alias:
      # `type P8 = Phantom[8]`, `type Q = P8`
      result = typeIsPhantom(body)
    else: result = false
  else: result = false

proc buildGlobalFn(kname: string, kernel, bridgeCall: NimNode, valueTypes, formalTypes: seq[NimNode], valueNames: seq[string]): NimNode =
  # Builds the `{.global.}` bridge proc, the call's value formals verbatim:
  #   proc <kname>(<value formals>) {.global.} = <bridgeCall>
  # The codegen lowers the formals, the binder binds the same values.
  # `bridgeCall` carries the kernel call in the kernel's formal order,
  # baked args inline, env args by name.
  var params = nnkFormalParams.newTree(newEmptyNode())
  for i, vtype in valueTypes:
    let argType = if i < formalTypes.len and not isBindManyArg(formalTypes[i]):
                    formalTypes[i]
                  else: vtype
    params.add(nnkIdentDefs.newTree(ident(valueNames[i]), argType, newEmptyNode()))
  let body = nnkStmtList.newTree(bridgeCall)

  result = nnkProcDef.newTree(
    ident(kname),
    newEmptyNode(), newEmptyNode(),
    params,
    nnkPragma.newTree(ident"global"),
    newEmptyNode(),
    body)

macro bridge(engine: HwEngine, node: untyped): untyped =
  ## Backend dispatch for the bridge source: emits the when/elif chain,
  ## branched on the engine's `backendKind`, calling the backend's lowering macro.
  ##
  ## `node` is untyped, not NimNode: the backend macros consume the raw bridge
  ## AST, a typed param would sem it before it reaches them.
  let metalCg = bindSym"metal"
  let cudaCg = bindSym"cuda"
  let openclCg = bindSym"opencl"
  let vulkanCg = bindSym"vulkan"
  let webgpuCg = bindSym"webgpu"
  result = quote do:
    when typeof(`engine`).backendKind == bkMetal: `metalCg`(`node`)
    elif typeof(`engine`).backendKind == bkCuda: `cudaCg`(`node`)
    elif typeof(`engine`).backendKind == bkOpenCL: `openclCg`(`node`)
    elif typeof(`engine`).backendKind == bkVulkan: `vulkanCg`(`node`)
    elif typeof(`engine`).backendKind == bkWGSL: `webgpuCg`(`node`)
    else: {.error: "launchDeviceFn: no lowering for this engine".}

macro crucibleBuild(engine: HwEngine, kernel: typed, gridN, blkN: untyped, values: varargs[typed]): untyped =
  ## Bridge construction and launch, behind `launchDeviceFn`:
  ##
  ## - one walk over the kernel's formals, the launch values matched positionally
  ## - `static T` formals and phantom-typed formals bake the call-site expression
  ##   into the bridge body, every other formal is an env member, literal included
  ## - the bridge lowers into a `{.global.}` proc on first launch, one bridge per
  ##   (kernel, baked args, env value types)
  ## - the launch values pass whole, the first one is the kernel's output,
  ##   the engine ingests the artifact when it differs
  let impl = kernel.getImpl()
  doAssert impl.kind in {nnkProcDef, nnkFuncDef}, "launchDeviceFn " & $kernel & ": the kernel must be a proc or func"

  var
    statics: seq[NimNode] = @[]
    valueNames: seq[string] = @[]
    valueTypes: seq[NimNode] = @[]
    formalTypes: seq[NimNode] = @[]
    envIdx: seq[int] = @[]
    slots: seq[NimNode] = @[]
  var si = 0
  for i in 1 ..< impl[3].len:
    let f = impl[3][i]
    let ftype = f[^2]
    let bakes = isStaticFormalType(ftype)
    for j in 0 ..< f.len - 2:
      doAssert si < values.len, "launchDeviceFn " & $kernel & ": " & $si & " formals but only " & $values.len & " launch values"
      if bakes or typeIsPhantom(values[si].getTypeInst()):
        statics.add(values[si])
        slots.add(values[si])
      else:
        valueNames.add($f[j])
        valueTypes.add(values[si].getTypeInst())
        formalTypes.add(ftype)
        envIdx.add(si)
        slots.add(ident($f[j]))
      inc si
  doAssert values.len == si, "launchDeviceFn " & $kernel & ": " & $values.len & " args for " & $si & " formals"
  doAssert envIdx.len > 0, "launchDeviceFn " & $kernel & ": the kernel needs an output formal"

  var staticRepr = ""
  for s in statics: staticRepr.add(repr(s))
  var typeRepr = ""
  for t in valueTypes: typeRepr.add(repr(t))
  # the bridge bakes the value types in, the cache key carries them
  # negative hashes contain a `-` which MSL rejects in identifiers
  let kname = ($kernel & "_" & $hash(staticRepr & "|" & typeRepr)).replace("-", "n")
  if not BridgeDefs.hasKey(kname):
    var bridgeCall = nnkCall.newTree(kernel)
    for s in slots: bridgeCall.add(s)
    let built = buildGlobalFn(kname, kernel, bridgeCall, valueTypes, formalTypes, valueNames)
    BridgeDefs[kname] = nnkStmtList.newTree(built)

  var bindings = initTable[string, NimNode]()
  for i, idx in envIdx: bindings[valueNames[i]] = values[idx]
  let grid = substituteIdents(gridN.copy(), bindings)
  let blk = substituteIdents(blkN.copy(), bindings)

  var argsTuple = nnkTupleConstr.newTree()
  for i in 1 ..< envIdx.len: argsTuple.add(values[envIdx[i]])
  let outputNode = values[envIdx[0]]

  let bridgeSource = ident("crucibleBridgeSource" & kname)
  let bridgeN = bindSym"bridge"
  let bridgeNode = BridgeDefs[kname].copy()

  result = quote do:
    when not declared(`bridgeSource`):
      const `bridgeSource` = `bridgeN`(`engine`, `bridgeNode`)
    block:
      if `engine`.getArtifact() != `bridgeSource`:
        `engine`.ingest(`bridgeSource`)
      `engine`.launchKernel(`kname`, `outputNode`, `argsTuple`,
                   LaunchConfig(grid: dim3(`grid`), blk: dim3(`blk`)))

macro launchDeviceFn*(engine: HwEngine, call: untyped): untyped =
  ## Launches a device function with whole values, schedule via a `do:` block
  ## or the chevron form:
  ##
  ##   engine.launchDeviceFn gemm(tiledMma(f32, f16, f16, 32, 32, 16), dD, dA, dB, 1) do:
  ##     grid = ((D.shape.m + 31) div 32, (D.shape.n + 31) div 32)
  ##     blk = (32, 1)
  ##
  ##   engine.launchDeviceFn<<grid, blk>> gemm(dD, dA, dB, 1)
  ##
  ## - the schedule block holds `grid =` / `blk =` assignments, both optional, default (1, 1)
  ## - the expressions evaluate at the launch site and may reference the kernel's formals
  var schedule: NimNode = nil
  var callNode = call
  if callNode[^1].kind == nnkStmtList:
    schedule = callNode[^1]
    callNode = nnkCall.newTree(call[0 ..< call.len - 1])
  doAssert schedule.kind in {nnkNilLit, nnkStmtList}, "launchDeviceFn: the schedule must be a `do:` block, got " & schedule.repr
  var gridN = nnkPar.newTree(newLit(1), newLit(1))
  var blkN = nnkPar.newTree(newLit(1), newLit(1))
  if schedule.kind == nnkStmtList:
    for a in schedule:
      doAssert a.kind == nnkAsgn and $a[0] in ["grid", "blk"], "launchDeviceFn: schedule expects grid = ... / blk = ..., got " & a.repr
      if $a[0] == "grid": gridN = a[1]
      else: blkN = a[1]
  let buildN = bindSym"crucibleBuild"
  result = nnkCall.newTree(buildN, engine, callNode[0], gridN, blkN)
  for i in 1 ..< callNode.len:
    result.add(callNode[i])

type
  DeviceFnSugar[E: HwEngine] = object
    ## Transient chevron sugar for launchDeviceFn launches, created per `engine.launchDeviceFn` access, no field on the engine
    engine: E

  DeviceFnProxy[E: HwEngine] = object
    ## Chevron intermediate:
    ## built by `<<`, consumed by `>>`.
    engine: E
    grid: Dim3
    blk: Dim3

proc makeDeviceFnProxy[E: HwEngine](engine: E, cfg: LaunchConfig): DeviceFnProxy[E] {.inline.} =
  ## DeviceFnProxy construction, generic param inference lives in this helper
  DeviceFnProxy[E](engine: engine, grid: cfg.grid, blk: cfg.blk)

proc launchDeviceFn*[E: HwEngine](engine: E): DeviceFnSugar[E] {.inline.} =
  ## Chevron form accessor: `engine.launchDeviceFn<<grid, blk>> fn(args)`.
  ## A proc resolves by signature, the 2-arg `do:` macro form is untouched.
  DeviceFnSugar[E](engine: engine)

macro `<<`*[E: HwEngine](r: DeviceFnSugar[E], cfg: untyped): untyped =
  ## Chevron launch geometry, positional `(grid, blk)` or `grid:` / `blk:` pairs:
  ##
  ##   engine.launchDeviceFn<<(grid, blk)>> fn(args)
  ##
  ## sharedMem and stream do not exist on the device-fn path, they are compile errors
  doAssert cfg.kind in {nnkPar, nnkTupleConstr}, "launchDeviceFn chevron takes (grid, blk), got: " & cfg.repr
  var named = 0
  var positional = 0
  for ch in cfg:
    if ch.kind == nnkExprColonExpr:
      doAssert $ch[0] in ["grid", "blk"], "launchDeviceFn chevron takes grid/blk only, '" & $ch[0] & "' does not exist on the device-fn path: " & ch.repr
      inc named
    else: inc positional
  doAssert (named == 0 and positional == 2) or positional == 0, "launchDeviceFn chevron takes (grid, blk) or grid:/blk: pairs, got: " & cfg.repr
  let mkSym = bindSym"makeDeviceFnProxy"
  let cfgSym = bindSym"chevronConfig"
  result = quote do:
    `mkSym`(`r`.engine, `cfgSym`(`cfg`))

macro `>>`*[E: HwEngine](proxy: DeviceFnProxy[E], call: untyped): untyped =
  ## Builds the launch: `engine.launchDeviceFn<<grid, blk>> fn(args...)`
  var callNode = call
  if callNode.kind == nnkPar and callNode.len == 1:
    callNode = callNode[0]
  doAssert callNode.kind == nnkCall, "launchDeviceFn: expected fn(args...), got " & callNode.repr
  let buildN = bindSym"crucibleBuild"
  result = nnkCall.newTree(buildN,
    nnkDotExpr.newTree(proxy, ident"engine"), callNode[0],
    nnkDotExpr.newTree(proxy, ident"grid"),
    nnkDotExpr.newTree(proxy, ident"blk"))
  for i in 1 ..< callNode.len:
    result.add(callNode[i])
