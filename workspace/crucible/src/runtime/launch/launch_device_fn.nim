# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Wrap-and-run: launch a device function through the chevron.

import std/[macros, hashes, strutils, tables]
import ../../codegen/gpu_compiler
import ../engines_common
import launch_kernel {.all.}
import launch_common {.all.}

export launch_kernel

proc substituteIdents(n: NimNode, bindings: Table[string, NimNode]): NimNode =
  # Replaces identifier leaves matched in `bindings`, everywhere except dots' right sides
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
  ## True when a formal's declared type is `static T`: such formals bake the call-site expression into the bridge body
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
  ## True when the type carries no runtime fields, its value lives in the type:
  ## empty phantom objects like `Phantom[8]` or ceramic's `Int[V]`.
  ## Bake decision predicate for formals. Instantiated generics resolve through `getImpl`.
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
      # a type alias whose body is the generic instantiation or another alias
      result = typeIsPhantom(body)
    else: result = false
  else: result = false

proc buildGlobalFn(kname: string, kernel, bridgeCall: NimNode, valueTypes, formalTypes: seq[NimNode], valueNames: seq[string]): NimNode =
  # Builds the `{.global.}` bridge proc, the call's value formals verbatim:
  #   proc <kname>(<value formals>) {.global.} = <bridgeCall>
  # `bridgeCall` carries the kernel call in the kernel's formal order, baked args inline, env args by name.
  # The codegen lowers the formals, the binder binds the same ones.
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
  ## Backend dispatch for the bridge source: emits the when/elif chain over
  ## the engine backend kind, calling the backend's lowering macro.
  ## `node` is untyped, not NimNode: a typed param would sem the bridge AST before it reaches the backend macros.
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
  ## Bridge construction and launch behind `launchDeviceFn`:
  ## - one walk over the kernel formals, the launch values matched positionally
  ## - `static T` formals and no-runtime-field types bake the call-site expression
  ##   into the bridge body, every other formal is an env member, literal included
  ## - the bridge lowers into a `{.global.}` proc, one per (kernel, baked args, env value types)
  ## - the first launch value is the kernel's output, the engine ingests the artifact
  let impl = kernel.getImpl()
  doAssert impl.kind in {nnkProcDef, nnkFuncDef}, "launchDeviceFn " & $kernel & ": the kernel must be a proc or func"

  var
    staticRepr = ""
    valueNames: seq[string] = @[]
    valueTypes: seq[NimNode] = @[]
    formalTypes: seq[NimNode] = @[]
    bindings = initTable[string, NimNode]()
    bridgeCall = nnkCall.newTree(kernel)
    argsTuple = nnkTupleConstr.newTree()
    outputNode: NimNode = nil
  var si = 0
  for i in 1 ..< impl[3].len:
    let f = impl[3][i]
    let ftype = f[^2]
    let bakes = isStaticFormalType(ftype)
    for j in 0 ..< f.len - 2:
      doAssert si < values.len, "launchDeviceFn " & $kernel & ": " & $si & " formals but only " & $values.len & " launch values"
      if bakes or typeIsPhantom(values[si].getTypeInst()):
        # baked: the call-site expression lands in the bridge body, no bridge param
        staticRepr.add(repr(values[si]))
        bridgeCall.add(values[si])
      else:
        # env member: bridge param + launch value, literal included
        let name = $f[j]
        valueNames.add(name)
        valueTypes.add(values[si].getTypeInst())
        formalTypes.add(ftype)
        bindings[name] = values[si]
        if outputNode.isNil:
          outputNode = values[si]
        else:
          argsTuple.add(values[si])
        bridgeCall.add(ident(name))
      inc si
  doAssert values.len == si, "launchDeviceFn " & $kernel & ": " & $values.len & " args for " & $si & " formals"
  doAssert not outputNode.isNil, "launchDeviceFn " & $kernel & ": the kernel needs a launch value formal as output"

  var typeRepr = ""
  for t in valueTypes: typeRepr.add(repr(t))
  # the bridge bakes the value types in, the name carries them
  # negative hashes contain a `-` which MSL rejects in identifiers
  let kname = ($kernel & "_" & $hash(staticRepr & "|" & typeRepr)).replace("-", "n")
  let bridgeNode = nnkStmtList.newTree(buildGlobalFn(kname, kernel, bridgeCall, valueTypes, formalTypes, valueNames))
  let grid = substituteIdents(gridN.copy(), bindings)
  let blk = substituteIdents(blkN.copy(), bindings)
  let gridD = quote do: dim3(`grid`)
  let blkD = quote do: dim3(`blk`)

  let bridgeSource = ident("crucibleBridgeSource" & kname)
  let bridgeN = bindSym"bridge"

  result = quote do:
    block:
      const `bridgeSource` = `bridgeN`(`engine`, `bridgeNode`)
      if `engine`.getArtifact() != `bridgeSource`:
        `engine`.ingest(`bridgeSource`)
      `engine`.launchKernelImpl(`kname`, `outputNode`, `argsTuple`,
                   LaunchConfig(grid: `gridD`, blk: `blkD`))

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
  DeviceFnSugar*[E: HwEngine] = object
    ## Transient chevron sugar for launchDeviceFn launches, created per `engine.launchDeviceFn` access, no field on the engine
    engine: E

macro launchDeviceFn*[E: HwEngine](engine: E): DeviceFnSugar[E] =
  ## Chevron form accessor for device-fn launches.
  ## - Chevrons:
  ##   `engine.launchDeviceFn<<(grid, blk)>> fn(args)`
  ## - Do form:
  ##     `engine.launchDeviceFn <kernel-expr> do:`
  ##       `grid = (ceil_div(m, 32), ceil_div(n, 32))`
  ##       `blk = (32, 1)`
  result = quote do: DeviceFnSugar[E](`engine`)

type
  DeviceFnProxy*[E: HwEngine] = object
    ## Chevron intermediate, built by `<<`, consumed by `>>`. Same shape as `LaunchConfig[E]`,
    ## a distinct type so `>>` can tell the device-fn and kernel launch paths apart.
    engine: E
    grid: Dim3
    blk: Dim3

proc `<<`*[E: HwEngine, G, B: int|tuple](r: DeviceFnSugar[E], cfg: tuple[grid: G, blk: B]): DeviceFnProxy[E] =
  ## Builds the device-fn proxy from the chevron left half.
  DeviceFnProxy[E](engine: r.engine, grid: dim3(cfg.grid), blk: dim3(cfg.blk))

macro `>>`*[E: HwEngine](proxy: DeviceFnProxy[E], call: untyped): untyped =
  ## Builds the device-fn launch from the chevron proxy.
  ## - Chevrons:
  ##   `engine.launchDeviceFn<<(grid, blk)>> fn(args)`
  ## - Do form:
  ##     `engine.launchDeviceFn <kernel-expr> do:`
  ##       `grid = (ceil_div(m, 32), ceil_div(n, 32))`
  ##       `blk = (32, 1)`
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
