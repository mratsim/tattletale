# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Wrap-and-run: launch a device function through the chevron.

import std/[macros, macrocache, algorithm, hashes, strutils, tables]
import ../codegen/gpu_compiler
import chevrons {.all.}
import engines

export chevrons

const BridgeDefs = CacheTable"crucible.wrap_and_run.bridges"
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

proc isStaticFormal(f: NimNode): bool =
  let ptype = f[^2]
  ptype.kind == nnkStaticTy or
    (ptype.kind == nnkCommand and $ptype[0] == "static")

proc isConstantTy(t: NimNode): bool =
  ## Whether the type node has no scalar leaf: only tuples and generic
  ## instantiations of fieldless objects, a compile-time constant
  ## crossing the launch boundary as no bytes
  case t.kind
  of nnkIntLit..nnkFloat128Lit:
    # a static parameter carries its value in the type
    result = true
  of nnkBracketExpr:
    # a generic instantiation, constant iff every argument is
    result = true
    for e in t[1 ..< t.len]:
      if not isConstantTy(e):
        return false
  of nnkTupleTy:
    result = true
    for d in t:
      if d.kind == nnkIdentDefs and not isConstantTy(d[^2]):
        return false
  of nnkPar, nnkTupleConstr:
    result = true
    for e in t:
      if not isConstantTy(e):
        return false
  else:
    result = false

proc constantValue(t: NimNode): NimNode =
  ## Builds the compile-time value of a constant type node: nested
  ## fieldless objects and tuples constructed from their argument types
  if t.kind == nnkBracketExpr:
    return nnkCall.newTree(t.copy())
  if t.kind in {nnkTupleTy, nnkPar, nnkTupleConstr}:
    var r = nnkTupleConstr.newTree()
    if t.kind == nnkTupleTy:
      for d in t:
        if d.kind == nnkIdentDefs:
          r.add(constantValue(d[^2]))
    else:
      for e in t:
        r.add(constantValue(e))
    return r
  error "constantValue: not a constant type node: " & t.repr

proc isBindMany(t: NimNode): bool =
  if t.kind == nnkDistinctTy: return true
  if t.kind == nnkSym and t.getTypeInst().kind == nnkDistinctTy: return true
  return false

proc buildGlobalFn(kname: string, kernel, statics: NimNode,
                   valueTypes, valueImpls, formalTypes: seq[NimNode],
                   valueNames: seq[string]): NimNode =
  # Builds the `{.global.}` bridge proc:
  #
  #   proc <kname>(<leaf params>) {.global.} =
  #     <kernel>(<static args>, <rebuilt formals>)
  var params = nnkFormalParams.newTree(newEmptyNode())
  var bridgeArgs: seq[NimNode] = @[]
  for i, vtype in valueTypes:
    let useFormal = i < formalTypes.len and not isBindMany(formalTypes[i])
    let argType = if useFormal: formalTypes[i] else: vtype
    let obj = if useFormal: formalTypes[i].getTypeImpl() else: valueImpls[i]
    if obj.kind == nnkObjectTy:
      var reconFields: seq[NimNode] = @[]
      for k in 2 ..< obj.len:
        let recList = obj[k]
        doAssert recList.kind in {nnkRecList, nnkIdentDefs},
          "buildGlobalFn: unsupported object field node " & recList.repr
        let defs = if recList.kind == nnkRecList: recList else:
            nnkRecList.newTree(recList)
        for d in defs:
          doAssert d.kind == nnkIdentDefs,
            "buildGlobalFn: unsupported field " & d.repr
          let fldType = d[^2]
          doAssert fldType.kind != nnkObjectTy,
            "buildGlobalFn: nested pointer structs are not supported: " & vtype.repr
          for m in 0 ..< d.len - 2:
            if isConstantTy(fldType):
              # a constant field crosses the launch boundary as no bytes,
              # the kernel reads its value from the type
              reconFields.add(nnkExprColonExpr.newTree(d[m].copy(),
                constantValue(fldType)))
            else:
              let pname = ident("cru" & valueNames[i] & "_" & $d[m])
              params.add(nnkIdentDefs.newTree(pname, fldType.copy(), newEmptyNode()))
              reconFields.add(nnkExprColonExpr.newTree(d[m].copy(), pname))
      var recon = nnkObjConstr.newTree(argType.copy())
      for rf in reconFields:
        recon.add(rf)
      bridgeArgs.add(recon)
    else:
      let pname = ident("cru" & valueNames[i])
      params.add(nnkIdentDefs.newTree(pname, argType.copy(), newEmptyNode()))
      bridgeArgs.add(pname)
  var bridgeCall = nnkCall.newTree(kernel)
  for s in statics:
    bridgeCall.add(s)
  for a in bridgeArgs:
    bridgeCall.add(a)
  let body = nnkStmtList.newTree(bridgeCall)

  result = nnkProcDef.newTree(
    ident(kname),
    newEmptyNode(), newEmptyNode(),
    params,
    nnkPragma.newTree(ident"global"),
    newEmptyNode(),
    body)

macro crucibleBuild(engine: HwEngine, kernel: typed, statics, formals, formalTys: untyped,
                    gridN, blkN: untyped, values: varargs[typed]): untyped =
  # Bridge construction and launch, behind `wrap_and_run`:
  #
  # - the kernel lowers into a `{.global.}` bridge on first launch,
  #   one bridge per launch shape
  # - the engine ingests the bridge's source when its artifact differs
  # - the launch values pass whole, the first one is the kernel's output
  # - static args bake into the bridge, they never cross the launch boundary,
  #   the backend context lives inside the bridge only
  var valueNames: seq[string] = @[]
  var valueTypes: seq[NimNode] = @[]
  var valueImpls: seq[NimNode] = @[]
  var formalTypeNodes: seq[NimNode] = @[]
  for f in formals:
    valueNames.add($f)
  for v in values:
    valueTypes.add(v.getTypeInst())
    valueImpls.add(v.getTypeImpl())
  for t in formalTys:
    formalTypeNodes.add(t)

  var staticRepr = ""
  for s in statics:
    staticRepr.add(repr(s))
  # negative hashes contain a `-` which MSL rejects in identifiers
  let kname = (if staticRepr.len == 0: $kernel
               else: $kernel & "_" & $hash(staticRepr)).replace("-", "n")
  if not BridgeDefs.hasKey(kname):
    BridgeDefs[kname] = nnkStmtList.newTree(
      buildGlobalFn(kname, kernel, statics, valueTypes, valueImpls, formalTypeNodes, valueNames))

  var bindings = initTable[string, NimNode]()
  for i, name in valueNames:
    bindings[name] = values[i]
  let grid = substituteIdents(gridN.copy(), bindings)
  let blk = substituteIdents(blkN.copy(), bindings)

  var argsTuple = nnkTupleConstr.newTree()
  for i in 1 ..< values.len:
    argsTuple.add(values[i])
  let outputNode = values[0]

  let bridgeSource = ident("crucibleBridgeSource" & kname)
  let metalCg = bindSym"metal"
  let cudaCg = bindSym"cuda"
  let openclCg = bindSym"opencl"
  let vulkanCg = bindSym"vulkan"
  let webgpuCg = bindSym"webgpu"
  let bridgeNode = BridgeDefs[kname].copy()

  result = quote do:
    when not declared(`bridgeSource`):
      const `bridgeSource` = when typeof(`engine`).backendKind == bkMetal: `metalCg`(`bridgeNode`)
                           elif typeof(`engine`).backendKind == bkCuda: `cudaCg`(`bridgeNode`)
                           elif typeof(`engine`).backendKind == bkOpenCL: `openclCg`(`bridgeNode`)
                           elif typeof(`engine`).backendKind == bkVulkan: `vulkanCg`(`bridgeNode`)
                           elif typeof(`engine`).backendKind == bkWGSL: `webgpuCg`(`bridgeNode`)
                           else: {.error: "wrap_and_run: no lowering for this engine".}
    block:
      if `engine`.getArtifact() != `bridgeSource`:
        `engine`.ingest(`bridgeSource`)
      `engine`.run(`kname`, `outputNode`, `argsTuple`,
                   LaunchConfig(grid: dim3(`grid`), blk: dim3(`blk`)))

macro crucibleSplit(engine: HwEngine, kernel: typed, call, gridN, blkN: untyped): untyped =
  # Static/value arg split, behind `wrap_and_run`:
  #
  # - the kernel symbol resolves alone, its impl gives the formal list and which formals are static, the launch args stay raw
  # - static args pass to the bridge builder raw, the backend context lives inside the bridge only
  # - value args pass positionally
  doAssert call.kind == nnkCall,
    "wrap_and_run: expected fn(args...), got " & call.repr
  let impl = kernel.getImpl()
  doAssert impl.kind in {nnkProcDef, nnkFuncDef},
    "wrap_and_run " & $kernel & ": the kernel must be a proc or func"

  var staticArgs: seq[NimNode] = @[]
  var valueArgs: seq[NimNode] = @[]
  var formalNames: seq[NimNode] = @[]
  var formalTypes: seq[NimNode] = @[]
  var si = 0
  for i in 1 ..< impl[3].len:
    let f = impl[3][i]
    let isStatic = isStaticFormal(f)
    for j in 0 ..< f.len - 2:
      doAssert call.len > 1 + si,
        "wrap_and_run " & $kernel & ": " & $(call.len - 1) & " args for " &
        $si & " formals so far"
      if isStatic:
        staticArgs.add(call[1 + si])
      else:
        valueArgs.add(call[1 + si])
        formalNames.add(f[j])
        formalTypes.add(f[^2])
      inc si
  doAssert call.len - 1 == si,
    "wrap_and_run " & $kernel & ": " & $(call.len - 1) & " args for " &
    $si & " formals"

  let statics = nnkBracket.newTree(staticArgs)
  let formals = nnkBracket.newTree(formalNames)
  let formalTys = nnkBracket.newTree(formalTypes)
  let buildN = bindSym"crucibleBuild"
  result = nnkCall.newTree(buildN, engine, kernel, statics, formals, formalTys, gridN, blkN)
  for a in valueArgs:
    result.add(a)

macro wrap_and_run*(engine: HwEngine, call: untyped): untyped =
  ## Launches a device function with whole values, schedule via a `do:` block:
  ##
  ##   engine.wrap_and_run gemm(tiledMma(f32, f16, f16, 32, 32, 16), dD, 2.0'f32, dA, dB, 1.0'f32, dC) do:
  ##     grid = ((D.shape.m + 31) div 32, (D.shape.n + 31) div 32)
  ##     blk = (32, 1)
  ##
  ## The schedule block:
  ##
  ## - binds to the kernel call, it holds `grid = <expr>` and `blk = <expr>` assignments, both optional, default (1, 1)
  ## - the expressions evaluate at the launch site and may reference the function's formals, the launch args substitute by name
  ## - anything but `grid =` / `blk =` assignments is rejected
  ##
  ## The chevron form `engine.wrap_and_run<<grid, blk>> fn(args...)`
  ## carries the same schedule positionally, host expressions only.
  var schedule: NimNode = nil
  var callNode = call
  if callNode[^1].kind == nnkStmtList:
    schedule = callNode[^1]
    callNode = nnkCall.newTree(call[0 ..< call.len - 1])
  doAssert schedule.kind in {nnkNilLit, nnkStmtList},
    "wrap_and_run: the schedule must be a `do:` block, got " & schedule.repr
  var gridN = nnkPar.newTree(newLit(1), newLit(1))
  var blkN = nnkPar.newTree(newLit(1), newLit(1))
  if schedule.kind == nnkStmtList:
    for a in schedule:
      doAssert a.kind == nnkAsgn and $a[0] in ["grid", "blk"],
        "wrap_and_run: schedule expects grid = ... / blk = ..., got " & a.repr
      if $a[0] == "grid": gridN = a[1]
      else: blkN = a[1]
  let splitN = bindSym"crucibleSplit"
  result = nnkCall.newTree(splitN, engine, callNode[0], callNode, gridN, blkN)

type
  WrapSugar[E: HwEngine] = object
    ## Transient chevron sugar for wrap_and_run launches, created per `engine.wrap_and_run` access, no field on the engine
    engine: E

  WrapProxy[E: HwEngine] = object
    ## Chevron intermediate:
    ## built by `<<`, consumed by `>>`.
    engine: E
    grid: Dim3
    blk: Dim3

proc makeWrapProxy[E: HwEngine](engine: E, grid, blk: Dim3): WrapProxy[E] =
  ## WrapProxy construction, generic param inference lives in this helper
  WrapProxy[E](engine: engine, grid: grid, blk: blk)

macro wrap_and_run*(engine: HwEngine): untyped =
  ## Chevron form accessor:
  ##
  ##   engine.wrap_and_run<<(gridX, gridY), (32, 1)>> gemm(dD, dC, dA, dB)
  result = quote do:
    WrapSugar[typeof(`engine`)](engine: `engine`)

macro `<<`*[E: HwEngine](r: WrapSugar[E], cfg: untyped): untyped =
  ## Chevron launch config, positional (grid, blk):
  ##
  ##   engine.wrap_and_run<<((1, 1), (32, 1))>>(gemm(dD, dC, dA, dB))
  # the chevron form takes host expressions only, the `do:` form alone has
  # a schedule that may reference the function's formals
  doAssert cfg.kind in {nnkPar, nnkTupleConstr} and cfg.len == 2,
    "wrap_and_run chevron takes (grid, blk), got: " & cfg.repr
  let dim3Sym = bindSym"dim3"
  let mkSym = bindSym"makeWrapProxy"
  result = quote do:
    `mkSym`(`r`.engine, `dim3Sym`(`cfg`[0]), `dim3Sym`(`cfg`[1]))

macro `>>`*[E: HwEngine](proxy: WrapProxy[E], call: untyped): untyped =
  ## Builds the launch: `engine.wrap_and_run<<grid, blk>> fn(args...)`
  var callNode = call
  if callNode.kind == nnkPar and callNode.len == 1:
    callNode = callNode[0]
  doAssert callNode.kind == nnkCall,
    "wrap_and_run: expected fn(args...), got " & callNode.repr
  let splitN = bindSym"crucibleSplit"
  result = nnkCall.newTree(splitN,
    nnkDotExpr.newTree(proxy, ident"engine"), callNode[0], callNode,
    nnkDotExpr.newTree(proxy, ident"grid"),
    nnkDotExpr.newTree(proxy, ident"blk"))
