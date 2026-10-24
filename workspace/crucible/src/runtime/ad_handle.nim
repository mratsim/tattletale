# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Runtime half of kernel-level autodiff.
## `engine.setupAD(pair)` returns an AD handle whose macros launch the pair's printed kernels.
##
## - `forward(output, primals...)`: the primal signature, output slot first
## - `backward(...)`: the pair's backward signature, primitive and convenience overloads by arity
##
## | Backward overload | Arity                                         | Behavior                                                          |
## | ----------------- | --------------------------------------------- | ----------------------------------------------------------------- |
## | primitive         | incoming cotangents + primals + all gradients | caller-owned buffers, the backward kernels accumulate into them   |
## | convenience       | incoming cotangents + primals                 | allocates and zeroes the gradients, returns them as a named tuple |
##
## Accumulation contracts:
##
## - the backward kernels accumulate (`d_a[i] += ...`), primitive callers pass pre-zeroed buffers for fresh-semantics gradients
## - the convenience's gradients start at zero
## - scalar cotangents run a per-call reduce-partials scratch, zeroed per call and never returned
## - each reduce kernel folds one scratch into the scalar's 1-element gradient buffer, no hidden host sync
##
## Launches are single-thread in v1: the scalar cotangents accumulate per-index partials without atomics, one thread keeps that race-free.

import std/macros

import workspace/crucible/src/codegen/kernel_pair
import workspace/crucible/src/runtime/engines/engines_api
import workspace/crucible/src/runtime/engines/metal_api
import workspace/crucible/src/runtime/engines/vulkan_api

# ═══════════════════════════════════════════════════════════════════════
# ▸ Types
# ═══════════════════════════════════════════════════════════════════════

type AdHandle*[E: HwEngine; kernels: static KernelPair] = object
  ## Runtime handle of one kernel pair.
  ##
  ## - `engine`: the engine that compiled the pair's kernels at `setupAD`
  ## - `kernels`: static parameter, its metadata drives the launch macros
  engine*: E

# ═══════════════════════════════════════════════════════════════════════
# ▸ Setup
# ═══════════════════════════════════════════════════════════════════════

proc setupAD*[E: HwEngine](engine: E, kernels: static KernelPair): AdHandle[E, kernels] =
  ## Compiles the pair's kernels through the engine's normal compile path
  ## and returns the AD handle. Each kernel ingests as one artifact:
  ##
  ## - the forward kernel
  ## - the backward kernel
  ## - every generated reduce kernel
  ##
  ## Engines without an autodiff pipeline refuse here at instantiation.
  when E is MetalEngine:
    engine.ingestADKernel(kernels.kernelName, kernels.fwdSource)
    engine.ingestADKernel(kernels.kernelName & "_bwd", kernels.bwdSource)
    for r in kernels.reduceSources:
      engine.ingestADKernel(r.name, r.source)
  elif E is VulkanEngine:
    engine.ingestADKernel(kernels.kernelName, kernels.fwdSource)
    engine.ingestADKernel(kernels.kernelName & "_bwd", kernels.bwdSource)
    for r in kernels.reduceSources:
      engine.ingestADKernel(r.name, r.source)
  else:
    {.error: "setupAD: this engine has no autodiff pipeline, this op gates Metal and Vulkan GLSL".}
  AdHandle[E, kernels](engine: engine)

# ═══════════════════════════════════════════════════════════════════════
# ▸ Pair metadata at macro expansion
# ═══════════════════════════════════════════════════════════════════════

func fieldNode(n: NimNode, name: string): NimNode =
  ## Value of one named field of an object-constructor node, nil when absent.
  for i in 1 ..< n.len:
    doAssert n[i].kind == nnkExprColonExpr,
      "ad_handle: unexpected pair-node shape, expected named fields"
    if n[i][0].eqIdent(name):
      return n[i][1]

func seqElems(n: NimNode): seq[NimNode] =
  ## Elements of a sequence-literal node: `@[...]` or `[...]` shapes.
  case n.kind
  of nnkBracket:
    for i in 0 ..< n.len:
      result.add n[i]
  of nnkCall, nnkCommand:
    doAssert n.len == 2 and n[0].eqIdent("@"),
      "ad_handle: unexpected pair-node shape, expected a sequence literal"
    for i in 0 ..< n[1].len:
      result.add n[1][i]
  else:
    doAssert false, "ad_handle: unexpected pair-node shape, expected a sequence literal"

func symOrOrd(n: NimNode): string =
  ## Name of one enum field of a pair node: the type-embedded constructor
  ## stores enum values as symbols or as their ordinal literals.
  case n.kind
  of nnkSym:
    result = $n
  of nnkIntLit:
    result = $n.intVal
  else:
    doAssert false, "ad_handle: unexpected enum node kind " & $n.kind

func parseBwdParamKind(n: NimNode): BwdParamKind =
  ## One `BwdParamKind` field of a pair node, see `symOrOrd`.
  result = BwdParamKind(n.intVal)

func parseBwdGradKind(n: NimNode): BwdGradKind =
  ## One `BwdGradKind` field of a pair node, see `symOrOrd`.
  result = BwdGradKind(n.intVal)

func parseBwdExtentSource(n: NimNode): BwdExtentSource =
  ## One `BwdExtentSource` field of a pair node, see `symOrOrd`.
  result = BwdExtentSource(n.intVal)

func readBwdParamMeta(n: NimNode): BwdParamMeta =
  result.name = fieldNode(n, "name").strVal
  result.kind = parseBwdParamKind(fieldNode(n, "kind"))
  result.gradName = fieldNode(n, "gradName").strVal
  result.gradKind = parseBwdGradKind(fieldNode(n, "gradKind"))
  result.extent = parseBwdExtentSource(fieldNode(n, "extent"))
  result.extentParam = fieldNode(n, "extentParam").strVal
  result.staticLen = fieldNode(n, "staticLen").intVal.int

func readScalarWorkspace(n: NimNode): ScalarWorkspace =
  result.gradName = fieldNode(n, "gradName").strVal
  result.workspaceName = fieldNode(n, "workspaceName").strVal
  result.tripCount = fieldNode(n, "tripCount").intVal.int
  result.tripCountParam = fieldNode(n, "tripCountParam").strVal
  result.tripCount64 = fieldNode(n, "tripCount64").intVal != 0

func readReduceSource(n: NimNode): ReduceSource =
  result.name = fieldNode(n, "name").strVal
  result.source = fieldNode(n, "source").strVal

func readPair(n: NimNode): KernelPair =
  ## Reconstructs the pair value of an `AdHandle` type at macro expansion:
  ## the handle type inlines the static pair as an object-constructor node.
  result.kernelName = fieldNode(n, "kernelName").strVal
  for e in seqElems(fieldNode(n, "reduceSources")):
    result.reduceSources.add readReduceSource(e)
  for e in seqElems(fieldNode(n, "params")):
    result.params.add readBwdParamMeta(e)
  for e in seqElems(fieldNode(n, "scalarWorkspaces")):
    result.scalarWorkspaces.add readScalarWorkspace(e)
  for e in seqElems(fieldNode(n, "bwdSignature")):
    result.bwdSignature.add e.strVal
  for e in seqElems(fieldNode(n, "bwdKernelParams")):
    result.bwdKernelParams.add e.strVal

# ═══════════════════════════════════════════════════════════════════════
# ▸ Call-site argument resolution
# ═══════════════════════════════════════════════════════════════════════

func isDeviceMemType(t: NimNode): bool =
  t.kind == nnkBracketExpr and t[0].eqIdent("DeviceMem")

func typedescNode(t: NimNode): NimNode =
  ## `t` as a `typedesc[t]` argument: type nodes resolved through
  ## `getTypeInst` splice as values, the typedesc wrapper binds as a type.
  nnkBracketExpr.newTree(bindSym"typedesc", t)

type BufferLeaf = object
  ## Device-memory leaf of one call-site buffer argument:
  ## `mem` is the DeviceMem-typed expression, `elemT` its element type node,
  ## `extentExpr` the wrapper's integer length field, nil for a bare DeviceMem.
  mem: NimNode
  elemT: NimNode
  extentExpr: NimNode

func hasDeviceMemLeaf(v: NimNode): bool =
  ## True when the call-site argument is a DeviceMem or a host wrapper
  ## carrying one. False for scalars.
  let t = v.getTypeInst()
  if isDeviceMemType(t):
    return true
  let impl = v.getTypeImpl()
  if impl.kind notin {nnkObjectTy, nnkTupleTy}:
    return false
  let recList = if impl.kind == nnkObjectTy: impl[^1] else: impl
  for f in recList:
    if f.kind == nnkIdentDefs and isDeviceMemType(f[^2]):
      return true
  false

func bufferLeaf(v: NimNode): BufferLeaf =
  ## Resolves one call-site buffer argument: a bare `DeviceMem`, or a host
  ## wrapper object in the launch bridge's `p`/`n` shape with one `DeviceMem`
  ## field and one integer length field.
  let t = v.getTypeInst()
  if isDeviceMemType(t):
    return BufferLeaf(mem: v, elemT: t[^1], extentExpr: nil)
  let impl = v.getTypeImpl()
  doAssert impl.kind in {nnkObjectTy, nnkTupleTy},
    "ad_handle: buffer arguments are DeviceMem or a host wrapper carrying one, got " &
    t.repr
  var memField: tuple[name, typ: NimNode]
  var extentField: tuple[name, typ: NimNode]
  let recList = if impl.kind == nnkObjectTy: impl[^1] else: impl
  for f in recList:
    f.expectKind nnkIdentDefs
    let ftype = f[^2]
    if isDeviceMemType(ftype):
      doAssert memField.name.isNil,
        "ad_handle: host wrappers carry exactly one DeviceMem field, got " & t.repr
      memField = (name: f[0], typ: ftype)
    elif ftype.kind == nnkSym and $ftype == "int":
      doAssert extentField.name.isNil,
        "ad_handle: host wrappers carry exactly one integer length field, got " & t.repr
      extentField = (name: f[0], typ: ftype)
  doAssert not memField.name.isNil,
    "ad_handle: host wrapper carries no DeviceMem field, got " & t.repr
  BufferLeaf(
    mem: v.newDotExpr(memField.name),
    elemT: memField.typ[^1],
    extentExpr: if extentField.name.isNil: nil else: v.newDotExpr(extentField.name))

func adLaunchCfg(): NimNode =
  ## Single-thread launch config of every AD launch. The scalar cotangents
  ## accumulate per-index partials without atomics and one thread keeps
  ## the accumulation race-free.
  nnkObjConstr.newTree(
    bindSym"LaunchConfig",
    nnkExprColonExpr.newTree(ident"grid", bindSym"dim3".newCall(newLit(1), newLit(1), newLit(1))),
    nnkExprColonExpr.newTree(ident"blk", bindSym"dim3".newCall(newLit(1), newLit(1), newLit(1))))

func allocAndZero(engine, dev, elemT, len: NimNode): NimNode =
  ## Zero-fill statements for a freshly allocated device buffer.
  ##
  ## The backward kernels accumulate, so gradient and scratch buffers
  ## start at zero.
  result = quote do:
    var zeroVals = newSeq[`elemT`](`len`)
    `engine`.copyToDeviceFromHost(`dev`, addr zeroVals[0], `len`)

# ═══════════════════════════════════════════════════════════════════════
# ▸ Launches
# ═══════════════════════════════════════════════════════════════════════

macro forward*(ad: typed, output: typed, primals: varargs[typed]): untyped =
  ## Launches the pair's forward kernel, the primal signature with the output
  ## slot first: `ad.forward(out, a, b, alpha)` for `out = alpha * a + b`.
  ##
  ## - `output`: the pair's primary output, its first forward output
  ## - `primals`: forward-parameter order with the output's own slot skipped,
  ##   buffers as device handles, scalars by value
  ##
  ## The launch ABI binds the launch output slot to the kernel's first parameter,
  ## so the printed parameter order is rebuilt here: the caller's output slots into its forward-parameter position,
  ## wherever the pair declares it. The caller keeps every buffer it passes.
  let kernels = ad.getTypeInst()[2].readPair()
  doAssert primals.len == kernels.params.len - 1,
    "ad_handle forward: expected " & $(kernels.params.len - 1) &
    " primals beside the output, got " & $primals.len
  var outSlot = 0
  for i, m in kernels.params:
    if m.kind == pkResidual:
      outSlot = i
      break
  var slotArgs: seq[NimNode]
  var pi = 0
  for i in 0 ..< kernels.params.len:
    if i == outSlot:
      slotArgs.add output
    else:
      slotArgs.add primals[pi]
      pi.inc
  let outMemN = if slotArgs[0].hasDeviceMemLeaf():
    slotArgs[0].bufferLeaf().mem
  else:
    slotArgs[0]
  var argsN = nnkTupleConstr.newTree()
  for n in slotArgs[1 ..< slotArgs.len]:
    if n.hasDeviceMemLeaf():
      let memN = n.bufferLeaf().mem
      argsN.add quote do: `memN`.mem
    else:
      argsN.add n
  let impl = bindSym"launchKernelImpl"
  let cfgN = adLaunchCfg()
  result = quote do:
    var argsVal = `argsN`
    `impl`(`ad`.engine, `ad`.kernels.kernelName, `outMemN`, argsVal, `cfgN`)

func bindMapAt(bindMap: seq[tuple[name: string, node: NimNode, isBuffer: bool]], name: string): tuple[node: NimNode, isBuffer: bool] =
  ## One printed-kernel parameter's bound argument.
  ## The lookup key is the pair's backward-signature name.
  for e in bindMap:
    if e.name == name:
      return (node: e.node, isBuffer: e.isBuffer)
  doAssert false, "ad_handle: the backward kernel binds '" & name &
    "', a name the backward signature does not carry"

macro backward*(ad: typed, args: varargs[typed]): untyped =
  ## Backward pass of the pair in the pair's backward signature.
  ## Arity selects between the primitive and convenience overloads:
  ##
  ## - primitive: `(d_<residuals...>, <primals...>, d_<buffers+scalars...>)`.
  ##   Every gradient is a caller-owned buffer the backward kernels
  ##   accumulate into. Pre-zero the buffers for fresh-semantics gradients.
  ## - convenience: `(d_<residuals...>, <primals...>)`. Allocates and zeroes
  ##   the remaining gradients, returns them as a named tuple of `DeviceMem`.
  ##   Allocation reads the size off each primal argument.
  ##   Grad-of-x has x's shape, scalar grads are 1-element device buffers.
  ##
  ## Example for `axpy(out, a, b: ptr UncheckedArray[float32], alpha: float32)`:
  ## `ad.backward(d_out, out, a, b, alpha)` returns the gradients
  ## `(d_a, d_b, d_alpha)`.
  let kernels = ad.getTypeInst()[2].readPair()
  result = nnkStmtList.newTree()

  var incoming: seq[BwdParamMeta]
  var allocatable: seq[BwdParamMeta]
  for m in kernels.params:
    if m.gradKind != gkNone:
      if m.kind == pkResidual:
        incoming.add m
      else:
        allocatable.add m

  let nConvenience = incoming.len + kernels.params.len
  let nPrimitive = nConvenience + allocatable.len
  doAssert args.len in {nConvenience, nPrimitive},
    "ad_handle backward: expected " & $nConvenience & " args (convenience) or " &
    $nPrimitive & " args (primitive), got " & $args.len
  let isPrimitive = args.len == nPrimitive

  # Argument table, backward-signature order: incoming cotangents, primals,
  # then caller-owned gradients for the primitive. Name -> call-site expr.
  var ai = 0
  var bindMap: seq[tuple[name: string, node: NimNode, isBuffer: bool]]
  for m in incoming:
    bindMap.add (name: m.gradName, node: args[ai], isBuffer: true)
    inc ai
  for m in kernels.params:
    bindMap.add (name: m.name, node: args[ai], isBuffer: m.kind != pkScalar)
    inc ai
  for m in allocatable:
    if isPrimitive:
      bindMap.add (name: m.gradName, node: args[ai], isBuffer: true)
      inc ai

  let engineN = ad.newDotExpr(ident"engine")
  let impl = bindSym"launchKernelImpl"
  let cfgN = adLaunchCfg()
  let bwdNameN = newLit(kernels.kernelName & "_bwd")

  # Name resolution across the whole backward surface.
  # Caller arguments resolve first.
  # Allocated gradients of the convenience resolve second, both by name.
  var gradDevs: seq[tuple[meta: BwdParamMeta, letN: NimNode]]

  proc resolveName(name: string): tuple[node: NimNode, isBuffer: bool, isOwned: bool] =
    ## `isOwned` marks the DeviceMem lets of this expansion.
    ## Such nodes are already the memory expression.
    ## Caller arguments unwrap through `bufferLeaf` instead.
    for e in bindMap:
      if e.name == name:
        return (node: e.node, isBuffer: e.isBuffer, isOwned: false)
    for g in gradDevs:
      if g.meta.gradName == name:
        return (node: g.letN, isBuffer: true, isOwned: true)
    doAssert false, "ad_handle backward: no argument for '" & name & "'"

  # Reduce-partials scratch: one zeroed buffer per scalar cotangent.
  # The length is the index-loop trip count, static or a named primal
  # parameter of the kernel.
  var scratchLets: seq[tuple[ws: ScalarWorkspace, letN: NimNode]]
  for ws in kernels.scalarWorkspaces:
    let tripN = if ws.tripCount >= 0:
      newLit(ws.tripCount)
    else:
      resolveName(ws.tripCountParam).node
    let scalarName = ws.gradName[2 ..< ws.gradName.len]
    let elemT = resolveName(scalarName).node.getTypeInst()
    let elemTD = typedescNode(elemT)
    let devN = genSym(nskLet, ws.workspaceName)
    let zeroN = allocAndZero(engineN, devN, elemT, tripN)
    result.add quote do:
      let `devN` = `engineN`.allocDevice(`elemTD`, `tripN`)
      `zeroN`
    scratchLets.add (ws: ws, letN: devN)

  # Gradient allocation for the convenience: sizes read off the primal args.
  if not isPrimitive:
    for m in allocatable:
      var elemT: NimNode
      var lenN: NimNode
      if m.gradKind == gkBuffer1:
        # Scalar cotangent: a 1-element device buffer of the scalar's type.
        elemT = resolveName(m.name).node.getTypeInst()
        lenN = newLit(1)
      else:
        let leaf = resolveName(m.name).node.bufferLeaf()
        elemT = leaf.elemT
        lenN = case m.extent
          of xeWrapperField:
            doAssert not leaf.extentExpr.isNil,
              "ad_handle backward: the convenience reads the length off the primal " &
              m.name & ", pass a host wrapper (p/n shape) instead of a bare DeviceMem"
            leaf.extentExpr
          of xeStaticShape:
            newLit(m.staticLen)
          of xeLenParam:
            resolveName(m.extentParam).node
      let elemTD = typedescNode(elemT)
      let devN = genSym(nskLet, m.gradName)
      let zeroN = allocAndZero(engineN, devN, elemT, lenN)
      result.add quote do:
        let `devN` = `engineN`.allocDevice(`elemTD`, `lenN`)
        `zeroN`
      gradDevs.add (meta: m, letN: devN)

  # Backward-kernel launch: the printed parameters bind in printed order.
  # The first printed slot is the output.
  # Names resolve against the backward signature.
  # Workspace names resolve against the scratch.
  doAssert kernels.bwdKernelParams.len > 0,
    "ad_handle backward: the kernels carries no printed backward signature"
  # The first printed slot may be an allocated gradient of this expansion,
  # a plain DeviceMem let with no type at macro time.
  # Such nodes bind `.mem` directly, caller arguments unwrap through `bufferLeaf`.
  let bwdOutResolved = resolveName(kernels.bwdKernelParams[0])
  let bwdOutput = if bwdOutResolved.isOwned:
    let eN = bwdOutResolved.node
    quote do: `eN`.mem
  else:
    bwdOutResolved.node.bufferLeaf().mem
  var bwdArgsN = nnkTupleConstr.newTree()
  for i in 1 ..< kernels.bwdKernelParams.len:
    let name = kernels.bwdKernelParams[i]
    var boundN: NimNode = nil
    for s in scratchLets:
      if s.ws.workspaceName == name:
        let sN = s.letN
        boundN = quote do: `sN`.mem
    if boundN.isNil:
      let e = resolveName(name)
      if e.isOwned:
        let eN = e.node
        boundN = quote do: `eN`.mem
      elif e.isBuffer:
        let memN = e.node.bufferLeaf().mem
        boundN = quote do: `memN`.mem
      else:
        boundN = e.node
    bwdArgsN.add boundN
  result.add quote do:
    var bwdArgsVal = `bwdArgsN`
    `impl`(`engineN`, `bwdNameN`, `bwdOutput`, bwdArgsVal, `cfgN`)

  # Reduce launches: each scratch folds into the scalar's 1-element gradient.
  for s in scratchLets:
    let scalarName = s.ws.gradName[2 ..< s.ws.gradName.len]
    let reduceName = kernels.kernelName & "_bwd_d_" & scalarName & "_reduce"
    var reduceKnown = false
    for r in kernels.reduceSources:
      if r.name == reduceName:
        reduceKnown = true
    doAssert reduceKnown, "ad_handle backward: no reduce kernel for " & s.ws.gradName
    let sN = s.letN
    let reduceNameN = newLit(reduceName)
    let nN = if s.ws.tripCount >= 0:
      if s.ws.tripCount64:
        newLit(int64(s.ws.tripCount))
      else:
        newLit(int32(s.ws.tripCount))
    else:
      resolveName(s.ws.tripCountParam).node
    let dAlpha = resolveName(s.ws.gradName)
    let dAlphaMem = if dAlpha.isOwned: dAlpha.node else: dAlpha.node.bufferLeaf().mem
    let dAlphaMemHandle = quote do: `dAlphaMem`.mem
    result.add quote do:
      var reduceArgsVal = (`nN`, `dAlphaMemHandle`)
      `impl`(`engineN`, `reduceNameN`, `sN`, reduceArgsVal, `cfgN`)

  # Convenience: the allocated gradients return, moved out of their lets.
  if not isPrimitive:
    var tupleN = nnkTupleConstr.newTree()
    for g in gradDevs:
      tupleN.add nnkExprColonExpr.newTree(ident(g.meta.gradName), g.letN)
    result.add tupleN
