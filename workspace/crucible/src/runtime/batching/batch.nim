# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Batch layer: the compile-time kernel stream and the runtime command buffer.
##
## Usage:
##   ```
##   var batch = engine.recordBatch()
##   batch.enqueue<<(G, 64)>> saxpy(out, x, y, 3.0'f32)
##   batch.submit()
##   ```
import std/[hashes, macros, sets]
import workspace/crucible/src/runtime/engines/engines_api
import workspace/crucible/src/runtime/launch/launch_device_fn {.all.}

# ═══════════════════════════════════════════════════════════════════════
#  Record
# ═══════════════════════════════════════════════════════════════════════

type
  BatchEntry*[E] = object
    entrypoint*: string
    config*: LaunchConfig
    launch*: proc(engine: E) {.closure.}

  Batch*[E] = ref object
    engine*: E
    entries*: seq[BatchEntry[E]]

proc makeBatch[E](engine: E): Batch[E] {.inline.} =
  Batch[E](engine: engine)

macro recordBatch*(engine: HwEngine): untyped =
  let makeS = bindSym"makeBatch"
  result = quote do:
    `makeS`(`engine`)

# ═══════════════════════════════════════════════════════════════════════
#  Compile-time stream
# ═══════════════════════════════════════════════════════════════════════

var ctgWrapperStash {.compileTime.}: seq[NimNode]
var ctgStashNames {.compileTime.}: HashSet[string]

proc stashWrapper(procdef: NimNode) {.compileTime.} =
  let name = $procdef[0]
  if name in ctgStashNames:
    return
  ctgWrapperStash.add(procdef)
  ctgStashNames.incl(name)

# ═══════════════════════════════════════════════════════════════════════
#  Enqueue
# ═══════════════════════════════════════════════════════════════════════

func batchCodegenKind(E: typedesc[MetalEngine]): CodegenKind {.compileTime.} =
  cgMetal

func batchCodegenKind(E: typedesc[VulkanEngine]): CodegenKind {.compileTime.} =
  cgSpirv

macro recordBatchEnqueue(batch: Batch, kernel: typed, statics: untyped, gridN, blkN: untyped, values: varargs[typed]): untyped =
  let wg = getWorkgroupPragma(blkN)
  var valuesN = nnkBracket.newTree()
  for v in values:
    valuesN.add(v)
  let procdef = buildGlobalFn(kernel, statics, valuesN)
  if not wg.isNil:
    procdef[4].add(wg)

  procdef[0] = ident($procdef[0] & "_" & base58(int(hash(blkN.repr))))
  procdef[6][0][0] = kernel
  let known = $procdef[0].strVal in ctgStashNames
  if not known:
    stashWrapper(procdef)

  let E = batch.getTypeInst()[1]
  let entryN = ident"batchEntry"
  let kname = newLit($procdef[0])
  let configCall = nnkObjConstr.newTree(bindSym"LaunchConfig",
    nnkExprColonExpr.newTree(ident"grid", bindSym"dim3".newCall(gridN)),
    nnkExprColonExpr.newTree(ident"blk", bindSym"dim3".newCall(blkN)))
  let argsTuple = nnkTupleConstr.newTree()
  for i in 1 ..< values.len:
    argsTuple.add(values[i])
  let outN = values[0]
  result = quote do:
    block:
      var `entryN` = BatchEntry[`E`](entrypoint: `kname`, config: `configCall`)
      `entryN`.launch = proc(e: `E`) =
        e.launchKernelImpl(`outN`, `argsTuple`, `configCall`, `kname`)
      `batch`.entries.add `entryN`

macro recordDeviceCall(batch: Batch, kernel: typed, gridN, blkN: untyped, args: varargs[untyped]): untyped =
  for a in args:
    doAssert a.kind notin {nnkExprEqExpr, nnkExprColonExpr},
      "batch enqueue: named arguments are not supported, pass arguments positionally"
  let k = kernel.resolveKernelOverload(args.len)
  let impl = k.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      formalTypes.add(f[^2])
  doAssert args.len <= formalTypes.len,
    "batch enqueue: " & $args.len & " args for " &
    $formalTypes.len & " parameters"
  var statics = nnkBracket.newTree()
  result = bindSym"recordBatchEnqueue".newCall(batch, k, statics, gridN, blkN)
  for i, a in args:
    if isCompileTime(formalTypes[i]):
      statics.add(a)
    else:
      result.add(a)

type
  EnqueueSugar*[E] = object
    ## Chevron stage between `batch.enqueue` and `<<`, wrapping the batch.
    batch: Batch[E]

  EnqueueProxy*[BX, BY, BZ: static int, E] = object
    ## Chevron stage between `<<` and the `>>` call
    batch: Batch[E]
    config: LaunchConfig

func dims3(n: NimNode): array[3, NimNode] =
  if n.kind in {nnkPar, nnkTupleConstr, nnkBracket} and n.len > 3:
    error("batch enqueue: a launch dim takes at most 3 axes, got " & $n.len)
  if n.kind in {nnkPar, nnkTupleConstr, nnkBracket}:
    for i in 0 ..< 3:
      result[i] = if i < n.len: n[i] else: newLit(1)
  else:
    result = [n, newLit(1), newLit(1)]

proc enqueue*[E](batch: Batch[E]): EnqueueSugar[E] {.inline.} =
  ## batch.enqueue<<(4, 16)>> saxpy(out, x, y, 2.0)
  EnqueueSugar[E](batch: batch)

macro `<<`*[E](r: EnqueueSugar[E], cfg: untyped): untyped =
  var gridN = nnkPar.newTree(newLit(1), newLit(1))
  var blkN = nnkPar.newTree(newLit(1), newLit(1))
  var c = cfg
  if c.kind in {nnkPar, nnkTupleConstr, nnkBracket} and c.len == 1:
    c = c[0]
  if c.kind in {nnkPar, nnkTupleConstr, nnkBracket}:
    var pos = 0
    for f in c:
      if f.kind == nnkExprColonExpr:
        case $f[0]
        of "grid":
          gridN = f[1]
        of "blk":
          blkN = f[1]
        else:
          error("batch enqueue: chevron schedule expects grid / blk, got " & f.repr)
      else:
        case pos
        of 0:
          gridN = f
        of 1:
          blkN = f
        else:
          error("batch enqueue: chevron schedule takes grid, blk, got " & f.repr)
        pos.inc
  else:
    error("batch enqueue: chevron schedule expects (grid, blk), got " & cfg.repr)
  let b = dims3(blkN)
  let E = r.getTypeInst()[1]
  let g = dims3(gridN)
  let bx = b[0]
  let by = b[1]
  let bz = b[2]
  let gridCall = bindSym"dim3".newCall(g[0], g[1], g[2])
  let blkCall = bindSym"dim3".newCall(bx, by, bz)
  let configCall = nnkObjConstr.newTree(bindSym"LaunchConfig",
    nnkExprColonExpr.newTree(ident"grid", gridCall),
    nnkExprColonExpr.newTree(ident"blk", blkCall))
  result = quote do:
    EnqueueProxy[`bx`, `by`, `bz`, `E`](batch: `r`.batch, config: `configCall`)

macro `>>`*(sugar: EnqueueProxy, call: untyped): untyped =
  let t = sugar.getTypeInst()
  var callNode = call
  if callNode.kind == nnkPar and callNode.len == 1:
    callNode = callNode[0]
  doAssert callNode.kind == nnkCall,
    "batch enqueue RHS must be a kernel call, got: " & callNode.repr
  let gridN = quote do: `sugar`.config.grid
  let blkN = nnkPar.newTree(t[1], t[2], t[3])
  result = bindSym"recordDeviceCall".newCall(
    sugar.newDotExpr(ident"batch"),
    callNode[0],
    gridN, blkN)
  for i in 1 ..< callNode.len:
    result.add(callNode[i])

# ═══════════════════════════════════════════════════════════════════════
#  Submission
# ═══════════════════════════════════════════════════════════════════════

macro submit*(batch: Batch): untyped =
  ## Flushes the batch: the launches run on the engine in enqueue order.
  ##
  ## Requires:
  ## - at least one `enqueue` in this compilation
  ## - submitting an empty batch is a no-op
  ##
  ## The compilation compiles its kernels once.
  ## Kernels enqueued later recompile it, every `submit` reuses the module.
  ##
  ##   ```
  ##   var batch = engine.recordBatch()
  ##   batch.enqueue<<(N div 64, 64)>> saxpy(out, x, y, 3.0'f32)
  ##   batch.submit()
  ##   ```
  if ctgWrapperStash.len == 0:
    error("submit: the stream holds no kernel, enqueue first", batch)
  var procdefsN = nnkStmtList.newTree()
  for i in 0 ..< ctgWrapperStash.len:
    procdefsN.add(ctgWrapperStash[i])
  let engineN = quote do: `batch`.engine
  result = quote do:
    block:
      const batchKind = batchCodegenKind(typeof(`engineN`))
      const batchArtifact = compileBatch(`engineN`, `procdefsN`)
      when batchKind == cgSpirv:
        if `engineN`.getSpirV() != batchArtifact:
          `engineN`.ingest(batchArtifact)
      else:
        if `engineN`.getArtifact() != batchArtifact:
          `engineN`.ingest(batchArtifact)
      `engineN`.submit(`batch`)
