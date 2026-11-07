# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Batch layer: the compile-time device function stream and the runtime command buffer.
##
## Usage:
##   ```
##   var batch = engine.recordBatch()
##   batch.enqueue<<(G, 64)>> saxpy(out, x, y, 3.0'f32)
##   batch.submit()
##   batch.finalize()
##   batch.replay()  # relaunches without recompiling
##   ```
import std/[macros]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/backends/backends_api
import workspace/crucible/src/codegen/backends/spirv_encoder
import workspace/crucible/src/codegen/frontend/nim_to_crucible
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
    when E is VulkanEngine:
      artifact*: SpirvWords
    else:
      artifact*: string

proc recordBatch*[E: HwEngine](engine: E): Batch[E] {.inline.} =
  Batch[E](engine: engine)

# ═══════════════════════════════════════════════════════════════════════
#  Compile-time stream
# ═══════════════════════════════════════════════════════════════════════
# Convention:
#   compile-time globals or functions that write to a compile-time global
#   are prefix with ctg

var ctgFrontend {.compileTime.}: CirFrontend
var ctgArtifactKind {.compileTime.}: CodegenKind
var ctgArtifactPrinted {.compileTime.}: bool
var ctgArtifactFnCount {.compileTime.}: int
var ctgTextArtifact {.compileTime.}: string
var ctgSpirvArtifact {.compileTime.}: SpirvWords

func batchCodegenKind(E: typedesc[MetalEngine]): CodegenKind {.compileTime.} =
  cgMetal

func batchCodegenKind(E: typedesc[VulkanEngine]): CodegenKind {.compileTime.} =
  cgSpirv

func batchCodegenKind(E: typedesc[WebgpuEngine]): CodegenKind {.compileTime.} =
  cgWebGPU

func batchCodegenKind(E: typedesc[OpenCLEngine]): CodegenKind {.compileTime.} =
  cgOpenCL

macro ctgAccumulateCir(procdef: typed): untyped =
  let kname = lowerNimAst(ctgFrontend, procdef)
  result = newLit(kname)

proc ctgAccumulateArtifact*(kind: static CodegenKind): auto {.compileTime.} =
  if ctgArtifactPrinted and ctgArtifactKind != kind:
    raiseAssert("ctgAccumulateArtifact: one codegen kind per compilation, " &
      $ctgArtifactKind & " already accumulated, asked for " & $kind)
  ctgArtifactKind = kind
  ctgArtifactPrinted = true
  when kind == cgSpirv:
    if ctgSpirvArtifact.len == 0 or ctgArtifactFnCount != ctgFrontend.getProcCount():
      var reg = PassRegistry()
      reg.registerGpuPasses(kind)
      let m = ctgFrontend.finish()
      reg.runPasses(m)
      ctgSpirvArtifact = codegenIR(m, kind)
      ctgArtifactFnCount = ctgFrontend.getProcCount()
    ctgSpirvArtifact
  else:
    if ctgTextArtifact.len == 0 or ctgArtifactFnCount != ctgFrontend.getProcCount():
      var reg = PassRegistry()
      reg.registerGpuPasses(kind)
      let m = ctgFrontend.finish()
      reg.runPasses(m)
      ctgTextArtifact = codegenSource(m, kind)
      ctgArtifactFnCount = ctgFrontend.getProcCount()
    ctgTextArtifact

# ═══════════════════════════════════════════════════════════════════════
#  Enqueue
# ═══════════════════════════════════════════════════════════════════════

template appendEntry[E](batch: Batch[E], fnName: string, cfg: LaunchConfig, output: auto, args: tuple) =
  var entry = BatchEntry[E](entrypoint: fnName, config: cfg)
  entry.launch = proc(engine: E) {.closure.} =
    engine.launchKernelImpl(output, args, cfg, fnName)
  batch.entries.add entry

macro enqueueDeviceFn[E](batch: Batch[E], deviceFn: typed, statics: untyped, cfg: LaunchConfig, blk: untyped, values: varargs[typed]): untyped =
  var valuesN = nnkBracket.newTree()
  for v in values:
    valuesN.add(v)
  let procdef = buildGlobalFn(deviceFn, statics, valuesN)
  let wg = getWorkgroupPragma(blk)
  if not wg.isNil:
    procdef[4].add(wg)
  procdef[6][0][0] = deviceFn
  let fnNameN = quote do: ctgAccumulateCir(`procdef`)
  let outN = values[0]
  var argsTuple = nnkTupleConstr.newTree()
  for i in 1 ..< values.len:
    argsTuple.add(values[i])
  result = quote do:
    block:
      const fnName = `fnNameN`
      `batch`.appendEntry(fnName, `cfg`, `outN`, `argsTuple`)

type
  EnqueueSugar*[E] = object
    ## Chevron stage between `batch.enqueue` and `<<`, wrapping the batch.
    batch: Batch[E]

  EnqueueProxy*[E] = object
    ## Chevron stage between `<<` and the `>>` call
    batch: Batch[E]
    config: LaunchConfig

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
  let E = r.getTypeInst()[1]
  let gridCall = bindSym"dim3".newCall(gridN)
  let blkCall = bindSym"dim3".newCall(blkN)
  let configCall = nnkObjConstr.newTree(bindSym"LaunchConfig",
    nnkExprColonExpr.newTree(ident"grid", gridCall),
    nnkExprColonExpr.newTree(ident"blk", blkCall))
  result = quote do:
    EnqueueProxy[`E`](batch: `r`.batch, config: `configCall`)

macro `>>`*(sugar: EnqueueProxy, call: untyped): untyped =
  var callNode = call
  if callNode.kind == nnkPar and callNode.len == 1:
    callNode = callNode[0]
  doAssert callNode.kind == nnkCall,
    "batch enqueue RHS must be a device function call, got: " & callNode.repr
  for a in callNode[1 ..< callNode.len]:
    doAssert a.kind notin {nnkExprEqExpr, nnkExprColonExpr},
      "batch enqueue: named arguments are not supported, pass arguments positionally"
  let k = callNode.resolveDeviceFnOverload()
  let impl = k.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      formalTypes.add(f[^2])
  doAssert callNode.len - 1 <= formalTypes.len,
    "batch enqueue: " & $(callNode.len - 1) & " args for " &
    $formalTypes.len & " parameters"
  var statics = nnkBracket.newTree()
  let batchN = quote do: `sugar`.batch
  let cfgN = quote do: `sugar`.config
  # Vulkan and WebGPU printers bake blk as the entry's LocalSize
  var blkN = newEmptyNode()
  for f in sugar:
    if f.kind == nnkExprColonExpr and f[0].kind == nnkIdent and f[0].strVal == "config":
      for g in f[1]:
        if g.kind == nnkExprColonExpr and g[0].kind == nnkIdent and g[0].strVal == "blk":
          blkN = g[1][1]
  result = bindSym"enqueueDeviceFn".newCall(
    batchN, k, statics, cfgN, blkN)
  for i, a in callNode[1 ..< callNode.len]:
    if not isCompileTime(formalTypes[i]):
      result.add(a)

# ═══════════════════════════════════════════════════════════════════════
#  Submission
# ═══════════════════════════════════════════════════════════════════════

macro finalize*(batch: Batch): untyped =
  ## Compiles the batch's device functions into one artifact and ingests it
  ## into the engine. Does not launch the entries.
  ##
  ## Requires:
  ## - at least one `enqueue` in this compilation
  ##
  ## Later `finalize` calls reuse the cached artifact.
  if ctgFrontend.isEmpty():
    error("finalize: the stream holds no device function, enqueue first", batch)
  let engineN = quote do: `batch`.engine
  result = quote do:
    block:
      const batchKind = batchCodegenKind(typeof(`engineN`))
      const batchArtifact = ctgAccumulateArtifact(batchKind)
      `batch`.artifact = batchArtifact
      when batchKind == cgSpirv:
        if `engineN`.getSpirV() != batchArtifact:
          `engineN`.ingest(batchArtifact)
      else:
        if `engineN`.getArtifact() != batchArtifact:
          `engineN`.ingest(batchArtifact)

proc replay*[E](batch: Batch[E]) {.inline.} =
  ## Launches the batch entries on the engine in enqueue order, without recompiling.
  ##
  ## Requires:
  ## - a prior `finalize` in this compilation
  ##
  ## Re-ingests the batch artifact when the engine ingested something else
  ## since `finalize`.
  if batch.artifact.len == 0:
    raiseAssert("replay: the batch holds no artifact, call finalize first")
  when E is VulkanEngine:
    if batch.engine.getSpirV() != batch.artifact:
      batch.engine.ingest(batch.artifact)
  else:
    if batch.engine.getArtifact() != batch.artifact:
      batch.engine.ingest(batch.artifact)
  for entry in batch.entries:
    entry.launch(batch.engine)

macro submit*(batch: Batch): untyped =
  ## Flushes the batch: finalizes the compiled artifact and launches the entries
  ## on the engine in enqueue order.
  ##
  ## Requires:
  ## - at least one `enqueue` in this compilation
  ## - submitting an empty batch is a no-op
  ##
  ## The compilation compiles its device functions once.
  ## Device functions enqueued later recompile it, every `submit` reuses the module.
  ##
  ##   ```
  ##   var batch = engine.recordBatch()
  ##   batch.enqueue<<(N div 64, 64)>> saxpy(out, x, y, 3.0'f32)
  ##   batch.submit()
  ##   ```
  result = nnkStmtList.newTree(
    bindSym"finalize".newCall(batch),
    bindSym"replay".newCall(batch))
