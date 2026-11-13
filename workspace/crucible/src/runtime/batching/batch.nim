# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## BatchCommands layer: the compile-time device function stream and the runtime command buffer.
##
## Usage:
##   ```
##   var batch = engine.recordBatch()
##   batch.enqueue<<(G, 64)>> saxpy(out, x, y, 3.0'f32)
##   batch.submit()
##   batch.finalize()
##   batch.replay()  # relaunches without recompiling
##   ```
import std/[macros, macrocache]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/backends/backends_api
import workspace/crucible/src/codegen/backends/spirv_encoder
import workspace/crucible/src/codegen/frontend/nim_to_crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/crucible/src/runtime/launch/launch_device_fn {.all.}
import workspace/crucible/src/runtime/batching/batch_metal
import workspace/crucible/src/runtime/batching/batch_vulkan
import workspace/crucible/src/runtime/batching/batch_opencl
import workspace/crucible/src/runtime/batching/batch_webgpu

# ═══════════════════════════════════════════════════════════════════════
#  Record
# ═══════════════════════════════════════════════════════════════════════

type
  BatchEntry*[E] = object
    entrypoint*: string
    config*: LaunchConfig
    launch*: proc(engine: E) {.closure.}
    when E is MetalEngine:
      record: proc(leg: MetalBatch) {.closure.}
    elif E is VulkanEngine:
      record: proc(leg: VulkanBatch) {.closure.}
    elif E is OpenCLEngine:
      record: proc(leg: OpenCLBatch) {.closure.}
    elif E is WebgpuEngine:
      record: proc(leg: WebgpuBatch) {.closure.}

  BatchCommands*[E] = ref object
    engine*: E
    entries*: seq[BatchEntry[E]]
    when E is VulkanEngine:
      artifact*: SpirvWords
    else:
      artifact*: string
    when E is MetalEngine:
      leg: MetalBatch
    elif E is VulkanEngine:
      leg: VulkanBatch
    elif E is OpenCLEngine:
      leg: OpenCLBatch
    elif E is WebgpuEngine:
      leg: WebgpuBatch

proc recordBatch*[E: HwEngine](engine: E): BatchCommands[E] {.inline.} =
  BatchCommands[E](engine: engine)

# ═══════════════════════════════════════════════════════════════════════
#  Compile-time stream
# ═══════════════════════════════════════════════════════════════════════
# Convention:
#   compile-time globals or functions that write to a compile-time global
#   are prefix with ctg

# Enqueued device functions wait here as tagged proc definitions,
# one entry per distinct (backend, bridge name).
var ctgPendingKernels {.compileTime.}: CacheSeq
var ctgPendingIndex {.compileTime.}: CacheTable

var ctgGenerationCount {.compileTime.}: array[CodegenKind, int]

func batchCodegenKind(E: typedesc[MetalEngine]): CodegenKind {.compileTime.} =
  cgMetal

func batchCodegenKind(E: typedesc[VulkanEngine]): CodegenKind {.compileTime.} =
  cgGlsl

func batchCodegenKind(E: typedesc[WebgpuEngine]): CodegenKind {.compileTime.} =
  cgWebGPU

func batchCodegenKind(E: typedesc[OpenCLEngine]): CodegenKind {.compileTime.} =
  cgOpenCL

macro ctgAccumulateCirImpl(procdef: untyped, backend: static CodegenKind): untyped =
  # A staged procdef stores untyped as a call description: kernels instantiate at materialization.
  let kname = $procdef[0]
  let key = $int(backend) & ":" & kname
  if not ctgPendingIndex.hasKey(key):
    ctgPendingIndex[key] = procdef
    ctgPendingKernels.add(nnkBracket.newTree(newLit(int(backend)), procdef, newLit(key)))
    ctgGenerationCount[backend].inc
  result = newLit(kname)

macro ctgAccumulateCir(procdef: untyped, backend: static CodegenKind): untyped =
  # Tile-typed expressions read the target through ccGetBackend during the enqueue expansion.
  crucibleCompileTarget = case backend
    of cgMetal: ctMetal
    of cgGlsl: ctGlsl
    of cgSpirv: ctSpirvShader
    of cgOpenCL: ctOpenCL
    of cgWebGPU: ctWebGPU
    else: ctNone
  # delayed symbol resolution
  result = newCall(bindSym"ctgAccumulateCirImpl", procdef, newLit(backend))

proc ctgLowerArtifact(kind: static CodegenKind): auto {.compileTime.} =
  # Every materialization lowers into a fresh frontend and passes it once:
  # passes are not idempotent over a frontend they already processed.
  crucibleCompileTarget = case kind
    of cgMetal: ctMetal
    of cgGlsl: ctGlsl
    of cgSpirv: ctSpirvShader
    of cgOpenCL: ctOpenCL
    of cgWebGPU: ctWebGPU
    else: ctNone
  var fe: CirFrontend
  for i in 0 ..< ctgPendingKernels.len:
    let entry = ctgPendingKernels[i]
    if entry[0].intVal == int(kind):
      discard lowerNimAst(fe, ctgPendingIndex[entry[2].strVal & ":typed"])
  var reg = PassRegistry()
  reg.registerGpuPasses(kind)
  let m = fe.finish()
  reg.runPasses(m)
  when kind == cgSpirv:
    codegenIR(m, kind)
  else:
    codegenSource(m, kind)

proc ctgCurrentGeneration*(kind: static CodegenKind): int {.compileTime.} =
  ## Stream generation of `kind`. Callers pass it as the static `gen` key
  ## of `ctgArtifact` to bind the artifact of their own enqueue prefix.
  ctgGenerationCount[kind]

macro ctgSemRelay(key: static string, procdef: typed): untyped =
  # Materialization-site record of a sem-checked wrapper, consumed by artifact lowering.
  ctgPendingIndex[key] = procdef
  result = newEmptyNode()

macro ctgSemStaged(kind: static CodegenKind): untyped =
  # A staged procdef references bound symbols only (callee symbol, parameter types, bindSym'd pragmas).
  # Kernels type without the enqueue site scope and instantiate exactly once each.
  crucibleCompileTarget = case kind
    of cgMetal: ctMetal
    of cgGlsl: ctGlsl
    of cgSpirv: ctSpirvShader
    of cgOpenCL: ctOpenCL
    of cgWebGPU: ctWebGPU
    else: ctNone
  result = nnkStmtList.newTree()
  for i in 0 ..< ctgPendingKernels.len:
    let entry = ctgPendingKernels[i]
    if entry[0].intVal == int(kind) and
        not ctgPendingIndex.hasKey(entry[2].strVal & ":typed"):
      result.add nnkCall.newTree(
        bindSym"ctgSemRelay", newLit(entry[2].strVal & ":typed"), entry[1])

proc ctgArtifact*(kind: static CodegenKind, gen: static int): auto =
  ## Returns the compiled artifact for `kind` at stream generation `gen`,
  ## baked into the binary as a const. One instantiation per (kind, gen):
  ## - sites at the same generation share one lowering
  ## - the artifact covers exactly the kernels enqueued before that generation
  ##
  ## The instantiation is the staging point: it sem-checks the staged wrappers,
  ## instantiating each stream kernel once, then lowers the typed nodes.
  ## Enqueueing more kernels and finalizing again compiles a fresh artifact.
  ctgSemStaged(kind)
  const artifact = static(ctgLowerArtifact(kind))
  artifact

# ═══════════════════════════════════════════════════════════════════════
#  Enqueue
# ═══════════════════════════════════════════════════════════════════════

template appendEntry[E](batch: BatchCommands[E], fnName: string, cfg: LaunchConfig, output: auto, args: tuple) =
  var entry = BatchEntry[E](entrypoint: fnName, config: cfg)
  entry.launch = proc(engine: E) {.closure.} =
    engine.launchKernelImpl(output, args, entry.config, fnName)
  when E is MetalEngine:
    entry.record = proc(leg: MetalBatch) {.closure.} =
      leg.recordGlobalFn(fnName, entry.config, output, args)
  batch.entries.add entry

template appendEntry[E](batch: BatchCommands[E], fnName: string, artifact: auto, cfg: LaunchConfig, output: auto, args: tuple) =
  var entry = BatchEntry[E](entrypoint: fnName, config: cfg)
  entry.launch = proc(engine: E) {.closure.} =
    engine.launchKernelImpl(output, args, entry.config, fnName)
  when E is VulkanEngine:
    entry.record = proc(leg: VulkanBatch) {.closure.} =
      leg.recordGlobalFn(artifact, fnName, entry.config, output, args)
  elif E is OpenCLEngine:
    entry.record = proc(leg: OpenCLBatch) {.closure.} =
      leg.recordGlobalFn(artifact, fnName, entry.config, output, args)
  elif E is WebgpuEngine:
    entry.record = proc(leg: WebgpuBatch) {.closure.} =
      leg.recordGlobalFn(artifact, fnName, entry.config, output, args)
  batch.entries.add entry

macro enqueueDeviceFn[E](batch: BatchCommands[E], deviceFn: typed, statics: typed, cfg: LaunchConfig, workgroup: untyped, values: varargs[typed]): untyped =
  var valuesN = nnkBracket.newTree()
  for v in values:
    valuesN.add(v)
  let procdef = buildGlobalFn(deviceFn, statics, valuesN)
  if workgroup.kind != nnkEmpty:
    procdef[4].add(workgroup)
  procdef[6][0][0] = deviceFn
  let backendN = quote do: batchCodegenKind(typeof(`batch`.engine))
  let fnNameN = quote do: ctgAccumulateCir(`procdef`, `backendN`)
  let outN = values[0]
  var argsTuple = nnkTupleConstr.newTree()
  for i in 1 ..< values.len:
    argsTuple.add(values[i])
  result = quote do:
    block:
      const fnName = `fnNameN`
      when typeof(`batch`.engine) is VulkanEngine:
        const entryArtifact = spirv(`procdef`)
        `batch`.appendEntry(fnName, entryArtifact, `cfg`, `outN`, `argsTuple`)
      elif typeof(`batch`.engine) is OpenCLEngine:
        const entryArtifact = opencl(`procdef`)
        `batch`.appendEntry(fnName, entryArtifact, `cfg`, `outN`, `argsTuple`)
      elif typeof(`batch`.engine) is WebgpuEngine:
        const entryArtifact = webgpu(`procdef`)
        `batch`.appendEntry(fnName, entryArtifact, `cfg`, `outN`, `argsTuple`)
      else:
        `batch`.appendEntry(fnName, `cfg`, `outN`, `argsTuple`)

type
  EnqueueSugar*[E] = object
    ## Chevron stage between `batch.enqueue` and `<<`, wrapping the batch.
    batch: BatchCommands[E]

  EnqueueProxy*[E] = object
    ## Chevron stage between `<<` and `>>`, runtime block size.
    batch: BatchCommands[E]
    config: LaunchConfig

  EnqueueProxyCt*[E; blkCt: static Dim3] = object
    ## Chevron stage between `<<` and `>>`, compile-time block size.
    batch: BatchCommands[E]
    config: LaunchConfig

proc enqueue*[E](batch: BatchCommands[E]): EnqueueSugar[E] {.inline.} =
  ## batch.enqueue<<(4, 16)>> saxpy(out, x, y, 2.0)
  EnqueueSugar[E](batch: batch)

macro `<<`*[E](r: EnqueueSugar[E], cfg: untyped): untyped =
  ## Chevron schedule: `(grid, blk)` or `(grid, blk, sharedMem)`, positional or named.
  ## Vulkan and WebGPU bake the block size into the artifact, it must be compile-time there.
  var gridN = nnkPar.newTree(newLit(1), newLit(1))
  var blkN = nnkPar.newTree(newLit(1), newLit(1))
  var sharedMemN = newLit(0)
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
        of "sharedMem":
          sharedMemN = f[1]
        else:
          error("batch enqueue: chevron schedule expects grid / blk / sharedMem, got " & f.repr)
      else:
        case pos
        of 0:
          gridN = f
        of 1:
          blkN = f
        of 2:
          sharedMemN = f
        else:
          error("batch enqueue: chevron schedule takes grid, blk, sharedMem, got " & f.repr)
        pos.inc
  else:
    error("batch enqueue: chevron schedule expects (grid, blk), got " & cfg.repr)
  let E = r.getTypeInst()[1]
  let batchN = nnkDotExpr.newTree(r, ident"batch")
  let gridRef = bindSym"dim3".newCall(gridN)
  let smemRef = sharedMemN
  result = quote do:
    when `blkN` is static:
      const blkCt = `blkN`
      EnqueueProxyCt[`E`, dim3(blkCt)](batch: `batchN`, config: LaunchConfig(
        grid: `gridRef`, blk: dim3(blkCt), sharedMem: `smemRef`))
    else:
      EnqueueProxy[`E`](batch: `batchN`, config: LaunchConfig(
        grid: `gridRef`, blk: dim3(`blkN`), sharedMem: `smemRef`))

func ctgEngineTarget(engineT: NimNode): CompileTarget {.compileTime.} =
  case $engineT
  of "MetalEngine": ctMetal
  of "VulkanEngine": ctGlsl
  of "WebgpuEngine": ctWebGPU
  of "OpenCLEngine": ctOpenCL
  else: ctNone

macro enqueueCall[E](batch: BatchCommands[E], cfg: LaunchConfig, workgroup: untyped, deviceFn: typed, values: varargs[untyped]): untyped =
  # Compile-time arguments sem-check in this expansion before the accumulate
  # macro runs: ccGetBackend (MMA atom resolution) reads the target.
  crucibleCompileTarget = ctgEngineTarget(batch.getTypeInst()[1])
  var callNode = nnkCall.newTree(deviceFn)
  for v in values:
    callNode.add(v)
  let k = callNode.resolveDeviceFnOverload()
  let impl = k.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      formalTypes.add(f[^2])
  # Tuple constr over a typed parameter: each compile-time argument sem-checks
  # at the enqueue site, where the handler generic parameters are in scope.
  var statics = nnkTupleConstr.newTree()
  let entry = bindSym"enqueueDeviceFn".newCall(batch, k, statics, cfg, workgroup)
  for i, v in values:
    if isCompileTime(formalTypes[i]):
      statics.add(v)
    else:
      entry.add(v)
  result = entry

macro enqueueCallCt[E](batch: BatchCommands[E], cfg: LaunchConfig, blkCt: static Dim3, deviceFn: typed, values: varargs[untyped]): untyped =
  let wg = getWorkgroupPragma(blkCt)
  result = bindSym"enqueueCall".newCall(batch, cfg,
    if wg.isNil(): newEmptyNode() else: wg, deviceFn)
  for v in values:
    result.add(v)

macro `>>`*(sugar: EnqueueProxyCt, call: untyped): untyped =
  var callNode = call
  if callNode.kind == nnkPar and callNode.len == 1:
    callNode = callNode[0]
  doAssert callNode.kind == nnkCall,
    "batch enqueue RHS must be a device function call, got: " & callNode.repr
  for a in callNode[1 ..< callNode.len]:
    doAssert a.kind notin {nnkExprEqExpr, nnkExprColonExpr},
      "batch enqueue: named arguments are not supported, pass arguments positionally"
  let batchN = nnkDotExpr.newTree(sugar, ident"batch")
  let cfgN = nnkDotExpr.newTree(sugar, ident"config")
  let blkCtN = sugar.getTypeInst()[2]
  result = bindSym"enqueueCallCt".newCall(batchN, cfgN, blkCtN, callNode[0])
  for a in callNode[1 ..< callNode.len]:
    result.add(a)

macro `>>`*[E: VulkanEngine or WebgpuEngine](sugar: EnqueueProxy[E], call: untyped): untyped =
  error("batch enqueue: Vulkan and WebGPU bake the block size into the artifact, " &
    "the schedule must be compile-time on those engines")

macro `>>`*[E](sugar: EnqueueProxy[E], call: untyped): untyped =
  var callNode = call
  if callNode.kind == nnkPar and callNode.len == 1:
    callNode = callNode[0]
  doAssert callNode.kind == nnkCall,
    "batch enqueue RHS must be a device function call, got: " & callNode.repr
  for a in callNode[1 ..< callNode.len]:
    doAssert a.kind notin {nnkExprEqExpr, nnkExprColonExpr},
      "batch enqueue: named arguments are not supported, pass arguments positionally"
  let batchN = nnkDotExpr.newTree(sugar, ident"batch")
  let cfgN = nnkDotExpr.newTree(sugar, ident"config")
  result = bindSym"enqueueCall".newCall(batchN, cfgN, newEmptyNode(), callNode[0])
  for a in callNode[1 ..< callNode.len]:
    result.add(a)

# ═══════════════════════════════════════════════════════════════════════
#  Submission
# ═══════════════════════════════════════════════════════════════════════

macro finalize*(batch: BatchCommands): untyped =
  ## Compiles the batch's device functions into one artifact and ingests it
  ## into the engine. Does not launch the entries.
  ##
  ## Finalize calls at the same stream generation share one artifact.
  ## Enqueueing more kernels and finalizing again compiles a fresh artifact
  ## over the complete stream.
  let engineN = quote do: `batch`.engine
  result = quote do:
    block:
      const batchKind = batchCodegenKind(typeof(`engineN`))
      const batchGen = ctgCurrentGeneration(batchKind)
      `batch`.artifact = ctgArtifact(batchKind, batchGen)
      `engineN`.ingest(`batch`.artifact)

proc replay*[E](batch: BatchCommands[E]) {.inline.} =
  ## Launches the batch entries on the engine in enqueue order, without recompiling.
  ##
  ## Requires:
  ## - a prior `finalize` in this compilation
  ##
  ## Re-ingests the batch artifact when the engine ingested something else
  ## since `finalize`.
  if batch.artifact.len == 0:
    raiseAssert("replay: the batch holds no artifact, call finalize first")
  batch.engine.ingest(batch.artifact)
  when E is MetalEngine:
    if batch.leg.isNil():
      batch.leg = newMetalBatch(batch.engine)
    else:
      batch.leg.resetRecording()
    for entry in batch.entries:
      entry.record(batch.leg)
    batch.leg.finalize()
    batch.leg.replay()
  elif E is VulkanEngine:
    if batch.leg.isNil():
      batch.leg = newVulkanBatch(batch.engine)
    else:
      batch.leg.resetRecording()
    for entry in batch.entries:
      entry.record(batch.leg)
    batch.leg.finalize()
    batch.leg.replay()
  elif E is OpenCLEngine:
    if batch.leg.isNil():
      batch.leg = newOpenCLBatch(batch.engine)
    else:
      batch.leg.resetRecording()
    for entry in batch.entries:
      entry.record(batch.leg)
    batch.leg.finalize()
    batch.leg.replay()
  elif E is WebgpuEngine:
    if batch.leg.isNil():
      batch.leg = newWebgpuBatch(batch.engine)
    else:
      batch.leg.resetRecording()
    for entry in batch.entries:
      entry.record(batch.leg)
    batch.leg.finalize()
    batch.leg.replay()
  else:
    for entry in batch.entries:
      entry.launch(batch.engine)

macro submit*(batch: BatchCommands): untyped =
  ## Flushes the batch: finalizes the compiled artifact and launches the entries
  ## on the engine in enqueue order.
  ##
  ## Requires:
  ## - at least one `enqueue` in this compilation, an empty batch raises
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
