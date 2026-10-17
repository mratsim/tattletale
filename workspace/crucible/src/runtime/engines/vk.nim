# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/[strutils, tables]

import workspace/crucible/src/abis/vulkan_abi as vk

import ../../codegen/ir/gpu_types
import ../exec/vulkan_runtime
import ../engines_common
import ../errors
import ../launch/launch_common

export vulkan_runtime
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  VulkanCtx = object
    ## RAII value wrapper, `=destroy` releases the device resources when the engine ref dies.
    ctx: VulkanContext

  VulkanCaches = object
    ## RAII value wrapper for the ingest-once / run-many caches.
    ## Holds a device ctx copy for the `=destroy` pipeline release.
    ctx: VulkanContext
    spirvCache: Table[string, seq[uint32]]
    pipelineCache: Table[(string, int, int), VulkanPipeline]

  VulkanEngine* = ref object
    ## A Vulkan engine, build with `bkVulkan.init()`: GLSL kernels compiled to
    ## SPIR-V and dispatched on one device. All GPU resources release when the engine dies.
    source: string
    spirv: seq[uint32]
    entryPoint: string  # GLSL entry point name, baked at ingest
    ctx: VulkanCtx
    bakedBlk: Dim3   # workgroup size baked into the shader at ingest
    caches: VulkanCaches

proc backendKind*(_: typedesc[VulkanEngine]): static BackendKind = bkVulkan

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc `=destroy`(c: var VulkanCtx) =
  ## Releases the Vulkan resources, nil-guarded against double teardown.
  if c.ctx.instance == nil:
    return
  let L = c.ctx.vkLib
  let vkDestroyPipelineCachePtr = cast[proc(device: VkDevice, pipelineCache: VkPipelineCache, pAllocator: pointer) {.cdecl.}](c.ctx.gpaAddr(c.ctx.instance, "vkDestroyPipelineCache"))
  let vkDestroyCommandPoolPtr = cast[proc(device: VkDevice, commandPool: VkCommandPool, pAllocator: pointer) {.cdecl.}](c.ctx.gpaAddr(c.ctx.instance, "vkDestroyCommandPool"))
  let vkDestroyDevicePtr = cast[
    proc(device: VkDevice, pAllocator: pointer) {.cdecl.}
  ](c.ctx.gpaAddr(c.ctx.instance, "vkDestroyDevice"))
  let vkDestroyInstancePtr = cast[
    proc(instance: VkInstance, pAllocator: pointer) {.cdecl.}
  ](c.ctx.gpaAddr(c.ctx.instance, "vkDestroyInstance"))
  if c.ctx.pipelineCache != nil and vkDestroyPipelineCachePtr != nil:
    vkDestroyPipelineCachePtr(c.ctx.device, c.ctx.pipelineCache, nil)
    c.ctx.pipelineCache = nil
  if c.ctx.commandPool != nil and vkDestroyCommandPoolPtr != nil:
    vkDestroyCommandPoolPtr(c.ctx.device, c.ctx.commandPool, nil)
    c.ctx.commandPool = nil
  if c.ctx.device != nil and vkDestroyDevicePtr != nil:
    vkDestroyDevicePtr(c.ctx.device, nil)
    c.ctx.device = nil
  if c.ctx.instance != nil and vkDestroyInstancePtr != nil:
    vkDestroyInstancePtr(c.ctx.instance, nil)
    c.ctx.instance = nil

proc `=destroy`(c: var VulkanCaches) =
  for p in c.pipelineCache.mvalues:
    p.destroyPipeline(c.ctx)
  `=destroy`(c.pipelineCache)
  `=destroy`(c.spirvCache)

proc newVulkanEngine(): VulkanEngine =
  let vctx = initVulkan()
  VulkanEngine(
    ctx: VulkanCtx(ctx: vctx),
    caches: VulkanCaches(ctx: vctx))

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: VulkanEngine, source: string) =
  ## Stores the source, glslangValidator → SPIR-V, re-entrant.
  if engine.spirv.len > 0:
    when defined(debug):
      echo "[INFO]: vulkan ingest: invalidating previous artifact"
  # Multi-kernel sources with by-value scalars misalign every kernel after
  # the first: the codegen unions all kernels' scalar params into one
  # file-scope push-constant block, the runtime packs only the invoked
  # kernel's scalars from offset 0. One kernel per source with scalars.
  if countKernels(source) > 1 and
     "layout(push_constant) uniform KernelParams" in source:
    failLoud("Vulkan: multi-kernel source with scalar params is unsupported, " &
         "use one kernel per source when passing by-value scalars")
  # The previous source built the cached pipelines and per-kernel SPIR-V,
  # release the pipelines (driver shader-compile objects) and drop both caches
  # before replacing the artifact.
  for p in engine.caches.pipelineCache.mvalues:
    p.destroyPipeline(engine.ctx.ctx)
  engine.caches.pipelineCache.clear()
  engine.caches.spirvCache.clear()
  engine.source = source
  engine.entryPoint = parseEntryPoint(source)
  engine.spirv = compileGlslToSpirV(source, engine.entryPoint)
  engine.bakedBlk = parseBakedBlk(source)

proc getArtifact*(engine: VulkanEngine): seq[uint32] =
  ## Returns the compiled SPIR-V.
  engine.spirv

proc deviceName*(engine: VulkanEngine): string {.inline.} =
  ## Returns the physical device name (e.g. "NVIDIA RTX PRO 6000 Blackwell ...").
  engine.ctx.ctx.deviceName(engine.ctx.ctx.physicalDevice)

# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────

proc parseBakedBlk(glsl: string): Dim3 =
  ## Extract the shader-baked workgroup size from the GLSL preamble:
  ## `layout(local_size_x = 256, local_size_y = 8, local_size_z = 1) in;`
  ## Returns (0, 0, 0) when absent (run will then fail blk validation loudly).
  const markers = ["local_size_x = ", "local_size_y = ", "local_size_z = "]
  for i, marker in markers:
    let j = glsl.find(marker)
    if j < 0:
      return Dim3(x: 0, y: 0, z: 0)
    var k = j + marker.len
    var n = 0
    while k < glsl.len and glsl[k] in {'0' .. '9'}:
      n = n * 10 + (ord(glsl[k]) - ord('0'))
      inc k
    case i
    of 0: result.x = n
    of 1: result.y = n
    else: result.z = n

proc countKernels(glsl: string): int =
  ## Counts kernel entry points, each kernel carries a `layout(local_size_x = ...) in;` preamble.
  const marker = "layout(local_size_x"
  var i = glsl.find(marker)
  while i >= 0:
    inc result
    i = glsl.find(marker, i + marker.len)

proc parseEntryPoint(glsl: string): string =
  ## Extracts the first kernel's entry point name from the GLSL.
  ## The codegen emits `layout(local_size_x ...) in;` before each kernel,
  ## device helpers are forward-declared, the first bare `void ` may name a helper.
  ## Without a kernel preamble the fallback is the first `void `, then "main".
  const preamble = "layout(local_size_x"
  const marker = "void "
  var startPos = 0
  let p = glsl.find(preamble)
  if p >= 0:
    startPos = p
  let i = glsl.find(marker, startPos)
  if i < 0:
    return "main"
  var j = i + marker.len
  while j < glsl.len and glsl[j] in {' ', '\t', '\n', '\r'}:
    inc j
  var name = ""
  while j < glsl.len and glsl[j] in {'a' .. 'z', 'A' .. 'Z', '0' .. '9', '_'}:
    name.add glsl[j]
    inc j
  if name.len == 0:
    name = "main"
  name


proc launchKernelImpl*(engine: VulkanEngine, kernel: string, output: VulkanBuffer, args: tuple, grid, blk = Dim3()) =
  ## Runs `kernel` on the engine's device, building the pipeline on first use:
  ## - `output` receives the kernel's result, `args` bind positionally
  ## - `grid`/`blk` carry the launch geometry, `blk` must match the shader's baked workgroup size
  ## - a plain run dispatches (1, 1)
  ## - example: engine.launchKernel<<(1024, 256)>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
  var vctx = engine.ctx.ctx

  # blk is shader-baked (local_size_xyz), the default blk (1, 1, 1) dispatches the baked size,
  # an explicit blk must match it exactly.
  let blk = if blk.x == 1 and blk.y == 1 and blk.z == 1:
              engine.bakedBlk
            else:
              blk
  if engine.bakedBlk.x == 0 or
     blk.x != engine.bakedBlk.x or blk.y != engine.bakedBlk.y or
     blk.z != engine.bakedBlk.z:
    failLoud("Vulkan run blk=" & $blk.x & "x" & $blk.y & "x" & $blk.z &
         " != baked workgroup size " & $engine.bakedBlk.x & "x" & $engine.bakedBlk.y &
         "x" & $engine.bakedBlk.z &
         ": launch config mismatch, blk is shader-baked on Vulkan)")

  # Entry point: the ingested SPIR-V serves the baked entry, other kernels
  # compile once per kernel name and cache.
  var entry = engine.entryPoint
  var spirv: seq[uint32]
  if kernel == engine.entryPoint:
    spirv = engine.spirv
  else:
    if not engine.caches.spirvCache.hasKey(kernel):
      engine.caches.spirvCache[kernel] = compileGlslToSpirV(engine.source, kernel)
    spirv = engine.caches.spirvCache[kernel]
    entry = kernel

  # byte chunks pack into the 128-byte push-constant block, buffers
  # cross as storage bindings
  type SinkCtx = object
    pushBuf: array[128, byte]
    pn: int
    inputBuffers: seq[VulkanBuffer]
  proc sinkArg(ctx: var SinkCtx, isBuffer: bool, h: VulkanBuffer, idx: int, p: pointer, len: int) {.nimcall.} =
    if isBuffer:
      if idx > 0:
        ctx.inputBuffers.add h
    else:
      if len mod 4 != 0:
        failLoud("Vulkan by-value (push-constant) args must be 4-byte-aligned std430 types, got " & $len & " bytes")
      if ctx.pn + len > 128:
        failLoud("Vulkan push constants exceed the spec-guaranteed 128-byte max")
      copyMem(ctx.pushBuf[ctx.pn].addr, p, len)
      inc ctx.pn, len
  var dctx = SinkCtx()
  bindArgs(dctx, bkVulkan, output, args, false, sinkArg)
  let pushConstLen = dctx.pn

  # The pipeline layout depends only on the shader module, the SSBO count
  # (output at binding 0 + buffer inputs) and the push-constant size.
  # The fully-created pipeline is cached by (kernel, ssboCount, pushConstSize)
  # and reused across runs.
  let pipelineKey = (kernel, dctx.inputBuffers.len + 1, pushConstLen)
  if not engine.caches.pipelineCache.hasKey(pipelineKey):
    # The shader module is only needed when building a new pipeline
    let vkCreateShaderModule = cast[
      proc(device: VkDevice, pCreateInfo: ptr VkShaderModuleCreateInfo,
           pAllocator: pointer,
           pShaderModule: ptr VkShaderModule): VkResult {.cdecl.}
    ](vctx.gpaAddr(vctx.instance, "vkCreateShaderModule"))
    var smCI = VkShaderModuleCreateInfo(
      sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      codeSize: csize_t(spirv.len * sizeof(uint32)),
      pCode: spirv[0].addr
    )
    var shaderModule: VkShaderModule
    check vkCreateShaderModule(vctx.device, smCI.addr, nil, shaderModule.addr)
    defer:
      let vkDestroyShaderModule = cast[
        proc(device: VkDevice, shaderModule: VkShaderModule, pAllocator: pointer) {.cdecl.}
      ](vctx.gpaAddr(vctx.instance, "vkDestroyShaderModule"))
      if shaderModule != nil and vkDestroyShaderModule != nil:
        vkDestroyShaderModule(vctx.device, shaderModule, nil)
    var pipeline = vctx.createPipeline(shaderModule, dctx.inputBuffers.len + 1, entry,
                                     pushConstLen)
    engine.caches.pipelineCache[pipelineKey] = pipeline

  # Output first (binding 0), then buffer inputs
  engine.caches.pipelineCache[pipelineKey].setArg(0, output, vctx)
  for i in 0 ..< dctx.inputBuffers.len:
    engine.caches.pipelineCache[pipelineKey].setArg(i + 1, dctx.inputBuffers[i], vctx)

  # Dispatch grid groups (runKernel computes groupCount = ceil(global/local))
  vctx.runKernel(engine.caches.pipelineCache[pipelineKey],
                 [uint32(grid.x * blk.x),
                  uint32(grid.y * blk.y),
                  uint32(grid.z * blk.z)],
                 [uint32(blk.x), uint32(blk.y), uint32(blk.z)],
                 if pushConstLen > 0: dctx.pushBuf[0].addr else: nil,
                 pushConstLen)

proc allocDevice*(engine: VulkanEngine, n: int): VulkanBuffer =
  var vctx = engine.ctx.ctx
  vctx.allocBuffer(n)

proc freeDevice*(engine: VulkanEngine, h: VulkanBuffer) =
  var h = h
  var vctx = engine.ctx.ctx
  h.dealloc(vctx)

proc copyToDeviceFromHost*(engine: VulkanEngine, dst: VulkanBuffer, src: pointer, n: int) =
  var dst = dst
  var vctx = engine.ctx.ctx
  vctx.writeBuffer(dst, src, n)

proc copyToHostFromDevice*(engine: VulkanEngine, dst: pointer, src: VulkanBuffer, n: int) =
  var vctx = engine.ctx.ctx
  let res = readBuffer[byte](vctx, src)
  copyMem(dst, res[0].addr, n)
