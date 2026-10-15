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
import ../chevrons

export vulkan_runtime
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  VulkanCtx = object
    ## RAII value wrapper, `=destroy` fires when the engine ref dies.
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
  if c.ctx.instance != nil:
    c.ctx.shutdown()

proc `=destroy`(c: var VulkanCaches) =
  ## Releases the cached pipelines before the engine's ctx shuts down,
  ## reverse declaration order destroys the caches first.
  ## The custom `=destroy` releases the tables explicitly.
  for p in c.pipelineCache.mvalues:
    p.destroyPipeline(c.ctx)
  `=destroy`(c.pipelineCache)
  `=destroy`(c.spirvCache)

proc newVulkanEngine(): VulkanEngine =
  ## Private factory, engines.nim reaches it via `import {.all.}`.
  let vctx = initVulkan()
  VulkanEngine(
    ctx: VulkanCtx(ctx: vctx),
    caches: VulkanCaches(ctx: vctx))

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: VulkanEngine, source: string) =
  ## glslangValidator → SPIR-V. Re-entrant: replaces the previous artifact
  ## (the device context persists, only the compiled artifact is replaced).
  if engine.spirv.len > 0:
    when defined(debug):
      echo "[INFO]: vulkan ingest: invalidating previous artifact"
  # Multi-kernel sources with by-value scalars misalign every kernel after
  # the first: the codegen unions all kernels' scalar params into one
  # file-scope push-constant block, the runtime packs only the invoked
  # kernel's scalars from offset 0. One kernel per source with scalars.
  if countKernels(source) > 1 and
     "layout(push_constant) uniform KernelParams" in source:
    quit("Vulkan: multi-kernel source with scalar params is unsupported, " &
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


proc run*(engine: VulkanEngine, kernel: string, output: VulkanBuffer, args: tuple, cfg = LaunchConfig()) =
  ## Runs `kernel` on the engine's device, building the pipeline on first use.
  ##
  ## Usage:
  ##   engine.run<<1024, 256>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
  ##
  ## `output` is the kernel's first argument, then the args in declaration order.
  ## Device buffers pass as-is, scalars copy by value into the push-constant block.
  ##
  ## Host seqs and arrays are not launch args:
  ## allocDevice + copyToDeviceFromHost upload them, copyToHostFromDevice reads results back.
  ##
  ## Plain `run` dispatches with the shader's workgroup size, chevron
  ## `run<<grid, blk>>` sets the grid, blk must match the shader's baked workgroup size.
  var vctx = engine.ctx.ctx

  # blk is shader-baked (local_size_xyz), a default cfg dispatches the baked size,
  # an explicit blk must match it exactly.
  let blk = if cfg.blk.x == 1 and cfg.blk.y == 1 and cfg.blk.z == 1:
              engine.bakedBlk
            else:
              cfg.blk
  if engine.bakedBlk.x == 0 or
     blk.x != engine.bakedBlk.x or blk.y != engine.bakedBlk.y or
     blk.z != engine.bakedBlk.z:
    quit("Vulkan run blk=" & $blk.x & "x" & $blk.y & "x" & $blk.z &
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

  var pushConstBytes = newSeq[byte]()
  var inputBuffers = newSeq[VulkanBuffer]()
  var argsCopy = args
  for f in fields(argsCopy):
    when typeof(f) is VulkanBuffer:
      inputBuffers.add f
    elif typeof(f) is bool:
      # std430 push-constant scalars are 4-byte aligned
      let v = int32(f)
      for j in 0 ..< sizeof(int32):
        pushConstBytes.add cast[ptr UncheckedArray[byte]](v.addr)[j]
    else:
      if sizeof(f) mod 4 != 0:
        quit("Vulkan by-value (push-constant) args must be 4-byte-aligned " &
             "std430 types, got " & $sizeof(f) & " bytes")
      for j in 0 ..< sizeof(f):
        pushConstBytes.add cast[ptr UncheckedArray[byte]](f.addr)[j]
  if pushConstBytes.len > 128:
    quit("Vulkan push constants exceed the spec-guaranteed 128-byte max (got " &
         $pushConstBytes.len & " bytes)")

  # The pipeline layout depends only on the shader module, the SSBO count
  # (output at binding 0 + buffer inputs) and the push-constant size.
  # The fully-created pipeline is cached by (kernel, ssboCount, pushConstSize)
  # and reused across runs.
  let pipelineKey = (kernel, inputBuffers.len + 1, pushConstBytes.len)
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
    var pipeline = vctx.createPipeline(shaderModule, inputBuffers.len + 1, entry,
                                     pushConstBytes.len)
    engine.caches.pipelineCache[pipelineKey] = pipeline

  # Output first (binding 0), then buffer inputs
  engine.caches.pipelineCache[pipelineKey].setArg(0, output, vctx)
  for i in 0 ..< inputBuffers.len:
    engine.caches.pipelineCache[pipelineKey].setArg(i + 1, inputBuffers[i], vctx)

  # Dispatch cfg.grid groups (runKernel computes groupCount = ceil(global/local))
  vctx.runKernel(engine.caches.pipelineCache[pipelineKey],
                 [uint32(cfg.grid.x * blk.x),
                  uint32(cfg.grid.y * blk.y),
                  uint32(cfg.grid.z * blk.z)],
                 [uint32(blk.x), uint32(blk.y), uint32(blk.z)],
                 if pushConstBytes.len > 0: pushConstBytes[0].addr else: nil,
                 pushConstBytes.len)

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
