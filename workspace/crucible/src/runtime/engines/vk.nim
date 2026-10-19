# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/[strutils, tables]

import workspace/crucible/src/abis/vulkan_abi as vk

import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/runtime/errors
import workspace/crucible/src/runtime/exec/vulkan_runtime
import workspace/crucible/src/runtime/launch/launch_serialization
import workspace/crucible/src/runtime/engines/engines_common

# ════════════════════════════════════════════════════════════════════════
# ▸ Types
# ════════════════════════════════════════════════════════════════════════

type
  VulkanCaches = object
    spirvCache: Table[string, seq[uint32]]
    pipelineCache: Table[(string, int), VulkanPipeline]

  VulkanEngine* = ref object
    source: string
    spirv: seq[uint32]
    entryPoint: string
    ctx: VulkanContext
    shaderSquadron: Dim3   # squadron baked into the shader at ingest
    caches: VulkanCaches

template backendKind*(_: typedesc[VulkanEngine]): BackendKind = bkVulkan

# ════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ════════════════════════════════════════════════════════════════════════

proc ingest*(engine: VulkanEngine, source: string) =
  ## Stores the source, glslangValidator → SPIR-V, re-entrant.
  if engine.spirv.len > 0:
    when defined(debug):
      echo "[INFO]: vulkan ingest: invalidating previous artifact"

  # multi-kernel sources cannot address per-kernel scalars in the shared
  # deduped push block, ingest quits loudly
  if countKernels(source) > 1:
    var inPush = false
    for line in source.splitLines():
      if "layout(push_constant" in line:
        inPush = true
      elif inPush and line.strip().startsWith("}"):
        inPush = false
      elif inPush:
        let t = line.strip()
        if t.len > 0 and t.endsWith(";") and not t.startsWith("Ref_"):
          failLoud("Vulkan: multi-kernel source with scalar params is unsupported, " &
                   "use one kernel per source when passing by-value scalars")

  engine.caches.pipelineCache = initTable[(string, int), VulkanPipeline]()
  engine.caches.spirvCache.clear()
  engine.source = source
  engine.entryPoint = parseEntryPoint(source)
  engine.spirv = compileGlslToSpirV(source, engine.entryPoint)
  engine.shaderSquadron = parseShaderSquadron(source)

proc getArtifact*(engine: VulkanEngine): string {.inline.} =
  ## Ingested GLSL source, the ingest-if-differs identity
  engine.source

proc deviceName*(engine: VulkanEngine): string {.inline.} =
  engine.ctx.deviceName(engine.ctx.physicalDevice)

proc allocDevice*[T](engine: VulkanEngine, _: typedesc[T], len: int): DeviceMem[VulkanBuffer, T] {.inline.} =
  ## Allocates len elements of T on the engine
  DeviceMem[VulkanBuffer, T](mem: engine.allocDeviceImpl(len * sizeof(T)))

proc copyToDeviceFromHost*[T](engine: VulkanEngine, dst: DeviceMem[VulkanBuffer, T], src: ptr T or ptr UncheckedArray[T], n: int) {.inline.} =
  ## Copies n elements from host to device
  engine.copyToDeviceFromHostImpl(dst.mem, src, n * sizeof(T))

proc copyToHostFromDevice*[T](engine: VulkanEngine, dst: ptr T or ptr UncheckedArray[T], src: DeviceMem[VulkanBuffer, T], n: int) {.inline.} =
  ## Copies n elements from device to host
  engine.copyToHostFromDeviceImpl(dst, src.mem, n * sizeof(T))

# ════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ════════════════════════════════════════════════════════════════════════

proc newVulkanEngine*(): VulkanEngine =
  let vctx = initVulkan()
  VulkanEngine(
    ctx: vctx,
    caches: VulkanCaches())

# ════════════════════════════════════════════════════════════════════════
# ▸ Launch serialization
# ════════════════════════════════════════════════════════════════════════

type SerializationCtx = object
  vctx: VulkanContext
  pushBuf: array[128, byte]
  pn: int

proc write8(ctx: var SerializationCtx, x: uint64) =
  ctx.pn = roundStepUp(ctx.pn, 8)
  if ctx.pn + 8 > 128:
    failLoud("Vulkan push constants exceed the spec-guaranteed 128-byte max")
  copyMem(ctx.pushBuf[ctx.pn].addr, x.unsafeAddr, 8)
  inc ctx.pn, 8

proc store(ctx: var SerializationCtx, v: VulkanBuffer, argIdx: int) =
  ctx.write8(vulkan_runtime.deviceAddress(v, ctx.vctx))

proc store[T](ctx: var SerializationCtx, v: T, argIdx: int) =
  when T is ptr|pointer:
    {.error: "Pointer kernel args are not supported, pass the DeviceMem handle from allocDevice, got " & $T.}
  const lay = deviceLayout(T, bkVulkan)
  if lay.align > 1:
    ctx.pn = roundStepUp(ctx.pn, lay.align)
  if ctx.pn + lay.size > 128:
    failLoud("Vulkan push constants exceed the spec-guaranteed 128-byte max")
  var blob: array[lay.size, byte]
  var off = 0
  when T is object|tuple and hasBufferLeaf(T, VulkanBuffer):
    for f in fields(v):
      when typeof(f) is DeviceMem:
        off = roundStepUp(off, 8)
        let a = deviceAddress(f.mem, ctx.vctx)
        copyMem(blob[off].addr, a.unsafeAddr, 8)
        inc off, 8
      else:
        blob.packFrom(off, f, bkVulkan)
    off = roundStepUp(off, lay.align)
    doAssert off == lay.size,
      "host packing of " & $T & " does not match the device layout"
  else:
    blob.packFrom(off, v, bkVulkan)
  if lay.size > 0:
    copyMem(ctx.pushBuf[ctx.pn].addr, blob[0].addr, lay.size)
  inc ctx.pn, lay.size

# ════════════════════════════════════════════════════════════════════════
# ▸ PRIVATE
# ════════════════════════════════════════════════════════════════════════

proc parseShaderSquadron(glsl: string): Dim3 =
  ## Extract the shader-baked squadron from the GLSL preamble:
  ## `layout(local_size_x = 256, local_size_y = 8, local_size_z = 1) in;`
  ## Returns (0, 0, 0) when absent (run will then fail squadron validation loudly).
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
  const marker = "layout(local_size_x"
  var i = glsl.find(marker)
  while i >= 0:
    inc result
    i = glsl.find(marker, i + marker.len)

proc parseEntryPoint(glsl: string): string =
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

proc launchKernelImpl*(engine: VulkanEngine, kernel: string, output: auto, args: var tuple, cfg: LaunchConfig) =
  if cfg.sharedMem != 0 or cfg.stream != 0:
    failLoud("Vulkan run: launch-time sharedMem/stream are unsupported, they are Metal/CUDA features")
  let squadron = if cfg.blk.x == 1 and cfg.blk.y == 1 and cfg.blk.z == 1:
              engine.shaderSquadron
            else:
              cfg.blk

  if engine.shaderSquadron.x == 0 or
     squadron.x != engine.shaderSquadron.x or squadron.y != engine.shaderSquadron.y or
     squadron.z != engine.shaderSquadron.z:
    failLoud("Vulkan run squadron=" & $squadron.x & "x" & $squadron.y & "x" & $squadron.z &
         " != baked squadron " & $engine.shaderSquadron.x & "x" & $engine.shaderSquadron.y &
         "x" & $engine.shaderSquadron.z &
         ": launch config mismatch, squadron is shader-baked on Vulkan)")

  # Entrypoint caching
  var entry = engine.entryPoint
  var spirv: seq[uint32]
  if kernel == engine.entryPoint:
    spirv = engine.spirv
  else:
    if not engine.caches.spirvCache.hasKey(kernel):
      engine.caches.spirvCache[kernel] = compileGlslToSpirV(engine.source, kernel)
    spirv = engine.caches.spirvCache[kernel]
    entry = kernel

  var dctx = SerializationCtx(vctx: engine.ctx)
  bindArgs(dctx, output, args, store, VulkanBuffer, bkVulkan)
  let pushConstLen = dctx.pn

  let pipelineKey = (kernel, pushConstLen)
  if not engine.caches.pipelineCache.hasKey(pipelineKey):
    let vkCreateShaderModule = cast[
      proc(device: VkDevice, pCreateInfo: ptr VkShaderModuleCreateInfo,
           pAllocator: pointer,
           pShaderModule: ptr VkShaderModule): VkResult {.cdecl.}
    ](engine.ctx.gpaAddr(engine.ctx.instance, "vkCreateShaderModule"))
    var smCI = VkShaderModuleCreateInfo(
      sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      codeSize: csize_t(spirv.len * sizeof(uint32)),
      pCode: spirv[0].addr
    )
    var shaderModule: VkShaderModule
    check vkCreateShaderModule(engine.ctx.device, smCI.addr, nil, shaderModule.addr)
    defer:
      let vkDestroyShaderModule = cast[
        proc(device: VkDevice, shaderModule: VkShaderModule, pAllocator: pointer) {.cdecl.}
      ](engine.ctx.gpaAddr(engine.ctx.instance, "vkDestroyShaderModule"))
      if shaderModule != nil and vkDestroyShaderModule != nil:
        vkDestroyShaderModule(engine.ctx.device, shaderModule, nil)
    var pipeline = engine.ctx.createPipeline(shaderModule, entry, pushConstLen)
    engine.caches.pipelineCache[pipelineKey] = pipeline

  # Dispatch cfg.grid groups (runKernel computes groupCount = ceil(global/local))
  engine.ctx.runKernel(engine.caches.pipelineCache[pipelineKey],
                 [uint32(cfg.grid.x * squadron.x),
                  uint32(cfg.grid.y * squadron.y),
                  uint32(cfg.grid.z * squadron.z)],
                 [uint32(squadron.x), uint32(squadron.y), uint32(squadron.z)],
                 if pushConstLen > 0: dctx.pushBuf[0].addr else: nil,
                 pushConstLen)

proc allocDeviceImpl(engine: VulkanEngine, n: int): VulkanBuffer {.inline.} =
  engine.ctx.allocBuffer(n)

proc copyToDeviceFromHostImpl(engine: VulkanEngine, dst: VulkanBuffer, src: pointer, n: int) =
  var dst = dst
  engine.ctx.writeBuffer(dst, src, n)

proc copyToHostFromDeviceImpl(engine: VulkanEngine, dst: pointer, src: VulkanBuffer, n: int) =
  let res = readBuffer[byte](engine.ctx, src)
  copyMem(dst, res[0].addr, n)
