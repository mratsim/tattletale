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
import ../launch/launch_serialization

export vulkan_runtime
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  VulkanCaches = object
    spirvCache: Table[string, seq[uint32]]
    pipelineCache: Table[(string, int, int), VulkanPipeline]

  VulkanEngine* = ref object
    source: string
    spirv: seq[uint32]
    entryPoint: string
    ctx: VulkanContext
    shaderSquadron: Dim3   # squadron baked into the shader at ingest
    caches: VulkanCaches

proc backendKind*(_: typedesc[VulkanEngine]): static BackendKind = bkVulkan

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════

proc newVulkanEngine(): VulkanEngine =
  let vctx = initVulkan()
  VulkanEngine(
    ctx: vctx,
    caches: VulkanCaches())

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: VulkanEngine, source: string) =
  ## Stores the source, glslangValidator → SPIR-V, re-entrant.
  if engine.spirv.len > 0:
    when defined(debug):
      echo "[INFO]: vulkan ingest: invalidating previous artifact"

  if countKernels(source) > 1 and "layout(push_constant) uniform KernelParams" in source:
    failLoud("Vulkan: multi-kernel source with scalar params is unsupported, " &
         "use one kernel per source when passing by-value scalars")

  engine.caches.pipelineCache = initTable[(string, int, int), VulkanPipeline]()
  engine.caches.spirvCache.clear()
  engine.source = source
  engine.entryPoint = parseEntryPoint(source)
  engine.spirv = compileGlslToSpirV(source, engine.entryPoint)
  engine.shaderSquadron = parseShaderSquadron(source)

proc getArtifact*(engine: VulkanEngine): seq[uint32] =
  engine.spirv

proc deviceName*(engine: VulkanEngine): string {.inline.} =
  engine.ctx.deviceName(engine.ctx.physicalDevice)

# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────

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


proc launchKernelImpl*(engine: VulkanEngine, kernel: string, output: VulkanBuffer, args: tuple, grid, squadron = Dim3()) =
  let squadron = if squadron.x == 1 and squadron.y == 1 and squadron.z == 1:
              engine.shaderSquadron
            else:
              squadron

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

  type SerializationCtx = object
    pushBuf: array[128, byte]
    pn: int
    inputBuffers: seq[VulkanBuffer]

  proc serializer(ctx: var SerializationCtx, bufferHandle: VulkanBuffer, p: pointer, len: int, argIdx: int, viaBuffer: bool) {.nimcall.} =
    if viaBuffer:
      if argIdx > 0:
        ctx.inputBuffers.add bufferHandle
    else:
      if len mod 4 != 0:
        failLoud("Vulkan by-value (push-constant) args must be 4-byte-aligned std430 types, got " & $len & " bytes")
      if ctx.pn + len > 128:
        failLoud("Vulkan push constants exceed the spec-guaranteed 128-byte max")
      copyMem(ctx.pushBuf[ctx.pn].addr, p, len)
      inc ctx.pn, len

  var dctx = SerializationCtx()
  bindArgs(dctx, output, args, serializer, bkVulkan, false)
  let pushConstLen = dctx.pn

  let pipelineKey = (kernel, dctx.inputBuffers.len + 1, pushConstLen)
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
    var pipeline = engine.ctx.createPipeline(shaderModule, dctx.inputBuffers.len + 1, entry,
                                     pushConstLen)
    engine.caches.pipelineCache[pipelineKey] = pipeline

  # Output first (binding 0), then buffer inputs
  engine.caches.pipelineCache[pipelineKey].setArg(0, output, engine.ctx)
  for i in 0 ..< dctx.inputBuffers.len:
    engine.caches.pipelineCache[pipelineKey].setArg(i + 1, dctx.inputBuffers[i], engine.ctx)

  # Dispatch grid groups (runKernel computes groupCount = ceil(global/local))
  engine.ctx.runKernel(engine.caches.pipelineCache[pipelineKey],
                 [uint32(grid.x * squadron.x),
                  uint32(grid.y * squadron.y),
                  uint32(grid.z * squadron.z)],
                 [uint32(squadron.x), uint32(squadron.y), uint32(squadron.z)],
                 if pushConstLen > 0: dctx.pushBuf[0].addr else: nil,
                 pushConstLen)

proc allocDevice*(engine: VulkanEngine, n: int): VulkanBuffer =
  engine.ctx.allocBuffer(n)

proc freeDevice*(engine: VulkanEngine, h: VulkanBuffer) =
  var h = h
  h.dealloc(engine.ctx)

proc copyToDeviceFromHost*(engine: VulkanEngine, dst: VulkanBuffer, src: pointer, n: int) =
  var dst = dst
  engine.ctx.writeBuffer(dst, src, n)

proc copyToHostFromDevice*(engine: VulkanEngine, dst: pointer, src: VulkanBuffer, n: int) =
  let res = readBuffer[byte](engine.ctx, src)
  copyMem(dst, res[0].addr, n)
