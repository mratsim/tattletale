# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/[strutils, tables]

import workspace/crucible/src/abis/vulkan_abi as vk

import workspace/crucible/src/crucible_codegen
import workspace/crucible/src/codegen/backends/spirv_encoder
import workspace/crucible/src/runtime/backends/vulkan_common
import workspace/crucible/src/runtime/backends/vulkan_compile {.all.}
import workspace/crucible/src/runtime/backends/vulkan_address_runtime {.all.}
import workspace/crucible/src/runtime/backends/vulkan_descriptor_runtime
import workspace/crucible/src/runtime/backends/runtime_utils
import workspace/crucible/src/runtime/launch/launch_serialization
import workspace/crucible/src/runtime/engines/engines_common

# ═══════════════════════════════════════════════════════════════════════
# ▸ Types
# ═══════════════════════════════════════════════════════════════════════

type
  VulkanCaches = object
    spirvCache: Table[string, SpirvWords]
    pipelineCache: Table[(string, int), VulkanPipeline]

  VulkanEngine* = ref object
    source: string
    spirv: SpirvWords
    entryPoint: string
    ctx: VulkanContext
    shaderSquadron: Dim3   # squadron baked into the shader at ingest
    entrySquadrons: Table[string, Dim3]  # LocalSize per entrypoint of a direct SPIR-V module, empty for source ingests
    isHlsl: bool
    caches: VulkanCaches

proc newVulkanEngine*(): VulkanEngine =
  let vctx = newVulkanContext()
  VulkanEngine(
    ctx: vctx,
    caches: VulkanCaches())

# ═══════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

proc resetCaches(engine: VulkanEngine) =
  if engine.spirv.len > 0 or engine.source.len > 0:
    when defined(debug):
      echo "[INFO]: vulkan ingest: invalidating previous artifact"
  engine.caches.pipelineCache = initTable[(string, int), VulkanPipeline]()
  engine.caches.spirvCache.clear()
  engine.entrySquadrons.clear()

proc ingest*(engine: VulkanEngine, source: string) =
  resetCaches(engine)
  engine.source = source
  if source.startsWith("#version"):
    ingestGlsl(engine, source)
  elif "[numthreads(" in source:
    ingestHlsl(engine, source)
  else:
    raiseAssert("Vulkan ingest: unrecognized source language, expected GLSL " &
             "(a `#version` first line) or HLSL (`[numthreads(`), got: " &
             source.splitLines()[0])

proc countKernels(glsl: string): int =
  const marker = "layout(local_size_x"
  var i = glsl.find(marker)
  while i >= 0:
    inc result
    i = glsl.find(marker, i + marker.len)

proc ingestGlsl(engine: VulkanEngine, glsl: string) =
  engine.entryPoint = findFirstFunctionName(glsl, "layout(local_size_x", "void ")
  if countKernels(glsl) > 1:
    # A member's offset in the deduped push block must be launch-invariant.
    # Only scalar members can violate it, so multi-kernel sources refuse at ingest.
    var inPush = false
    for line in glsl.splitLines():
      if "layout(push_constant" in line:
        inPush = true
      elif inPush and line.strip().startsWith("}"):
        inPush = false
      elif inPush:
        let t = line.strip()
        if t.len > 0 and t.endsWith(";") and not t.startsWith("Ref_"):
          raiseAssert("Vulkan: multi-kernel source with scalar params is unsupported, " &
                   "use one kernel per source when passing by-value scalars")
  engine.spirv = compileGlslToSpirvViaGlslang(glsl, engine.entryPoint)
  engine.shaderSquadron = parseShaderSquadron(glsl)
  engine.isHlsl = false

proc ingestHlsl(engine: VulkanEngine, hlsl: string) =
  engine.entryPoint = findFirstFunctionName(hlsl, "[numthreads(", "void ")
  engine.spirv = compileHlslToSpirvViaDxc(hlsl, engine.entryPoint)
  engine.shaderSquadron = parseHlslSquadron(hlsl)
  engine.isHlsl = true

proc ingest*(engine: VulkanEngine, spirv: SpirvWords) =
  ## Ingests a ready-made SPIR-V module produced by a `spirv:` block.
  const spvMagic = 0x07230203'u32
  resetCaches(engine)
  if spirv.len < 5 or spirv[0] != spvMagic:
    raiseAssert("Vulkan ingest: not a SPIR-V module, expected the 0x07230203 " &
             "magic as word 0 and the 5-word header, got " &
             (if spirv.len == 0: "an empty module"
              else: $spirv.len & " words, word 0 = " & spirv[0].toHex()))
  let sq = parseSpirvEntryDescriptor(spirv)
  if sq.entryPoints == 0:
    raiseAssert("Vulkan ingest: the module declares no OpEntryPoint")
  if sq.entryPoints > 1:
    engine.entrySquadrons = parseSpirvEntrySizes(spirv)
  engine.source = ""
  engine.spirv = spirv
  engine.entryPoint = sq.name
  engine.shaderSquadron = sq.localSize
  engine.isHlsl = false

proc getArtifact*(engine: VulkanEngine): string {.inline.} =
  engine.source

func getSpirV*(engine: VulkanEngine): SpirvWords {.inline.} =
  engine.spirv

proc deviceName*(engine: VulkanEngine): string {.inline.} =
  engine.ctx.deviceName(engine.ctx.physicalDevice)

proc allocDevice*[T](engine: VulkanEngine, _: typedesc[T], len: int): DeviceMem[VulkanBuffer, T] {.inline.} =
  DeviceMem[VulkanBuffer, T](mem: engine.allocDeviceImpl(len * sizeof(T)))

proc copyToDeviceFromHost*[T](engine: VulkanEngine, dst: DeviceMem[VulkanBuffer, T], src: openArray[T]) {.inline.} =
  engine.copyToDeviceFromHostImpl(dst.mem, cast[ptr T](unsafeAddr src[0]), src.len * sizeof(T))

proc copyToHostFromDevice*[T](engine: VulkanEngine, dst: var openArray[T], src: DeviceMem[VulkanBuffer, T], offsetElems: int = 0) {.inline.} =
  engine.copyToHostFromDeviceImpl(cast[ptr T](addr dst[0]), src.mem, dst.len * sizeof(T), offsetElems * sizeof(T))

# ═══════════════════════════════════════════════════════════════════════
# ▸ Launch serialization
# ═══════════════════════════════════════════════════════════════════════

type SerializationCtx = object
  vctx: VulkanContext
  pushBuf: array[128, byte]
  pn: int

proc write8(ctx: var SerializationCtx, x: uint64) =
  ctx.pn = ctx.pn.round_multiple_up(8)
  if ctx.pn + 8 > 128:
    raiseAssert("Vulkan push constants exceed the spec-guaranteed 128-byte max")
  copyMem(ctx.pushBuf[ctx.pn].addr, x.unsafeAddr, 8)
  inc ctx.pn, 8

proc store(ctx: var SerializationCtx, v: VulkanBuffer, _: int) =
  ctx.write8(deviceAddress(v, ctx.vctx))

proc store[T](ctx: var SerializationCtx, v: T, _: int) =
  when T is ptr|pointer:
    {.error: "Pointer kernel args are not supported, pass the DeviceMem handle from allocDevice, got " & $T.}
  const lay = computeDeviceLayout(T, cgGlsl)
  if lay.align > 1:
    ctx.pn = ctx.pn.round_multiple_up(lay.align)
  if ctx.pn + lay.size > 128:
    raiseAssert("Vulkan push constants exceed the spec-guaranteed 128-byte max")
  var blob: array[lay.size, byte]
  var off = 0
  when T is object|tuple and hasBufferLeaf(T, VulkanBuffer):
    for f in fields(v):
      when typeof(f) is DeviceMem:
        off = off.round_multiple_up(8)
        let a = deviceAddress(f.mem, ctx.vctx)
        copyMem(blob[off].addr, a.unsafeAddr, 8)
        inc off, 8
      else:
        blob.packFrom(off, f, cgGlsl)
    off = off.round_multiple_up(lay.align)
    doAssert off == lay.size,
      "host packing of " & $T & " does not match the device layout"
  else:
    blob.packFrom(off, v, cgGlsl)
  if lay.size > 0:
    copyMem(ctx.pushBuf[ctx.pn].addr, blob[0].addr, lay.size)
  inc ctx.pn, lay.size

# ═══════════════════════════════════════════════════════════════════════
# ▸ PRIVATE
# ═══════════════════════════════════════════════════════════════════════

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

proc parseHlslSquadron(hlsl: string): Dim3 =
  const marker = "[numthreads("
  let i = hlsl.find(marker)
  if i < 0:
    return Dim3(x: 0, y: 0, z: 0)
  var fields: array[3, int]
  var f = 0
  var n = 0
  var k = i + marker.len
  while k < hlsl.len:
    let c = hlsl[k]
    if c in {'0' .. '9'}:
      n = n * 10 + (ord(c) - ord('0'))
    elif c == ',':
      fields[f] = n
      n = 0
      inc f
      if f > 2:
        return Dim3(x: 0, y: 0, z: 0)
    elif c == ')':
      fields[f] = n
      return Dim3(x: fields[0], y: fields[1], z: fields[2])
    elif c notin {' ', '\t'}:
      return Dim3(x: 0, y: 0, z: 0)
    inc k
  return Dim3(x: 0, y: 0, z: 0)

proc findFirstFunctionName(src: string, preamble, marker: string): string =
  ## Reads the function name at the first `marker` after `preamble`,
  ## "main" when the marker is missing or the scan finds no identifier.
  var startPos = 0
  let p = src.find(preamble)
  if p >= 0:
    startPos = p
  let i = src.find(marker, startPos)
  if i < 0:
    return "main"
  var j = i + marker.len
  while j < src.len and src[j] in {' ', '\t', '\n', '\r'}:
    inc j
  var name = ""
  while j < src.len and src[j] in {'a' .. 'z', 'A' .. 'Z', '0' .. '9', '_'}:
    name.add src[j]
    inc j
  if name.len == 0:
    name = "main"
  name

proc launchKernelImpl*(engine: VulkanEngine, output: auto, args: tuple, cfg: LaunchConfig, kernel: string) =
  if cfg.sharedMem != 0 or cfg.stream != 0:
    raiseAssert("Vulkan run: launch-time sharedMem/stream are unsupported, they are Metal/CUDA features")
  var entry = engine.entryPoint
  var spirv: SpirvWords
  var baked = engine.shaderSquadron
  if engine.entrySquadrons.len > 0:
    if not engine.entrySquadrons.hasKey(kernel):
      raiseAssert("Vulkan: \"" & kernel & "\" is not an entrypoint of the ingested module")
    spirv = engine.spirv
    entry = kernel
    baked = engine.entrySquadrons[kernel]
  elif engine.isHlsl:
    spirv = engine.spirv
  elif kernel == engine.entryPoint:
    spirv = engine.spirv
  else:
    if engine.source.len == 0:
      raiseAssert("Vulkan: a direct SPIR-V module must declare exactly one entrypoint, " &
               "launch it as \"" & engine.entryPoint & "\", got \"" & kernel & "\"")
    if not engine.caches.spirvCache.hasKey(kernel):
      engine.caches.spirvCache[kernel] = compileGlslToSpirvViaGlslang(engine.source, kernel)
    spirv = engine.caches.spirvCache[kernel]
    entry = kernel

  let squadron = if cfg.blk.x == 1 and cfg.blk.y == 1 and cfg.blk.z == 1:
              baked
            else:
              cfg.blk

  if baked.x == 0 or
     squadron.x != baked.x or squadron.y != baked.y or
     squadron.z != baked.z:
    raiseAssert("Vulkan run squadron=" & $squadron.x & "x" & $squadron.y & "x" & $squadron.z &
         " != baked squadron " & $baked.x & "x" & $baked.y &
         "x" & $baked.z &
         ": launch config mismatch, squadron is shader-baked on Vulkan)")

  if usesPhysicalStorageBuffer(spirv):
    launchAddressRoute(engine, kernel, entry, spirv, output, args, cfg, squadron)
  else:
    launchDescriptorRoute(engine, kernel, entry, spirv, output, args, cfg, squadron)

proc launchAddressRoute(engine: VulkanEngine, kernel, entry: string, spirv: SpirvWords,
                        output: auto, args: tuple, cfg: LaunchConfig, squadron: Dim3) =
  var dctx = SerializationCtx(vctx: engine.ctx)
  bindArgs(dctx, output, args, store, VulkanBuffer, cgGlsl)
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

proc launchDescriptorRoute(engine: VulkanEngine, kernel, entry: string, spirv: SpirvWords,
                           output: auto, args: tuple, cfg: LaunchConfig, squadron: Dim3) =
  var bufs: seq[VulkanBuffer]
  template collectBuffer(ctx: var auto, v: VulkanBuffer, _: int) =
    bufs.add v
  template collectBuffer[T](ctx: var auto, v: T, _: int) =
    raiseAssert("descriptor route: scalar kernel args are not supported, " &
             "the uniform-block layout model does not exist yet")
  bindArgs(bufs, output, args, collectBuffer, VulkanBuffer, cgGlsl)

  let setLayout = engine.ctx.createDescriptorSetLayout(uint32(bufs.len))
  let pool = engine.ctx.createDescriptorPool(uint32(bufs.len))
  let descSet = engine.ctx.allocateDescriptorSet(pool, setLayout)
  engine.ctx.updateDescriptorSet(descSet, bufs, uint32(bufs.len))

  let pipelineKey = (kernel, 0)
  if not engine.caches.pipelineCache.hasKey(pipelineKey):
    let shaderModule = engine.ctx.createShaderModule(spirv)
    engine.caches.pipelineCache[pipelineKey] =
      engine.ctx.createPipeline(shaderModule, entry, 0, setLayout)
    engine.ctx.destroyShaderModule(shaderModule)

  # Dispatch cfg.grid groups (runKernel computes groupCount = ceil(global/local))
  engine.ctx.runKernel(engine.caches.pipelineCache[pipelineKey],
                 [uint32(cfg.grid.x * squadron.x),
                  uint32(cfg.grid.y * squadron.y),
                  uint32(cfg.grid.z * squadron.z)],
                 [uint32(squadron.x), uint32(squadron.y), uint32(squadron.z)],
                 nil, 0, descSet)

  # Sets and pools are per-launch and die once runKernel's fence wait returns,
  # and the pipeline outlives them in the engine's pipeline cache
  engine.ctx.destroyDescriptorPool(pool)
  engine.ctx.destroyDescriptorSetLayout(setLayout)

proc allocDeviceImpl(engine: VulkanEngine, n: int): VulkanBuffer {.inline.} =
  engine.ctx.allocBuffer(n)

proc copyToDeviceFromHostImpl(engine: VulkanEngine, dst: VulkanBuffer, src: pointer, n: int) =
  var dst = dst
  engine.ctx.writeBuffer(dst, src, n)

proc copyToHostFromDeviceImpl(engine: VulkanEngine, dst: pointer, src: VulkanBuffer, n: int, offsetBytes: int = 0) =
  let res = readBuffer[byte](engine.ctx, src)
  copyMem(dst, res[offsetBytes].addr, n)
