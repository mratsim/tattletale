# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/[strutils, tables]

import workspace/crucible/src/abis/vulkan_abi as vk

import workspace/crucible/src/codegen/backends/targets
import workspace/crucible/src/runtime/errors
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
    spirvCache: Table[string, seq[uint32]]
    pipelineCache: Table[(string, int), VulkanPipeline]

  VulkanEngine* = ref object
    source: string
    spirv: seq[uint32]
    entryPoint: string
    ctx: VulkanContext
    shaderSquadron: Dim3   # squadron baked into the shader at ingest
    isHlsl: bool
    stagesGrid: bool       # source reads the dispatch grid, stage cfg.grid
                           # into the uniform block (the materialized parameter)
    caches: VulkanCaches

template backendKind(_: typedesc[VulkanEngine]): BackendKind = bkGlsl

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

proc ingest*(engine: VulkanEngine, source: string) =
  resetCaches(engine)
  engine.source = source
  if source.startsWith("#version"):
    ingestGlsl(engine, source)
  elif "[numthreads(" in source:
    ingestHlsl(engine, source)
  else:
    failLoud("Vulkan ingest: unrecognized source language, expected GLSL " &
             "(a `#version` first line) or HLSL (`[numthreads(`), got: " &
             source.splitLines()[0])

proc ingestGlsl(engine: VulkanEngine, glsl: string) =
  # multi-kernel sources cannot address per-kernel scalars in the shared
  # deduped push block, ingest quits loudly
  if countKernels(glsl) > 1:
    var inPush = false
    for line in glsl.splitLines():
      if "layout(push_constant" in line:
        inPush = true
      elif inPush and line.strip().startsWith("}"):
        inPush = false
      elif inPush:
        let t = line.strip()
        if t.len > 0 and t.endsWith(";") and not t.startsWith("Ref_"):
          failLoud("Vulkan: multi-kernel source with scalar params is unsupported, " &
                   "use one kernel per source when passing by-value scalars")

  engine.entryPoint = parseEntryPoint(glsl)
  engine.spirv = compileGlslToSpirvViaGlslang(glsl, engine.entryPoint)
  engine.shaderSquadron = parseShaderSquadron(glsl)
  engine.isHlsl = false

proc ingestHlsl(engine: VulkanEngine, hlsl: string) =
  engine.entryPoint = "main"
  engine.spirv = compileHlslToSpirvViaDxc(hlsl)
  engine.shaderSquadron = parseHlslSquadron(hlsl)
  engine.isHlsl = true
  engine.stagesGrid = "crucible_numworkgroups" in hlsl

proc ingest*(engine: VulkanEngine, spirv: seq[uint32], stagedGrid: bool = false) =
  ## Ingests a ready-made SPIR-V module, the artifact of a `spirv:` block or of a compiled fvVulkan HLSL flavor source.
  ##
  ## Args:
  ## - `stagedGrid`: true when the module expects the dispatch group counts
  ##   staged at the end of its push constant block, the fvVulkan flavor's
  ##   materialized `crucible_numworkgroups` member
  ##
  ## The words hold no such signal, the compile side passes the flag explicitly,
  ## the address route stages the counts at launch.
  const spvMagic = 0x07230203'u32
  resetCaches(engine)
  if spirv.len < 5 or spirv[0] != spvMagic:
    failLoud("Vulkan ingest: not a SPIR-V module, expected the 0x07230203 " &
             "magic as word 0 and the 5-word header, got " &
             (if spirv.len == 0: "an empty module"
              else: $spirv.len & " words, word 0 = " & spirv[0].toHex()))
  let sq = getSpirvSquadron(spirv)
  if sq.entryPoints != 1:
    failLoud("Vulkan ingest: a direct SPIR-V module carries exactly one " &
             "OpEntryPoint, got " & $sq.entryPoints)
  engine.source = ""
  engine.spirv = spirv
  engine.entryPoint = sq.name
  engine.shaderSquadron = sq.localSize
  engine.isHlsl = false
  engine.stagesGrid = stagedGrid

proc getArtifact*(engine: VulkanEngine): string {.inline.} =
  ## Ingested GLSL source, the ingest-if-differs identity
  engine.source

func getSpirV*(engine: VulkanEngine): seq[uint32] {.inline.} =
  engine.spirv

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

# ═══════════════════════════════════════════════════════════════════════
# ▸ Launch serialization
# ═══════════════════════════════════════════════════════════════════════

type SerializationCtx = object
  vctx: VulkanContext
  pushBuf: array[128, byte]
  pn: int

proc write8(ctx: var SerializationCtx, x: uint64) =
  ctx.pn = round_multiple_up(ctx.pn, 8)
  if ctx.pn + 8 > 128:
    failLoud("Vulkan push constants exceed the spec-guaranteed 128-byte max")
  copyMem(ctx.pushBuf[ctx.pn].addr, x.unsafeAddr, 8)
  inc ctx.pn, 8

proc store(ctx: var SerializationCtx, v: VulkanBuffer, argIdx: int) =
  ctx.write8(deviceAddress(v, ctx.vctx))

proc store[T](ctx: var SerializationCtx, v: T, argIdx: int) =
  when T is ptr|pointer:
    {.error: "Pointer kernel args are not supported, pass the DeviceMem handle from allocDevice, got " & $T.}
  const lay = deviceLayout(T, bkGlsl)
  if lay.align > 1:
    ctx.pn = round_multiple_up(ctx.pn, lay.align)
  if ctx.pn + lay.size > 128:
    failLoud("Vulkan push constants exceed the spec-guaranteed 128-byte max")
  var blob: array[lay.size, byte]
  var off = 0
  when T is object|tuple and hasBufferLeaf(T, VulkanBuffer):
    for f in fields(v):
      when typeof(f) is DeviceMem:
        off = round_multiple_up(off, 8)
        let a = deviceAddress(f.mem, ctx.vctx)
        copyMem(blob[off].addr, a.unsafeAddr, 8)
        inc off, 8
      else:
        blob.packFrom(off, f, bkGlsl)
    off = round_multiple_up(off, lay.align)
    doAssert off == lay.size,
      "host packing of " & $T & " does not match the device layout"
  else:
    blob.packFrom(off, v, bkGlsl)
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
  if engine.isHlsl:
    spirv = engine.spirv
    entry = "main"
  elif kernel == engine.entryPoint:
    spirv = engine.spirv
  else:
    if engine.source.len == 0:
      failLoud("Vulkan: a direct SPIR-V module carries exactly one entry point, " &
               "launch it as \"" & engine.entryPoint & "\", got \"" & kernel & "\"")
    if not engine.caches.spirvCache.hasKey(kernel):
      engine.caches.spirvCache[kernel] = compileGlslToSpirvViaGlslang(engine.source, kernel)
    spirv = engine.caches.spirvCache[kernel]
    entry = kernel

  if usesPhysicalStorageBuffer(spirv):
    launchAddressRoute(engine, kernel, entry, spirv, output, args, cfg, squadron)
  else:
    launchDescriptorRoute(engine, kernel, entry, spirv, output, args, cfg, squadron)

proc launchAddressRoute(engine: VulkanEngine, kernel, entry: string, spirv: seq[uint32],
                        output: auto, args: var tuple, cfg: LaunchConfig, squadron: Dim3) =
  var dctx = SerializationCtx(vctx: engine.ctx)
  bindArgs(dctx, output, args, store, VulkanBuffer, bkGlsl)
  var pushConstLen = dctx.pn
  if engine.stagesGrid:
    # staged dispatch grid size, 3 x i32 behind the packed kernel args,
    # at the 16-byte offset where the fvVulkan printer locks the grid member
    let gridOff = round_multiple_up(dctx.pn, 16)
    if gridOff + 12 > 128:
      failLoud("Vulkan push constants exceed the spec-guaranteed 128-byte max")
    let groups = [ceil_div(cfg.grid.x * squadron.x, squadron.x),
                  ceil_div(cfg.grid.y * squadron.y, squadron.y),
                  ceil_div(cfg.grid.z * squadron.z, squadron.z)]
    for i, g in groups:
      let gi = int32(g)
      copyMem(dctx.pushBuf[gridOff + i * 4].addr, gi.unsafeAddr, 4)
    pushConstLen = gridOff + 12

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

proc launchDescriptorRoute(engine: VulkanEngine, kernel, entry: string, spirv: seq[uint32],
                           output: auto, args: var tuple, cfg: LaunchConfig, squadron: Dim3) =
  # Buffer leaves bind as storage descriptors 0 ..< bufs.len,
  # value leaves pack into one uniform blob staged behind them,
  # binding bufs.len, the printed Push block's register(bN) ordinal.
  var bufs: seq[VulkanBuffer]
  var blob: array[hlslUniformBlobCap(typeof(output), typeof(args), VulkanBuffer), byte]
  var off = 0
  template collectDescriptorLeaf(ctx: var auto, v: auto, argIdx: int) =
    when typeof(v) is VulkanBuffer:
      bufs.add v
    elif typeof(v) is DeviceMem:
      bufs.add v.mem
    else:
      blob.packHlslPush(off, v)
  bindArgs(bufs, output, args, collectDescriptorLeaf, VulkanBuffer, bkWebGPU)
  if not engine.isHlsl and off > 0:
    failLoud("descriptor route: scalar kernel args are not supported, " &
             "the uniform-block layout model does not exist yet")

  var uniforms: seq[VulkanBuffer]
  if engine.isHlsl:
    if engine.stagesGrid:
      # materialized dispatch grid size, 3 x i32 with the vector rule:
      # 4-byte aligned, never straddling a 16-byte register.
      # staged values are the dispatch group counts runKernel computes, what a threadgroups_per_grid read means.
      off = (off + 3) div 4 * 4
      if off mod 16 + 12 > 16:
        off = (off + 15) div 16 * 16
      let groups = [ceil_div(cfg.grid.x * squadron.x, squadron.x),
                    ceil_div(cfg.grid.y * squadron.y, squadron.y),
                    ceil_div(cfg.grid.z * squadron.z, squadron.z)]
      for g in groups:
        let gi = int32(g)
        copyMem(blob[off].addr, gi.unsafeAddr, 4)
        off += 4
    if off > 0:
      var uniform = engine.allocDeviceImpl(off)
      engine.copyToDeviceFromHostImpl(uniform, blob[0].addr, off)
      uniforms.add uniform

  let setLayout = engine.ctx.createDescriptorSetLayout(uint32(bufs.len),
    uint32(uniforms.len))
  let pool = engine.ctx.createDescriptorPool(uint32(bufs.len), uint32(uniforms.len))
  let descSet = engine.ctx.allocateDescriptorSet(pool, setLayout)
  engine.ctx.updateDescriptorSet(descSet, bufs, uint32(bufs.len), uniforms)

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

  # Sets, pools and the uniform scratch die once runKernel's fence wait returns,
  # pipeline outlives them in the engine's pipeline cache
  for buf in uniforms.mitems:
    buf.dealloc()
  engine.ctx.destroyDescriptorPool(pool)
  engine.ctx.destroyDescriptorSetLayout(setLayout)

proc allocDeviceImpl(engine: VulkanEngine, n: int): VulkanBuffer {.inline.} =
  engine.ctx.allocBuffer(n)

proc copyToDeviceFromHostImpl(engine: VulkanEngine, dst: VulkanBuffer, src: pointer, n: int) =
  var dst = dst
  engine.ctx.writeBuffer(dst, src, n)

proc copyToHostFromDeviceImpl(engine: VulkanEngine, dst: pointer, src: VulkanBuffer, n: int) =
  let res = readBuffer[byte](engine.ctx, src)
  copyMem(dst, res[0].addr, n)
