# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## VulkanEngine — Vulkan (SPIR-V) runtime compilation and execution (moved
## from codegen/vk.nim, decoupled from the compile-time DSL: no gpu_compiler
## import).
##
## ingest = glslangValidator subprocess → SPIR-V (keep the existing
## mechanism) + parse the shader-baked workgroup size. getArtifact = SPIR-V.
## run = pipeline + vkCmdDispatch; blk is shader-baked (local_size_x):
## run's blk is validated against the baked workgroup size and fails loudly
## via check. grid = vkCmdDispatch group count. By-value scalars (ArgBlob
## size < 0) are packed into the push-constant range (4-byte only, see
## runImpl); pointer args get SSBO bindings 1..N, output at binding 0
## (output first, per CONVENTIONS.md).
##
## Structure: PUBLIC API block first (exported `*`); PRIVATE machinery below
## (no `*`). `{.experimental: "codeReordering".}` lifts Nim's
## declaration-before-use rule so the private types/helpers may follow the
## public surface that calls them.
{.experimental: "codeReordering".}

import std/strutils

import workspace/crucible/src/abis/vulkan_abi as vk

import ../exec/vulkan_runtime
import ../exec/runtime_utils
import ./arg_blobs
import ../chevrons

export vulkan_runtime
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  VulkanCtx = object
    ## RAII value wrapper — `=destroy` fires when the engine ref dies.
    ctx: VulkanContext

  VulkanEngine* = ref object
    ## Fields directly (no Obj indirection); resources in the RAII value field.
    source: string
    spirv: seq[uint32]
    entryPoint: string  # GLSL entry point name, baked at ingest
    ctx: VulkanCtx
    grid, blk: int   # engine-default geometry for the plain `run`
    bakedBlk: int    # workgroup size baked into the shader at ingest

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc `=destroy`(c: var VulkanCtx) =
  if c.ctx.instance != nil:
    c.ctx.shutdown()

proc newVulkanEngine(grid, blk: int): VulkanEngine =
  ## Private factory — engines.nim reaches it via `import {.all.}`.
  VulkanEngine(ctx: VulkanCtx(ctx: initVulkan()), grid: grid, blk: blk)

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: VulkanEngine, source: string) =
  ## glslangValidator → SPIR-V. Re-entrant: replaces the previous artifact
  ## (the device context persists — only the compiled artifact is replaced).
  if engine.spirv.len > 0:
    when defined(debug):
      echo "[INFO]: vulkan ingest: invalidating previous artifact"
  engine.source = source
  engine.entryPoint = parseEntryPoint(source)
  engine.spirv = compileGlslToSpirV(source, engine.entryPoint)
  engine.bakedBlk = parseBakedBlk(source)

proc getArtifact*(engine: VulkanEngine): seq[uint32] =
  ## The compiled SPIR-V.
  engine.spirv

template run*[T](engine: VulkanEngine, kernel: string, output: var T, args: untyped,
              cfg: LaunchConfig): untyped =
  var blobStorage: seq[byte]   # backing store for by-value scalars; lives until scope exit
  runImpl(engine, kernel, outBlob(output), flattenBlobs(args, blobStorage), cfg)

template run*[T](engine: VulkanEngine, kernel: string, output: var T, args: untyped): untyped =
  run(engine, kernel, output, args,
      LaunchConfig(grid: Dim3(x: engine.grid), blk: Dim3(x: engine.blk)))

# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────

proc parseBakedBlk(glsl: string): int =
  ## Extract the shader-baked workgroup size from the GLSL preamble:
  ## `layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;`
  ## Returns 0 when absent (run will then fail blk validation loudly).
  const marker = "local_size_x = "
  let i = glsl.find(marker)
  if i < 0:
    return 0
  var j = i + marker.len
  var n = 0
  while j < glsl.len and glsl[j] in {'0' .. '9'}:
    n = n * 10 + (ord(glsl[j]) - ord('0'))
    inc j
  n

proc parseEntryPoint(glsl: string): string =
  ## Extract the entry point name from the GLSL: `void <name>() { ... }`.
  ## The vulkan codegen emits exactly one compute entry per kernel.
  const marker = "void "
  let i = glsl.find(marker)
  if i < 0:
    return "main"
  var j = i + marker.len
  var name = ""
  while j < glsl.len and glsl[j] notin {'(', ' ', '\t', '\n'}:
    name.add glsl[j]
    inc j
  name

proc runImpl(engine: VulkanEngine, kernel: string, output: ArgBlob,
             blobs: seq[ArgBlob], cfg: LaunchConfig) =
  ## Pipeline + vkCmdDispatch. Bindings follow the shader: the output is
  ## the first SSBO binding (output first, per CONVENTIONS.md), then the
  ## pointer args in order. By-value scalar args (size < 0) are packed into
  ## the push-constant range (codegen emits `layout(push_constant) uniform
  ## KernelParams`).
  var vctx = engine.ctx.ctx

  # blk is shader-baked (local_size_x): validate loudly (relax later)
  if cfg.blk.x != engine.bakedBlk:
    quit("Vulkan run blk=" & $cfg.blk.x & " != baked workgroup size " & $engine.bakedBlk &
         " — launch config mismatch (blk is shader-baked on Vulkan)")
  if cfg.blk.y != 1 or cfg.blk.z != 1:
    quit("Vulkan blk y/z must be 1 (shader-baked 1D workgroup)")

  # Entry point: reuse the ingested SPIR-V when the run kernel matches the
  # baked entry; multi-kernel GLSL recompiles on demand with the kernel name.
  var spirv = engine.spirv
  var entry = engine.entryPoint
  if kernel != engine.entryPoint:
    spirv = compileGlslToSpirV(engine.source, kernel)
    entry = kernel

  let outSize = abs(output.size)

  # Create shader module from the ingested SPIR-V
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
      proc(device: VkDevice, shaderModule: VkShaderModule,
           pAllocator: pointer) {.cdecl.}
    ](vctx.gpaAddr(vctx.instance, "vkDestroyShaderModule"))
    if shaderModule != nil and vkDestroyShaderModule != nil:
      vkDestroyShaderModule(vctx.device, shaderModule, nil)

  # Inputs: by-value scalars (size < 0) → push constants; pointers → SSBOs.
  # Scalars never occupy a descriptor binding — they go in the push-constant
  # block (binding 0 is the output).
  var pushConstBytes = newSeq[byte]()
  var inputBuffers = newSeq[VulkanBuffer]()
  for i in 0 ..< blobs.len:
    if blobs[i].size < 0:
      let sz = -blobs[i].size
      # Strict contract: 4-byte scalars only (std430 push-constant block
      # members are 4-byte aligned). Vec/struct by-value args would need a
      # real std430 layout computation — fail loudly rather than misalign.
      if sz != 4:
        quit("Vulkan by-value (push-constant) args must be 4-byte scalars, got " &
             $sz & " bytes")
      for j in 0 ..< 4:
        pushConstBytes.add cast[ptr UncheckedArray[byte]](blobs[i].data)[j]
    else:
      var buf = vctx.allocBuffer(blobs[i].size)
      if blobs[i].size > 0:
        vctx.writeBuffer(buf, blobs[i].data, blobs[i].size)
      inputBuffers.add buf
  if pushConstBytes.len > 128:
    quit("Vulkan push constants exceed the spec-guaranteed 128-byte max (got " &
         $pushConstBytes.len & " bytes)")
  var outBuf = vctx.allocBuffer(outSize)
  # Upload the output's current contents before launch (in-place β·C)
  if outSize > 0:
    vctx.writeBuffer(outBuf, output.data, outSize)

  defer:
    for buf in inputBuffers.mitems:
      buf.dealloc(vctx)
    outBuf.dealloc(vctx)

  # Entry point is baked at ingest (the GLSL has one compute entry)
  var pipeline = vctx.createPipeline(shaderModule, inputBuffers.len + 1, entry,
                                   pushConstBytes.len)
  defer:
    pipeline.destroyPipeline(vctx)

  # Output first (binding 0), then pointer inputs
  pipeline.setArg(0, outBuf, vctx)
  for i in 0 ..< inputBuffers.len:
    pipeline.setArg(i + 1, inputBuffers[i], vctx)

  # Dispatch cfg.grid groups (runKernel computes groupCount = ceil(global/local))
  vctx.runKernel(pipeline, [uint32(cfg.grid.x * cfg.blk.x)], [uint32(cfg.blk.x), 1'u32, 1'u32],
                 if pushConstBytes.len > 0: pushConstBytes[0].addr else: nil,
                 pushConstBytes.len)

  # Read output
  if outSize > 0:
    let res = readBuffer[byte](vctx, outBuf)
    copyMem(output.data, res[0].addr, outSize)
