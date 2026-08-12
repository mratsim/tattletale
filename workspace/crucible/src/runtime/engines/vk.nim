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
## ingest = glslangValidator subprocess → SPIR-V + parse the shader-baked
## workgroup size. getArtifact = SPIR-V.
## run = pipeline + vkCmdDispatch; blk is shader-baked (local_size_x):
## run's blk is validated against the baked workgroup size and fails loudly
## on mismatch. grid = vkCmdDispatch group count. By-value scalars (ArgBlob
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
    bakedBlk: Dim3   # workgroup size baked into the shader at ingest

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc `=destroy`(c: var VulkanCtx) =
  if c.ctx.instance != nil:
    c.ctx.shutdown()

proc newVulkanEngine(): VulkanEngine =
  ## Private factory — engines.nim reaches it via `import {.all.}`.
  VulkanEngine(ctx: VulkanCtx(ctx: initVulkan()))

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: VulkanEngine, source: string) =
  ## glslangValidator → SPIR-V. Re-entrant: replaces the previous artifact
  ## (the device context persists — only the compiled artifact is replaced).
  if engine.spirv.len > 0:
    when defined(debug):
      echo "[INFO]: vulkan ingest: invalidating previous artifact"
  # Multi-kernel sources with by-value scalars misalign every kernel after
  # the first: the codegen unions all kernels' scalar params into one
  # file-scope push-constant block, but the runtime packs only the invoked
  # kernel's scalars contiguously from offset 0. Enforce the documented
  # contract (one kernel per source when using scalars) loudly; pointer-only
  # multi-kernel sources are unaffected.
  if countKernels(source) > 1 and
     "layout(push_constant) uniform KernelParams" in source:
    quit("Vulkan: multi-kernel source with scalar params is unsupported, " &
         "use one kernel per source when passing by-value scalars")
  engine.source = source
  engine.entryPoint = parseEntryPoint(source)
  engine.spirv = compileGlslToSpirV(source, engine.entryPoint)
  engine.bakedBlk = parseBakedBlk(source)

proc getArtifact*(engine: VulkanEngine): seq[uint32] =
  ## The compiled SPIR-V.
  engine.spirv

proc deviceName*(engine: VulkanEngine): string {.inline.} =
  ## The physical device name (e.g. "NVIDIA RTX PRO 6000 Blackwell ...").
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
  ## Count kernel entry points: each kernel gets its own
  ## `layout(local_size_x = ...) in;` preamble (vulkan_lang.nim).
  const marker = "layout(local_size_x"
  var i = glsl.find(marker)
  while i >= 0:
    inc result
    i = glsl.find(marker, i + marker.len)

proc parseEntryPoint(glsl: string): string =
  ## Extract the first kernel's entry point name from the GLSL. The codegen
  ## emits `layout(local_size_x ...) in;` immediately before each kernel's
  ## `void <name>() { ... }` and forward-declares device helpers
  ## (`void helper(params);`) before all kernels, so the first bare `void `
  ## may be a helper name. Fall back to the first `void ` when no kernel
  ## preamble is present, then to "main".
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

proc runImpl(engine: VulkanEngine, kernel: string, output: ArgBlob,
             blobs: seq[ArgBlob], cfg: LaunchConfig) =
  ## Pipeline + vkCmdDispatch. Bindings follow the shader: the output is
  ## the first SSBO binding (output first, per CONVENTIONS.md), then the
  ## pointer args in order. By-value scalar args (size < 0) are packed into
  ## the push-constant range (codegen emits `layout(push_constant) uniform
  ## KernelParams`).
  var vctx = engine.ctx.ctx

  # blk is shader-baked (local_size_xyz). A default cfg (plain run) dispatches
  # with the baked size; an explicit blk must match it exactly.
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
         " — launch config mismatch (blk is shader-baked on Vulkan)")

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
  vctx.runKernel(pipeline,
                 [uint32(cfg.grid.x * blk.x),
                  uint32(cfg.grid.y * blk.y),
                  uint32(cfg.grid.z * blk.z)],
                 [uint32(blk.x), uint32(blk.y), uint32(blk.z)],
                 if pushConstBytes.len > 0: pushConstBytes[0].addr else: nil,
                 pushConstBytes.len)

  # Read output
  if outSize > 0:
    let res = readBuffer[byte](vctx, outBuf)
    copyMem(output.data, res[0].addr, outSize)
