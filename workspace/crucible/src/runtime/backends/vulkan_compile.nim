# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.


import std/[os, osproc, streams, strutils, tempfiles]
import ../errors
import ./runtime_utils

# ═══════════════════════════════════════════════════════════════════════
# SPIR-V compilation
# ═══════════════════════════════════════════════════════════════════════

proc compileGlslToSpirvViaGlslang*(glsl: string, entryPoint: string = "main"): seq[uint32] =
  ## Compiles GLSL to SPIR-V via ``glslangValidator``
  ## (``libshaderc_shared`` does not support compute shaders on this platform).
  ## Returns SPIR-V words:
  ## - subgroup sources (the KHR subgroup extensions the codegen emits)
  ##   compile with the vulkan1.1 target env, shuffles are SPIR-V 1.3
  ## - every other source stays on the SPIR-V 1.0 default
  let glslang = findExe("glslangValidator")
  if glslang.len == 0:
    failLoud("glslangValidator not found on PATH, the Vulkan engine needs it to compile GLSL sources")
  let tmpDir = getKernelDir("vulkan")
  # Private temp dir: 0700 so other local users cannot plant symlinks
  # at deterministic paths (TOCTOU), and unique names so concurrent compiles
  # never collide on the same file.
  setFilePermissions(tmpDir, {fpUserExec, fpUserWrite, fpUserRead})
  let srcPath = genTempPath(sanitizePath(entryPoint) & "_", ".comp", tmpDir)
  let spvPath = genTempPath(sanitizePath(entryPoint) & "_", ".spv", tmpDir)

  writeFile(srcPath, glsl)
  defer:
    if fileExists(srcPath):
      removeFile(srcPath)
    if fileExists(spvPath):
      removeFile(spvPath)
  var args = @["-V", "-e", entryPoint, "--source-entrypoint", "main"]
  if "GL_KHR_shader_subgroup" in glsl:
    args.add "--target-env"
    args.add "vulkan1.1"
  args.add "-o"
  args.add spvPath
  args.add srcPath
  let p = startProcess(glslang, args = args,
    options = {poStdErrToStdOut})
  let compOut = p.outputStream.readAll()
  let exitCode = p.waitForExit()
  defer: p.close()
  if exitCode != 0:
    failLoud("glslangValidator failed (exit=" & $exitCode & "):\n" & compOut)

  let raw = readFile(spvPath)
  result = newSeq[uint32](raw.len div 4)
  copyMem(result[0].addr, raw[0].addr, raw.len)

proc compileHlslToSpirvViaDxc*(hlsl: string): seq[uint32] =
  ## Compiles HLSL to SPIR-V via ``dxc``, the SPIR-V codegen backend, no DXIL leg.
  let dxc = findExe("dxc")
  if dxc.len == 0:
    failLoud("dxc not found on PATH, the Vulkan engine needs it to compile HLSL sources")
  let tmpDir = getKernelDir("vulkan")
  setFilePermissions(tmpDir, {fpUserExec, fpUserWrite, fpUserRead})
  let srcPath = genTempPath("main_", ".hlsl", tmpDir)
  let spvPath = genTempPath("main_", ".spv", tmpDir)

  writeFile(srcPath, hlsl)
  defer:
    if fileExists(srcPath):
      removeFile(srcPath)
    if fileExists(spvPath):
      removeFile(spvPath)
  let args = ["-T", "cs_6_4", "-E", "main", "-spirv", "-Fo", spvPath, srcPath]
  let p = startProcess(dxc, args = args, options = {poStdErrToStdOut})
  let compOut = p.outputStream.readAll()
  let exitCode = p.waitForExit()
  defer: p.close()
  if exitCode != 0:
    failLoud("dxc failed (exit=" & $exitCode & "):\n" & compOut)

  let raw = readFile(spvPath)
  result = newSeq[uint32](raw.len div 4)
  copyMem(result[0].addr, raw[0].addr, raw.len)

proc usesPhysicalStorageBuffer*(spv: seq[uint32]): bool =
  ## True when the SPIR-V module declares the PhysicalStorageBufferAddresses capability (OpCapability 17, capability 5347, spirv-dis-verified).
  const opCapability = 17
  const capPhysicalStorageBufferAddresses = 5347
  # The 5-word module header (magic, version, generator, bound, schema) holds no instructions, the instruction stream starts after it
  var i = 5
  while i < spv.len:
    let wordCount = spv[i] shr 16
    if wordCount == 0:
      break
    if (spv[i] and 0xFFFF) == opCapability and
       i + 1 < spv.len and
       spv[i + 1] == capPhysicalStorageBufferAddresses:
      return true
    inc i, wordCount.int
