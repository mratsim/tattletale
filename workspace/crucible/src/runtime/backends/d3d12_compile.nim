# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.


import std/[os, osproc, streams, tempfiles]
import ../errors
import ./runtime_utils

proc compileHlslToDxilViaDxc*(hlsl: string): seq[uint32] =
  let dxc = findExe("dxc")
  if dxc.len == 0:
    failLoud("dxc not found on PATH, the D3D12 engine needs it to compile HLSL sources to DXIL")
  let tmpDir = getKernelDir("d3d12")
  # 0700 on the temp dir blocks other local users from planting a file
  # at a deterministic path this process reads afterwards (TOCTOU).
  # Temp file names are unique, so concurrent compiles never write the same file.
  setFilePermissions(tmpDir, {fpUserExec, fpUserWrite, fpUserRead})
  let srcPath = genTempPath("main_", ".hlsl", tmpDir)
  let csoPath = genTempPath("main_", ".cso", tmpDir)

  writeFile(srcPath, hlsl)
  defer:
    if fileExists(srcPath):
      removeFile(srcPath)
    if fileExists(csoPath):
      removeFile(csoPath)
  let args = ["-T", "cs_6_4", "-E", "main", "-Fo", csoPath, srcPath]
  let p = startProcess(dxc, args = args, options = {poStdErrToStdOut})
  let compOut = p.outputStream.readAll()
  let exitCode = p.waitForExit()
  defer: p.close()
  if exitCode != 0:
    failLoud("dxc failed (exit=" & $exitCode & "):\n" & compOut)

  let raw = readFile(csoPath)
  if raw.len < 4 or raw[0 ..< 4] != "DXBC":
    failLoud("dxc produced " & $raw.len & " bytes without the DXBC container magic, the D3D12 leg needs a DXBC/DXIL container")
  if raw.len mod 4 != 0:
    failLoud("dxc produced " & $raw.len & " bytes, not a multiple of 4; container truncated")
  result = newSeq[uint32](raw.len div 4)
  copyMem(result[0].addr, raw[0].addr, raw.len)
