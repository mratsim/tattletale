## Tattletale
## Copyright (c) 2026 Mamy Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Downloads the wgpu-native shared library for WebGPU/WGSL execution on CPU.
##
## Run with: nim c -r workspace/crucible/vendor/wgpu_installer.nim
##
## Downloads the release zip from https://github.com/gfx-rs/wgpu-native/releases
## and extracts `libwgpu_native.so` (or .dylib / .dll) + `webgpu.h` into
## `workspace/crucible/vendor/wgpu/`.
##
## No GPU required — wgpu-native runs via Vulkan software rasterisation (SwiftShader)
## or its CPU-based shader execution path (Naga IR interpreter).

import
  std/[httpclient, os, strutils, strformat],
  zip/zipfiles

{.passl: "-lz".}

# ############################################################
#
#  Platform detection
#
# ############################################################

type
  Platform = enum
    LinuxX64
    LinuxAarch64
    MacOSX64
    MacOSAarch64
    WindowsX64Msvc
    WindowsX64Gnu
    WindowsAarch64

proc detectPlatform(): Platform =
  when defined(linux) and defined(amd64):
    LinuxX64
  elif defined(linux) and defined(arm64):
    LinuxAarch64
  elif defined(macosx) and defined(arm64):
    MacOSAarch64
  elif defined(macosx) and defined(amd64):
    MacOSX64
  elif defined(windows) and defined(amd64):
    when defined(gcc):
      WindowsX64Gnu
    else:
      WindowsX64Msvc
  elif defined(windows) and defined(arm64):
    WindowsAarch64
  else:
    {.fatal: "Unsupported platform for wgpu-native. See https://github.com/gfx-rs/wgpu-native/releases".}

proc zipName(platform: Platform, release = true): string =
  ## Returns the GitHub release asset name for this platform.
  ## e.g. "wgpu-linux-x86_64-release.zip"
  let suffix = if release: "release" else: "debug"
  case platform
  of LinuxX64:         &"wgpu-linux-x86_64-{suffix}.zip"
  of LinuxAarch64:     &"wgpu-linux-aarch64-{suffix}.zip"
  of MacOSX64:         &"wgpu-macos-x86_64-{suffix}.zip"
  of MacOSAarch64:     &"wgpu-macos-aarch64-{suffix}.zip"
  of WindowsX64Msvc:   &"wgpu-windows-x86_64-msvc-{suffix}.zip"
  of WindowsX64Gnu:    &"wgpu-windows-x86_64-gnu-{suffix}.zip"
  of WindowsAarch64:   &"wgpu-windows-aarch64-msvc-{suffix}.zip"

proc libName(platform: Platform): string =
  ## Name of the shared library file inside the archive.
  case platform
  of LinuxX64, LinuxAarch64:     "libwgpu_native.so"
  of MacOSX64, MacOSAarch64:     "libwgpu_native.dylib"
  of WindowsX64Msvc, WindowsX64Gnu, WindowsAarch64: "wgpu_native.dll"

# ############################################################
#
#  Download & extract
#
# ############################################################

const
  WGPU_VERSION = "v29.0.0.0"
  RELEASE_BASE = "https://github.com/gfx-rs/wgpu-native/releases/download"

proc getProjectDir(): string {.compileTime.} =
  currentSourcePath().parentDir()

proc onProgressChanged(total, progress, speed: BiggestInt) =
  echo &"Downloaded {progress} of {total} ({100 * progress div max(total, 1)}%)"
  echo &"Current rate: {speed.float64 / (1000*1000):4.3f} MiBi/s"

proc downloadFile(url, targetDir, filename: string) =
  var client = newHttpClient()
  defer:
    client.close()
  client.onProgressChanged = onProgressChanged
  echo "Starting download of \"", url, '"'
  echo "Storing temporary into: \"", targetDir, '"'
  client.downloadFile(url, targetDir / filename)

proc extractFile(targetDir, zipPath: string, platform: Platform) =
  let libFile = "lib/" & libName(platform)
  let headerFile = "include/webgpu/webgpu.h"
  echo "Extracting \"", zipPath, "\" into \"", targetDir, "\""
  echo "  extracting: ", libFile
  echo "  extracting: ", headerFile

  var z: ZipArchive
  if not z.open(zipPath):
    raise newException(IOError, &"Could not open zip file: \"{zipPath}\"")
  defer:
    z.close()

  createDir(targetDir / "lib")
  createDir(targetDir / "include/webgpu")
  z.extractFile(libFile, targetDir / libFile)
  z.extractFile(headerFile, targetDir / headerFile)

  echo "Removing \"", zipPath, '"'
  removeFile(zipPath)
  echo "Done."

# ############################################################
#
#  Main
#
# ############################################################

when isMainModule:
  let platform = detectPlatform()
  echo "Detected platform: ", platform
  echo "Target version: ", WGPU_VERSION

  let filename = platform.zipName(release = true)
  let url = &"{RELEASE_BASE}/{WGPU_VERSION}/{filename}"
  let target = getProjectDir() / "wgpu"

  createDir(target)
  let zipPath = target / filename
  if not fileExists(zipPath):
    downloadFile(url, target, filename)
  else:
    echo "File already downloaded: ", zipPath

  extractFile(target, zipPath, platform)

  echo ""
  echo "wgpu-native installed in: ", target
  echo "  ", target, "/lib/", platform.libName()
  echo "  ", target, "/include/webgpu/webgpu.h"
  echo ""
  echo "To use with Nim, set:"
  echo "  export LD_LIBRARY_PATH=\"", target, "/lib\":$LD_LIBRARY_PATH"
  echo "Or add to your nim project:"
  echo "  --passL:\"-L", target, "/lib\""
  echo "  --passL:\"-lwgpu_native\""
