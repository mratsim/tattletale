# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[httpclient, strformat, os],
  zip/zipfiles

{.passl: "-lz".}

type
  Acceleration* = enum
    Cpu = "cpu"
    Cuda92 = "cu92"
    Cuda101 = "cu101"
    Cuda102 = "cu102"
    Cuda110 = "cu110"
    Cuda111 = "cu111"
    Cuda113 = "cu113"
    Cuda118 = "cu118"
    Cuda121 = "cu121"
    Cuda124 = "cu124"
    Cuda126 = "cu126"
    Cuda128 = "cu128"
    Cuda129 = "cu129"
    Cuda130 = "cu130"

  ABI* = enum
    # Up until PyTorch 2.8.0, default libtorch was built without C++11 ABI to support CentOS
    # This corresponds to cu128. Afterwards, CentOS was EOL and all libtorch uses C++11 ABI
    Cpp = "shared-with-deps"
    Cpp11 = "cxx11-abi-shared-with-deps"

proc getProjectDir(): string {.compileTime.} =
  currentSourcePath.parentDir()

proc onProgressChanged(total, progress, speed: BiggestInt) =
  echo &"Downloaded {progress} of {total}"
  echo &"Current rate: {speed.float64 / (1000*1000):4.3f} MiBi/s"

proc downloadTo(url, targetDir, filename: string) =
  var client = newHttpClient()
  defer:
    client.close()
  client.onProgressChanged = onProgressChanged
  echo "Starting download of \"", url, '"'
  echo "Storing temporary into: \"", targetDir, '"'
  client.downloadFile(url, targetDir / filename)

proc getUrlAndFilename(version = "latest", accel = Cuda130, abi = Cpp): tuple[url, filename: string] =
  result.filename = "libtorch-"

  when defined(linux):
    result.filename &= &"{abi}-{version}"
    if accel != Cuda102 and version != "latest":
      result.filename &= &"%2B{accel}"
    result.filename &= ".zip"
  elif defined(windows):
    let abi = Cpp
    doAssert abi == Cpp, "LibTorch for Windows does not support the C++11 ABI"
    result.filename &= &"win-{abi}-{version}"
    result.filename &= &"%2B{accel}"
    result.filename &= ".zip"
  elif defined(osx):
    doAssert accel == Cpu, "LibTorch for MacOS does not support GPU acceleration"
    result.filename &= &"macos-{version}.zip"

  if version != "latest":
    result.url = &"https://download.pytorch.org/libtorch/{accel}/{result.filename}"
  else:
    result.url = &"https://download.pytorch.org/libtorch/nightly/{accel}/{result.filename}"

proc downloadLibTorch(url, targetDir, filename: string) =
  if not fileExists(targetDir / filename):
    url.downloadTo(targetDir, filename)
  else:
    echo "File is already downloaded"

proc uncompress(targetDir, filename: string, delete = false) =
  var z: ZipArchive
  let tmp = targetDir / filename
  echo "Decompressing \"", tmp, "\" and storing into \"", targetDir, "\""
  if not z.open(tmp):
    raise newException(IOError, &"Could not open zip file: \"{tmp}\"")
  defer:
    z.close()
  z.extractAll(targetDir)
  echo "Done."
  if delete:
    echo "Deleting \"", tmp, '"'
    removeFile(tmp)
  else:
    echo "Not deleting \"", tmp, '"'

when isMainModule:
  when defined(osx) or defined(macosx):
    let (url, filename) = getUrlAndFilename(accel = Cpu)
  else:
    let (url, filename) = getUrlAndFilename()
  let target = getProjectDir()
  downloadLibTorch(url, target, filename)
  uncompress(target, filename, true)
