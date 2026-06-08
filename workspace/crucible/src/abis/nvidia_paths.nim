# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## NVIDIA CUDA toolkit path discovery.

import std/os, std/strutils

## Resolves the root CUDA toolkit directory at compile time.
##
## Priority:
##   1. Derive from ``nvcc`` on PATH (must be on PATH to build ``.cu`` files)
##   2. ``CUDA_HOME`` env var (user may set explicitly)
##   3. Known standard locations: ``/usr/local/cuda``, ``/opt/cuda``
##   4. ``""`` — caller must handle gracefully
const CudaHome* = block:
  let nvccPath = staticExec("command -v nvcc 2>/dev/null || true").strip()
  if nvccPath.len > 0:
    nvccPath.parentDir().parentDir()
  else:
    let envHome = getEnv("CUDA_HOME")
    if envHome.len > 0: envHome
    elif dirExists("/usr/local/cuda"): "/usr/local/cuda"
    elif dirExists("/opt/cuda"): "/opt/cuda"
    else: ""

## Returns ``lib64`` or ``lib`` whichever exists under ``CudaHome``.
const CudaLibDir* = block:
  if CudaHome.len > 0:
    if dirExists(CudaHome / "lib64"): "lib64"
    elif dirExists(CudaHome / "lib"): "lib"
    else: ""
  else:
    ""

## ``-L<libdir> -lcudart`` or just ``-lcudart`` when CUDA is not found.
const CudaLibFlag* =
  if CudaLibDir.len > 0: "-L" & (CudaHome / CudaLibDir) & " -lcudart"
  else: "-lcudart"

## Tries common locations for ``libcudadevrt.a`` and returns the first existing path.
## Returns ``""`` if not found.
proc findCudaDevrt*(): string =
  ## Compile-time path resolver for the CUDA device runtime archive.
  if CudaHome.len > 0:
    for subdir in ["targets/x86_64-linux/lib", CudaLibDir, "lib64", "lib"]:
      if subdir.len > 0 and fileExists(CudaHome / subdir / "libcudadevrt.a"):
        return CudaHome / subdir / "libcudadevrt.a"
  for root in ["/usr/local/cuda", "/opt/cuda"]:
    if dirExists(root):
      for subdir in ["targets/x86_64-linux/lib", "lib64", "lib"]:
        if fileExists(root / subdir / "libcudadevrt.a"):
          return root / subdir / "libcudadevrt.a"
  return ""
