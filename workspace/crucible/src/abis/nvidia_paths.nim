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
## Priority (explicit beats derived):
##   1. ``CUDA_HOME`` env var — the user's explicit choice, wins
##   2. Derive from ``nvcc`` on PATH (convenience when CUDA_HOME is unset)
##   3. Known standard locations: ``/usr/local/cuda``, ``/opt/cuda``
##   4. ``""`` — caller must handle gracefully
##
## Note: NVRTC compiles ``.cu`` at runtime through ``libnvrtc.so`` (dynlib)
## and never invokes the ``nvcc`` executable. ``CudaHome`` only drives the
## link-time pieces: ``libcudadevrt.a`` (cuLinkAddFile) and ``-lcudart``
## (positron static lib). nvcc-on-PATH is thus a fallback, never an
## override of an explicit ``CUDA_HOME``.
const CudaHome* = block:
  let envHome = getEnv("CUDA_HOME")
  if envHome.len > 0:
    envHome
  else:
    let nvccPath = staticExec("command -v nvcc 2>/dev/null || true").strip()
    if nvccPath.len > 0:
      nvccPath.parentDir().parentDir()
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
