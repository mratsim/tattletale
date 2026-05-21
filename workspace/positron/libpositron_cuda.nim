# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/os
import std/options
import std/strutils
import workspace/libtorch

from ./src/kernels/portable/hadamard_transforms import INV_SQRT_128

# This file wraps the static lib built by make_positron_cuda

const BuildDir = currentSourcePath()
                  .parentDir()
                  .parentDir()
                  .parentDir()/
                  "build"
# CUDA runtime library discovery for linking libpositron_cuda.a
# Priority:
#   1. Derive from nvcc on PATH (must be on PATH to build the .cu file)
#   2. CUDA_HOME env var (user may set this explicitly)
#   3. Known standard locations: /usr/local/cuda, /opt/cuda
#   4. Blind -lcudart (user must have LIBRARY_PATH / system install)
const CudaHome = block:
  let nvccPath = staticExec("command -v nvcc 2>/dev/null || true").strip()
  if nvccPath.len > 0:
    nvccPath.parentDir().parentDir()
  else:
    let envHome = getEnv("CUDA_HOME")
    if envHome.len > 0:
      envHome
    elif dirExists("/usr/local/cuda"):
      "/usr/local/cuda"
    elif dirExists("/opt/cuda"):
      "/opt/cuda"
    else:
      ""
const CudaLibDir = block:
  if CudaHome.len > 0:
    (if dirExists(CudaHome / "lib64"): "lib64" else: "lib")
  else:
    ""
const CudaLibFlag =
  if CudaLibDir.len > 0:
    "-L" & CudaHome / CudaLibDir & " -lcudart"
  else:
    "-lcudart"

{.passL: BuildDir / "libpositron_cuda.a " & CudaLibFlag.}

# ─── Positron kernel library (C ABI) ────────────────────────────

proc pkl_rms_norm_fp16_cuda(
       x, w, y: pointer, epsilon: float32,
       rows, dim: int32
     ): int {.importc, cdecl, discardable.}

# ─── Positron kernel library (Tensor) ────────────────────────────

proc pkl_rms_norm_fp16_cuda*(x, weight: Tensor, eps: float64): Tensor =
  ## RMSNorm via Positron CUDA kernel (linked from libpositron.a).
  ## x, weight are moved to fp16 on CUDA.
  let x_f16 = x.to(kFloat16).to(kCUDA).contiguous()
  let w_f16 = weight.to(kFloat16).to(kCUDA).contiguous()
  result = empty_like(x_f16)
  let status = pkl_rms_norm_fp16_cuda(
    x_f16.dataPtr(),
    w_f16.dataPtr(),
    result.dataPtr(),
    eps.float32,
    (x_f16.numel div x_f16.size(-1)).int32,
    x_f16.size(-1).int32)
  doAssert status == 0, "[ttt] Internal error when calling RMSNorm"

# ─── FWHT-128 (EXL3 Hadamard) CUDA kernel ──────────────────────

proc pkl_hadamard_rotate_128_cuda(
       input, output, pre_scale, post_scale: pointer,
       r_scale: float32, rows, cols: int32
     ): int {.importc, cdecl, discardable.}

proc hadamard_rotate_128_cuda*(
    x: Tensor,
    pre_scale: Option[Tensor],
    post_scale: Option[Tensor],
    norm = INV_SQRT_128
  ): Tensor =
  ## Apply 128-block Walsh-Hadamard transform on CUDA.
  ## Matches ext.had_r_128(input, output, pre_scale, post_scale, norm).
  ## x: [batch, dim] fp16 on CUDA, dim must be multiple of 128.
  ## pre_scale: optional [dim] scale applied before FWHT (fp16).
  ## post_scale: optional [dim] scale applied after FWHT + norm (fp16).
  let x_f16 = x.to(kFloat16).contiguous()
  result = empty_like(x_f16)
  let rows = (x_f16.numel div x_f16.size(-1)).int32
  let cols = x_f16.size(-1).int32
  let pre = if pre_scale.isSome:
               pre_scale.unsafeGet().to(kFloat16).contiguous().dataPtr()
             else:
               nil
  let post = if post_scale.isSome:
                post_scale.unsafeGet().to(kFloat16).contiguous().dataPtr()
              else:
                nil
  let status = pkl_hadamard_rotate_128_cuda(
    x_f16.dataPtr(), result.dataPtr(),
    pre, post,
    norm, rows, cols)
  doAssert status == 0, "[ttt] hadamard_rotate_128_cuda failed"