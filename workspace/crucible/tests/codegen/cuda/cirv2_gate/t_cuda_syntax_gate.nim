# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CUDA-C++ syntax fixtures over the emission surface recorded by the v1
## CUDA emitter audit, type-checked on the Mac with Apple clang as plain C++
## with the NVRTC stub header (`src/codegen/backends/cuda_nvrtc_stub.h`).
##
## The positive kernel exercises every audited construct:
##
## - the `extern "C" __global__` signature and the parenthesized x-major
##   coordinate arithmetic
## - `__shared__` with `__syncthreads()`, the atomicAdd int and float sites
## - a warpReduceAdd call site with the template definition text verbatim
##   from the v1 emitter, a constexpr struct, a braced init list
##
## The rejection fixtures keep the check honest:
##
## - an atomicAdd call passing a pointer as the addend, no overload accepts it
## - a builtin the stub does not declare, the stub must grow with the emitted
##   surface or the fixtures fail (the co-evolution contract)
##
## - the check validates syntax and types only, CUDA-dialect semantics plus
##   PTX production plus execution need nvcc on the Linux workstation
##
## - Run: `nim c -r --outdir:build/tests tests/codegen/cuda/cirv2_gate/t_cuda_syntax_gate.nim`

import std/[unittest]
import workspace/crucible/tests/codegen/cuda/cuda_syntax_gate

const reduceKernel = """
template <typename T> __forceinline__ __device__ T warpReduceAdd(T v) {
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, delta, 32);
  }
  return v;
}

struct Scale {
  int n;
  float a;
};

constexpr Scale scale = {4, 1.5f};

extern "C" __global__ void reduce_kernel(float* out, const float* in, int* hits) {
  __shared__ float tile[64];
  int gx = (int)(blockIdx.x*blockDim.x+threadIdx.x);
  int flat = (int)(threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x);
  int3 g = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
  int3 origin{(int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z};
  tile[flat] = in[gx];
  __syncthreads();
  float v = tile[flat] * scale.a;
  v = warpReduceAdd(v);
  (void)atomicAdd(&out[g.x], v);
  (void)atomicAdd(hits, 1);
  if (flat == 0) {
    out[0] = v + origin.x;
  }
}
"""

const illTypedKernel = """
extern "C" __global__ void ill_typed(float* out) {
  float v = atomicAdd(out, out);
  out[0] = v;
}
"""

const stubDriftKernel = """
extern "C" __global__ void drift_kernel(int3* out) {
  int3 g = make_int4(1, 2, 3, 4);
  out[0] = g;
}
"""

suite "CUDA-C++ syntax fixtures":

  test "representative kernel: the audited emission surface type-checks":
    gateCudaSyntax("reduce_kernel", reduceKernel)

  test "rejection: atomicAdd(float*, float*) has no overload":
    gateCudaSyntaxReject("ill_typed_atomicadd", illTypedKernel)

  test "rejection: make_int4 is not on the stub surface":
    gateCudaSyntaxReject("stub_drift_make_int4", stubDriftKernel)
