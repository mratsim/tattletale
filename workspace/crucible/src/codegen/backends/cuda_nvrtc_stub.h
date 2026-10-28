// Tattletale
// Copyright (c) 2026 Mamy André-Ratsimbazafy
// Licensed and distributed under either of
//   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
//   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
// at your option. This file may not be copied, modified, or distributed except according to those terms.

// Validation-only stub of the NVRTC builtin surface, mirroring the CUDA-C++
// constructs the CUDA route emits. Apple clang type-checks the emitted source
// as plain C++ with this header force-included:
//
//   clang++ -x c++ -fsyntax-only -std=c++17 -include cuda_nvrtc_stub.h kernel.cu
//
// Co-evolution contract: this header is never shipped and never included by
// any src module. It stands next to the future CUDA spelling tables on
// purpose, a change to the emitted CUDA-C++ surface (qualifier set, builtin
// names, signatures) updates this stub in the same change, the syntax
// fixtures under tests/codegen/cuda/ fail otherwise. The surface below covers
// only what the fixtures exercise, nothing speculative:
//
// - the execution-space qualifiers, defined away
// - uint3 coordinate builtins with .x/.y/.z, make_int3, size_t
// - __syncthreads, __shfl_xor_sync / __shfl_down_sync (int, unsigned, float)
// - atomicAdd overloads (int, unsigned, float, double, long long)
// - the warpReduceAdd / warpReduceMax template declarations
//
// The check validates syntax and types only: no CUDA-dialect semantics, no
// PTX, no execution. Those need nvcc on the Linux workstation.

#ifndef CRUCIBLE_CUDA_NVRTC_STUB_H
#define CRUCIBLE_CUDA_NVRTC_STUB_H

// Execution-space qualifiers: plain C++ on the host compiler, no meaning here.
#define __global__
#define __device__
#define __shared__
#define __constant__
#define __forceinline__

typedef __SIZE_TYPE__ size_t;

struct uint3 { unsigned x, y, z; };
struct int3 { int x, y, z; };

// The runtime materializes these per thread, the fixtures only read them.
extern const uint3 threadIdx;
extern const uint3 blockIdx;
extern const uint3 blockDim;
extern const uint3 gridDim;

inline int3 make_int3(int x, int y, int z) {
  int3 v;
  v.x = x; v.y = y; v.z = z;
  return v;
}

inline void __syncthreads() {}

inline int __shfl_xor_sync(unsigned mask, int value, int delta, int width = 32) {
  (void)mask; (void)delta; (void)width; return value;
}
inline unsigned __shfl_xor_sync(unsigned mask, unsigned value, int delta, int width = 32) {
  (void)mask; (void)delta; (void)width; return value;
}
inline float __shfl_xor_sync(unsigned mask, float value, int delta, int width = 32) {
  (void)mask; (void)delta; (void)width; return value;
}

inline int __shfl_down_sync(unsigned mask, int value, int delta, int width = 32) {
  (void)mask; (void)delta; (void)width; return value;
}
inline unsigned __shfl_down_sync(unsigned mask, unsigned value, int delta, int width = 32) {
  (void)mask; (void)delta; (void)width; return value;
}
inline float __shfl_down_sync(unsigned mask, float value, int delta, int width = 32) {
  (void)mask; (void)delta; (void)width; return value;
}

inline int atomicAdd(int* address, int value) {
  int old = *address; *address = old + value; return old;
}
inline unsigned atomicAdd(unsigned* address, unsigned value) {
  unsigned old = *address; *address = old + value; return old;
}
inline float atomicAdd(float* address, float value) {
  float old = *address; *address = old + value; return old;
}
inline double atomicAdd(double* address, double value) {
  double old = *address; *address = old + value; return old;
}
inline long long atomicAdd(long long* address, long long value) {
  long long old = *address; *address = old + value; return old;
}

template <typename T> __device__ T warpReduceAdd(T v);
template <typename T> __device__ T warpReduceMax(T v);

#endif // CRUCIBLE_CUDA_NVRTC_STUB_H
