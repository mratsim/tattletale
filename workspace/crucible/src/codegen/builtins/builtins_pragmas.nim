# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

#import std / [macros, strutils, sequtils, options, sugar, tables, strformat, hashes, sets]
#
#import ./gpu_types
#import ./backends/backends
#import ./nim_to_gpu
#
#export gpu_types

## Pragma templates for the GPU DSL. The address-space pragmas (`smem`,
## `rmem`, `const_mem`) set a var declaration's address space. Unannotated
## vars default to device memory (`asDevice`).
template nimonly*(): untyped {.pragma.}
template cudaName*(s: string): untyped {.pragma.}

template global*() {.pragma.}
template workgroup*(size: untyped): untyped {.pragma.}
  ## Workgroup size annotation for a kernel proc: `{.workgroup: (X, Y, Z).}`
  ## sets the threads per threadgroup. Backend spellings:
  ##   `local_size_xyz` on Vulkan
  ##   `@workgroup_size` on WebGPU
  ## On Metal and CUDA the size is set host-side at dispatch, so the
  ## annotation is not baked into the shader.
  ## Without the annotation, the generated shader declares a default workgroup size:
  ##   256×1×1 on Vulkan
  ##   64×1×1 on WebGPU
template device*() {.pragma.}
template forceinline*() {.pragma.}

template builtin*() {.pragma.}
  ## Marks a function, type, or variable as a builtin provided by the
  ## target backend. The compiler generates no code for it.
template const_mem*(): untyped {.pragma.}
  ## Constant memory pragma for a var declaration inside a GPU block.
  ## Allocates the var in read-only constant memory:
  ##   `constant` in MSL
  ##   `uniform` in WGSL
  ##   `__constant__` in CUDA
  ## Useful for a constant defined without an initializer, then filled
  ## before kernel launch (`cudaMemcpyToSymbol` / `copyToSymbol`).
template smem*(): untyped {.pragma.}
  ## Shared memory (smem) pragma for a var declaration inside a GPU block.
  ## Allocates the var in block/threadgroup-scope memory:
  ##   `threadgroup` in MSL
  ##   `workgroup` in WGSL
  ##   `__shared__` in CUDA
  ##
  ## Usage:
  ##   var scratch {.smem.}: array[64, uint32]
template rmem*(): untyped {.pragma.}
  ## Register memory (rmem) pragma for a var declaration inside a GPU block.
  ## Allocates the var in per-thread storage:
  ##   `thread` in MSL
  ##   `function` in WGSL
  ##   `__local__` in CUDA
  ##
  ## Usage:
  ##   var tile {.rmem.}: array[16, half]
