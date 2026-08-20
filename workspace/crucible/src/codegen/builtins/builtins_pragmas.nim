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

template nimonly*(): untyped {.pragma.}
template cudaName*(s: string): untyped {.pragma.}

# Dummy data for the typed nature of the `cuda` macro. These define commonly used
# CUDA specific names so that they produce valid Nim code in the context of a typed macro.
template global*() {.pragma.}
template workgroup*(size: untyped): untyped {.pragma.}
  ## `{.workgroup: (X, Y, Z).}` on a kernel proc bakes the workgroup size
  ## into the generated shader, local_size_xyz on Vulkan or @workgroup_size
  ## on WebGPU. Absent → per-backend default, Vulkan 256 or WebGPU 64,
  ## both 1D.
template device*() {.pragma.}
template forceinline*() {.pragma.}

template builtin*() {.pragma.}
  ## If attached to a function, type or variable it will refer to a built in
  ## in the target backend. This is used for all the functions, types and variables
  ## defined below to indicate that we do not intend to generate code for them.
template const_mem*(): untyped {.pragma.}
  ## If attached to a `var` it will be treated as constant memory
  ## (MSL `constant` / CUDA `__constant__` / WGSL `uniform`). Only useful
  ## if you want to define a constant without initializing it (and then
  ## use `cudaMemcpyToSymbol` / `copyToSymbol` to initialize it before
  ## executing the kernel)
template smem*(): untyped {.pragma.}
  ## Address-space pragmas for variable declarations inside GPU blocks.
  ## The var's address space resolves to the unified `AddressSpace` enum:
  ## `{.smem.}` → asSMEM (block/threadgroup shared memory), `{.rmem.}` →
  ## asRMEM (per-thread storage), `{.const_mem.}` → asConstant. The default
  ## (no pragma) is asDevice.
template rmem*(): untyped {.pragma.}
  ## Per-thread (register) storage — see `smem` for the address-space
  ## pragma family.
