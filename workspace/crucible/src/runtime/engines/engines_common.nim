# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared leaf: the launch modules, the engines and the public API import it.
## DeviceMem and the per-backend buffer release share this module, the type
## and its allocator live in one scope like the stdlib's memalloc and seqs_v2,
## the release procs are private and never re-exported.

import workspace/crucible/src/codegen/ir/ast_datatypes
export ast_datatypes.BackendKind

import workspace/crucible/src/abis/nvidia_abi
import workspace/crucible/src/abis/objc_abi as objc
import workspace/crucible/vendor/wgpu
import workspace/crucible/src/runtime/errors
import workspace/crucible/src/runtime/exec/opencl_runtime
import workspace/crucible/src/runtime/exec/vulkan_runtime

type
  Dim3* = object
    x*, y*, z* = 1

  LaunchConfig* = object
    grid*, blk*: Dim3
    sharedMem*, stream*: int

template dim3*(a, b, c: SomeInteger = 1): Dim3 =
  Dim3(x: int(a), y: int(b), z: int(c))

template dim3*(t: tuple[a: SomeInteger]): Dim3 = Dim3(x: int(t[0]))
template dim3*(t: tuple[a, b: SomeInteger]): Dim3 = Dim3(x: int(t[0]), y: int(t[1]))
template dim3*(t: tuple[a, b, c: SomeInteger]): Dim3 = Dim3(x: int(t[0]), y: int(t[1]), z: int(t[2]))
template dim3*(d: Dim3): Dim3 = d

proc freeDeviceImpl(h: CUdeviceptr) {.inline.} =
  check cuMemFree(h)

proc freeDeviceImpl(h: OpenCLBuffer) {.inline.} =
  var h = h
  h.dealloc()

proc freeDeviceImpl(h: VulkanBuffer) {.inline.} =
  var h = h
  h.dealloc()

proc freeDeviceImpl(h: WGPUBuffer) {.inline.} =
  wgpuBufferRelease(h)

proc freeDeviceImpl(h: objc.ID) {.inline.} =
  if not objc.isNil(h):
    objc.release(h)

type
  DeviceMem*[H, T] = object
    ## Typed on-device memory buffer owning its memory: released on scope
    ## exit and move-only.
    mem*: H

proc `=destroy`*[H, T](mem: var DeviceMem[H, T]) {.inline.} =
  freeDeviceImpl(mem.mem)

proc `=copy`*[H, T](dst: var DeviceMem[H, T], src: DeviceMem[H, T]) {.
    error: "DeviceMem owns its buffer and is move-only".}
