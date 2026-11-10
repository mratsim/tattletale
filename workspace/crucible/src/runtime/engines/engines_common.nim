# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared leaf. Launch modules, engines, and the public API import it.
## DeviceMem and its per-backend release procs share one scope, like the stdlib's memalloc and seqs_v2.
## Each release proc is private and never re-exported.

import std/[typetraits, importutils]

import workspace/crucible/src/crucible_codegen
export crucible_codegen.CodegenKind

import workspace/crucible/src/abis/nvidia_abi
import workspace/crucible/src/abis/opencl_abi
import workspace/crucible/src/abis/objc_abi as objc
import workspace/crucible/vendor/wgpu
import workspace/crucible/src/runtime/backends/opencl_runtime
import workspace/crucible/src/runtime/backends/vulkan_common

from workspace/crucible/src/runtime/backends/opencl_runtime import check

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
  if culonglong(h) != 0:
    check cuMemFree(h)

proc freeDeviceImpl(h: OpenCLBuffer) {.inline.} =
  var h = h
  h.dealloc()

proc freeDeviceImpl(h: VulkanBuffer) {.inline.} =
  var h = h
  h.dealloc()

proc freeDeviceImpl(h: WGPUBuffer) {.inline.} =
  if not h.isNil:
    wgpuBufferRelease(h)

when defined(macosx):
  # objc.ID device handles only exist on Metal and objc_abi's release is macOS-only,
  # Linux builds of the engines layer fail to compile this overload without the guard
  proc freeDeviceImpl(h: objc.ID) {.inline.} =
    if not objc.isNil(h):
      objc.release(h)

type
  DeviceMem*[H, T] = object
    ## Typed on-device memory buffer owning its memory: released on scope
    ## exit and move-only.
    mem*: H

  DeviceBorrowsObj = object
    ## Device buffers a recorded launch borrows: retained at enqueue so the batch's launches can
    ## run after the enqueue site's owner left scope. Released when the last copy of the entry dies.
    handles*: seq[pointer]
    releaseImpl*: proc (h: pointer) {.nimcall.}

  DeviceBorrows* = ref DeviceBorrowsObj

proc deviceRetain*(h: objc.ID): pointer {.inline.} =
  ## +1 on a borrowed Metal buffer. Returns the raw handle for the borrow book.
  discard objc.retain(h)
  cast[pointer](h)

proc deviceRelease*(h: objc.ID) {.inline.} =
  freeDeviceImpl(h)

proc deviceRetain*(h: OpenCLBuffer): pointer {.inline.} =
  ## +1 on a borrowed OpenCL buffer. Returns the raw cl_mem for the borrow book.
  privateAccess(OpenCLBuffer)
  let status = retainMemObject(h.mem)
  opencl_runtime.check(status)
  h.mem

proc deviceRelease*(h: OpenCLBuffer) {.inline.} =
  privateAccess(OpenCLBuffer)
  if not h.mem.isNil:
    discard releaseMemObject(h.mem)

proc deviceRetain*(h: WGPUBuffer): pointer {.inline.} =
  ## +1 on a borrowed WebGPU buffer. Returns the raw handle for the borrow book.
  wgpuBufferReference(h)
  h

proc deviceRelease*(h: WGPUBuffer) {.inline.} =
  wgpuBufferRelease(h)

proc deviceRetain*(h: CUdeviceptr): pointer {.inline.} =
  ## CUDA device pointers are not reference counted: the DeviceMem's cuMemFree ownership is the only lifetime.
  nil

proc deviceRelease*(h: CUdeviceptr) {.inline.} =
  discard

proc deviceRetain*(h: VulkanBuffer): pointer {.inline.} =
  ## Vulkan buffers are not reference counted: the allocator or pool owns their lifetime.
  nil

proc deviceRelease*(h: VulkanBuffer) {.inline.} =
  discard

proc releaseBorrowed[H](h: pointer) {.nimcall.} =
  ## Releases one raw borrow recorded under handle type `H`.
  when H is OpenCLBuffer:
    discard releaseMemObject(cast[Pmem](h))
  else:
    deviceRelease(cast[H](h))

proc addBorrow*[H](cell: DeviceBorrows, h: H) {.inline.} =
  ## Retains one device buffer for the batch's lifetime and registers the matching release.
  ## Handle types without reference counting record nothing.
  let raw = deviceRetain(h)
  if not raw.isNil:
    cell.handles.add raw
    if cell.releaseImpl.isNil:
      cell.releaseImpl = releaseBorrowed[H]

proc `=destroy`(cell: var DeviceBorrowsObj) =
  if not cell.releaseImpl.isNil:
    for h in cell.handles:
      cell.releaseImpl(h)

proc collectBorrows*(cell: var DeviceBorrows, v: auto) =
  ## Retains every DeviceMem leaf of `v` (a launch output or its args tuple)
  ## into `cell`, creating the cell on the first borrow.
  when typeof(v) is DeviceMem:
    if cell.isNil:
      cell = DeviceBorrows()
    cell.addBorrow(v.mem)
  elif typeof(v) is object or typeof(v) is tuple:
    for f in fields(v):
      collectBorrows(cell, f)

proc `=destroy`*[H, T](mem: var DeviceMem[H, T]) {.inline.} =
  freeDeviceImpl(mem.mem)

proc `=copy`*[H, T](dst: var DeviceMem[H, T], src: DeviceMem[H, T]) {.
    error: "DeviceMem owns its buffer and is move-only".}
