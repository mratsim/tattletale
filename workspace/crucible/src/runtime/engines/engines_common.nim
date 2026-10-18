# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared leaf: the launch modules, the engines and the public API import it.
## Imports gpu_types for BackendKind only (the IR does not import back).

import ../../codegen/ir/gpu_types
export gpu_types.BackendKind

type
  Dim3* = object
    ## A 3D launch dims, `Dim3(x: n)` defaults y/z to 1.
    x*, y*, z* = 1

  LaunchConfig*[E] = object
    engine*: E
    grid*, blk*: Dim3

template dim3*(a, b, c: SomeInteger = 1): Dim3 =
  Dim3(x: int(a), y: int(b), z: int(c))

template dim3*(t: tuple[a: SomeInteger]): Dim3 = Dim3(x: int(t[0]))
template dim3*(t: tuple[a, b: SomeInteger]): Dim3 = Dim3(x: int(t[0]), y: int(t[1]))
template dim3*(t: tuple[a, b, c: SomeInteger]): Dim3 = Dim3(x: int(t[0]), y: int(t[1]), z: int(t[2]))
template dim3*(d: Dim3): Dim3 = d

type
  DeviceMem*[H, T] = object
    ## Typed on-device memory buffer
    mem*: H

type
  HwEngine* = concept engine, type E
    proc ingest(engine: E, source: string)
    proc getArtifact(engine: E)
    proc deviceName(engine: E): string
    proc allocDeviceImpl(engine: E, nbytes: int)
    proc freeDeviceImpl(engine: E, handle: auto)
    proc copyToDeviceFromHostImpl(engine: E, dst: auto, src: pointer, n: int)
    proc copyToHostFromDeviceImpl(engine: E, dst: pointer, src: auto, n: int)

mixin newCudaEngine, newOpenCLEngine, newVulkanEngine, newWgpuEngine, newMetalEngine

proc init*(backend: static BackendKind): auto =
  when backend == bkCuda:
    result = newCudaEngine()
  elif backend == bkOpenCL:
    result = newOpenCLEngine()
  elif backend == bkVulkan:
    result = newVulkanEngine()
  elif backend == bkWGSL:
    result = newWgpuEngine()
  elif backend == bkMetal:
    result = newMetalEngine()
  else:
    {.error: "init: unknown BackendKind".}

