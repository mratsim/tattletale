# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
import ../codegen/ir/gpu_types
import ./chevrons
import ./launch_common
import ./engines/nvrtc {.all.}
import ./engines/cl {.all.}
import ./engines/vk {.all.}
import ./engines/wgpu {.all.}
import ./engines/metal {.all.}


export gpu_types
export chevrons
export launch_common

export ingest, getArtifact, launchKernel, check, deviceName
export allocDevice, freeDevice, backendKind, copyToDeviceFromHost, copyToHostFromDevice

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════

type
  HwEngine* = concept engine, type E
    ## A hardware engine: compiles GPU source and launches its kernels on one device.
    ## One concrete type per backend (`bkMetal.init()`, `bkCuda.init()`, `bkOpenCL.init()`, `bkVulkan.init()`, `bkWGSL.init()`).
    ##
    ## Usage:
    ##   engine.ingest(source)              # compile kernels, replaces the previous artifact.
    ##   engine.launchKernel("kernel", out, args)    # launch, or engine.launchKernel<<grid, blk>> for geometry.
    ##   engine.deviceName()                # the device this engine drives.
    ##
    ## `ingest` compiles the whole source in one shot, one call per engine.
    ## `run` takes the output buffer first, then the kernel args in declaration order.
    proc ingest(engine: E, source: string)
    proc getArtifact(engine: E)
    proc deviceName(engine: E): string
    proc launchKernel(engine: E): LaunchSugar[E]
  # Concept procs cannot use `auto` returns, bare signatures only.

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════


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

proc init*(backend: static BackendKind, source: string): auto =
  result = init(backend)
  result.ingest(source)

