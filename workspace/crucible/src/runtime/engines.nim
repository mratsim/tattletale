# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
import ../codegen/ir/gpu_types
import ./engines_common
import ./launch/launch_kernel
import ./launch/launch_serialization
import ./launch/launch_device_fn
import ./engines/nvrtc {.all.}
import ./engines/cl {.all.}
import ./engines/vk {.all.}
import ./engines/wgpu {.all.}
import ./engines/metal {.all.}


export gpu_types
# Curated launch surface: geometry types, the two launch verbs,
# chevron and do forms. launch_kernel and launch_device_fn export
# exactly the facade via `*` marks, tests import runtime/launch modules directly.
export engines_common, launch_kernel, launch_device_fn

export ingest, getArtifact, launchKernelImpl, check, deviceName
export allocDevice, freeDevice, backendKind, copyToDeviceFromHost, copyToHostFromDevice

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════


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

