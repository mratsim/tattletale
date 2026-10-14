# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible/src/abis/nvidia_abi

proc cudaGetComputeCapability*(): int =
  ## Compute capability of device 0 packed as one int (120 for 12.0),
  ## used to pick the NVRTC `--gpu-architecture` target at compile time.
  check cuInit(0)
  var device: CUdevice
  check cuDeviceGet(device, 0)
  var major, minor: int32
  check cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
  check cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
  result = int(major) * 10 + int(minor)
