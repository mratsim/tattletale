# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CUDA driver bindings used by the NVRTC engine (engines/nvrtc.nim).
##
## The legacy macro layer (execCuda/execCudaImpl/maybeWrap/CudaDim3/dim3 and
## the argument-marshalling helpers they used) is deleted: the HwEngine's
## chevron LaunchConfig carries the full 3D launch extents and `runImpl`
## marshals ArgBlobs directly. This module is now a thin shim importing the
## low-level driver API (cuModuleLoadData, cuModuleGetFunction, cuMemAlloc,
## cuMemcpyHtoD, cuMemcpyDtoH, cuMemFree, cuLaunchKernel, cuCtxSynchronize,
## cuCtxSetCurrent, cuCtxDestroy, cuModuleUnload, the cuEvent* timing procs
## and the `check` template) from the NVIDIA ABI binding.

import workspace/crucible/src/abis/nvidia_abi
