# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# NOTE: For the moment we import and export builtins here for all backends.
# Once we change the code to make single backends importable on their own,
# this will be changed and these builtins will be imported/exported in the
# corresponding CUDA/WGSL etc module the user needs to import.
import ./builtins_pragmas
import ./cuda_builtins
import ./wgsl_builtins
import ./opencl_builtins
import ./vulkan_builtins
import ./multigpu_builtins

export builtins_pragmas
export cuda_builtins
export wgsl_builtins
export opencl_builtins
export vulkan_builtins
export multigpu_builtins
