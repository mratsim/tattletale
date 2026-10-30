# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Target vocabulary shared across the crucible boundary, plus the codegen
## pipeline context that the readers below import.
##
## Readers:
##
## - the public builtins (`builtins_functions`, `builtins_gpu_types`) import
##   `BackendKind` and `GpuTypeKind`
## - the runtime engines import `BackendKind`
## - the backend printers receive the compilation state through `GpuContext`

import ./cirv2_lang

type
  BackendKind* = enum
    bkCuda,   ## CUDA C++ backend, NVRTC compiled
    bkWebGPU, ## WebGPU backend, WGSL emitted
    bkOpenCL, ## OpenCL C backend
    bkGlsl,   ## GLSL backend, executed on Vulkan
    bkMetal,  ## Metal backend, MSL emitted
    bkHlsl,   ## HLSL backend, glslang compiled

  GpuTypeKind* = enum
    gtVoid,
    gtBool, gtUint8, gtUint16, gtInt16, gtUint32, gtInt32, gtUint64, gtInt64, gtFloat32, gtFloat64,
    gtFloat16, gtBf16, gtF8E4M3, gtF8E5M2, gtSize_t,
    gtStatic       # Static integer value (used for generic params)
    gtArray,       # Static array `array[N, dtype]` -> `dtype[N]`
    gtString,
    gtObject,      # Struct types
    gtPtr,         # Pointer type over an inner type
    gtUncheckedArray,          # UncheckedArray, mapped to runtime sized arrays
    gtSpan,         # Runtime-length array view (openArray/varargs)
    gtGenericInst, # Instantiated generic type with one or more generic arguments (instantiated!)
    gtVoidPtr      # Opaque void pointer
    gtInvalid      # Can be returned to indicate a call to `nimToGpuType` failed to determine a type
    gtCompileTimeValue  # a compile-time argument value in the parameter's type slot

  GpuContext* = ref object
    ## Compilation state for the backend printers.
    ##
    ## The `module` field is the CIRv2 module after the pass registry: legalization, normalization, optimization, lowering, validation.
    ##
    ## Flow:
    ## - the `metal:`-family macros build the module through the frontend
    ## - the pass registry runs over the module
    ## - `backends_api.codegen` prints `ctx.module`, never mutating it
    module*: CirModule
