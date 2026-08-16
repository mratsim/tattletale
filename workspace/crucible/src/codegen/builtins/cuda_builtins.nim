# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

type
  ## Dummy objects to make CUDA block/thread index access possible in the
  ## *typed* `cuda` macro. Fields are plain `int` (Nim int64) so
  ## threadIdx.x/blockIdx.x match the layout templates' `int or Int` params
  ## without casts. NOTE: do NOT introduce a named alias here — the codegen
  ## resolves field types by name and emits C++ tuple struct names from it
  ## (Tuple_F0_Dim vs Tuple_F0_int mismatch broke tuple-concat emission).
  ## The codegen emits these by name, so the width only affects Nim-side
  ## typing — at emission `int` maps to int32 anyway
  ## (toGpuTypeKind: ntyInt -> gtInt32, gpu_type_constructors.nim), so this
  ## is emission-neutral (verified: emitted PTX is all-32-bit index math).
  ## The int32/SomeInteger idea remains only to make the Nim-side typing
  ## match the emitted width — the IR currently types these int64 while
  ## the generated CUDA is 32-bit (benign at harness sizes; latent for
  ## grids >= 2^32 threads). NOT a GPU perf issue.
  NvBlockIdx* = object
    x*, y*, z*: int
  NvBlockDim = object
    x*, y*, z*: int
  NvThreadIdx* = object
    x*, y*, z*: int
  NvGridDim = object
    x*, y*, z*: int


## These are dummy elements to make CUDA block / thread index / dim
## access possible in the *typed* `cuda` macro. It cannot be `const`,
## because then the typed code would evaluate the values before we
## can work with it from the typed macro.
let blockIdx* {.builtin, compileTime.} = NvBlockIdx()
let blockDim* {.builtin, compileTime.} = NvBlockDim()
let gridDim* {.builtin, compileTime.} = NvGridDim()
let threadIdx* {.builtin, compileTime.} = NvThreadIdx()

## Similar for procs. They don't need any implementation, as they won't ever be actually called.
proc printf*(fmt: string) {.varargs, builtin.} = discard
proc memcpy*(dst, src: pointer, size: int) {.builtin.} = discard

## While you can use `malloc` on device with small sizes, it is usually not
## recommended to do so.
proc malloc*(size: csize_t): pointer {.builtin.}  = discard
proc free*(p: pointer) {.builtin.} = discard
proc syncthreads*() {.cudaName: "__syncthreads", builtin.} = discard
proc cvtaGenericToShared*(p: pointer): uint32 {.cudaName: "__cvta_generic_to_shared", builtin.} = discard
