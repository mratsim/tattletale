# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ./safetensors,
  std/memfiles,
  std/strformat,
  std/tables,
  workspace/libtorch

# #######################################################################
#
#               Safetensors + libtorch syntactic sugar
#
# #######################################################################

# TODO: this will likely evolve and be put at a higher level in the stack
#       so that we can accelerate loading with multiple workers / async CUDA streams

proc toTorchType*(dtype: Dtype): ScalarKind =
  ## Convert safetensors dtype to libtorch ScalarKind.
  ## Raises ValueError if no direct mapping exists.

  case dtype:
    of BOOL: kBool
    of U8:   kUint8
    of I8:   kInt8
    of I16:  kInt16
    of F16:  kFloat16
    of BF16: kBfloat16
    of I32:  kInt32
    of F32:  kFloat32
    of C64:  kComplexF64
    of F64:  kFloat64
    of I64:  kInt64
    else:
      raise newException(
        ValueError,
        &"No direct libtorch mapping for safetensors dtype: {dtype}"
      )

proc getTensor*(st: Safetensor, memFile: MemFile, dataSectionOffset: int, tensorName: string): TorchTensor =
  let view = st.getMmapView(memFile, dataSectionOffset, tensorName)
  let info = st.tensors[tensorName]
  let torchType = info.dtype
  return view.data.from_blob(info.shape.asTorchView(), info.dType.toTorchType())
