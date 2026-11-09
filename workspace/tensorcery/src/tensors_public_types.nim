# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Public tensor types: the element-type vocabulary for construction
##
## - Dtype is the leaf of the type tower, the ifaces and the erased user
##   face consume it
## - everything else (Span, the impl types, the ifaces) is internal

import workspace/crucible/src/builtins/builtins_gpu_types

type Dtype* = enum
    kFp16, kBf16, kFp32

func toDtype*(_: typedesc[float32]): Dtype =
  kFp32

func toDtype*(_: typedesc[bfloat16]): Dtype =
  kBf16
