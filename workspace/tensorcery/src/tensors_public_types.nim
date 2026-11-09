# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible/src/builtins/builtins_gpu_types
import workspace/crucible/src/runtime/engines/engines_api

export engines_api.Runtime

type Dtype* = enum
    kFp16, kBf16, kFp32

func toDtype*(_: typedesc[float32]): Dtype =
  kFp32

func toDtype*(_: typedesc[bfloat16]): Dtype =
  kBf16
