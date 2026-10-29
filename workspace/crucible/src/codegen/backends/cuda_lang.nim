# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CUDA C++ printer for NVRTC: the `backends_api` entry point over a CIRv2 module.
##
## Interim state: the `cuda:` block refuses until the CUDA emission re-homes onto CIRv2.
##
## The Metal printer (`msl_lang.nim`) is the CIRv2 emission reference.
## V1's `cuda_lang` emission rules are the differential contract.

import std/macros
import ../ir/ast_datatypes

proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as CUDA C++ source.
  ## Refuses until the CUDA emission re-homes onto CIRv2.
  error("cuda_lang: the CUDA printer is not re-homed onto CIRv2 yet, " &
    "msl_lang.nim is the CIRv2 emission reference and v1 cuda_lang.nim " &
    "the differential contract")
