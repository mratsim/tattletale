# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            Backend kinds
#
# ############################################################

type
  CodegenKind* = enum
    ## One GPU target kind, the key of every per-target table:
    ## the `TargetVocabs` array, the engine dispatch, the printer sections.
    cgCuda
    cgWebGPU
    cgOpenCL
    cgGlsl
    cgMetal
    cgHlsl
    cgSpirv
    cgCpu      ## CPU backend, C or Nim text for the host compiler
    cgLean
    ## Lean 4 verification backend: emits a proof module, never executable shader code.
