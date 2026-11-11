# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Tensorcery ops_blas ops on OpenCL

import workspace/crucible
import workspace/ceramic
import ./ops_gemm_template

proc main =
  runGemmTest(runOpenCL, "OpenCL")

when isMainModule:
  main()
