# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_sum/t_kernel_sum_axis_metal.nim (from tattletale)

import workspace/crucible
import workspace/ceramic/tests/kernels/k_sum/kernel_sum_template

proc main =
  var engine = bkMetal.newEngine()
  run_sum_axis_tests(engine, moduleName = "test_k_sum_axis_metal")

when isMainModule:
  main()
