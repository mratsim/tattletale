# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test:
##   tile_reductions' two sums on Metal
##
## Test that the full and last-axis tensor sums match the host references on Metal.
##
## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_sum/t_kernel_sum.nim (from tattletale)

import std/strformat
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_sum/k_sum_kernels
import workspace/ceramic/tests/kernels/k_sum/t_kernel_sum_template

proc main =
  var engine = bkMetal.newEngine()
  run_sum_tests(engine, moduleName = "test_k_sum_metal")

when isMainModule:
  main()
