# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal fixture replay for the einsum macro's nine elementary recipes.
##
## Test that every recipe replays its fixtures through the `einsum` macro,
## matching the numpy f64 reference, stride-permuted operands included.
##
## Run from the repository root:
##   nim c -r --hints:off --warnings:off --threads:off --outdir:build/tests \
##     --nimcache:nimcache/tests/k_einsum \
##     workspace/ceramic/tests/kernels/k_einsum/t_kernel_einsum_metal.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_einsum/kernel_einsum_template

proc replayAllFixtures() =
  var engine = newEngine(runMetal)
  const fixtures = [
    "transpose_rm", "transpose_cm",
    "sum_total_rm",
    "sum_row_rm", "sum_row_cm",
    "sum_col_rm", "sum_col_cm",
    "matvec_cm",
    "mm_rm", "mm_cm",
    "dot_rm",
    "hadamard_cm",
    "outer_rm",
  ]
  for name in fixtures:
    engine.replayEinsumFixture(name)
  echo "all einsum fixture replays pass (9 recipes, ", fixtures.len, " fixtures)"

replayAllFixtures()
