# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal RMSNorm kernel: layout matrix vs the f64 host reference
##
## Test that the tile-reduce RMSNorm matches the f64 host reference on Metal
## over the full shape/stride/axis case matrix, ragged rows, wide multi-tile rows
## and strided destinations, size-1 axes, wide dynamic range, weighted and unweighted.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/k_rmsnorm/t_kernel_rmsnorm_metal.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_rmsnorm/kernel_rmsnorm_template

proc runTest =
  var engine = bkMetal.newEngine()
  run_rmsnorm_tests(engine, moduleName = "test_k_rmsnorm_metal")

when isMainModule:
  runTest()
