# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Run:
##   nim c -r -d:release --hints:off --warnings:off --outdir:build/tests/t_ops_metadata_opencl \
##     --nimcache:nimcache/t_ops_metadata_opencl workspace/tensorcery/tests/ops_metadata/t_ops_metadata_opencl.nim


import workspace/crucible
import workspace/ceramic
import ./ops_swapaxes_template
import ./ops_views_template

proc main =
  runSwapaxesTest(runOpenCL, "OpenCL")
  runViewsTest(runOpenCL, "OpenCL")

when isMainModule:
  main()
