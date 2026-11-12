# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Tensorcery ops_metadata ops on Metal
##
## Run:
##   nim c -r -d:release --hints:off --warnings:off --outdir:build/tests/t_ops_metadata_metal \
##     --nimcache:nimcache/t_ops_metadata_metal workspace/tensorcery/tests/ops_metadata/t_ops_metadata_metal.nim

import workspace/crucible
import workspace/ceramic
import ./ops_swapaxes_template
import ./ops_views_template

proc main =
  runSwapaxesTest(runMetal, "Metal")
  runViewsTest(runMetal, "Metal")

when isMainModule:
  main()
