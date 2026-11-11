# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Tensorcery ops_0i1o factories on Metal
##
## Test that the factory faces construct, eval and read back correct values, that invalid calls refuse and that size-0 shapes raise.
##
## Run:
##   nim c -r -d:release --hints:off --warnings:off \
##     --nimcache:nimcache/t_ops_0i1o_metal workspace/tensorcery/tests/ops_0i1o/t_ops_0i1o_metal.nim

import workspace/crucible
import workspace/ceramic
import ./ops_factories_template

proc main =
  runFactoriesTest(runMetal, "Metal")

when isMainModule:
  main()
