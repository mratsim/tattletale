# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Tensorcery ops_neuralnets ops on Metal

import workspace/crucible
import workspace/ceramic
import ./ops_rmsnorm_template
import ./ops_linear_template

proc main =
  runRmsnormTest(runMetal, "Metal")
  runLinearTest(runMetal, "Metal")

when isMainModule:
  main()
