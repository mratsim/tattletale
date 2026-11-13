# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Never-eval check: toTensor only, no `+`, no eval on Metal
## Run:
##   nim c -r -d:release --hints:off --warnings:off --outdir:build/tests/t_graph_never_eval_metal \
##     --nimcache:nimcache/t_graph_never_eval_metal workspace/tensorcery/tests/graph/t_graph_never_eval_metal.nim

import workspace/crucible
import workspace/ceramic
import workspace/tensorcery

proc main =
  var x = @[1.0'f32, 2.0, 3.0, 4.0].toTensor([4], runMetal)
  doAssert x.dtype() == kFp32
  echo "tensorcery never-eval probe (Metal): passed"

when isMainModule:
  main()
