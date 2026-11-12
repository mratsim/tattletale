# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## One-add check: toTensor + `+` + eval + item on Metal
## Run:
##   nim c -r -d:release --hints:off --warnings:off --outdir:build/tests/t_graph_one_add_metal \
##     --nimcache:nimcache/t_graph_one_add_metal workspace/tensorcery/tests/graph/t_graph_one_add_metal.nim
##
## Dumps the ingested MSL artifact to build/poc/one_add_artifact.msl.
## Byte-compare it against the eager enqueue-time lowering on master.

from std/importutils import privateAccess
import workspace/crucible
import workspace/ceramic
import workspace/iface
import workspace/tensorcery

privateAccess(Tensor)

proc main =
  var x = @[1.0'f32, 2.0, 3.0, 4.0].toTensor([4], runMetal)
  let y = x + x
  doAssert y.item(float32) == 2.0'f32
  let engine = x.engine.to(MetalEngine)
  writeFile("build/poc/one_add_artifact.msl", engine.getArtifact())
  echo "tensorcery one-add probe (Metal): passed"

when isMainModule:
  main()
