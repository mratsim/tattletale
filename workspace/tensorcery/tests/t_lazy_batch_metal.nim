# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Batch artifact generation is deferred to runtime.
##
## - enqueueing without `submit` compiles without running the codegen pass pipeline
## - the first `submit` materializes the artifact and the kernel launches
## Run:
##   nim c -r -d:release --hints:off --warnings:off --outdir:build/tests/t_lazy_batch_metal \
##     --nimcache:nimcache/t_lazy_batch_metal workspace/tensorcery/tests/t_lazy_batch_metal.nim

from std/importutils import privateAccess
import workspace/crucible
import workspace/ceramic
import workspace/tensorcery
import workspace/tensorcery/src/compute_graph/batching

privateAccess(Tensor)

proc main =
  const run = runMetal

  # Enqueue only, no submit: must not compile the artifact
  block:
    var x = @[1.0'f32, 2.0].toTensor(run, [2])
    let y = x + x
    let engine = newEngine(runMetal)
    var batch = engine.recordBatch()
    engine.enqueueNode(batch, y.node)

  # Submit: the artifact materializes at runtime and the kernel launches
  block:
    var x = @[1.0'f32, 2.0].toTensor(run, [2])
    let y = x + x
    doAssert y.allClose(@[2.0'f32, 4.0].toTensor(run, [2]))

  echo "tensorcery lazy batch (Metal): all checks passed"

when isMainModule:
  main()
