# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Eval sites dispatch ops first marked between earlier eval sites
##
## Test that a second eval site dispatches an op whose first call sits
## between the two sites, through the item, allClose and eval surfaces
## whose proc chain dispatches on the threaded snapshot.
##
## Run:
##   nim c -r -d:release --hints:off --warnings:off --outdir:build/tests/t_snapshot_threading_metal \
##     --nimcache:nimcache/t_snapshot_threading_metal workspace/tensorcery/tests/graph/t_snapshot_threading_metal.nim

import std/importutils
import workspace/crucible
import workspace/ceramic
import workspace/tensorcery

privateAccess(Tensor)

proc main =
  const run = runMetal

  block: # item: site 2 walks a graph holding the op first called between the sites
    var x = @[1.0'f32, 2].toTensor([2], run)
    let y = x + x
    doAssert y.item(float32) == 2.0'f32
    let o = ones([2], run)
    let z = y + o
    doAssert z.item(float32) == 3.0'f32

  block: # allClose: same ordering over the second evaluating surface
    var x = @[1.0'f32, 2].toTensor([2], run)
    let y = x + x
    doAssert y.allClose(@[2.0'f32, 4].toTensor([2], run))
    let o = full([2], 1.0'f32, run)
    let z = y + o
    doAssert z.allClose(@[3.0'f32, 5].toTensor([2], run))
    doAssert not z.allClose(@[4.0'f32, 6].toTensor([2], run))

  block: # eval: same ordering over the varargs surface
    var x = @[1.0'f32, 2].toTensor([2], run)
    let y = x + x
    eval(y)
    let o = arange(2.0'f32, run)
    let z = y + o
    eval(z)
    doAssert z.node.isPreAllocated(), "eval must replace the root by its computed leaf"
    doAssert z.item(float32) == 2.0'f32

  echo "tensorcery snapshot threading (Metal): all checks passed"

main()
