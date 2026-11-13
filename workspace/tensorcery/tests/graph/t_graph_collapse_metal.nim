# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Eval collapses each root into a leaf over its computed impl.
## Run:
##   nim c -r -d:release --hints:off --warnings:off --outdir:build/tests/t_graph_collapse_metal \
##     --nimcache:nimcache/t_graph_collapse_metal workspace/tensorcery/tests/graph/t_graph_collapse_metal.nim

import std/importutils
import workspace/crucible
import workspace/ceramic
import workspace/iface
import workspace/tensorcery

privateAccess(Tensor)

proc main =
  block: # Re-evaluating a collapsed root reads the leaf, not the graph
    var x = @[1.0'f32, 2, 3, 4].toTensor([4], runMetal)
    let y = x + x
    doAssert not y.node.isPreAllocated()
    doAssert y.item(float32) == 2.0'f32
    doAssert y.node.isPreAllocated(), "eval must replace the root by its computed leaf"
    doAssert y.item(float32) == 2.0'f32

  block: # A tensor built after the eval captures the leaf
    var x = @[1.0'f32, 2, 3, 4].toTensor([4], runMetal)
    let y = x + x
    doAssert y.item(float32) == 2.0'f32
    let z = y + x
    doAssert z.item(float32) == 3.0'f32

  block: # Views over computed roots survive the collapse
    var x = @[1.0'f32, 2, 3, 4, 5, 6].toTensor([2, 3], runMetal)
    let v = x.narrow(1, 0, 2)
    doAssert v.item(float32) == 1.0'f32
    doAssert v.item(float32) == 1.0'f32

  block: # A view over an unevaluated computed root survives the collapse
    var x = @[1.0'f32, 2, 3, 4, 5, 6].toTensor([2, 3], runMetal)
    let y = x + x
    let v = y.narrow(1, 0, 2)
    doAssert v.item(float32) == 2.0'f32
    doAssert v.item(float32) == 2.0'f32

  block: # Deeper graph, two roots sharing a subtree
    var x = @[1.0'f32, 2].toTensor([2], runMetal)
    let a = x + x
    let b = a + a
    doAssert b.item(float32) == 4.0'f32
    doAssert a.item(float32) == 2.0'f32
    doAssert b.item(float32) == 4.0'f32

  echo "tensorcery graph collapse (Metal): all checks passed"

main()
