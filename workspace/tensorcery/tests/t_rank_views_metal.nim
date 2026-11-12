# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Rank-changing and negative-stride views through eval on Metal
## Run:
##   nim c -r -d:release --hints:off --warnings:off --outdir:build/tests/t_rank_views_metal \
##     --nimcache:nimcache/t_rank_views_metal workspace/tensorcery/tests/t_rank_views_metal.nim

from std/importutils import privateAccess
import workspace/crucible
import workspace/ceramic
import workspace/tensorcery

privateAccess(Tensor)

proc main =
  const run = runMetal

  # Squeeze rank-3 -> rank-2 over a pre-allocated source, read back with item
  block:
    var x = @[7.0'f32].toTensor([1, 1, 1], run, bfloat16)
    let vNode = x.engine.newViewNode(x.node, [1, 1], [1, 1], 0, "squeeze")
    let sx = Tensor[run](engine: x.engine, node: vNode)
    doAssert sx.shape()[0] == 1 and sx.shape()[1] == 1
    doAssert sx.item(bfloat16) == 7.0'f32.to(bfloat16)

  # Squeeze rank-3 -> rank-2 over an op output: the view binds the source
  # data at eval, through enqueueView
  block:
    var x = @[3.5'f32].toTensor([1, 1, 1], run, bfloat16)
    let y = x + x
    let vNode = x.engine.newViewNode(y.node, [1, 1], [1, 1], 0, "squeeze")
    let sy = Tensor[run](engine: x.engine, node: vNode)
    doAssert sy.item(bfloat16) == 7.0'f32.to(bfloat16)

  # Reshape rank-2 -> rank-3 over an op output, read back with allClose
  block:
    var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
    var x = xd.toTensor([2, 4], run, bfloat16)
    let y = x + x
    let vNode = x.engine.newViewNode(y.node, [2, 2, 2], [4, 2, 1], 0, "reshape")
    let r = Tensor[run](engine: x.engine, node: vNode)
    var expected: array[8, bfloat16]
    for i in 0 ..< 8:
      expected[i] = (2.0'f32 * xd[i]).to(bfloat16)
    doAssert r.allClose(expected.toTensor([2, 2, 2], run, bfloat16))

  # Negative-stride view of shape (4,), stride -1, offset 3 over a 4-element source
  block:
    var xd = @[10.0'f32, 20, 30, 40]
    var x = xd.toTensor([4], run, bfloat16)
    let y = x + x
    let vNode = x.engine.newViewNode(y.node, [4], [-1], 3, "negstride")
    let v = Tensor[run](engine: x.engine, node: vNode)
    var expected: array[4, bfloat16]
    for i in 0 ..< 4:
      expected[i] = (2.0'f32 * xd[3 - i]).to(bfloat16)
    doAssert v.allClose(expected.toTensor([4], run, bfloat16))

  echo "tensorcery rank-changing views (Metal): all checks passed"

when isMainModule:
  main()
