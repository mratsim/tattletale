# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/tensorcery/src/tensorcery
import workspace/tensorcery/src/tensors_internals
import workspace/tensorcery/src/tensors_globals
import workspace/tensorcery/src/compute_graph/dispatch
import workspace/tensorcery/src/compute_graph/ops_2i1o/ops_add
import workspace/tensorcery/src/compute_graph/eval
export eval.eval

export tensorcery
# erased layer imports, the vtable mixins resolve at the outermost scope
export tensors_internals

proc runTest*(run: static Runtime, name: string) =
  var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  var yd = @[2.0'f32, 2, 2, 2, 2, 2, 2, 2]
  var xq: array[8, bfloat16]
  var yq: array[8, bfloat16]
  for i in 0 ..< 8:
    xq[i] = xd[i].to(bfloat16)
    yq[i] = yd[i].to(bfloat16)
  let x = toTensor(xq, @[2, 4])
  let y = toTensor(yq, @[2, 4])

  doAssert x.shape().len == 2
  doAssert x.shape()[0] == 2
  doAssert x.shape()[1] == 4
  doAssert x.strides()[0] == 4
  doAssert x.strides()[1] == 1
  doAssert x.dtype() == kBf16

  let z = x + y
  let parts = split(x, 0, 2)
  doAssert parts.len == 2
  for p in parts:
    doAssert p.shape()[0] == 1
    doAssert p.shape()[1] == 4

  let small = toTensor(yq[0 ..< 4], @[1, 4])
  # face checks are contracts, they raise AssertionDefect
  doAssertRaises AssertionDefect:
    discard x + small
  doAssertRaises AssertionDefect:
    discard split(x, 0, 3)
  doAssertRaises AssertionDefect:
    discard toTensor(xq[0 ..< 7], @[2, 4])

  for i in 0 ..< 8:
    let xi = xd[i].to(bfloat16).to(float32)
    let yi = yd[i].to(bfloat16).to(float32)
    doAssert z.item(i div 4, i mod 4, bfloat16) == (xi + yi).to(bfloat16), name & ": add " & $i
    doAssert x.item(i div 4, i mod 4, bfloat16) == xd[i].to(bfloat16), name & ": split source " & $i
  for c in 0 ..< 4:
    doAssert parts[0].item(0, c, bfloat16) == xd[c].to(bfloat16), name & ": part0 " & $c
    doAssert parts[1].item(0, c, bfloat16) == xd[4 + c].to(bfloat16), name & ": part1 " & $c
  # inner-axis split makes strided views, strides inherit the source
  let cols = split(x, 1, 2)
  let w = cols[0] + cols[1]
  for r in 0 ..< 2:
    for c in 0 ..< 2:
      let e0 = xd[r * 4 + c].to(bfloat16).to(float32)
      let e1 = xd[r * 4 + c + 2].to(bfloat16).to(float32)
      doAssert w.item(r, c, bfloat16) == (e0 + e1).to(bfloat16), name & ": colsplit " & $r & "," & $c

  # Memoize computation and drop ancestor nodes
  let d = x + y
  let e = d + x
  eval(e)
  eval(e)
  for i in 0 ..< 8:
    doAssert e.item(i div 4, i mod 4, bfloat16) ==
      (xd[i].to(bfloat16).to(float32) * 2'f32 + yd[i].to(bfloat16).to(float32)).to(bfloat16),
      name & ": diamond " & $i

  echo "tensorcery Tensor (", name, "): all checks passed"


proc runDedupeTest*(run: static Runtime, name: string) =
  ## One-enqueue semantics of the per-eval visited set.
  ## Kernel launches count as batch entries: 1 entry per enqueue.
  var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  var yd = @[10.0'f32, 20, 30, 40, 50, 60, 70, 80]
  var xq: array[8, bfloat16]
  var yq: array[8, bfloat16]
  for i in 0 ..< 8:
    xq[i] = xd[i].to(bfloat16)
    yq[i] = yd[i].to(bfloat16)
  let engine = newEngine(run)
  var shape = @[2, 4]
  let x = engine.newTensorNode(kBf16, shape.toSpan())
  let y = engine.newTensorNode(kBf16, shape.toSpan())
  engine.upload(x, xq)
  engine.upload(y, yq)

  # diamond: p = x+y, z1 = p+x, z2 = p+y, both roots in one eval, p enqueues once
  let p = engine.tn_add(x, y)
  let z1 = engine.tn_add(p, x)
  let z2 = engine.tn_add(p, y)
  var batch = engine.recordBatch()
  var visited: VisitedNodes
  var nz1 = z1
  var nz2 = z2
  engine.evalNode(batch, nz1, visited)
  engine.evalNode(batch, nz2, visited)
  doAssert batch.entries.len == 3,
    name & ": diamond enqueued " & $batch.entries.len & " kernels, expected 3"
  batch.submit()
  var gz1, gz2: array[8, bfloat16]
  engine.download(gz1, nz1)
  engine.download(gz2, nz2)

  # aliased inputs: w = x+y, e = w+w, w enqueues once
  let w = engine.tn_add(x, y)
  let e = engine.tn_add(w, w)
  var aliasBatch = engine.recordBatch()
  var aliasVisited: VisitedNodes
  var nw = w
  var ne = e
  engine.evalNode(aliasBatch, nw, aliasVisited)
  engine.evalNode(aliasBatch, ne, aliasVisited)
  doAssert aliasBatch.entries.len == 2,
    name & ": aliased inputs enqueued " & $aliasBatch.entries.len & " kernels, expected 2"

  # cross-eval memo: re-walking the replaced root enqueues nothing
  var rememoBatch = engine.recordBatch()
  var rememoVisited: VisitedNodes
  var again = ne
  engine.evalNode(rememoBatch, again, rememoVisited)
  doAssert rememoBatch.entries.len == 0,
    name & ": re-eval enqueued " & $rememoBatch.entries.len & " kernels, expected 0"

  aliasBatch.submit()
  var gw, ge: array[8, bfloat16]
  engine.download(gw, nw)
  engine.download(ge, ne)
  for i in 0 ..< 8:
    let pVal = (xd[i].to(bfloat16).to(float32) + yd[i].to(bfloat16).to(float32)).to(bfloat16)
    let z1Exp = (pVal.to(float32) + xd[i].to(bfloat16).to(float32)).to(bfloat16)
    let z2Exp = (pVal.to(float32) + yd[i].to(bfloat16).to(float32)).to(bfloat16)
    doAssert gz1[i] == z1Exp, name & ": diamond z1[" & $i & "]"
    doAssert gz2[i] == z2Exp, name & ": diamond z2[" & $i & "]"
    let wExp = pVal
    let eExp = (wExp.to(float32) + wExp.to(float32)).to(bfloat16)
    doAssert gw[i] == wExp, name & ": aliased w[" & $i & "]"
    doAssert ge[i] == eExp, name & ": aliased e[" & $i & "]"

  echo "tensorcery dedupe (", name, "): all checks passed"
