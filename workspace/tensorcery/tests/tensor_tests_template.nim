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
import workspace/tensorcery/src/compute_graph/ops_Ni1o/ops_embed_gather
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

proc runEmbedGatherTest*(run: static Runtime, name: string) =
  ## Embed gather end to end: a (16, 8) bf16 weight, ids [3, 0, 7, 3]
  ## (the duplicate id reads one weight entry twice).
  ##
  ## Checks:
  ## - public-face eval: toTensor, embed_gather, eval and item readback equal
  ##   the weight entry at the id
  ## - face contracts raise AssertionDefect: an empty id list, ids outside
  ##   the vocabulary
  ## - fan-out enqueue count: two add consumers of one gather output enqueue
  ##   the gather exactly once (3 batch entries), numerics read back
  var wq = newSeq[bfloat16](16 * 8)
  for i in 0 ..< 16 * 8:
    wq[i] = float32(i).to(bfloat16)

  # public face: toTensor + eval + item readback
  let w = toTensor(wq, @[16, 8])
  let selected = embed_gather(w, @[3, 0, 7, 3])
  doAssert selected.shape().len == 2, name & ": gather rank"
  doAssert selected.shape()[0] == 4, name & ": gather tokens"
  doAssert selected.shape()[1] == 8, name & ": gather hidden"
  eval(selected)
  for t in 0 ..< 4:
    for c in 0 ..< 8:
      doAssert selected.item(t, c, bfloat16) == float32([3, 0, 7, 3][t] * 8 + c).to(bfloat16),
        name & ": gather out " & $t & "," & $c

  # face contracts raise AssertionDefect
  doAssertRaises AssertionDefect:
    discard embed_gather(w, @[])
  doAssertRaises AssertionDefect:
    discard embed_gather(w, @[16])

  # fan-out: two consumers of one gather output enqueue the gather ONCE
  let engine = newEngine(run)
  var wShape = @[16, 8]
  let wNode = engine.newTensorNode(kBf16, wShape.toSpan())
  engine.upload(wNode, wq)
  var xq: array[4 * 8, bfloat16]
  for i in 0 ..< 4 * 8:
    xq[i] = float32(i).to(bfloat16)
  var gShape = @[4, 8]
  let x = engine.newTensorNode(kBf16, gShape.toSpan())
  let y = engine.newTensorNode(kBf16, gShape.toSpan())
  engine.upload(x, xq)
  engine.upload(y, xq)

  let g = engine.tn_embed_gather(wNode, @[3, 0, 7, 3])
  let z1 = engine.tn_add(g, x)
  let z2 = engine.tn_add(g, y)
  var batch = engine.recordBatch()
  var visited: VisitedNodes
  var nz1 = z1
  var nz2 = z2
  engine.evalNode(batch, nz1, visited)
  engine.evalNode(batch, nz2, visited)
  doAssert batch.entries.len == 3,
    name & ": gather fan-out enqueued " & $batch.entries.len & " kernels, expected 3"
  batch.submit()
  var gz1, gz2: array[4 * 8, bfloat16]
  engine.download(gz1, nz1)
  engine.download(gz2, nz2)
  for t in 0 ..< 4:
    for c in 0 ..< 8:
      let gVal = float32([3, 0, 7, 3][t] * 8 + c).to(bfloat16)
      let xVal = float32(t * 8 + c).to(bfloat16)
      let exp = (gVal.to(float32) + xVal.to(float32)).to(bfloat16)
      doAssert gz1[t * 8 + c] == exp, name & ": gather z1 " & $t & "," & $c
      doAssert gz2[t * 8 + c] == exp, name & ": gather z2 " & $t & "," & $c

  echo "tensorcery embed gather (", name, "): all checks passed"

proc runRankNTest*(run: static Runtime, name: string) =
  ## Construct-upload-item-download round-trips at ranks 1, 3, 4, 5, bf16.
  ## The payloads are the integers 1..32, exact in bf16: every comparison is exact.
  let xd = [1.0'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17.0'f32, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
  var xq: array[32, bfloat16]
  for i in 0 ..< 32:
    xq[i] = xd[i].to(bfloat16)

  # rank 1: a vector, an rmsnorm weight shape
  let v = toTensor(xq[0 ..< 8], @[8])
  var vShape = @[8]
  var vStrides = @[1]
  doAssert v.shape() == vShape.toSpan(), name & ": rank1 shape"
  doAssert v.strides() == vStrides.toSpan(), name & ": rank1 strides"
  doAssert v.item(0, bfloat16) == xd[0].to(bfloat16), name & ": rank1 first"
  doAssert v.item(4, bfloat16) == xd[4].to(bfloat16), name & ": rank1 mid"
  doAssert v.item(7, bfloat16) == xd[7].to(bfloat16), name & ": rank1 last"

  # rank 3: a q-like projection (batch, heads, head_dim)
  let q = toTensor(xq[0 ..< 8], @[2, 2, 2])
  var qStrides = @[4, 2, 1]
  doAssert q.strides() == qStrides.toSpan(), name & ": rank3 strides"
  doAssert q.item(0, 0, 0, bfloat16) == xd[0].to(bfloat16), name & ": rank3 first"
  doAssert q.item(1, 0, 1, bfloat16) == xd[5].to(bfloat16), name & ": rank3 mid"
  doAssert q.item(1, 1, 1, bfloat16) == xd[7].to(bfloat16), name & ": rank3 last"

  # rank 4: a qkv projection (batch, heads, seq, head_dim)
  let qkv = toTensor(xq[0 ..< 16], @[2, 2, 2, 2])
  var qkvStrides = @[8, 4, 2, 1]
  doAssert qkv.strides() == qkvStrides.toSpan(), name & ": rank4 strides"
  doAssert qkv.item(0, 0, 0, 0, bfloat16) == xd[0].to(bfloat16), name & ": rank4 first"
  doAssert qkv.item(1, 0, 1, 0, bfloat16) == xd[10].to(bfloat16), name & ": rank4 mid"
  doAssert qkv.item(1, 1, 1, 1, bfloat16) == xd[15].to(bfloat16), name & ": rank4 last"

  # rank 5: a pool page slab
  let page = toTensor(xq, @[2, 2, 2, 2, 2])
  var pageStrides = @[16, 8, 4, 2, 1]
  doAssert page.strides() == pageStrides.toSpan(), name & ": rank5 strides"
  doAssert page.item(0, 0, 0, 0, 0, bfloat16) == xd[0].to(bfloat16), name & ": rank5 first"
  doAssert page.item(1, 0, 1, 0, 1, bfloat16) == xd[21].to(bfloat16), name & ": rank5 mid"
  doAssert page.item(1, 1, 1, 1, 1, bfloat16) == xd[31].to(bfloat16), name & ": rank5 last"

  # add walks ranks 3 and 5: one enqueued kernel per node, rank-generic
  let qy = toTensor(xq[8 ..< 16], @[2, 2, 2])
  let qz = q + qy
  doAssert qz.item(1, 1, 1, bfloat16) == (xd[7] + xd[15]).to(bfloat16), name & ": rank3 add"
  let py = toTensor(xq, @[2, 2, 2, 2, 2])
  let pz = page + py
  doAssert pz.item(1, 1, 1, 1, 1, bfloat16) == (xd[31] + xd[31]).to(bfloat16), name & ": rank5 add"

  # erased-engine upload/download round-trip per rank
  let engine = gEngine(run)
  for shape in @[@[8], @[2, 2, 2], @[2, 2, 2, 2], @[2, 2, 2, 2, 2]]:
    var numel = 1
    for d in shape:
      numel *= d
    var src = newSeq[bfloat16](numel)
    for i in 0 ..< numel:
      src[i] = xd[i].to(bfloat16)
    let node = engine.newTensorNode(kBf16, shape.toSpan())
    engine.upload(node, src)
    var dst = newSeq[bfloat16](numel)
    engine.download(dst, node)
    for i in 0 ..< numel:
      doAssert dst[i] == src[i], name & ": rank" & $shape.len & " download " & $i

  # single-element readback through the erased engine: a client module
  # consumes the new AnyEngine vtable method from the public face
  var vec: array[4, bfloat16]
  for i in 0 ..< 4:
    vec[i] = xd[i].to(bfloat16)
  var vecShape = @[4]
  let vecNode = engine.newTensorNode(kBf16, vecShape.toSpan())
  engine.upload(vecNode, vec)
  var got: bfloat16
  engine.downloadItem(got, vecNode, 2)
  doAssert got == xd[2].to(bfloat16), name & ": erased downloadItem"

  echo "tensorcery rank-N (", name, "): all checks passed"
