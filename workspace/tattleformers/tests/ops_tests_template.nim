# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/ceramic
import workspace/tensorcery/src/tensor_datatypes
import workspace/tattleformers/src/ops_2i1o/add
import workspace/tattleformers/src/ops_Ni1o/embed_gather
import workspace/tattleformers/src/layers/embedding

export tensor_datatypes

proc runTest*[E: HwEngine](engine: E, name: string) =
  # weight entries with a distinct value per (entry, col), bf16 end to end
  const Vocab = 7
  const Hidden = 4
  var wHost = newSeq[bfloat16](Vocab * Hidden)
  for r in 0 ..< Vocab:
    for c in 0 ..< Hidden:
      wHost[r * Hidden + c] = (float32(r * 10 + c) + 0.5).to(bfloat16)
  let w = newTensor(engine, @[Vocab, Hidden], kBf16)
  w.copyToDeviceFromHost(wHost)

  # ids with duplicates and both boundary ids, judged against the host entries
  const ids = [6, 0, 3, 6, 6, 2]
  let g = Tensor(engine: w.engine,
    node: tn_embed_gather(w.engine, w.node, ids))
  doAssert g.node.getImpl().shape().len == 2
  doAssert g.node.getImpl().shape()[0] == ids.len
  doAssert g.node.getImpl().shape()[1] == Hidden
  doAssert g.node.getImpl().dtype() == kBf16

  var gRoots = @[g.node]
  g.engine.eval(gRoots.toSpan())
  var got = newSeq[bfloat16](ids.len * Hidden)
  copyToHostFromDevice(got, g)
  for t in 0 ..< ids.len:
    for c in 0 ..< Hidden:
      doAssert got[t * Hidden + c].to(float32) == wHost[ids[t] * Hidden + c].to(float32),
        name & ": gather token " & $t & " col " & $c

  # out-of-vocabulary and empty inputs raise at the face
  doAssertRaises ValueError:
    discard tn_embed_gather(w.engine, w.node, [Vocab])
  doAssertRaises ValueError:
    discard tn_embed_gather(w.engine, w.node, [-1])

  # gather and add share one graph: both handlers registered, each node dispatches to its own op
  let doubled = g + g
  var dRoots = @[doubled.node]
  doubled.engine.eval(dRoots.toSpan())
  var gotD = newSeq[bfloat16](ids.len * Hidden)
  copyToHostFromDevice(gotD, doubled)
  for i in 0 ..< ids.len * Hidden:
    let want = 2.0 * wHost[ids[i div Hidden] * Hidden + i mod Hidden].to(float32)
    doAssert gotD[i].to(float32) == want.to(bfloat16).to(float32),
      name & ": gathered add " & $i

  # Embedding layer port: (batch,) ids give (batch, hidden_size)
  let layer = Embedding.init(w)
  doAssert layer.vocab_size == Vocab
  doAssert layer.hidden_size == Hidden
  let e1 = layer.forward(ids, @[ids.len])
  doAssert e1.node.getImpl().shape().len == 2
  doAssert e1.node.getImpl().shape()[0] == ids.len
  doAssert e1.node.getImpl().shape()[1] == Hidden
  var e1Roots = @[e1.node]
  e1.engine.eval(e1Roots.toSpan())
  var gotE1 = newSeq[bfloat16](ids.len * Hidden)
  copyToHostFromDevice(gotE1, e1)
  for t in 0 ..< ids.len:
    for c in 0 ..< Hidden:
      doAssert gotE1[t * Hidden + c].to(float32) == wHost[ids[t] * Hidden + c].to(float32),
        name & ": layer (batch,) token " & $t & " col " & $c

  # (batch, seq) ids give a (batch, seq, hidden_size) reshape view
  # over the gathered weight entries
  let e2 = layer.forward(ids, @[2, 3])
  doAssert e2.node.getImpl().shape().len == 3
  doAssert e2.node.getImpl().shape()[0] == 2
  doAssert e2.node.getImpl().shape()[1] == 3
  doAssert e2.node.getImpl().shape()[2] == Hidden
  var e2Roots = @[e2.node]
  e2.engine.eval(e2Roots.toSpan())
  var gotE2 = newSeq[bfloat16](ids.len * Hidden)
  copyToHostFromDevice(gotE2, e2)
  for t in 0 ..< ids.len:
    for c in 0 ..< Hidden:
      doAssert gotE2[t * Hidden + c].to(float32) == wHost[ids[t] * Hidden + c].to(float32),
        name & ": layer (batch, seq) token " & $t & " col " & $c

  # reshape views a dense source in place: same buffer cell, new layout
  block:
    let r = w.reshape([Hidden, Vocab])
    doAssert r.node.getImpl().shape()[0] == Hidden
    doAssert r.node.getImpl().shape()[1] == Vocab
    doAssert r.node.getImpl().strides()[0] == Vocab
    doAssert r.node.getImpl().strides()[1] == 1
    type MemW = typeof(engine.allocDevice(bfloat16, 0))
    let wImpl = w.node.getImpl().to(TensorImpl[MemW, Layout[(int, int), (int, int)]])
    let rImpl = r.node.getImpl().to(TensorImpl[MemW, Layout[(int, int), (int, int)]])
    doAssert rImpl.data == wImpl.data, name & ": reshape must alias the source cell"
    var gotR = newSeq[bfloat16](Vocab * Hidden)
    copyToHostFromDevice(gotR, r)
    for i in 0 ..< Vocab * Hidden:
      doAssert gotR[i].to(float32) == wHost[i].to(float32), name & ": reshape " & $i
    doAssertRaises ValueError:
      discard w.reshape([Vocab * Hidden + 1])

  echo "tattleformers ops (", name, "): all checks passed"
