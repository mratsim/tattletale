# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/times
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/iface
import workspace/tensorcery/src/tensorcery
import workspace/tensorcery/src/tensors_internals
import workspace/tensorcery/src/tensors_globals
import workspace/tensorcery/src/compute_graph/dispatch
import workspace/tensorcery/src/compute_graph/ops_2i1o/ops_add
import workspace/tensorcery/src/compute_graph/ops_2i1o/ops_matmul
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
  # item bounds are contracts: negative and past-the-end indices raise
  doAssertRaises AssertionDefect:
    discard x.item(-1, float32)
  doAssertRaises AssertionDefect:
    discard x.item(0, 4, float32)

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

proc runCoordPassRebuildTest*(run: static Runtime, name: string) =
  ## Two artifact builds in one binary over one set of device functions.
  ##
  ## Sequence:
  ## - first build: the matmul op's submit compiles the tile gemm chain.
  ##   Its device functions gain their coordinate parameters
  ## - between the builds: a harvest-only enqueue of the op's kernels adds
  ##   new wrappers with literal statics to the compilation, never executed
  ## - second build: the add op's submit re-runs the MSL lowering pass over
  ##   the accumulated module, which still contains the gemm chain's device
  ##   functions
  ##
  ## Both results read back exact, so the second build serves the first
  ## build's kernels unchanged.
  var xh = newSeq[bfloat16](4 * 8)
  var wh = newSeq[bfloat16](6 * 8)
  for i in 0 ..< 4 * 8:
    xh[i] = (float32((i * 37 mod 19) - 9) * 0.125'f32).to(bfloat16)
  for i in 0 ..< 6 * 8:
    wh[i] = (float32((i * 29 mod 17) - 8) * 0.125'f32).to(bfloat16)
  let engine = newEngine(run)
  var xShape = @[4, 8]
  var wShape = @[6, 8]
  let xNode = engine.newTensorNode(kBf16, xShape.toSpan())
  let wNode = engine.newTensorNode(kBf16, wShape.toSpan())
  engine.upload(xNode, xh)
  engine.upload(wNode, wh)

  # first build: the matmul chain, whose device functions gain their coordinate parameters here
  let y = engine.tn_matmul(xNode, wNode)
  var batch1 = engine.recordBatch()
  var visited: VisitedNodes
  var ny = y
  engine.evalNode(batch1, ny, visited)
  doAssert batch1.entries.len == 2, name & ": first build enqueued " &
    $batch1.entries.len & ", expected 2"
  batch1.submit()
  for mm in 0 ..< 4:
    for nn in 0 ..< 6:
      var acc = 0.0'f32
      for kk in 0 ..< 8:
        acc += xh[mm * 8 + kk].to(float32) * wh[nn * 8 + kk].to(float32)
      var got: bfloat16
      engine.downloadItem(got, ny, mm * 6 + nn)
      doAssert got == acc.to(bfloat16),
        name & ": first build matmul " & $mm & "," & $nn

  # between the builds: enqueue the op's kernels with literal statics into a batch that never submits,
  # so the compilation gains new wrappers and the second build re-runs the lowering pass over the first chain
  block:
    var oShape = @[4, 6]
    let oMan = engine.newTensorNode(kBf16, oShape.toSpan())
    let
      rx = xNode.getImpl().to(TensorImpl[typeof(engine.allocDevice(bfloat16, 0)), Layout[(int, int), (int, int)]])
      rw = wNode.getImpl().to(TensorImpl[typeof(engine.allocDevice(bfloat16, 0)), Layout[(int, int), (int, int)]])
      roMan = oMan.getImpl().to(TensorImpl[typeof(engine.allocDevice(bfloat16, 0)), Layout[(int, int), (int, int)]])
      xv = initTensorBridge(rx.data[], rx.offset, rx.layout)
      wv = initTensorBridge(rw.data[], rw.offset, rw.layout)
      ovMan = initTensorBridge(roMan.data[], roMan.offset, roMan.layout)
    type MemAcc = typeof(engine.allocDevice(float32, 0))
    let rAcc = engine.makeTensorImpl(kFp32, oShape.toSpan())
      .to(TensorImpl[MemAcc, Layout[(int, int), (int, int)]])
    let accV = initTensorBridge(rAcc.data[], rAcc.offset, rAcc.layout)
    var batchMan = engine.recordBatch()
    batchMan.enqueue<<(1, 32)>> k_matmul(accV, xv, wv, tiledMma(float32, bfloat16, bfloat16, 32, 32, 16))
    batchMan.enqueue<<(4, 32)>> k_round_to_output(ovMan, accV)
    doAssert batchMan.entries.len == 2, name & ": harvest enqueue " &
      $batchMan.entries.len & ", expected 2"

  # second build: the add chain shares the gemm chain's device functions,
  # so the lowering pass re-runs over a module containing both chains
  var ah = newSeq[bfloat16](2 * 3)
  var bh = newSeq[bfloat16](2 * 3)
  for i in 0 ..< 2 * 3:
    ah[i] = (float32(i + 1) * 0.125'f32).to(bfloat16)
    bh[i] = (float32(10 * (i + 1)) * 0.125'f32).to(bfloat16)
  var aShape = @[2, 3]
  let aNode = engine.newTensorNode(kBf16, aShape.toSpan())
  let bNode = engine.newTensorNode(kBf16, aShape.toSpan())
  engine.upload(aNode, ah)
  engine.upload(bNode, bh)
  let c = engine.tn_add(aNode, bNode)
  var batch2 = engine.recordBatch()
  var visited2: VisitedNodes
  var nc = c
  engine.evalNode(batch2, nc, visited2)
  doAssert batch2.entries.len == 1, name & ": second build enqueued " &
    $batch2.entries.len & ", expected 1"
  batch2.submit()
  for i in 0 ..< 2 * 3:
    var got: bfloat16
    engine.downloadItem(got, nc, i)
    let want = (ah[i].to(float32) + bh[i].to(float32)).to(bfloat16)
    doAssert got == want, name & ": second build add " & $i

  echo "tensorcery coord pass rebuild (", name, "): all checks passed"

func bf16UlpKey(x: bfloat16): int32 =
  ## Ordered key of a bfloat16 word: sign-magnitude bits mapped so keys
  ## increase with the value, -0 and +0 collide, and the distance of two
  ## keys counts bfloat16 ulps
  let u = cast[uint16](x)
  if (u and 0x8000'u16) != 0'u16:
    -int32(u and 0x7FFF'u16)
  else:
    int32(u)

proc runMatmulTest*(run: static Runtime, name: string) =
  ## Matmul op end to end: x (tokens, in) x w (out, in)^T -> (tokens, out).
  ##
  ## Checks:
  ## - public face: the toTensor, linear, eval and item readback results
  ##   equal the sequential f32 host reference rounded once into bfloat16, exactly:
  ##   every f32 product and partial sum of the 2^-3-granular
  ##   fixtures is exact in f32, and K = 8 fits one 16-deep k-tile
  ## - face contracts raise AssertionDefect: a K mismatch, a dtype mismatch
  ## - LMHead-shaped smoke, M = 8, N = 151936, K = 1024, bf16: the enqueue
  ##   launches the f32 accumulation and the rounding pass, and the spot-checked
  ##   outputs sit within the band of the f32 host reference.
  ##   The f32 partial-sum order differs from the kernel's k-tile order, so a 0-ulp match is not expected
  ## - fan-out: two add consumers of one matmul output see the matmul's two
  ##   launches enqueued exactly once (4 batch entries), and both results
  ##   read back exact
  var xh = newSeq[bfloat16](4 * 8)
  var wh = newSeq[bfloat16](6 * 8)
  for i in 0 ..< 4 * 8:
    xh[i] = (float32((i * 37 mod 19) - 9) * 0.125'f32).to(bfloat16)
  for i in 0 ..< 6 * 8:
    wh[i] = (float32((i * 29 mod 17) - 8) * 0.125'f32).to(bfloat16)

  # public face: toTensor + linear + eval + item readback
  let x = toTensor(xh, @[4, 8])
  let w = toTensor(wh, @[6, 8])
  let y = linear(x, w)
  doAssert y.shape().len == 2, name & ": linear rank"
  doAssert y.shape()[0] == 4, name & ": linear tokens"
  doAssert y.shape()[1] == 6, name & ": linear outFeatures"
  doAssert y.dtype() == kBf16, name & ": linear dtype"
  eval(y)
  for mm in 0 ..< 4:
    for nn in 0 ..< 6:
      var acc = 0.0'f32
      for kk in 0 ..< 8:
        acc += xh[mm * 8 + kk].to(float32) * wh[nn * 8 + kk].to(float32)
      doAssert y.item(mm, nn, bfloat16) == acc.to(bfloat16),
        name & ": linear out " & $mm & "," & $nn

  # face contracts raise AssertionDefect: an inFeatures mismatch, a dtype mismatch
  let wShort = toTensor(wh[0 ..< 6 * 7], @[6, 7])
  doAssertRaises AssertionDefect:
    discard linear(x, wShort)
  var wHost32 = newSeq[float32](6 * 8)
  for i in 0 ..< 6 * 8:
    wHost32[i] = wh[i].to(float32)
  let wF32 = toTensor(wHost32, @[6, 8])
  doAssertRaises AssertionDefect:
    discard linear(x, wF32)

  # LMHead-shaped smoke: enqueue and spot-check against the f32 reference
  block:
    const
      LM = 8
      LN = 151936
      LK = 1024
    var lx = newSeq[bfloat16](LM * LK)
    var lw = newSeq[bfloat16](LN * LK)
    # positive fill: the running partial sum grows monotonically without cancellation,
    # so the band is relative to the reference's own magnitude
    for i in 0 ..< LM * LK:
      lx[i] = (float32((i * 37 mod 19) + 1) * 0.125'f32).to(bfloat16)
    for i in 0 ..< LN * LK:
      lw[i] = (float32((i * 29 mod 17) + 1) * 0.125'f32).to(bfloat16)
    let engine = newEngine(run)
    var lxShape = @[LM, LK]
    var lwShape = @[LN, LK]
    let lxNode = engine.newTensorNode(kBf16, lxShape.toSpan())
    let lwNode = engine.newTensorNode(kBf16, lwShape.toSpan())
    engine.upload(lxNode, lx)
    engine.upload(lwNode, lw)
    let lo = engine.tn_matmul(lxNode, lwNode)
    doAssert lo.getImpl().shape().len == 2, name & ": lmhead rank"
    doAssert lo.getImpl().shape()[0] == LM, name & ": lmhead tokens"
    doAssert lo.getImpl().shape()[1] == LN, name & ": lmhead outFeatures"
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var lno = lo
    let tEnq0 = epochTime()
    engine.evalNode(batch, lno, visited)
    let tEnq1 = epochTime()
    doAssert batch.entries.len == 2,
      name & ": lmhead enqueued " & $batch.entries.len & ", expected 2"
    batch.submit()
    let tWall = epochTime()
    const spots = [(0, 0), (4, LN - 1), (7, LN div 2), (3, 12345)]
    var worstRel = 0.0'f32
    var worstUlp = 0'i32
    for (mm, nn) in spots:
      var acc = 0.0'f32
      for kk in 0 ..< LK:
        acc += lx[mm * LK + kk].to(float32) * lw[nn * LK + kk].to(float32)
      var got: bfloat16
      engine.downloadItem(got, lno, mm * LN + nn)
      let g = got.to(float32)
      doAssert g == g and abs(g) < 1e30'f32,
        name & ": lmhead spot " & $mm & "," & $nn & " is not finite"
      let rel = abs(g - acc) / abs(acc)
      let want = acc.to(bfloat16)
      let dist = abs(bf16UlpKey(got) - bf16UlpKey(want))
      if rel > worstRel:
        worstRel = rel
      if dist > worstUlp:
        worstUlp = dist
      doAssert rel <= 0.05'f32,
        name & ": lmhead spot " & $mm & "," & $nn & ": got " & $g &
        ", reference " & $acc & ", drift " & $(rel * 100.0'f32) & "%"
    echo name, ": lmhead enqueue ", (tEnq1 - tEnq0), " s, wall ",
      (tWall - tEnq0), " s, worst spot drift ",
      (worstRel * 100.0'f32), "% / ", worstUlp, " bf16 ulps"

  # fan-out: two add consumers of one matmul output enqueue the matmul once
  # (its 2 launches: the f32 accumulation and the rounding pass)
  block:
    let engine = newEngine(run)
    var wShape = @[6, 8]
    let wNode = engine.newTensorNode(kBf16, wShape.toSpan())
    engine.upload(wNode, wh)
    var xShape = @[4, 8]
    let xNode = engine.newTensorNode(kBf16, xShape.toSpan())
    engine.upload(xNode, xh)
    var ch = newSeq[bfloat16](4 * 6)
    for i in 0 ..< 4 * 6:
      ch[i] = (float32((i * 23 mod 13) - 6) * 0.125'f32).to(bfloat16)
    var cShape = @[4, 6]
    let cx = engine.newTensorNode(kBf16, cShape.toSpan())
    let cy = engine.newTensorNode(kBf16, cShape.toSpan())
    engine.upload(cx, ch)
    engine.upload(cy, ch)
    let p = engine.tn_matmul(xNode, wNode)
    let z1 = engine.tn_add(p, cx)
    let z2 = engine.tn_add(p, cy)
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var nz1 = z1
    var nz2 = z2
    engine.evalNode(batch, nz1, visited)
    engine.evalNode(batch, nz2, visited)
    doAssert batch.entries.len == 4,
      name & ": fan-out enqueued " & $batch.entries.len & ", expected 4"
    batch.submit()
    var gz1, gz2: array[4 * 6, bfloat16]
    engine.download(gz1, nz1)
    engine.download(gz2, nz2)
    for mm in 0 ..< 4:
      for nn in 0 ..< 6:
        var acc = 0.0'f32
        for kk in 0 ..< 8:
          acc += xh[mm * 8 + kk].to(float32) * wh[nn * 8 + kk].to(float32)
        let pVal = acc.to(bfloat16)
        let cVal = ch[mm * 6 + nn].to(float32)
        let exp = (pVal.to(float32) + cVal).to(bfloat16)
        doAssert gz1[mm * 6 + nn] == exp, name & ": fan-out z1 " & $mm & "," & $nn
        doAssert gz2[mm * 6 + nn] == exp, name & ": fan-out z2 " & $mm & "," & $nn

  echo "tensorcery matmul (", name, "): all checks passed"

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
