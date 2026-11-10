# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[math, times, random]
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/iface
import workspace/tensorcery/src/tensorcery
import workspace/tensorcery/src/tensors_internals
import workspace/tensorcery/src/tensors_globals
import workspace/tensorcery/src/compute_graph/dispatch
import workspace/tensorcery/src/compute_graph/ops_2i1o/ops_add
import workspace/tensorcery/src/compute_graph/ops_2i1o/ops_matmul
import workspace/tensorcery/src/compute_graph/ops_2i1o/ops_silu_mul
import workspace/tensorcery/src/compute_graph/ops_Ni1o/ops_embed_gather
import workspace/tensorcery/src/compute_graph/ops_1i1o/ops_rmsnorm
import workspace/tensorcery/src/compute_graph/ops_Ni1o/ops_rope
import workspace/tensorcery/src/compute_graph/ops_Ni1o/ops_pool_write
import workspace/tensorcery/src/compute_graph/ops_Ni1o/ops_attention
import workspace/tensorcery/src/compute_graph/views
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

proc runRmsnormTest*(run: static Runtime, name: string) =
  ## RmsNorm op end to end: x (tokens, width) -> normed, per-axis weight.
  ##
  ## Checks:
  ## - public face at (4, 8), bfloat16: the readback equals the f32 host reference exactly,
  ##   one rounding into the output. The numerics record is the composition's single-rounding order:
  ##   exact upcast, f32 normalize, one rounding at the final store
  ## - f32 input: one launch of the stock weighted kernel,
  ##   the readback inside a 1e-6 relative band for the kernel's native rsqrt against the reference's 1/sqrt
  ## - face contracts raise AssertionDefect on a non-rank-2 input, a non-rank-1 weight, a weight size mismatch and a dtype mismatch
  ## - wide smoke at (32, 1024) through manual enqueues of the op's kernels, exact vs the f32 reference,
  ##   and the fan-out memo: two add consumers of one rmsnorm output see its three launches enqueued
  ##   exactly once (5 batch entries) and both results read back exact
  var xh = newSeq[bfloat16](4 * 8)
  var wh = newSeq[bfloat16](8)
  for i in 0 ..< 4 * 8:
    xh[i] = (float32((i * 37 mod 19) - 9) * 0.125'f32).to(bfloat16)
  for i in 0 ..< 8:
    wh[i] = (float32((i * 29 mod 17) - 8) * 0.125'f32).to(bfloat16)

  # public face: toTensor + rmsnorm + eval + item readback, exact
  let x = toTensor(xh, @[4, 8])
  let w = toTensor(wh, @[8])
  let y = rmsnorm(x, w)
  doAssert y.shape().len == 2, name & ": rmsnorm rank"
  doAssert y.shape()[0] == 4, name & ": rmsnorm tokens"
  doAssert y.shape()[1] == 8, name & ": rmsnorm width"
  doAssert y.dtype() == kBf16, name & ": rmsnorm dtype"
  eval(y)
  for t in 0 ..< 4:
    var ss = 0.0'f32
    for c in 0 ..< 8:
      let x = xh[t * 8 + c].to(float32)
      ss += x * x
    let scale = 1.0'f32 / sqrt(ss / 8.0'f32 + 1e-6'f32)
    for c in 0 ..< 8:
      let want = (xh[t * 8 + c].to(float32) * scale * wh[c].to(float32)).to(bfloat16)
      doAssert y.item(t, c, bfloat16) == want,
        name & ": rmsnorm out " & $t & "," & $c

  # face contracts raise AssertionDefect
  let x3 = toTensor(xh[0 ..< 8], @[2, 2, 2])
  doAssertRaises AssertionDefect:
    discard rmsnorm(x3, w)
  let w2 = toTensor(wh, @[2, 4])
  doAssertRaises AssertionDefect:
    discard rmsnorm(x, w2)
  let wShort = toTensor(wh[0 ..< 7], @[7])
  doAssertRaises AssertionDefect:
    discard rmsnorm(x, wShort)
  var wf = newSeq[float32](8)
  for i in 0 ..< 8:
    wf[i] = wh[i].to(float32)
  let wF32 = toTensor(wf, @[8])
  doAssertRaises AssertionDefect:
    discard rmsnorm(x, wF32)

  # f32 input: single launch of the stock weighted kernel
  block:
    var xf = newSeq[float32](4 * 8)
    for i in 0 ..< 4 * 8:
      xf[i] = xh[i].to(float32)
    let x32 = toTensor(xf, @[4, 8])
    let w32 = toTensor(wf, @[8])
    let y32 = rmsnorm(x32, w32)
    doAssert y32.dtype() == kFp32, name & ": rmsnorm f32 dtype"
    eval(y32)
    var worstRel = 0.0'f32
    for t in 0 ..< 4:
      var ss = 0.0'f32
      for c in 0 ..< 8:
        ss += xf[t * 8 + c] * xf[t * 8 + c]
      let scale = 1.0'f32 / sqrt(ss / 8.0'f32 + 1e-6'f32)
      for c in 0 ..< 8:
        let want = xf[t * 8 + c] * scale * wf[c]
        let got = y32.item(t, c, float32)
        if want == 0.0'f32:
          # zero fixture weights produce zero outputs where a relative band
          # would divide by zero, so those elements compare exactly
          doAssert got == want,
            name & ": rmsnorm f32 out " & $t & "," & $c & ": got " & $got &
            ", reference " & $want
        else:
          let rel = abs(got - want) / abs(want)
          if rel > worstRel:
            worstRel = rel
          doAssert rel <= 1e-6'f32,
            name & ": rmsnorm f32 out " & $t & "," & $c & ": got " & $got &
            ", reference " & $want
    echo name, ": rmsnorm f32 worst rel ", worstRel

  # wide smoke at (32, 1024) through manual enqueues of the op's kernels:
  # composition numerics at the transformer width, exact vs the f32 reference
  block:
    const R = 32
    const D = 1024
    var wx = newSeq[bfloat16](R * D)
    var ww = newSeq[bfloat16](D)
    for i in 0 ..< R * D:
      wx[i] = (float32((i * 37 mod 19) - 9) * 0.125'f32).to(bfloat16)
    for i in 0 ..< D:
      ww[i] = (float32((i * 29 mod 17) - 8) * 0.125'f32).to(bfloat16)
    let engine = newEngine(run)
    var xShape = @[R, D]
    var wShape = @[D]
    let xNode = engine.newTensorNode(kBf16, xShape.toSpan())
    let wNode = engine.newTensorNode(kBf16, wShape.toSpan())
    engine.upload(xNode, wx)
    engine.upload(wNode, ww)
    let oNode = engine.newTensorNode(kBf16, xShape.toSpan())
    let
      rx = xNode.getImpl().to(TensorImpl[typeof(engine.allocDevice(bfloat16, 0)), Layout[(int, int), (int, int)]])
      rw = wNode.getImpl().to(TensorImpl[typeof(engine.allocDevice(bfloat16, 0)), Layout[(int,), (int,)]])
      ro = oNode.getImpl().to(TensorImpl[typeof(engine.allocDevice(bfloat16, 0)), Layout[(int, int), (int, int)]])
      xv = initTensorBridge(rx.data[], rx.offset, rx.layout)
      wv = initTensorBridge(rw.data[], rw.offset, rw.layout)
      ov = initTensorBridge(ro.data[], ro.offset, ro.layout)
    let rAcc = engine.makeTensorImpl(kFp32, xShape.toSpan())
      .to(TensorImpl[typeof(engine.allocDevice(float32, 0)), Layout[(int, int), (int, int)]])
    let accV = initTensorBridge(rAcc.data[], rAcc.offset, rAcc.layout)
    var batch = engine.recordBatch()
    batch.enqueue<<(R, 32)>> k_cast(accV, xv)
    batch.enqueue<<(R, 32)>> k_rmsnorm_unweighted(accV, accV, 1e-6'f32)
    batch.enqueue<<(R, 32)>> k_weighted_round_to_output(ov, accV, wv)
    doAssert batch.entries.len == 3, name & ": wide smoke enqueued " &
      $batch.entries.len & ", expected 3"
    batch.submit()
    for t in 0 ..< R:
      var ss = 0.0'f32
      for c in 0 ..< D:
        let x = wx[t * D + c].to(float32)
        ss += x * x
      let scale = 1.0'f32 / sqrt(ss / float32(D) + 1e-6'f32)
      for c in 0 ..< D:
        let want = (wx[t * D + c].to(float32) * scale * ww[c].to(float32)).to(bfloat16)
        var got: bfloat16
        engine.downloadItem(got, oNode, t * D + c)
        doAssert got == want,
          name & ": wide smoke out " & $t & "," & $c & ": got " &
          $got.to(float32) & ", reference " & $want.to(float32)

  # Wide op-path dispatch: the retained-buffer fix keeps multi-launch handlers
  # correct at wide shapes (the small shape above covers the op dispatch)
  block:
    const R = 32
    const D = 1024
    var wx = newSeq[bfloat16](R * D)
    var ww = newSeq[bfloat16](D)
    for i in 0 ..< R * D:
      wx[i] = (float32((i * 37 mod 19) - 9) * 0.125'f32).to(bfloat16)
    for i in 0 ..< D:
      ww[i] = (float32((i * 29 mod 17) - 8) * 0.125'f32).to(bfloat16)
    let engine = newEngine(run)
    var xShape = @[R, D]
    var wShape = @[D]
    let xNode = engine.newTensorNode(kBf16, xShape.toSpan())
    let wNode = engine.newTensorNode(kBf16, wShape.toSpan())
    engine.upload(xNode, wx)
    engine.upload(wNode, ww)
    let y = engine.tn_rmsnorm(xNode, wNode, 1e-6'f32)
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var ny = y
    engine.evalNode(batch, ny, visited)
    doAssert batch.entries.len == 3, name & ": wide op enqueued " &
      $batch.entries.len & ", expected 3"
    batch.submit()
    for t in 0 ..< R:
      var ss = 0.0'f32
      for c in 0 ..< D:
        let x = wx[t * D + c].to(float32)
        ss += x * x
      let scale = 1.0'f32 / sqrt(ss / float32(D) + 1e-6'f32)
      for c in 0 ..< D:
        let want = (wx[t * D + c].to(float32) * scale * ww[c].to(float32)).to(bfloat16)
        var got: bfloat16
        engine.downloadItem(got, ny, t * D + c)
        doAssert got == want,
          name & ": wide op out " & $t & "," & $c & ": got " &
          $got.to(float32) & ", reference " & $want.to(float32)

  # fan-out: two add consumers of one rmsnorm output enqueue its three
  # launches exactly once
  block:
    let engine = newEngine(run)
    var xShape = @[4, 8]
    var wShape = @[8]
    let xNode = engine.newTensorNode(kBf16, xShape.toSpan())
    let wNode = engine.newTensorNode(kBf16, wShape.toSpan())
    engine.upload(xNode, xh)
    engine.upload(wNode, wh)
    var ch = newSeq[bfloat16](4 * 8)
    for i in 0 ..< 4 * 8:
      ch[i] = (float32((i * 23 mod 13) - 6) * 0.125'f32).to(bfloat16)
    let cx = engine.newTensorNode(kBf16, xShape.toSpan())
    let cy = engine.newTensorNode(kBf16, xShape.toSpan())
    engine.upload(cx, ch)
    engine.upload(cy, ch)
    let p = engine.tn_rmsnorm(xNode, wNode, 1e-6'f32)
    let z1 = engine.tn_add(p, cx)
    let z2 = engine.tn_add(p, cy)
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var nz1 = z1
    var nz2 = z2
    engine.evalNode(batch, nz1, visited)
    engine.evalNode(batch, nz2, visited)
    doAssert batch.entries.len == 5,
      name & ": fan-out enqueued " & $batch.entries.len & ", expected 5"
    batch.submit()
    var gz1, gz2: array[4 * 8, bfloat16]
    engine.download(gz1, nz1)
    engine.download(gz2, nz2)
    for t in 0 ..< 4:
      var ss = 0.0'f32
      for c in 0 ..< 8:
        let x = xh[t * 8 + c].to(float32)
        ss += x * x
      let scale = 1.0'f32 / sqrt(ss / 8.0'f32 + 1e-6'f32)
      for c in 0 ..< 8:
        let pVal = (xh[t * 8 + c].to(float32) * scale * wh[c].to(float32)).to(bfloat16)
        let cVal = ch[t * 8 + c].to(float32)
        let exp = (pVal.to(float32) + cVal).to(bfloat16)
        doAssert gz1[t * 8 + c] == exp, name & ": fan-out z1 " & $t & "," & $c
        doAssert gz2[t * 8 + c] == exp, name & ": fan-out z2 " & $t & "," & $c

  echo "tensorcery rmsnorm (", name, "): all checks passed"

proc ropeTables(Tokens, Half, Dim: int, theta: float64): tuple[c, s: seq[float32]] =
  ## Half-width RoPE tables computed host-side.
  ## Entry (t, i) pairs with the columns (i, i + dim div 2) of token t,
  ## invFreq = theta^(-2i/dim)
  result.c.newSeq(Tokens * Half)
  result.s.newSeq(Tokens * Half)
  for t in 0 ..< Tokens:
    for j in 0 ..< Half:
      let invFreq = pow(theta, -(2.0 * float64(j)) / float64(Dim))
      let angle = float64(t) * invFreq
      result.c[t * Half + j] = float32(cos(angle))
      result.s[t * Half + j] = float32(sin(angle))

proc runRopeTest*(run: static Runtime, name: string) =
  ## RoPE op end to end: NEOX half-repeat rotation over q's last axis.
  ##
  ## Numerics checks, each exact against the f32 host reference with one
  ## rounding into bfloat16:
  ##
  ## - public face at (4, 2, 8), theta 100, kernel pairing and op order
  ## - sentinels at (2, 3, 8): cos=1/sin=0 leaves q bit for bit unchanged,
  ##   cos=0/sin=1 swaps the halves with the sign flip on the even columns
  ## - wide smoke at the Qwen3 geometry (8, 16, 128), theta 1e6, one launch
  ##   through the op path
  ##
  ## Contract and dispatch checks:
  ##
  ## - face contracts raise AssertionDefect on a rank-2 q, an odd last axis,
  ##   a float32 q, a bfloat16 table, a table token count or width mismatch
  ## - fan-out memo: two add consumers of one rope output see the rope's
  ##   single launch enqueued exactly once (3 batch entries)

  # public face: toTensor + rope + eval + item readback, exact
  block:
    const Tokens = 4
    const Heads = 2
    const Dim = 8
    const Half = Dim div 2
    let (cosH, sinH) = ropeTables(Tokens, Half, Dim, 100.0)
    var qH = newSeq[bfloat16](Tokens * Heads * Dim)
    for i in 0 ..< qH.len:
      qH[i] = (float32((i * 37) mod 31 - 15) * 0.25'f32).to(bfloat16)
    let q = toTensor(qH, @[Tokens, Heads, Dim])
    let cosT = toTensor(cosH, @[Tokens, Half])
    let sinT = toTensor(sinH, @[Tokens, Half])
    let o = rope(q, cosT, sinT)
    doAssert o.shape().len == 3, name & ": rope rank"
    doAssert o.shape()[0] == Tokens, name & ": rope tokens"
    doAssert o.shape()[1] == Heads, name & ": rope heads"
    doAssert o.shape()[2] == Dim, name & ": rope dim"
    doAssert o.dtype() == kBf16, name & ": rope dtype"
    eval(o)
    for t in 0 ..< Tokens:
      for h in 0 ..< Heads:
        for i in 0 ..< Dim:
          let partner = if i < Half: i + Half else: i - Half
          let qv = qH[t * Heads * Dim + h * Dim + i].to(float32)
          let pv = qH[t * Heads * Dim + h * Dim + partner].to(float32)
          let c = cosH[t * Half + i mod Half]
          let s = sinH[t * Half + i mod Half]
          let want = (if i < Half: qv * c - pv * s else: qv * c + pv * s).to(bfloat16)
          doAssert o.item(t, h, i, bfloat16) == want,
            name & ": rope out " & $t & "," & $h & "," & $i & ": got " &
            $o.item(t, h, i, bfloat16).to(float32) & ", want " & $want.to(float32)

  # sentinels: identity and half swap
  block:
    const Tokens = 2
    const Heads = 3
    const Dim = 8
    const Half = Dim div 2
    var pqH = newSeq[bfloat16](Tokens * Heads * Dim)
    for i in 0 ..< pqH.len:
      pqH[i] = (float32((i * 29) mod 19 - 9) * 0.5'f32).to(bfloat16)
    let pq = toTensor(pqH, @[Tokens, Heads, Dim])
    var ones = newSeq[float32](Tokens * Half)
    var zeros = newSeq[float32](Tokens * Half)
    for i in 0 ..< ones.len:
      ones[i] = 1.0'f32
      zeros[i] = 0.0'f32
    let cosId = toTensor(ones, @[Tokens, Half])
    let sinId = toTensor(zeros, @[Tokens, Half])
    let oid = rope(pq, cosId, sinId)
    eval(oid)
    for i in 0 ..< pqH.len:
      doAssert oid.item(i div (Heads * Dim), (i div Dim) mod Heads, i mod Dim, bfloat16).to(float32) ==
        pqH[i].to(float32),
        name & ": rope identity " & $i & ": got " &
        $oid.item(i div (Heads * Dim), (i div Dim) mod Heads, i mod Dim, bfloat16).to(float32) &
        ", want " & $pqH[i].to(float32)
    let cosSw = toTensor(zeros, @[Tokens, Half])
    let sinSw = toTensor(ones, @[Tokens, Half])
    let osw = rope(pq, cosSw, sinSw)
    eval(osw)
    for t in 0 ..< Tokens:
      for h in 0 ..< Heads:
        for i in 0 ..< Half:
          let base = t * Heads * Dim + h * Dim
          doAssert osw.item(t, h, i, bfloat16).to(float32) ==
            (-pqH[base + i + Half].to(float32)).to(bfloat16).to(float32),
            name & ": rope swap even " & $t & "," & $h & "," & $i
          doAssert osw.item(t, h, i + Half, bfloat16).to(float32) ==
            pqH[base + i].to(float32),
            name & ": rope swap odd " & $t & "," & $h & "," & $i

  # face contracts raise AssertionDefect
  block:
    const Tokens = 4
    const Heads = 2
    const Dim = 8
    const Half = Dim div 2
    let (cosH, sinH) = ropeTables(Tokens, Half, Dim, 100.0)
    var qH = newSeq[bfloat16](Tokens * Heads * Dim)
    for i in 0 ..< qH.len:
      qH[i] = (float32((i * 37) mod 31 - 15) * 0.25'f32).to(bfloat16)
    let q = toTensor(qH, @[Tokens, Heads, Dim])
    let cosT = toTensor(cosH, @[Tokens, Half])
    let sinT = toTensor(sinH, @[Tokens, Half])
    let q2 = toTensor(qH[0 ..< Tokens * Half], @[Tokens, Half])
    doAssertRaises AssertionDefect:
      discard rope(q2, cosT, sinT)
    let qOdd = toTensor(qH & qH[0 ..< 8], @[Tokens, Heads, Dim + 1])
    doAssertRaises AssertionDefect:
      discard rope(qOdd, cosT, sinT)
    var qf = newSeq[float32](Tokens * Heads * Dim)
    for i in 0 ..< qf.len:
      qf[i] = qH[i].to(float32)
    let q32 = toTensor(qf, @[Tokens, Heads, Dim])
    doAssertRaises AssertionDefect:
      discard rope(q32, cosT, sinT)
    var cosBfH = newSeq[bfloat16](Tokens * Half)
    for i in 0 ..< cosBfH.len:
      cosBfH[i] = cosH[i].to(bfloat16)
    let cosBf = toTensor(cosBfH, @[Tokens, Half])
    doAssertRaises AssertionDefect:
      discard rope(q, cosBf, sinT)
    let cosShort = toTensor(cosH[0 ..< (Tokens - 1) * Half], @[Tokens - 1, Half])
    doAssertRaises AssertionDefect:
      discard rope(q, cosShort, sinT)
    let cosNarrow = toTensor(cosH[0 ..< Tokens * (Half - 1)], @[Tokens, Half - 1])
    doAssertRaises AssertionDefect:
      discard rope(q, cosNarrow, sinT)

  # wide smoke at the Qwen3 geometry: (8, 16, 128), theta 1e6, single launch
  block:
    const Tokens = 8
    const Heads = 16
    const Dim = 128
    const Half = Dim div 2
    let (cosH, sinH) = ropeTables(Tokens, Half, Dim, 1e6)
    var qH = newSeq[bfloat16](Tokens * Heads * Dim)
    for i in 0 ..< qH.len:
      qH[i] = (float32((i * 37) mod 31 - 15) * 0.25'f32).to(bfloat16)
    let q = toTensor(qH, @[Tokens, Heads, Dim])
    let cosT = toTensor(cosH, @[Tokens, Half])
    let sinT = toTensor(sinH, @[Tokens, Half])
    let o = rope(q, cosT, sinT)
    eval(o)
    for t in 0 ..< Tokens:
      for h in 0 ..< Heads:
        for i in 0 ..< Dim:
          let partner = if i < Half: i + Half else: i - Half
          let qv = qH[t * Heads * Dim + h * Dim + i].to(float32)
          let pv = qH[t * Heads * Dim + h * Dim + partner].to(float32)
          let c = cosH[t * Half + i mod Half]
          let s = sinH[t * Half + i mod Half]
          let want = (if i < Half: qv * c - pv * s else: qv * c + pv * s).to(bfloat16)
          let got = o.item(t, h, i, bfloat16)
          doAssert got == want,
            name & ": rope wide " & $t & "," & $h & "," & $i & ": got " &
            $got.to(float32) & ", want " & $want.to(float32)

  # fan-out: two add consumers of one rope output enqueue the rope ONCE
  block:
    const Tokens = 4
    const Heads = 2
    const Dim = 8
    const Half = Dim div 2
    let (cosH, sinH) = ropeTables(Tokens, Half, Dim, 100.0)
    var qH = newSeq[bfloat16](Tokens * Heads * Dim)
    for i in 0 ..< qH.len:
      qH[i] = (float32((i * 37) mod 31 - 15) * 0.25'f32).to(bfloat16)
    var cH = newSeq[bfloat16](Tokens * Heads * Dim)
    for i in 0 ..< cH.len:
      cH[i] = (float32((i * 23) mod 13 - 6) * 0.25'f32).to(bfloat16)
    let engine = newEngine(run)
    var qShape = @[Tokens, Heads, Dim]
    let qNode = engine.newTensorNode(kBf16, qShape.toSpan())
    engine.upload(qNode, qH)
    var cShape = @[Tokens, Half]
    let cosNode = engine.newTensorNode(kFp32, cShape.toSpan())
    engine.upload(cosNode, cosH)
    let sinNode = engine.newTensorNode(kFp32, cShape.toSpan())
    engine.upload(sinNode, sinH)
    let cx = engine.newTensorNode(kBf16, qShape.toSpan())
    let cy = engine.newTensorNode(kBf16, qShape.toSpan())
    engine.upload(cx, cH)
    engine.upload(cy, cH)
    let p = engine.tn_rope(qNode, cosNode, sinNode)
    let z1 = engine.tn_add(p, cx)
    let z2 = engine.tn_add(p, cy)
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var nz1 = z1
    var nz2 = z2
    engine.evalNode(batch, nz1, visited)
    engine.evalNode(batch, nz2, visited)
    doAssert batch.entries.len == 3,
      name & ": rope fan-out enqueued " & $batch.entries.len & ", expected 3"
    batch.submit()
    var gz1, gz2: array[Tokens * Heads * Dim, bfloat16]
    engine.download(gz1, nz1)
    engine.download(gz2, nz2)
    for t in 0 ..< Tokens:
      for h in 0 ..< Heads:
        for i in 0 ..< Dim:
          let partner = if i < Half: i + Half else: i - Half
          let qv = qH[t * Heads * Dim + h * Dim + i].to(float32)
          let pv = qH[t * Heads * Dim + h * Dim + partner].to(float32)
          let c = cosH[t * Half + i mod Half]
          let s = sinH[t * Half + i mod Half]
          let pVal = (if i < Half: qv * c - pv * s else: qv * c + pv * s).to(bfloat16)
          let cVal = cH[t * Heads * Dim + h * Dim + i].to(float32)
          let exp = (pVal.to(float32) + cVal).to(bfloat16)
          doAssert gz1[t * Heads * Dim + h * Dim + i] == exp,
            name & ": rope fan-out z1 " & $t & "," & $h & "," & $i
          doAssert gz2[t * Heads * Dim + h * Dim + i] == exp,
            name & ": rope fan-out z2 " & $t & "," & $h & "," & $i

  echo "tensorcery rope (", name, "): all checks passed"

proc runSiluMulTest*(run: static Runtime, name: string) =
  ## Silu·mul op end to end: out = silu(gate)·up for the gated FFN activation.
  ##
  ## Numerics checks, judged against the f64 host reference
  ## (silu(g) = g / (1 + exp(-g)) in f64, one rounding into bfloat16):
  ##
  ## - public face at (4, 8), bfloat16: exact
  ## - the exp2 boundary set at the float32 limits: silu inputs from -200
  ##   up to the bfloat16 max against multipliers up to the bfloat16 max,
  ##   at a 2 bf16 ulp band. Above the exp2 overflow threshold,
  ##   silu(-89)·1e38 reads -19.75, never -0
  ## - f32 input: same face, a 1e-6 relative band
  ##
  ## Contract and dispatch checks:
  ##
  ## - face contracts raise AssertionDefect: a rank-3 operand, a shape mismatch, a dtype mismatch
  ## - fan-out memo: with two add consumers downstream, one silu·mul output enqueues exactly once (3 batch entries), readback exact
  block:
    const Rows = 4
    const Cols = 8
    var gh = newSeq[bfloat16](Rows * Cols)
    var vh = newSeq[bfloat16](Rows * Cols)
    for i in 0 ..< Rows * Cols:
      gh[i] = (float32((i * 37) mod 41 - 20) * 0.5'f32).to(bfloat16)
      vh[i] = (float32((i * 29) mod 31 - 15) * 0.25'f32).to(bfloat16)
    let gate = toTensor(gh, @[Rows, Cols])
    let up = toTensor(vh, @[Rows, Cols])
    let o = silu_mul(gate, up)
    doAssert o.shape().len == 2, name & ": silu rank"
    doAssert o.shape()[0] == Rows, name & ": silu tokens"
    doAssert o.shape()[1] == Cols, name & ": silu cols"
    doAssert o.dtype() == kBf16, name & ": silu dtype"
    eval(o)
    for t in 0 ..< Rows:
      for c in 0 ..< Cols:
        let g = float64(gh[t * Cols + c].to(float32))
        let v = float64(vh[t * Cols + c].to(float32))
        let want = float32(g / (1.0 + exp(-g)) * v).to(bfloat16)
        doAssert o.item(t, c, bfloat16) == want,
          name & ": silu out " & $t & "," & $c & ": got " &
          $o.item(t, c, bfloat16).to(float32) & ", want " & $want.to(float32)

  # exp2 boundary set at the float32 limits. The exp2 of the stock formula
  # overflows float32 once -g·Log2e passes 128, so silu inputs below -88.72.
  # Each product must keep the reference value and must never read -0.
  # From -192 down, both sides round to -0 in bfloat16
  block:
    const Rows = 2
    const Cols = 14
    let gh = [(3.3e38'f32).to(bfloat16), (-3.3e38'f32).to(bfloat16),
              (-100.0'f32).to(bfloat16), (-90.0'f32).to(bfloat16),
              (-89.0'f32).to(bfloat16), (-88.5'f32).to(bfloat16),
              (-88.0'f32).to(bfloat16), (88.0'f32).to(bfloat16),
              (0.0'f32).to(bfloat16), (-150.0'f32).to(bfloat16),
              (-176.0'f32).to(bfloat16), (-192.0'f32).to(bfloat16),
              (-193.0'f32).to(bfloat16), (-200.0'f32).to(bfloat16)]
    let vh = [(1.0'f32).to(bfloat16), (2.0'f32).to(bfloat16),
              (1e38'f32).to(bfloat16), (1e38'f32).to(bfloat16),
              (1e38'f32).to(bfloat16), (1e38'f32).to(bfloat16),
              (1e38'f32).to(bfloat16), (2.0'f32).to(bfloat16),
              (3.0'f32).to(bfloat16), (1e38'f32).to(bfloat16),
              (3.3e38'f32).to(bfloat16), (3.3e38'f32).to(bfloat16),
              (3.3e38'f32).to(bfloat16), (3.3e38'f32).to(bfloat16)]
    var ghFull = newSeq[bfloat16](Rows * Cols)
    var vhFull = newSeq[bfloat16](Rows * Cols)
    var refO = newSeq[float32](Rows * Cols)
    for t in 0 ..< Rows:
      for c in 0 ..< Cols:
        ghFull[t * Cols + c] = gh[c]
        vhFull[t * Cols + c] = vh[c]
        let g = float64(gh[c].to(float32))
        let v = float64(vh[c].to(float32))
        refO[t * Cols + c] = float32(g / (1.0 + exp(-g)) * v)
    let gate = toTensor(ghFull, @[Rows, Cols])
    let up = toTensor(vhFull, @[Rows, Cols])
    let o = silu_mul(gate, up)
    eval(o)
    for t in 0 ..< Rows:
      for c in 0 ..< Cols:
        let got = o.item(t, c, bfloat16)
        let want = refO[t * Cols + c].to(bfloat16)
        let dist = abs(bf16UlpKey(got) - bf16UlpKey(want))
        doAssert dist <= 2,
          name & ": silu boundary " & $t & "," & $c & " (gate " &
          $gh[c].to(float32) & "): got " & $got.to(float32) & ", want " &
          $want.to(float32) & ", " & $dist & " bf16 ulps"
      # silu inputs above the exp2 overflow threshold, large multipliers:
      # each product is an ordinary magnitude, never -0
      for c in [2, 3, 4, 9, 10]:
        doAssert o.item(t, c, bfloat16).to(float32) != 0.0'f32,
          name & ": silu boundary " & $t & "," & $c & " (gate " &
          $gh[c].to(float32) & ") collapsed to zero"

  # f32 input: same face, math in f32 against the f64 reference
  block:
    const Rows = 4
    const Cols = 8
    var gh = newSeq[float32](Rows * Cols)
    var vh = newSeq[float32](Rows * Cols)
    for i in 0 ..< Rows * Cols:
      gh[i] = float32((i * 37) mod 41 - 20) * 0.5'f32
      vh[i] = float32((i * 29) mod 31 - 15) * 0.25'f32
    let gate = toTensor(gh, @[Rows, Cols])
    let up = toTensor(vh, @[Rows, Cols])
    let o = silu_mul(gate, up)
    doAssert o.dtype() == kFp32, name & ": silu f32 dtype"
    eval(o)
    var worstRel = 0.0'f32
    for t in 0 ..< Rows:
      for c in 0 ..< Cols:
        let g = float64(gh[t * Cols + c])
        let v = float64(vh[t * Cols + c])
        let want = float32(g / (1.0 + exp(-g)) * v)
        let got = o.item(t, c, float32)
        if want == 0.0'f32:
          doAssert got == want, name & ": silu f32 out " & $t & "," & $c
        else:
          let rel = abs(got - want) / abs(want)
          if rel > worstRel:
            worstRel = rel
          doAssert rel <= 1e-6'f32,
            name & ": silu f32 out " & $t & "," & $c & ": got " & $got &
            ", reference " & $want
    echo name, ": silu f32 worst rel ", worstRel

  # face contracts raise AssertionDefect
  block:
    const Rows = 4
    const Cols = 8
    var gh = newSeq[bfloat16](Rows * Cols)
    var vh = newSeq[bfloat16](Rows * Cols)
    for i in 0 ..< Rows * Cols:
      gh[i] = (float32((i * 37) mod 41 - 20) * 0.5'f32).to(bfloat16)
      vh[i] = (float32((i * 29) mod 31 - 15) * 0.25'f32).to(bfloat16)
    let gate = toTensor(gh, @[Rows, Cols])
    let up = toTensor(vh, @[Rows, Cols])
    let gate3 = toTensor(gh[0 ..< 8], @[2, 2, 2])
    doAssertRaises AssertionDefect:
      discard silu_mul(gate3, up)
    let upShort = toTensor(vh[0 ..< Rows * (Cols - 1)], @[Rows, Cols - 1])
    doAssertRaises AssertionDefect:
      discard silu_mul(gate, upShort)
    var vf = newSeq[float32](Rows * Cols)
    for i in 0 ..< Rows * Cols:
      vf[i] = vh[i].to(float32)
    let upF32 = toTensor(vf, @[Rows, Cols])
    doAssertRaises AssertionDefect:
      discard silu_mul(gate, upF32)

  # fan-out: two add consumers of one silu·mul output enqueue it ONCE
  block:
    const Rows = 4
    const Cols = 8
    var gh = newSeq[bfloat16](Rows * Cols)
    var vh = newSeq[bfloat16](Rows * Cols)
    var ch = newSeq[bfloat16](Rows * Cols)
    for i in 0 ..< Rows * Cols:
      gh[i] = (float32((i * 37) mod 41 - 20) * 0.5'f32).to(bfloat16)
      vh[i] = (float32((i * 29) mod 31 - 15) * 0.25'f32).to(bfloat16)
      ch[i] = (float32((i * 23) mod 13 - 6) * 0.25'f32).to(bfloat16)
    let engine = newEngine(run)
    var shape = @[Rows, Cols]
    let gateNode = engine.newTensorNode(kBf16, shape.toSpan())
    let upNode = engine.newTensorNode(kBf16, shape.toSpan())
    let cxNode = engine.newTensorNode(kBf16, shape.toSpan())
    let cyNode = engine.newTensorNode(kBf16, shape.toSpan())
    engine.upload(gateNode, gh)
    engine.upload(upNode, vh)
    engine.upload(cxNode, ch)
    engine.upload(cyNode, ch)
    let p = engine.tn_silu_mul(gateNode, upNode)
    let z1 = engine.tn_add(p, cxNode)
    let z2 = engine.tn_add(p, cyNode)
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var nz1 = z1
    var nz2 = z2
    engine.evalNode(batch, nz1, visited)
    engine.evalNode(batch, nz2, visited)
    doAssert batch.entries.len == 3,
      name & ": silu fan-out enqueued " & $batch.entries.len & ", expected 3"
    batch.submit()
    var gz1, gz2: array[Rows * Cols, bfloat16]
    engine.download(gz1, nz1)
    engine.download(gz2, nz2)
    for t in 0 ..< Rows:
      for c in 0 ..< Cols:
        let g = float64(gh[t * Cols + c].to(float32))
        let v = float64(vh[t * Cols + c].to(float32))
        let pVal = float32(g / (1.0 + exp(-g)) * v).to(bfloat16)
        let cVal = ch[t * Cols + c].to(float32)
        let exp = (pVal.to(float32) + cVal).to(bfloat16)
        doAssert gz1[t * Cols + c] == exp, name & ": silu fan-out z1 " & $t & "," & $c
        doAssert gz2[t * Cols + c] == exp, name & ": silu fan-out z2 " & $t & "," & $c

  echo "tensorcery silu·mul (", name, "): all checks passed"

proc runViewTest*(run: static Runtime, name: string) =
  ## Shared-buffer views: view, reshape, transpose, permute, narrow, expand.
  ##
  ## Checks, all bfloat16 with item() readback against the fixture values.
  ## The read checks:
  ##
  ## - narrow reads the window, and the element offset shifts down the axis
  ## - transpose round-trips: the transpose of the transpose is the original
  ## - a view chain (narrow of transpose of narrow) stacks offset on offset
  ## - reshape keeps the element order over a dense source, and expand
  ##   broadcasts a size-1 axis at stride 0
  ## - an op consumes a stride-0 expand view per position on the device,
  ##   with the stride-0 axis last and first
  ##
  ## The window and split checks:
  ##
  ## - the general view face places positive- and negative-stride windows,
  ##   and rejects the windows that leave the backing buffer, in both signs
  ## - split of a narrowed tensor keeps the narrowed offset in every part
  ##
  ## The dispatch checks:
  ##
  ## - a view allocates nothing: evaluating any view of a materialized
  ##   tensor enqueues zero kernels
  ## - two views of one producer in one eval enqueue the producer exactly once,
  ##   and a re-eval enqueues nothing
  ##
  ## The contract checks raise AssertionDefect:
  ##
  ## - a backing-capacity overflow, and negative dimensions at construction
  ##   and at the view face
  ## - a narrow window outside its axis, and a shape/strides rank mismatch
  ## - a reshape of a strided source or mismatched element count
  ## - a repeated permutation axis, and an inexpandable target shape
  var xd: array[8, float32] = [1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  var xq: array[8, bfloat16]
  for i in 0 ..< 8:
    xq[i] = xd[i].to(bfloat16)
  let x = toTensor(xq, @[2, 4])

  # narrow: the window with its element offset shifted down the axis
  block:
    var nShape = @[2, 2]
    var nStrides = @[4, 1]
    let n = narrow(x, 1, 1, 2)
    doAssert n.shape() == nShape.toSpan(), name & ": narrow shape"
    doAssert n.strides() == nStrides.toSpan(), name & ": narrow strides"
    for r in 0 ..< 2:
      for c in 0 ..< 2:
        doAssert n.item(r, c, bfloat16) == xd[r * 4 + 1 + c].to(bfloat16),
          name & ": narrow out " & $r & "," & $c

  # transpose round-trip: the transpose of the transpose is the original
  block:
    var tShape = @[4, 2]
    var tStrides = @[1, 4]
    let t = transpose(x)
    doAssert t.shape() == tShape.toSpan(), name & ": transpose shape"
    doAssert t.strides() == tStrides.toSpan(), name & ": transpose strides"
    for i in 0 ..< 4:
      for j in 0 ..< 2:
        doAssert t.item(i, j, bfloat16) == xd[j * 4 + i].to(bfloat16),
          name & ": transpose out " & $i & "," & $j
    var ttShape = @[2, 4]
    var ttStrides = @[4, 1]
    let tt = transpose(t)
    doAssert tt.shape() == ttShape.toSpan(), name & ": round-trip shape"
    doAssert tt.strides() == ttStrides.toSpan(), name & ": round-trip strides"
    for i in 0 ..< 8:
      doAssert tt.item(i div 4, i mod 4, bfloat16) == xd[i].to(bfloat16),
        name & ": round-trip out " & $i

  # view chain: narrow of transpose of narrow, offset stacked on offset
  block:
    let v = narrow(transpose(narrow(x, 1, 1, 2)), 0, 1, 1)
    # inner narrow (2,2):(4,1) at offset 1, transpose (2,2):(1,4) keeps offset 1,
    # outer narrow (1,2):(1,4) adds 1*1, so the buffer offset reads element 2
    var vShape = @[1, 2]
    var vStrides = @[1, 4]
    doAssert v.shape() == vShape.toSpan(), name & ": chain shape"
    doAssert v.strides() == vStrides.toSpan(), name & ": chain strides"
    for c in 0 ..< 2:
      doAssert v.item(0, c, bfloat16) == xd[2 + c * 4].to(bfloat16),
        name & ": chain out " & $c

  # reshape keeps the element order, and a dense narrowed source keeps its offset
  block:
    let r = reshape(x, @[8])
    for i in 0 ..< 8:
      doAssert r.item(i, bfloat16) == xd[i].to(bfloat16),
        name & ": reshape out " & $i
    let nr = reshape(narrow(x, 0, 1, 1), @[4])
    for i in 0 ..< 4:
      doAssert nr.item(i, bfloat16) == xd[4 + i].to(bfloat16),
        name & ": reshape offset out " & $i

  # permute: one index per axis, result axis i reads the source axis axes[i]
  block:
    let p = permute(x, @[1, 0])
    for i in 0 ..< 4:
      for j in 0 ..< 2:
        doAssert p.item(i, j, bfloat16) == xd[j * 4 + i].to(bfloat16),
          name & ": permute out " & $i & "," & $j

  # expand: a size-1 axis broadcasts at stride 0, readback on the host
  block:
    let col = narrow(x, 1, 0, 1)
    let e = expand(col, @[2, 4])
    var eShape = @[2, 4]
    var eStrides = @[4, 0]
    doAssert e.shape() == eShape.toSpan(), name & ": expand shape"
    doAssert e.strides() == eStrides.toSpan(), name & ": expand strides"
    for r in 0 ..< 2:
      for c in 0 ..< 4:
        doAssert e.item(r, c, bfloat16) == xd[r * 4].to(bfloat16),
          name & ": expand out " & $r & "," & $c

  # a stride-0 expand consumed by an op: the device reads the broadcast
  # operand per position, in both axis orientations
  block:
    let col = narrow(x, 1, 0, 1)
    let lastAxis = expand(col, @[2, 4]) + x
    eval(lastAxis)
    for r in 0 ..< 2:
      for c in 0 ..< 4:
        doAssert lastAxis.item(r, c, bfloat16) ==
          (xd[r * 4].to(float32) + xd[r * 4 + c].to(float32)).to(bfloat16),
          name & ": expand add last axis " & $r & "," & $c
    let row = narrow(x, 0, 0, 1)
    let firstAxis = expand(row, @[2, 4]) + x
    eval(firstAxis)
    for r in 0 ..< 2:
      for c in 0 ..< 4:
        doAssert firstAxis.item(r, c, bfloat16) ==
          (xd[c].to(float32) + xd[r * 4 + c].to(float32)).to(bfloat16),
          name & ": expand add first axis " & $r & "," & $c

  # general view face: windows in both stride signs, offsets included
  block:
    let y = toTensor(xq[0 ..< 4], @[1, 4])
    let pos = view(y, @[3], @[1], 1)
    for i in 0 ..< 3:
      doAssert pos.item(i, bfloat16) == xd[1 + i].to(bfloat16),
        name & ": view pos out " & $i
    let neg = view(y, @[4], @[-1], 3)
    for i in 0 ..< 4:
      doAssert neg.item(i, bfloat16) == xd[3 - i].to(bfloat16),
        name & ": view neg out " & $i

  # split of a narrowed tensor keeps the narrowed offset in every part
  block:
    let n = narrow(x, 0, 1, 1)
    let parts = split(n, 1, 2)
    doAssert parts.len == 2, name & ": split of narrow part count"
    for c in 0 ..< 2:
      doAssert parts[0].item(0, c, bfloat16) == xd[4 + c].to(bfloat16),
        name & ": split of narrow part0 " & $c
      doAssert parts[1].item(0, c, bfloat16) == xd[6 + c].to(bfloat16),
        name & ": split of narrow part1 " & $c

  # contracts raise AssertionDefect
  block:
    let y = toTensor(xq[0 ..< 4], @[1, 4])
    # a window that leaves the backing buffer, positive and negative strides
    doAssertRaises AssertionDefect:
      discard view(y, @[4], @[1], 1)
    doAssertRaises AssertionDefect:
      discard view(y, @[4], @[-1], 4)
    # negative dimensions, at construction and at the view face
    doAssertRaises AssertionDefect:
      discard toTensor(xq, @[-2, -4])
    doAssertRaises AssertionDefect:
      discard view(x, @[-2, 4], @[4, 1], 0)
    # a narrow window outside its axis, and an axis outside the rank
    doAssertRaises AssertionDefect:
      discard narrow(x, 1, 3, 2)
    doAssertRaises AssertionDefect:
      discard narrow(x, 2, 0, 1)
    # a shape/strides rank mismatch
    doAssertRaises AssertionDefect:
      discard view(x, @[2], @[2, 1], 0)
    # a rank outside 1..5, on view and reshape
    doAssertRaises AssertionDefect:
      discard view(x, @[], @[], 0)
    doAssertRaises AssertionDefect:
      discard view(x, @[2, 2, 2, 2, 2, 2], @[8, 4, 2, 1, 1, 1], 0)
    doAssertRaises AssertionDefect:
      discard reshape(x, @[2, 2, 2, 2, 2, 2])
    # reshape of a strided source, and a mismatched element count
    doAssertRaises AssertionDefect:
      discard reshape(transpose(x), @[8])
    doAssertRaises AssertionDefect:
      discard reshape(x, @[7])
    # a repeated permutation axis
    doAssertRaises AssertionDefect:
      discard permute(x, @[0, 0])
    # an inexpandable target shape
    doAssertRaises AssertionDefect:
      discard expand(x, @[3, 4])

  # a view allocates nothing: evaluating any view of a materialized tensor
  # enqueues zero kernels
  block:
    let engine = newEngine(run)
    var shape = @[2, 4]
    let xn = engine.newTensorNode(kBf16, shape.toSpan())
    engine.upload(xn, xq)
    let views = @[engine.tn_narrow(xn, 1, 1, 2),
                  engine.tn_transpose(xn),
                  engine.tn_permute(xn, @[1, 0]),
                  engine.tn_reshape(xn, @[8]),
                  engine.tn_expand(engine.tn_narrow(xn, 1, 0, 1), @[2, 4]),
                  engine.tn_view(xn, @[8], @[1], 0)]
    for v in views:
      var nv = v
      var batch = engine.recordBatch()
      var visited: VisitedNodes
      engine.evalNode(batch, nv, visited)
      doAssert batch.entries.len == 0,
        name & ": view eval enqueued " & $batch.entries.len & " kernels, expected 0"

  # eval-through-view kernel counts: two views of one producer enqueue it once,
  # a re-eval enqueues nothing, source and view in one eval enqueue once
  block:
    var yd: array[8, float32] = [2.0'f32, 2, 2, 2, 2, 2, 2, 2]
    var yq: array[8, bfloat16]
    for i in 0 ..< 8:
      yq[i] = yd[i].to(bfloat16)
    let engine = newEngine(run)
    var shape = @[2, 4]
    let xn = engine.newTensorNode(kBf16, shape.toSpan())
    let yn = engine.newTensorNode(kBf16, shape.toSpan())
    engine.upload(xn, xq)
    engine.upload(yn, yq)

    let p = engine.tn_add(xn, yn)
    let v1 = engine.tn_narrow(p, 1, 1, 2)
    let v2 = engine.tn_transpose(p)
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var nv1 = v1
    var nv2 = v2
    engine.evalNode(batch, nv1, visited)
    engine.evalNode(batch, nv2, visited)
    doAssert batch.entries.len == 1,
      name & ": two views of one producer enqueued " & $batch.entries.len &
      " kernels, expected 1"
    batch.submit()
    for r in 0 ..< 2:
      for c in 0 ..< 2:
        let want = (xd[r * 4 + 1 + c] + yd[r * 4 + 1 + c]).to(bfloat16)
        var got: bfloat16
        engine.downloadItem(got, nv1, r * 4 + c)
        doAssert got == want, name & ": narrow of add " & $r & "," & $c
    for i in 0 ..< 4:
      for j in 0 ..< 2:
        let want = (xd[j * 4 + i] + yd[j * 4 + i]).to(bfloat16)
        var got: bfloat16
        engine.downloadItem(got, nv2, i + j * 4)
        doAssert got == want, name & ": transpose of add " & $i & "," & $j

    # a re-eval of both views enqueues nothing, since the buffers are materialized
    var reBatch = engine.recordBatch()
    var reVisited: VisitedNodes
    var rv1 = nv1
    var rv2 = nv2
    engine.evalNode(reBatch, rv1, reVisited)
    engine.evalNode(reBatch, rv2, reVisited)
    doAssert reBatch.entries.len == 0,
      name & ": view re-eval enqueued " & $reBatch.entries.len & " kernels, expected 0"

    # source and view in one eval, source root first: still one kernel
    let p2 = engine.tn_add(xn, yn)
    let v3 = engine.tn_transpose(p2)
    var batch2 = engine.recordBatch()
    var visited2: VisitedNodes
    var np2 = p2
    var nv3 = v3
    engine.evalNode(batch2, np2, visited2)
    engine.evalNode(batch2, nv3, visited2)
    doAssert batch2.entries.len == 1,
      name & ": source and view enqueued " & $batch2.entries.len &
      " kernels, expected 1"

  # a view chain consumed twice through one eval: the producer still
  # enqueues exactly once, and the chain's offsets compose at readback
  block:
    var yd: array[8, float32] = [2.0'f32, 2, 2, 2, 2, 2, 2, 2]
    var yq: array[8, bfloat16]
    for i in 0 ..< 8:
      yq[i] = yd[i].to(bfloat16)
    let engine = newEngine(run)
    var shape = @[2, 4]
    let xn = engine.newTensorNode(kBf16, shape.toSpan())
    let yn = engine.newTensorNode(kBf16, shape.toSpan())
    engine.upload(xn, xq)
    engine.upload(yn, yq)

    let p = engine.tn_add(xn, yn)
    let chain = engine.tn_narrow(engine.tn_transpose(engine.tn_narrow(p, 1, 1, 2)), 0, 0, 2)
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var c1 = chain
    var c2 = chain
    engine.evalNode(batch, c1, visited)
    engine.evalNode(batch, c2, visited)
    doAssert batch.entries.len == 1,
      name & ": chain consumed twice enqueued " & $batch.entries.len &
      " kernels, expected 1"
    batch.submit()
    for i in 0 ..< 2:
      for j in 0 ..< 2:
        let want = (xd[j * 4 + 1 + i] + yd[j * 4 + 1 + i]).to(bfloat16)
        var got: bfloat16
        engine.downloadItem(got, c1, i + j * 4)
        doAssert got == want, name & ": chain readback " & $i & "," & $j

  # offset composition: a view's origin includes the source's own offset,
  # so a view of a narrowed tensor reads through the composed origin,
  # not the buffer origin
  block:
    let t = toTensor(xq, @[2, 4])
    let v = view(narrow(t, 1, 1, 2), @[2, 2], @[2, 1], 1)
    # composed origin = narrow's offset 1 + view's offset 1 = 2
    # a dropped source term would read from 1
    doAssert v.item(0, 0, bfloat16) == xq[2], name & ": composed origin (0,0)"
    doAssert v.item(1, 1, bfloat16) == xq[5], name & ": composed origin (1,1)"

  echo "tensorcery views (", name, "): all checks passed"

proc k_write_once_probe*(dst: distinct TensorView, hits: ptr UncheckedArray[int32]) =
  ## Write-once test of the pool write kernel's grid-stride skeleton.
  ##
  ## Contract:
  ##
  ## - keeps the kernel's exact launch shape and stride expression
  ## - counts the writes per entry through an atomic
  ## - a stride that does not cover the thread space exactly once fails
  ##
  ## The op's copy is idempotent: a bad stride writes each entry several
  ## times concurrently with identical values, which no value comparison can observe.
  let
    numel = dst.layout.shape[0]
    tid = thread_position_in_grid.x
    stride = threadgroups_per_grid.x * threads_per_threadgroup.x
  for i in countup(tid, int32(numel - 1), stride):
    dst[(int(i),)] = (float32(i) * 0.5'f32).to(dst.T)
    discard atomicFetchAdd(hits[i].addr, 1'i32)

proc runPoolWriteTest*(run: static Runtime, name: string) =
  ## Pool write op checks:
  ##
  ## - grid-stride write-once at multi-workgroup launches
  ## - page-decomposition round-trips
  ## - face contracts raise
  ## - the fan-out memo dedupes a re-walk to one batch entry
  ##
  ## Everything runs on one concrete engine: the write-once test, the round-trips and the fan-out write their kernels into one artifact set.
  const
    NumPages = 4
    NumLayers = 2
    KvHeads = 2
    HeadDim = 4
    PageSize = 256
    SlotElems = PageSize * KvHeads * HeadDim
    SlabElems = NumPages * NumLayers * SlotElems

  proc slabIdx(page, layer, pos, h, d: int): int =
    (((page * NumLayers + layer) * PageSize + pos) * KvHeads + h) * HeadDim + d

  proc srcIdx(t, h, d: int): int =
    (t * KvHeads + h) * HeadDim + d

  let engine = newEngine(run)
  var slabShapeSeq = @[NumPages, NumLayers, PageSize, KvHeads, HeadDim]
  let slabShape = slabShapeSeq.toSpan()
  var srcShapeSeq = @[1, 10, KvHeads, HeadDim]
  let srcShape = srcShapeSeq.toSpan()

  proc slabNode(seed: int): TensorNode =
    ## Sentinel-filled slab, device and host copies in lockstep
    var host = newSeq[bfloat16](SlabElems)
    for i in 0 ..< SlabElems:
      host[i] = (float32((i * 31 + seed) mod 251) * 0.125'f32 - 8.0'f32).to(bfloat16)
    result = engine.newTensorNode(kBf16, slabShape)
    engine.upload(result, host)

  proc slabHostCopy(seed: int): seq[bfloat16] =
    result = newSeq[bfloat16](SlabElems)
    for i in 0 ..< SlabElems:
      result[i] = (float32((i * 31 + seed) mod 251) * 0.125'f32 - 8.0'f32).to(bfloat16)

  proc srcNodes(seqLen: int, seed: int): tuple[k, v: TensorNode, kh, vh: seq[bfloat16]] =
    ## (1, seq_len, kv_heads, head_dim) src entries, device and host copies
    var shapeSeq = @[1, seqLen, KvHeads, HeadDim]
    let shape = shapeSeq.toSpan()
    result.kh = newSeq[bfloat16](seqLen * KvHeads * HeadDim)
    result.vh = newSeq[bfloat16](seqLen * KvHeads * HeadDim)
    for i in 0 ..< result.kh.len:
      result.kh[i] = (float32((i * 7 + seed) mod 97) * 0.25'f32 - 4.0'f32).to(bfloat16)
      result.vh[i] = (float32((i * 11 + seed) mod 89) * 0.25'f32 - 4.0'f32).to(bfloat16)
    result.k = engine.newTensorNode(kBf16, shape)
    result.v = engine.newTensorNode(kBf16, shape)
    engine.upload(result.k, result.kh)
    engine.upload(result.v, result.vh)

  proc evalOne(n: TensorNode): TensorNode =
    ## One batch, one root, the node's replacement returned
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var nn = n
    engine.evalNode(batch, nn, visited)
    batch.submit()
    nn

  # grid-stride write-once at a multi-workgroup launch: 4096 entries over
  # 16 threadgroups of 256 threads. Each element must carry exactly one
  # write and the copied value must be exact.
  block:
    const Numel = 4096
    let
      dstDev = engine.allocDevice(bfloat16, Numel)
      hitsDev = engine.allocDevice(int32, Numel)
      db = initTensorBridge(dstDev, 0, make_layout((int(Numel),), (int(1),)))
    var zeroHits = newSeq[int32](Numel)
    engine.copyToDeviceFromHost(hitsDev, zeroHits)
    var batch = engine.recordBatch()
    batch.enqueue<<((Numel + 255) div 256, 256)>> k_write_once_probe(db, hitsDev)
    batch.submit()
    var gotHits = newSeq[int32](Numel)
    engine.copyToHostFromDevice(gotHits, hitsDev, 0)
    var gotDst = newSeq[bfloat16](Numel)
    engine.copyToHostFromDevice(gotDst, dstDev, 0)
    for i in 0 ..< Numel:
      doAssert gotHits[i] == 1,
        "entry " & $i & " written " & $gotHits[i] & " times, expected exactly 1"
      doAssert gotDst[i].to(float32) == (float32(i) * 0.5'f32).to(bfloat16).to(float32),
        "entry " & $i & " value mismatch"

  # multi-page write across three pages at 19 threadgroups
  block:
    let kSlab = slabNode(4)
    let vSlab = slabNode(5)
    let kHost = slabHostCopy(4)
    let (kSrc, vSrc, kh, vh) = srcNodes(600, 6)
    let w = engine.tn_pool_write(kSlab, vSlab, kSrc, vSrc, 1, 200, 600, 0, @[0, 1, 2, 3])
    discard evalOne(w)
    var gotK = newSeq[bfloat16](SlabElems)
    var gotV = newSeq[bfloat16](SlabElems)
    engine.download(gotK, kSlab)
    engine.download(gotV, vSlab)
    for t in 0 ..< 600:
      let globalPos = 200 + t
      let page = globalPos div PageSize
      let pos = globalPos mod PageSize
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(page, 1, pos, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
            "K multi (" & $t & " @" & $globalPos & "," & $h & "," & $d & ")"
          doAssert gotV[slabIdx(page, 1, pos, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
            "V multi (" & $t & " @" & $globalPos & "," & $h & "," & $d & ")"
    for page in 0 .. 2:
      for pos in 0 ..< PageSize:
        for h in 0 ..< KvHeads:
          for d in 0 ..< HeadDim:
            doAssert gotK[slabIdx(page, 0, pos, h, d)].to(float32) == kHost[slabIdx(page, 0, pos, h, d)].to(float32),
              "K multi layer0 sentinel (" & $page & "," & $pos & ")"

  # remapped page table: sequence page 0 -> pool slot 2, page 1 -> slot 0
  block:
    let kSlab = slabNode(9)
    let vSlab = slabNode(10)
    let (kSrc, vSrc, kh, vh) = srcNodes(PageSize + 10, 11)
    let w = engine.tn_pool_write(kSlab, vSlab, kSrc, vSrc, 1, 0, PageSize + 10, 0, @[2, 0, 1, 3])
    discard evalOne(w)
    var gotK = newSeq[bfloat16](SlabElems)
    var gotV = newSeq[bfloat16](SlabElems)
    engine.download(gotK, kSlab)
    engine.download(gotV, vSlab)
    for t in 0 ..< PageSize + 10:
      let seqPage = t div PageSize
      let poolPage = [2, 0, 1, 3][seqPage]
      let pos = t mod PageSize
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(poolPage, 1, pos, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
            "K remapped (" & $t & " seqPage " & $seqPage & " slot " & $poolPage & ")"
          doAssert gotV[slabIdx(poolPage, 1, pos, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
            "V remapped (" & $t & " seqPage " & $seqPage & " slot " & $poolPage & ")"

  # cached-prefix skip: offset 100, 10 entries, cachedTokens 105
  block:
    let kSlab = slabNode(7)
    let vSlab = slabNode(8)
    let kHost = slabHostCopy(7)
    let (kSrc, vSrc, kh, vh) = srcNodes(10, 12)
    let w = engine.tn_pool_write(kSlab, vSlab, kSrc, vSrc, 0, 100, 10, 105, @[0, 1, 2, 3])
    discard evalOne(w)
    var gotK = newSeq[bfloat16](SlabElems)
    var gotV = newSeq[bfloat16](SlabElems)
    engine.download(gotK, kSlab)
    engine.download(gotV, vSlab)
    # writeStart = max(0, 105 - 100) = 5: global positions 105..109 written
    for t in 0 ..< 10:
      let pos = (100 + t) mod PageSize
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          if t < 5:
            doAssert gotK[slabIdx(0, 0, pos, h, d)].to(float32) == kHost[slabIdx(0, 0, pos, h, d)].to(float32),
              "K cached sentinel (" & $t & "," & $h & "," & $d & ")"
          else:
            doAssert gotK[slabIdx(0, 0, pos, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
              "K cached (" & $t & "," & $h & "," & $d & ")"
            doAssert gotV[slabIdx(0, 0, pos, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
              "V cached (" & $t & "," & $h & "," & $d & ")"

  # contracts raise: dtype, pool geometry, layer, write range, page table
  block:
    let kSlab = slabNode(1)
    let vSlab = slabNode(2)
    let (kSrc, vSrc, _, _) = srcNodes(10, 3)
    var f32ShapeSeq = @[NumPages, NumLayers, PageSize, KvHeads, HeadDim]
    let f32Slab = engine.newTensorNode(kFp32, f32ShapeSeq.toSpan())
    engine.upload(f32Slab, newSeq[float32](SlabElems))
    var smallShapeSeq = @[NumPages, NumLayers, PageSize, KvHeads, HeadDim + 1]
    let smallSlab = engine.newTensorNode(kBf16, smallShapeSeq.toSpan())
    engine.upload(smallSlab, newSeq[bfloat16](NumPages * NumLayers * PageSize * KvHeads * (HeadDim + 1)))
    doAssertRaises AssertionDefect:
      discard engine.tn_pool_write(kSlab, smallSlab, kSrc, vSrc, 0, 0, 10, 0, @[0, 1, 2, 3])
    doAssertRaises AssertionDefect:
      discard engine.tn_pool_write(kSlab, vSlab, f32Slab, vSrc, 0, 0, 10, 0, @[0, 1, 2, 3])
    doAssertRaises AssertionDefect:
      discard engine.tn_pool_write(kSlab, vSlab, kSrc, vSrc, 2, 0, 10, 0, @[0, 1, 2, 3])
    doAssertRaises AssertionDefect:
      discard engine.tn_pool_write(kSlab, vSlab, kSrc, vSrc, 0, 0, 0, 0, @[0, 1, 2, 3])
    doAssertRaises AssertionDefect:
      discard engine.tn_pool_write(kSlab, vSlab, kSrc, vSrc, 0, 999, 10, 0, @[0, 1, 2])
    doAssertRaises AssertionDefect:
      discard engine.tn_pool_write(kSlab, vSlab, kSrc, vSrc, 0, 0, 10, 0, @[0, 1, 2, 4])
    doAssertRaises AssertionDefect:
      discard engine.tn_pool_write(kSlab, vSlab, kSrc, vSrc, 0, 0, 10, 0, @[0, 1, 2, -1])

  # Fan-out memo: a re-walk of the pool-write node in one eval must not
  # re-stage the metadata or re-enqueue the launch. One batch entry, the write happens once.
  block:
    let kSlab = slabNode(13)
    let vSlab = slabNode(14)
    let (kSrc, vSrc, kh, vh) = srcNodes(10, 15)
    let w = engine.tn_pool_write(kSlab, vSlab, kSrc, vSrc, 0, 0, 10, 0, @[0, 1, 2, 3])
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var n1 = w
    var n2 = w
    engine.evalNode(batch, n1, visited)
    engine.evalNode(batch, n2, visited)
    doAssert batch.entries.len == 1, name & ": re-walk enqueued " &
      $batch.entries.len & ", expected 1 (the visited memo dedupes the write)"
    batch.submit()
    var gotK = newSeq[bfloat16](SlabElems)
    var gotV = newSeq[bfloat16](SlabElems)
    engine.download(gotK, kSlab)
    engine.download(gotV, vSlab)
    for t in 0 ..< 10:
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(0, 0, t, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
            "fan-out K (" & $t & "," & $h & "," & $d & ")"
          doAssert gotV[slabIdx(0, 0, t, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
            "fan-out V (" & $t & "," & $h & "," & $d & ")"

  echo "pool write op (", name, "): all checks passed"

proc runFactoryTest*(run: static Runtime, name: string) =
  ## Host tensor factories and the layout-aware host readback (toSeq).
  ##
  ## Checks:
  ## - zeros, ones, full and the like-faces round-trip: toSeq returns
  ##   the constructed values, bf16 and f32
  ## - arange, linspace and eye are exact at the boundaries: a negative-step
  ##   arange excludes the stop, linspace endpoints are included,
  ##   eye is the identity
  ## - toSeq of narrowed+transposed, negative-stride and stride-0 views
  ##   equals the host-computed strided read
  ## - toSeq materializes an op output first, and the contracts raise:
  ##   rank outside 1..5, zero arange step, linspace num below 2, negative eye,
  ##   dtype-mismatched toSeq
  var xd: array[32, float32] = [1.0'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17.0'f32, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32]
  var xq: array[32, bfloat16]
  for i in 0 ..< 32:
    xq[i] = xd[i].to(bfloat16)

  # zeros/ones/full round-trip through toSeq, bf16 and f32
  block:
    let z = zeros(@[2, 4], run)
    var zShape = @[2, 4]
    var zStrides = @[4, 1]
    doAssert z.shape() == zShape.toSpan(), name & ": zeros shape"
    doAssert z.strides() == zStrides.toSpan(), name & ": zeros strides"
    doAssert z.dtype() == kBf16, name & ": zeros dtype"
    var gz = toSeq(z, bfloat16)
    doAssert gz.len == 8, name & ": zeros numel"
    for i in 0 ..< gz.len:
      doAssert gz[i] == 0'f32.to(bfloat16), name & ": zeros " & $i
    let o = ones(@[2, 2, 2], run, float32)
    doAssert o.dtype() == kFp32, name & ": ones dtype"
    var go = toSeq(o, float32)
    for i in 0 ..< go.len:
      doAssert go[i] == 1.0'f32, name & ": ones " & $i
    let f = full(@[2, 3], 2.5'f32, run)
    var gf = toSeq(f, bfloat16)
    for i in 0 ..< gf.len:
      doAssert gf[i] == 2.5'f32.to(bfloat16), name & ": full " & $i
    let ff = full(@[4], -1.5'f32, run, float32)
    var gff = toSeq(ff, float32)
    for i in 0 ..< gff.len:
      doAssert gff[i] == -1.5'f32, name & ": full f32 " & $i

  # like-faces: the source's shape and dtype, a fresh dense buffer
  block:
    let x = toTensor(xq[0 ..< 8], @[2, 4])
    let zl = zeros_like(x)
    var zlStrides = @[4, 1]
    doAssert zl.dtype() == kBf16, name & ": zeros_like dtype"
    doAssert zl.strides() == zlStrides.toSpan(), name & ": zeros_like strides"
    var gzl = toSeq(zl, bfloat16)
    for i in 0 ..< gzl.len:
      doAssert gzl[i] == 0'f32.to(bfloat16), name & ": zeros_like " & $i
    let ol = ones_like(x)
    var gol = toSeq(ol, bfloat16)
    for i in 0 ..< gol.len:
      doAssert gol[i] == 1'f32.to(bfloat16), name & ": ones_like " & $i
    let fl = full_like(x, 3.5'f32)
    var gfl = toSeq(fl, bfloat16)
    for i in 0 ..< gfl.len:
      doAssert gfl[i] == 3.5'f32.to(bfloat16), name & ": full_like " & $i
    # a like-face over a view source: the view's shape, dense strides again
    let n = narrow(x, 0, 1, 1)
    let nl = ones_like(n)
    var nlShape = @[1, 4]
    doAssert nl.shape() == nlShape.toSpan(), name & ": ones_like of narrow shape"
    doAssert nl.strides() == zlStrides.toSpan(), name & ": ones_like of narrow strides"
    var gnl = toSeq(nl, bfloat16)
    for i in 0 ..< gnl.len:
      doAssert gnl[i] == 1'f32.to(bfloat16), name & ": ones_like of narrow " & $i
    # f32 source
    let xf = toTensor(@[1.5'f32, 2.5], @[2], run, float32)
    let zfl = zeros_like(xf)
    doAssert zfl.dtype() == kFp32, name & ": zeros_like f32 dtype"
    var gzfl = toSeq(zfl, float32)
    for i in 0 ..< gzfl.len:
      doAssert gzfl[i] == 0.0'f32, name & ": zeros_like f32 " & $i

  # arange: exact values, a negative step excludes the stop
  block:
    let a = arange(0, 8, 1'f32, run, float32)
    var aShape = @[8]
    doAssert a.shape() == aShape.toSpan(), name & ": arange shape"
    var ga = toSeq(a, float32)
    for i in 0 ..< 8:
      doAssert ga[i] == float32(i), name & ": arange " & $i
    let aneg = arange(10, 0, -2'f32, run)
    var ganeg = toSeq(aneg, bfloat16)
    doAssert ganeg.len == 5, name & ": arange negative step count"
    for i in 0 ..< 5:
      doAssert ganeg[i] == float32(10 - 2 * i).to(bfloat16),
        name & ": arange negative step " & $i
    let astep = arange(0, 1, 0.25'f32, run, float32)
    var gastep = toSeq(astep, float32)
    doAssert gastep.len == 4, name & ": arange f32 step count"
    for i in 0 ..< 4:
      doAssert gastep[i] == float32(i) * 0.25'f32, name & ": arange f32 step " & $i

  # linspace: both endpoints included. The last element is exactly the stop.
  block:
    let l = linspace(0, 1, 5, run, float32)
    var gl = toSeq(l, float32)
    doAssert gl.len == 5, name & ": linspace count"
    doAssert gl[0] == 0.0'f32, name & ": linspace first endpoint"
    doAssert gl[4] == 1.0'f32, name & ": linspace last endpoint"
    for i in 1 ..< 4:
      doAssert gl[i] == float32(i) * 0.25'f32, name & ": linspace " & $i
    let lb = linspace(2, 3, 3, run)
    var glb = toSeq(lb, bfloat16)
    doAssert glb[0] == 2'f32.to(bfloat16), name & ": linspace bf16 start"
    doAssert glb[1] == 2.5'f32.to(bfloat16), name & ": linspace bf16 mid"
    doAssert glb[2] == 3'f32.to(bfloat16), name & ": linspace bf16 stop"

  # eye: the identity
  block:
    let e = eye(3, run, float32)
    var eShape = @[3, 3]
    doAssert e.shape() == eShape.toSpan(), name & ": eye shape"
    var ge = toSeq(e, float32)
    for i in 0 ..< 3:
      for j in 0 ..< 3:
        doAssert ge[i * 3 + j] == (if i == j: 1.0'f32 else: 0.0'f32),
          name & ": eye " & $i & "," & $j
    let eb = eye(2, run)
    var geb = toSeq(eb, bfloat16)
    doAssert geb[0] == 1'f32.to(bfloat16), name & ": eye bf16 diag"
    doAssert geb[1] == 0'f32.to(bfloat16), name & ": eye bf16 off"

  # toSeq of views: the strided read matches the host-computed layout walk
  block:
    let x = toTensor(xq, @[4, 8])
    # narrow the first axis to entries 1..2, then transpose: (8,2) strides (1,8), offset 8
    let v = transpose(narrow(x, 0, 1, 2))
    var vShape = @[8, 2]
    var vStrides = @[1, 8]
    doAssert v.shape() == vShape.toSpan(), name & ": view gather shape"
    doAssert v.strides() == vStrides.toSpan(), name & ": view gather strides"
    let gv = toSeq(v, bfloat16)
    doAssert gv.len == 16, name & ": view gather numel"
    for i in 0 ..< 8:
      for j in 0 ..< 2:
        doAssert gv[i * 2 + j] == xd[(1 + j) * 8 + i].to(bfloat16),
          name & ": view gather " & $i & "," & $j
    # a negative-stride window: the readback stays inside the buffer
    let y = toTensor(xq[0 ..< 4], @[1, 4])
    let neg = view(y, @[4], @[-1], 3)
    var gneg = toSeq(neg, bfloat16)
    for i in 0 ..< 4:
      doAssert gneg[i] == xd[3 - i].to(bfloat16),
        name & ": negative-stride gather " & $i
    # a stride-0 expand: host readback only, element (r, c) reads the source element at (r, 0)
    let e = expand(narrow(x, 1, 0, 1), @[4, 8])
    var ge = toSeq(e, bfloat16)
    doAssert ge.len == 32, name & ": expand gather numel"
    for r in 0 ..< 4:
      for c in 0 ..< 8:
        doAssert ge[r * 8 + c] == xd[r * 8].to(bfloat16),
          name & ": expand gather " & $r & "," & $c

  # toSeq materializes an op output before reading it
  block:
    var yd: array[8, float32] = [2.0'f32, 2, 2, 2, 2, 2, 2, 2]
    var yq: array[8, bfloat16]
    for i in 0 ..< 8:
      yq[i] = yd[i].to(bfloat16)
    let x = toTensor(xq[0 ..< 8], @[2, 4])
    let y = toTensor(yq, @[2, 4])
    let s = x + y
    let gs = toSeq(s, bfloat16)
    for i in 0 ..< 8:
      doAssert gs[i] == (xd[i] + yd[i]).to(bfloat16), name & ": add gather " & $i

  # contracts raise AssertionDefect
  block:
    let z = zeros(@[2, 4], run)
    doAssertRaises AssertionDefect:
      discard zeros(@[], run)
    doAssertRaises AssertionDefect:
      discard zeros(@[2, 2, 2, 2, 2, 2], run)
    doAssertRaises AssertionDefect:
      discard zeros(@[-2, 4], run)
    doAssertRaises AssertionDefect:
      discard arange(0, 10, 0'f32, run)
    doAssertRaises AssertionDefect:
      discard linspace(0, 1, 1, run)
    doAssertRaises AssertionDefect:
      discard eye(-1, run)
    doAssertRaises AssertionDefect:
      discard toSeq(z, float32)

  echo "tensorcery factories (", name, "): all checks passed"

proc runJaggedAttentionTest*(run: static Runtime, name: string) =
  ## Jagged attention op end to end over the page pool slabs.
  ##
  ## Test that every scenario replays through the public face inside
  ## the pre-stated bf16 band (rtol 1e-2, abstol 1e-2).
  ##
  ## The reference side computes in f64 over the same bf16-rounded inputs.
  ##
  ## Staging: covered pool tokens random in [-1, 1]. Every other slot
  ## stays NaN garbage, so a 0*NaN in the P*V mma would poison the output.
  ##
  ## Checks:
  ##
  ## - five scenarios (decode, prefill, mixed shared-prefix, grid-stride multi-workgroup, MLA)
  ## - face contracts raise on shape, dtype, statics, head counts and metadata
  ## - fan-out memo: two add consumers of one attention output enqueue it once, 3 batch entries
  const
    H = 16
    D = 128
    PageSize = 256
    NumLayers = 2
    AttnTile = 32
    Bf16Rtol = 1e-2'f64
    Bf16Abstol = 1e-2'f64

  type JaggedScenario = object
    ## One replay's geometry and page table, host-authored.
    qLens: seq[int]
    kvLens: seq[int]
    pageIds: seq[int]
    kvPageBounds: seq[int]
    layer: int
    numPages: int
    nkv: int

  proc poolPos(pageId, layer, pos, kvHead: int, sc: JaggedScenario): int =
    (((pageId * NumLayers + layer) * PageSize + pos) * sc.nkv + kvHead) * D

  proc jaggedTables(sc: JaggedScenario):
      tuple[qOffsets, kvOffsets, kvIndices, kvSeqlens: seq[int]] =
    ## Cumulative q offsets and the JaggedKv metadata tables of one scenario.
    result.qOffsets = @[0]
    for s in 0 ..< sc.qLens.len:
      result.qOffsets.add result.qOffsets[^1] + sc.qLens[s]
    result.kvOffsets = @[0]
    result.kvSeqlens = @[]
    for s in 0 ..< sc.qLens.len:
      for b in sc.kvPageBounds[s] ..< sc.kvPageBounds[s + 1]:
        result.kvIndices.add sc.pageIds[b]
      result.kvOffsets.add result.kvOffsets[^1] +
        sc.kvPageBounds[s + 1] - sc.kvPageBounds[s]
      result.kvSeqlens.add sc.kvLens[s]

  proc jaggedStage(sc: JaggedScenario, seed: int):
      tuple[q, k, v: seq[bfloat16]] =
    ## Seeded inputs: q and the covered pool tokens random in [-1, 1],
    ## every other slot NaN garbage. The kernel bounds at the covered tokens.
    var rng = initRand(seed)
    template randBf16(): bfloat16 =
      rng.rand(-1.0 .. 1.0).to(bfloat16)
    template nanBf16(): bfloat16 =
      cast[bfloat16](uint16(0x7FC0))
    var qTotal = 0
    for s in 0 ..< sc.qLens.len:
      qTotal += sc.qLens[s]
    result.q = newSeq[bfloat16](qTotal * H * D)
    for i in 0 ..< result.q.len:
      result.q[i] = randBf16()
    let slabElems = sc.numPages * NumLayers * PageSize * sc.nkv * D
    result.k = newSeq[bfloat16](slabElems)
    result.v = newSeq[bfloat16](slabElems)
    for i in 0 ..< slabElems:
      result.k[i] = nanBf16()
      result.v[i] = nanBf16()
    for s in 0 ..< sc.qLens.len:
      for t in 0 ..< sc.kvLens[s]:
        let
          pageIdx = sc.kvPageBounds[s] + t div PageSize
          base = poolPos(sc.pageIds[pageIdx], sc.layer, t mod PageSize, 0, sc)
        for h in 0 ..< sc.nkv:
          for d in 0 ..< D:
            let idx = base + h * D + d
            result.k[idx] = randBf16()
            result.v[idx] = randBf16()

  proc jaggedHostReference(sc: JaggedScenario, qH, kH, vH: seq[bfloat16]):
      seq[float64] =
    ## f64 reference over the same bf16-rounded inputs:
    ## - prefill (qLen >= 2): query token j attends keys [0, kvLen - qLen + j],
    ##   flash-attn varlen semantics
    ## - decode (qLen == 1): attends keys [0, kvLen)
    ## - GQA: kvHead = head div (H div Nkv)
    ## - the softmax runs in the exp form, identical in exact arithmetic
    ##   to the kernel's exp2 shape up to one log2(e) factor
    let (qOffsets, _, _, _) = jaggedTables(sc)
    result = newSeq[float64](qOffsets[^1] * H * D)
    let scale = 1.0 / sqrt(float64(D))
    for s in 0 ..< sc.qLens.len:
      let
        q0 = qOffsets[s]
        qLen = sc.qLens[s]
        kvLen = sc.kvLens[s]
      for h in 0 ..< H:
        let kvHead = h div (H div sc.nkv)
        for j in 0 ..< qLen:
          let lastKey = (if qLen == 1: kvLen - 1 else: kvLen - qLen + j)
          var scores = newSeq[float64](lastKey + 1)
          var m = -1.0e300
          for k in 0 .. lastKey:
            var acc = 0.0'f64
            for d in 0 ..< D:
              let pageIdx = sc.kvPageBounds[s] + k div PageSize
              let pos = poolPos(sc.pageIds[pageIdx], sc.layer,
                                k mod PageSize, kvHead, sc)
              acc += float64(qH[((q0 + j) * H + h) * D + d].to(float32)) *
                     float64(kH[pos + d].to(float32))
            scores[k] = acc * scale
            m = max(m, scores[k])
          var l = 0.0'f64
          var ps = newSeq[float64](lastKey + 1)
          for k in 0 .. lastKey:
            ps[k] = exp(scores[k] - m)
            l += ps[k]
          for d in 0 ..< D:
            var acc = 0.0'f64
            for k in 0 .. lastKey:
              let pageIdx = sc.kvPageBounds[s] + k div PageSize
              let pos = poolPos(sc.pageIds[pageIdx], sc.layer,
                                k mod PageSize, kvHead, sc)
              acc += ps[k] * float64(vH[pos + d].to(float32))
            result[((q0 + j) * H + h) * D + d] = acc / l

  proc replay(sc: JaggedScenario, seed: int, label: string) =
    ## One scenario through the public face, per-element band judgment.
    let (qOffsets, kvOffsets, kvIndices, kvSeqlens) = jaggedTables(sc)
    let (qH, kH, vH) = jaggedStage(sc, seed)
    let refO = jaggedHostReference(sc, qH, kH, vH)
    let
      q = toTensor(qH, @[qOffsets[^1], H, D])
      poolK = toTensor(kH, @[sc.numPages, NumLayers, PageSize, sc.nkv, D])
      poolV = toTensor(vH, @[sc.numPages, NumLayers, PageSize, sc.nkv, D])
    let kvMeta = JaggedKvMeta(
      offsets: kvOffsets, indices: kvIndices, seqlens: kvSeqlens,
      batchSize: sc.qLens.len)
    let o = jagged_attention(q, poolK, poolV, qOffsets, kvMeta,
      1.0'f32 / sqrt(float32(D)), sc.layer, H, sc.nkv)
    doAssert o.shape().len == 3, name & ": " & label & " output rank"
    doAssert o.shape()[0] == qOffsets[^1], name & ": " & label & " output tokens"
    doAssert o.shape()[1] == H, name & ": " & label & " output heads"
    doAssert o.shape()[2] == D, name & ": " & label & " output dim"
    doAssert o.dtype() == kBf16, name & ": " & label & " output dtype"
    eval(o)
    let got = toSeq(o, bfloat16)
    var oBad = 0
    var worst = 0.0'f64
    for i in 0 ..< got.len:
      let g = float64(got[i].to(float32))
      let want = refO[i]
      let bar = Bf16Rtol * abs(want) + Bf16Abstol
      let diff = abs(g - want)
      # NaN counts as out-of-band only under the not-(<=) form.
      # A `>` comparison would let a NaN output pass judgment silently
      if not (diff <= bar):
        inc oBad
      elif diff / bar > worst:
        worst = diff / bar
    echo "  " & label & ": worstBar=" & $worst & " mismatches=" & $oBad & "/" & $got.len
    doAssert oBad == 0,
      name & ": " & label & ": " & $oBad & " of " & $got.len &
      " output elements outside the band"

  # decode: one segment, q_len 1, kv_len 300 over 2 pages, the last page
  # partially covered (300 = 256 + 44), one spare page pure garbage
  replay(JaggedScenario(
    qLens: @[1], kvLens: @[300], pageIds: @[0, 1], kvPageBounds: @[0, 2],
    layer: 0, numPages: 3, nkv: 8), 20261111, "decode_partial_last_page")

  # prefill: one segment, q_len 8 == kv_len 8, layer 1
  replay(JaggedScenario(
    qLens: @[8], kvLens: @[8], pageIds: @[0], kvPageBounds: @[0, 1],
    layer: 1, numPages: 2, nkv: 8), 20261112, "prefill_layer1")

  # mixed batch sharing a prefix page, GQA 16Q/8KV:
  # - segment 0: prefill q_len 8, kv_len 8, page [0]
  # - segment 1: prefill q_len 8, kv_len 264, pages [0, 2], shared page 0,
  #   second page partially covered (264 = 256 + 8)
  # - segment 2: decode q_len 1, kv_len 520, pages [0, 1, 3], shared page 0
  #   (520 = 2*256 + 8)
  replay(JaggedScenario(
    qLens: @[8, 8, 1], kvLens: @[8, 264, 520],
    pageIds: @[0, 0, 2, 0, 1, 3], kvPageBounds: @[0, 1, 3, 6],
    layer: 0, numPages: 4, nkv: 8), 20261113, "mixed_shared_prefix")

  # grid-stride / multi-workgroup: the shared-prefix batch with one
  # segment grown to a 40-token prefill. The op-computed launch grid
  # spans several query tiles per head, one softmax state per tile
  replay(JaggedScenario(
    qLens: @[40, 8, 1], kvLens: @[40, 264, 520],
    pageIds: @[0, 0, 2, 0, 1, 3], kvPageBounds: @[0, 1, 3, 6],
    layer: 0, numPages: 4, nkv: 8), 20261114, "grid_stride_multi_workgroup")

  # MLA shape: kv_heads 1, the compressed latent in the K slab.
  # Every head reads kv head 0. Prefill and decode share one batch
  replay(JaggedScenario(
    qLens: @[3, 1], kvLens: @[3, 300], pageIds: @[0, 1, 2],
    kvPageBounds: @[0, 1, 3],
    layer: 0, numPages: 3, nkv: 1), 20261115, "mla_kv_heads_1")

  # face contracts raise
  block:
    let engine = newEngine(run)
    var qShape = @[1, 16, 128]
    let qNode = engine.newTensorNode(kBf16, qShape.toSpan())
    var slabShape = @[1, 1, 256, 8, 128]
    let kNode = engine.newTensorNode(kBf16, slabShape.toSpan())
    let vNode = engine.newTensorNode(kBf16, slabShape.toSpan())
    var qH = newSeq[bfloat16](16 * 128)
    for i in 0 ..< qH.len:
      qH[i] = (float32((i * 13) mod 17 - 8) * 0.25'f32).to(bfloat16)
    engine.upload(qNode, qH)
    var slabH = newSeq[bfloat16](256 * 8 * 128)
    for i in 0 ..< slabH.len:
      slabH[i] = (float32((i * 7) mod 31 - 15) * 0.125'f32).to(bfloat16)
    engine.upload(kNode, slabH)
    engine.upload(vNode, slabH)
    let kvMeta = JaggedKvMeta(
      offsets: @[0, 1], indices: @[0], seqlens: @[1], batchSize: 1)
    proc attnValid(): auto =
      engine.tn_jagged_attention(qNode, kNode, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    discard attnValid()
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kNode, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8, windowSize = 64)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kNode, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 1, 16, 8)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kNode, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 15, 8)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kNode, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 3)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kNode, vNode, @[0], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kNode, vNode, @[0, 2], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    let kvBad = JaggedKvMeta(
      offsets: @[0, 1], indices: @[5], seqlens: @[1], batchSize: 1)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kNode, vNode, @[0, 1], kvBad,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    let kvEmpty = JaggedKvMeta(
      offsets: @[0, 1], indices: @[0], seqlens: @[0], batchSize: 1)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kNode, vNode, @[0, 1], kvEmpty,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    # a float32 q and a rank-2 q refuse at the face
    var f32Shape = @[1, 16, 128]
    let qF32 = engine.newTensorNode(kFp32, f32Shape.toSpan())
    var f32H = newSeq[float32](16 * 128)
    for i in 0 ..< f32H.len:
      f32H[i] = float32(i mod 7) * 0.25'f32
    engine.upload(qF32, f32H)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qF32, kNode, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    var r2Shape = @[16, 128]
    let qR2 = engine.newTensorNode(kBf16, r2Shape.toSpan())
    var r2H = newSeq[bfloat16](16 * 128)
    for i in 0 ..< r2H.len:
      r2H[i] = (float32(i mod 5) * 0.25'f32).to(bfloat16)
    engine.upload(qR2, r2H)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qR2, kNode, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    # a rank-4 slab and mismatched slab shapes refuse at the face
    var r4Shape = @[1, 256, 8, 128]
    let kR4 = engine.newTensorNode(kBf16, r4Shape.toSpan())
    var r4H = newSeq[bfloat16](256 * 8 * 128)
    for i in 0 ..< r4H.len:
      r4H[i] = (float32(i mod 11) * 0.125'f32).to(bfloat16)
    engine.upload(kR4, r4H)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kR4, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    var smallShape = @[1, 1, 128, 8, 128]
    let kSmall = engine.newTensorNode(kBf16, smallShape.toSpan())
    var smallH = newSeq[bfloat16](128 * 8 * 128)
    for i in 0 ..< smallH.len:
      smallH[i] = (float32(i mod 9) * 0.125'f32).to(bfloat16)
    engine.upload(kSmall, smallH)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kNode, kSmall, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    # page-size and head-dim statics refuse other geometries
    var pgShape = @[1, 1, 128, 8, 128]
    let kPg = engine.newTensorNode(kBf16, pgShape.toSpan())
    engine.upload(kPg, smallH)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qNode, kPg, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    var d64QShape = @[1, 16, 64]
    let qD64 = engine.newTensorNode(kBf16, d64QShape.toSpan())
    var d64H = newSeq[bfloat16](16 * 64)
    for i in 0 ..< d64H.len:
      d64H[i] = (float32(i mod 5) * 0.25'f32).to(bfloat16)
    engine.upload(qD64, d64H)
    doAssertRaises AssertionDefect:
      discard engine.tn_jagged_attention(qD64, kNode, vNode, @[0, 1], kvMeta,
        1.0'f32 / sqrt(64.0'f32), 0, 16, 8)

  # fan-out: two add consumers of one attention output see one
  # attention launch enqueued exactly once
  block:
    let engine = newEngine(run)
    var qShape = @[1, 16, 128]
    let qNode = engine.newTensorNode(kBf16, qShape.toSpan())
    var qH = newSeq[bfloat16](16 * 128)
    for i in 0 ..< qH.len:
      qH[i] = (float32((i * 13) mod 17 - 8) * 0.25'f32).to(bfloat16)
    engine.upload(qNode, qH)
    var slabShape = @[2, 2, 256, 8, 128]
    let kNode = engine.newTensorNode(kBf16, slabShape.toSpan())
    let vNode = engine.newTensorNode(kBf16, slabShape.toSpan())
    var kH = newSeq[bfloat16](2 * 2 * 256 * 8 * 128)
    var vH = newSeq[bfloat16](2 * 2 * 256 * 8 * 128)
    for i in 0 ..< kH.len:
      kH[i] = (float32((i * 7) mod 31 - 15) * 0.125'f32).to(bfloat16)
      vH[i] = (float32((i * 11) mod 29 - 14) * 0.125'f32).to(bfloat16)
    engine.upload(kNode, kH)
    engine.upload(vNode, vH)
    let kvMeta = JaggedKvMeta(
      offsets: @[0, 2], indices: @[0, 1], seqlens: @[300], batchSize: 1)
    let p = engine.tn_jagged_attention(qNode, kNode, vNode, @[0, 1], kvMeta,
      1.0'f32 / sqrt(128.0'f32), 0, 16, 8)
    var cx = engine.newTensorNode(kBf16, qShape.toSpan())
    var cy = engine.newTensorNode(kBf16, qShape.toSpan())
    var cH = newSeq[bfloat16](16 * 128)
    for i in 0 ..< cH.len:
      cH[i] = (float32((i * 23) mod 13 - 6) * 0.25'f32).to(bfloat16)
    engine.upload(cx, cH)
    engine.upload(cy, cH)
    let z1 = engine.tn_add(p, cx)
    let z2 = engine.tn_add(p, cy)
    var batch = engine.recordBatch()
    var visited: VisitedNodes
    var nz1 = z1
    var nz2 = z2
    engine.evalNode(batch, nz1, visited)
    engine.evalNode(batch, nz2, visited)
    doAssert batch.entries.len == 3,
      name & ": attention fan-out enqueued " & $batch.entries.len & ", expected 3"
    batch.submit()
    # numerics: the reference output plus the addend, one bf16 rounding
    var sc = JaggedScenario(
      qLens: @[1], kvLens: @[300], pageIds: @[0, 1], kvPageBounds: @[0, 2],
      layer: 0, numPages: 2, nkv: 8)
    let refO = jaggedHostReference(sc, qH, kH, vH)
    var gz1, gz2: seq[bfloat16]
    gz1 = newSeq[bfloat16](16 * 128)
    gz2 = newSeq[bfloat16](16 * 128)
    engine.download(gz1, nz1)
    engine.download(gz2, nz2)
    for i in 0 ..< refO.len:
      # band judgment: the attention output stays inside the pre-stated
      # bf16 band around the f64 reference. The add on top is exact.
      let want = refO[i] + float64(cH[i].to(float32))
      let bar = Bf16Rtol * abs(want) + Bf16Abstol
      doAssert abs(float64(gz1[i].to(float32)) - want) <= bar,
        name & ": fan-out z1 " & $i & ": got " & $gz1[i].to(float32) &
        ", want " & $want
      doAssert abs(float64(gz2[i].to(float32)) - want) <= bar,
        name & ": fan-out z2 " & $i & ": got " & $gz2[i].to(float32) &
        ", want " & $want

  echo "tensorcery jagged attention (", name, "): all checks passed"
