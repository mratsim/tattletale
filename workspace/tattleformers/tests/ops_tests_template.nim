# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[math, options]

{.experimental: "callOperator".}

import workspace/crucible
import workspace/ceramic
import workspace/tensorcery/src/tensor_datatypes
import workspace/tattleformers/src/ops_2i1o/add
import workspace/tattleformers/src/ops_2i1o/matmul
import workspace/ceramic/src/tile_algebra
import workspace/tattleformers/src/ops_Ni1o/embed_gather
import workspace/tattleformers/src/ops_1i1o/rmsnorm
import workspace/tattleformers/src/layers/embedding
import workspace/tattleformers/src/layers/linear
import workspace/tattleformers/src/layers/norm

export tensor_datatypes

func ulpKey16*(x: bfloat16): int32 =
  ## Ordered key of a bfloat16 word.
  ##
  ## Args:
  ## - x: a bfloat16 value
  ##
  ## Returns: sign-magnitude bits mapped so keys increase with the value,
  ## -0 and +0 collide. Key distance counts bfloat16 ulps.
  let u = cast[uint16](x)
  if (u and 0x8000'u16) != 0'u16:
    -int32(u and 0x7FFF'u16)
  else:
    int32(u)

proc checkBf16Ulp*(got: openArray[bfloat16], refVals: openArray[float32], ulpBand: int, msg: string) =
  ## Asserts every element within `ulpBand` bfloat16 ulps of its reference.
  ##
  ## Args:
  ## - got: device readback
  ## - refVals: sequential f32 host reference
  ## - ulpBand: per-stage drift allowance in ulps, 4 for a kReduction-class
  ##   check at depth 1
  doAssert got.len == refVals.len, msg & ": length mismatch"
  var worst = 0
  var worstAt = 0
  for i in 0 ..< got.len:
    let want = refVals[i].to(bfloat16)
    let dist = abs(ulpKey16(got[i]) - ulpKey16(want))
    if dist > worst:
      worst = dist
      worstAt = i
    doAssert dist <= ulpBand,
      msg & " elem " & $i & ": got " & $got[i].to(float32) &
      ", reference " & $refVals[i] & ", distance " & $dist & " bf16 ulps"
  echo msg, ": worst ", worst, " bf16 ulps at elem ", worstAt

proc runTest*[Mem](any: AnyEngine, name: string) =
  # Mem: explicit type argument, erased engine.
  # Why not a typed engine parameter:
  # - the converter instantiates twice, here and in the driver
  # - both emit the AnyEngine vtable under the driver's call-site name
  # - the linker rejects the duplicate symbol
  # weight entries with a distinct value per (entry, col), bf16 end to end
  const Vocab = 7
  const Hidden = 4
  var wHost = newSeq[bfloat16](Vocab * Hidden)
  for r in 0 ..< Vocab:
    for c in 0 ..< Hidden:
      wHost[r * Hidden + c] = (float32(r * 10 + c) + 0.5).to(bfloat16)
  let w = newTensor(any, @[Vocab, Hidden], kBf16)
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
    type MemW = Mem
    let wImpl = w.node.getImpl().to(TensorImpl[MemW, Layout[(int, int), (int, int)]])
    let rImpl = r.node.getImpl().to(TensorImpl[MemW, Layout[(int, int), (int, int)]])
    doAssert rImpl.data == wImpl.data, name & ": reshape must alias the source cell"
    var gotR = newSeq[bfloat16](Vocab * Hidden)
    copyToHostFromDevice(gotR, r)
    for i in 0 ..< Vocab * Hidden:
      doAssert gotR[i].to(float32) == wHost[i].to(float32), name & ": reshape " & $i
    doAssertRaises ValueError:
      discard w.reshape([Vocab * Hidden + 1])

  # ── matmul op and Linear layer ────────────────────────────────────────
  # operands with a distinct value per cell, judged against a sequential
  # f32 host reference within the kReduction-class ulp band
  block:
    const
      M = 3
      K = 8
      N = 5
    var aHost = newSeq[bfloat16](M * K)
    var bHost = newSeq[bfloat16](K * N)
    for i in 0 ..< M * K:
      aHost[i] = (float32(i) * 0.25'f32 - 1.0'f32).to(bfloat16)
    for i in 0 ..< K * N:
      bHost[i] = (float32(i mod 7) * 0.5'f32 - 1.5'f32).to(bfloat16)
    var refO = newSeq[float32](M * N)
    for m in 0 ..< M:
      for n in 0 ..< N:
        var acc = 0.0'f32
        for k in 0 ..< K:
          acc += aHost[m * K + k].to(float32) * bHost[k * N + n].to(float32)
        refO[m * N + n] = acc

    let a = newTensor(any, @[M, K], kBf16)
    a.copyToDeviceFromHost(aHost)
    let b = newTensor(any, @[K, N], kBf16)
    b.copyToDeviceFromHost(bHost)
    let c = Tensor(engine: a.engine, node: tn_matmul(a.engine, a.node, b.node))
    doAssert c.node.getImpl().shape()[0] == M
    doAssert c.node.getImpl().shape()[1] == N
    doAssert c.node.getImpl().dtype() == kBf16
    var cRoots = @[c.node]
    c.engine.eval(cRoots.toSpan())
    var gotC = newSeq[bfloat16](M * N)
    copyToHostFromDevice(gotC, c)
    gotC.checkBf16Ulp(refO, 4, name & ": matmul")

    # face contract: a K mismatch and a dtype mismatch raise
    let bShort = newTensor(any, @[K + 1, N], kBf16)
    doAssertRaises ValueError:
      discard tn_matmul(any, a.node, bShort.node)
    let bF32 = newTensor(any, @[K, N], kFp32)
    doAssertRaises ValueError:
      discard tn_matmul(any, a.node, bF32.node)

    # matmul and add compose in one graph through the open dispatch
    let bias = newTensor(any, @[M, N], kBf16)
    var biasHost = newSeq[bfloat16](M * N)
    for i in 0 ..< M * N:
      biasHost[i] = (float32(i mod 5) * 0.125'f32).to(bfloat16)
    bias.copyToDeviceFromHost(biasHost)
    let summed = c + bias
    var sRoots = @[summed.node]
    summed.engine.eval(sRoots.toSpan())
    var gotS = newSeq[bfloat16](M * N)
    copyToHostFromDevice(gotS, summed)
    for i in 0 ..< M * N:
      let want = (refO[i] + biasHost[i].to(float32)).to(bfloat16).to(float32)
      doAssert gotS[i].to(float32) == want, name & ": matmul add " & $i

  # Linear layer port: y = x @ weight^T over the erased face
  block:
    const
      OutF = 5
      InF = 8
    var wHost = newSeq[bfloat16](OutF * InF)
    for i in 0 ..< OutF * InF:
      wHost[i] = (float32(i) * 0.125'f32 - 2.0'f32).to(bfloat16)
    let w = newTensor(any, @[OutF, InF], kBf16)
    w.copyToDeviceFromHost(wHost)

    let layer = Linear.init(w)
    doAssert layer.in_features == InF
    doAssert layer.out_features == OutF

    # GEMM weight operand: a transposed view of the stored weight,
    # same buffer cell, no allocation
    block:
      let wt = w.transpose()
      doAssert wt.node.getImpl().shape()[0] == InF
      doAssert wt.node.getImpl().shape()[1] == OutF
      doAssert wt.node.getImpl().strides()[0] == 1
      doAssert wt.node.getImpl().strides()[1] == InF
      type MemW = Mem
      let wImpl = w.node.getImpl().to(TensorImpl[MemW, Layout[(int, int), (int, int)]])
      let wtImpl = wt.node.getImpl().to(TensorImpl[MemW, Layout[(int, int), (int, int)]])
      doAssert wtImpl.data == wImpl.data, name & ": the transposed weight view must alias the weight cell"

    # (tokens, in) -> (tokens, out)
    block:
      var xHost = newSeq[bfloat16](3 * InF)
      for i in 0 ..< 3 * InF:
        xHost[i] = (float32(i) * 0.5'f32 - 6.0'f32).to(bfloat16)
      let x = newTensor(any, @[3, InF], kBf16)
      x.copyToDeviceFromHost(xHost)
      let y = layer.forward(x)
      doAssert y.node.getImpl().shape()[0] == 3
      doAssert y.node.getImpl().shape()[1] == OutF
      var yRoots = @[y.node]
      y.engine.eval(yRoots.toSpan())
      var gotY = newSeq[bfloat16](3 * OutF)
      copyToHostFromDevice(gotY, y)
      for m in 0 ..< 3:
        for n in 0 ..< OutF:
          var acc = 0.0'f32
          for k in 0 ..< InF:
            acc += xHost[m * InF + k].to(float32) * wHost[n * InF + k].to(float32)
          let want = acc.to(bfloat16).to(float32)
          doAssert abs(gotY[m * OutF + n].to(float32) - want) <= 4.0 * max(abs(want), 1.0'f32) * 0.008,
            name & ": linear (" & $m & "," & $n & "): got " &
            $gotY[m * OutF + n].to(float32) & ", want " & $want

    # (batch, seq, in) -> (batch, seq, out), reference (..., in_features) contract
    block:
      var xHost = newSeq[bfloat16](2 * 3 * InF)
      for i in 0 ..< 2 * 3 * InF:
        xHost[i] = (float32(i) * 0.25'f32 - 4.0'f32).to(bfloat16)
      let x = newTensor(any, @[2, 3, InF], kBf16)
      x.copyToDeviceFromHost(xHost)
      let y = layer.forward(x)
      doAssert y.node.getImpl().shape()[0] == 2
      doAssert y.node.getImpl().shape()[1] == 3
      doAssert y.node.getImpl().shape()[2] == OutF
      var yRoots = @[y.node]
      y.engine.eval(yRoots.toSpan())
      var gotY = newSeq[bfloat16](2 * 3 * OutF)
      copyToHostFromDevice(gotY, y)
      for t in 0 ..< 2 * 3:
        for n in 0 ..< OutF:
          var acc = 0.0'f32
          for k in 0 ..< InF:
            acc += xHost[t * InF + k].to(float32) * wHost[n * InF + k].to(float32)
          let want = acc.to(bfloat16).to(float32)
          doAssert abs(gotY[t * OutF + n].to(float32) - want) <= 4.0 * max(abs(want), 1.0'f32) * 0.008,
            name & ": linear 3d (" & $t & "," & $n & "): got " &
            $gotY[t * OutF + n].to(float32) & ", want " & $want

    # bias branch raises, Qwen3 runs bias-free
    block:
      let b = newTensor(any, @[OutF], kBf16)
      let biased = Linear.init(w, some(b))
      let x = newTensor(any, @[2, InF], kBf16)
      doAssertRaises ValueError:
        discard biased.forward(x)


  # ── rmsnorm op and RmsNorm layer ─────────────────────────────────────
  # bf16 x (32, 1024), eps 1e-6, bf16 weight: judged against an f64 host
  # reference at the kElementwise-class band (2 bf16 ulps)
  block:
    const
      Rows = 32
      N = 1024
    var xHost = newSeq[bfloat16](Rows * N)
    var wHost = newSeq[bfloat16](N)
    for i in 0 ..< Rows * N:
      xHost[i] = (float32((i * 37) mod 25 - 12) * 0.5'f32).to(bfloat16)
    for i in 0 ..< N:
      wHost[i] = (float32(i mod 11) * 0.1'f32 + 0.5'f32).to(bfloat16)

    # f64 host reference, one axis strip at a time
    var refO = newSeq[float32](Rows * N)
    for r in 0 ..< Rows:
      var ss = 0.0'f64
      for i in 0 ..< N:
        let x = float64(xHost[r * N + i].to(float32))
        ss += x * x
      let scale = 1.0 / sqrt(ss / float64(N) + 1e-6)
      for i in 0 ..< N:
        let x = float64(xHost[r * N + i].to(float32))
        let w = float64(wHost[i].to(float32))
        refO[r * N + i] = float32(x * scale * w)

    let x = newTensor(any, @[Rows, N], kBf16)
    x.copyToDeviceFromHost(xHost)
    let w = newTensor(any, @[N], kBf16)
    w.copyToDeviceFromHost(wHost)

    let o = Tensor(engine: x.engine,
      node: tn_rmsnorm(x.engine, x.node, w.node, 1e-6'f32))
    doAssert o.node.getImpl().shape().len == 2
    doAssert o.node.getImpl().shape()[0] == Rows
    doAssert o.node.getImpl().shape()[1] == N
    doAssert o.node.getImpl().dtype() == kBf16
    var oRoots = @[o.node]
    o.engine.eval(oRoots.toSpan())
    var gotO = newSeq[bfloat16](Rows * N)
    copyToHostFromDevice(gotO, o)
    gotO.checkBf16Ulp(refO, 2, name & ": rmsnorm (32, 1024)")

    # rank-1 and rank-3 sources walk at their own rank, the last axis is normalized
    block:
      let x1 = newTensor(any, @[N], kBf16)
      x1.copyToDeviceFromHost(xHost[0 ..< N])
      let o1 = Tensor(engine: x1.engine,
        node: tn_rmsnorm(x1.engine, x1.node, w.node, 1e-6'f32))
      doAssert o1.node.getImpl().shape().len == 1
      var o1Roots = @[o1.node]
      o1.engine.eval(o1Roots.toSpan())
      var got1 = newSeq[bfloat16](N)
      copyToHostFromDevice(got1, o1)
      got1.checkBf16Ulp(refO[0 ..< N], 2, name & ": rmsnorm rank 1")

      let x3 = newTensor(any, @[2, 3, N], kBf16)
      var x3Host = newSeq[bfloat16](2 * 3 * N)
      for i in 0 ..< 2 * 3 * N:
        x3Host[i] = xHost[i mod (Rows * N)]
      x3.copyToDeviceFromHost(x3Host)
      let o3 = Tensor(engine: x3.engine,
        node: tn_rmsnorm(x3.engine, x3.node, w.node, 1e-6'f32))
      doAssert o3.node.getImpl().shape().len == 3
      var o3Roots = @[o3.node]
      o3.engine.eval(o3Roots.toSpan())
      var got3 = newSeq[bfloat16](2 * 3 * N)
      copyToHostFromDevice(got3, o3)
      got3.checkBf16Ulp(refO[0 ..< 2 * 3 * N], 2, name & ": rmsnorm rank 3")

    # face contract: weight size, weight rank and dtype mismatches raise
    block:
      let wShort = newTensor(any, @[N - 1], kBf16)
      doAssertRaises ValueError:
        discard tn_rmsnorm(any, x.node, wShort.node, 1e-6'f32)
      let w2d = newTensor(any, @[1, N], kBf16)
      doAssertRaises ValueError:
        discard tn_rmsnorm(any, x.node, w2d.node, 1e-6'f32)
      let wF32 = newTensor(any, @[N], kFp32)
      doAssertRaises ValueError:
        discard tn_rmsnorm(any, x.node, wF32.node, 1e-6'f32)

    # RmsNorm layer port: RmsNorm(w)(x) through the erased face
    block:
      let rms = RmsNorm.init(w)
      doAssert rms.hidden_size == N
      let y = rms.forward(x)
      doAssert y.node.getImpl().shape()[0] == Rows
      doAssert y.node.getImpl().shape()[1] == N
      var yRoots = @[y.node]
      y.engine.eval(yRoots.toSpan())
      var gotY = newSeq[bfloat16](Rows * N)
      copyToHostFromDevice(gotY, y)
      gotY.checkBf16Ulp(refO, 2, name & ": RmsNorm layer (32, 1024)")

      # fused residual addition + RMSNorm: (normed, new_residual)
      block:
        var rHost = newSeq[bfloat16](Rows * N)
        for i in 0 ..< Rows * N:
          rHost[i] = (float32((i * 53) mod 17 - 8) * 0.25'f32).to(bfloat16)
        let residual = newTensor(any, @[Rows, N], kBf16)
        residual.copyToDeviceFromHost(rHost)
        let (normed, newResidual) = rms.forward_with_residual(x, residual)
        doAssert normed.node.getImpl().shape()[0] == Rows
        doAssert newResidual.node.getImpl().shape()[0] == Rows
        var nRoots = @[normed.node, newResidual.node]
        normed.engine.eval(nRoots.toSpan())
        var gotRes = newSeq[bfloat16](Rows * N)
        copyToHostFromDevice(gotRes, newResidual)
        for i in 0 ..< Rows * N:
          let want = (xHost[i].to(float32) + rHost[i].to(float32)).to(bfloat16).to(float32)
          doAssert gotRes[i].to(float32) == want,
            name & ": rmsnorm residual " & $i & ": got " & $gotRes[i].to(float32) &
            ", want " & $want
        # normed output: the same rmsnorm over x + residual
        var gotNormed = newSeq[bfloat16](Rows * N)
        copyToHostFromDevice(gotNormed, normed)
        var refRes = newSeq[float32](Rows * N)
        for r in 0 ..< Rows:
          var ss = 0.0'f64
          for i in 0 ..< N:
            let v = float64((xHost[r * N + i].to(float32) + rHost[r * N + i].to(float32)).to(bfloat16).to(float32))
            ss += v * v
          let scale = 1.0 / sqrt(ss / float64(N) + 1e-6)
          for i in 0 ..< N:
            let v = float64((xHost[r * N + i].to(float32) + rHost[r * N + i].to(float32)).to(bfloat16).to(float32))
            let w = float64(wHost[i].to(float32))
            refRes[r * N + i] = float32(v * scale * w)
        gotNormed.checkBf16Ulp(refRes, 2, name & ": RmsNorm residual fused")

  echo "tattleformers ops (", name, "): all checks passed"
