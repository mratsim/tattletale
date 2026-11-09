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
import workspace/tattleformers/src/ops_2i1o/silu_mul
import workspace/ceramic/src/tile_algebra
import workspace/tattleformers/src/ops_Ni1o/embed_gather
import workspace/tattleformers/src/ops_1i1o/rmsnorm
import workspace/tattleformers/src/layers/embedding
import workspace/tattleformers/src/layers/linear
import workspace/tattleformers/src/layers/norm
import workspace/tattleformers/src/layers/ffn
import workspace/tattleformers/src/ops_Ni1o/rope
import workspace/tattleformers/src/layers/rope

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


  # ── silu·mul op: the gated FFN activation ─────────────────────────────
  # bf16 (64, 3072) at the Qwen3 FFN intermediate width: an f64 host
  # reference judges it at the kElementwise-class band (2 bf16 ulps)
  block:
    const
      Tokens = 64
      N = 3072
    var gHost = newSeq[bfloat16](Tokens * N)
    var vHost = newSeq[bfloat16](Tokens * N)
    for i in 0 ..< Tokens * N:
      gHost[i] = (float32((i * 37) mod 41 - 20) * 0.5'f32).to(bfloat16)
      vHost[i] = (float32((i * 29) mod 31 - 15) * 0.25'f32).to(bfloat16)
    # f64 host reference of the reference silu path
    # (F.silu(gate_out) * up_out): silu(g) = g / (1 + exp(-g))
    var refO = newSeq[float32](Tokens * N)
    for i in 0 ..< Tokens * N:
      let g = float64(gHost[i].to(float32))
      let v = float64(vHost[i].to(float32))
      refO[i] = float32(g / (1.0 + exp(-g)) * v)

    let gate = newTensor(any, @[Tokens, N], kBf16)
    gate.copyToDeviceFromHost(gHost)
    let value = newTensor(any, @[Tokens, N], kBf16)
    value.copyToDeviceFromHost(vHost)
    let o = Tensor(engine: gate.engine,
      node: tn_silu_mul(gate.engine, gate.node, value.node))
    doAssert o.node.getImpl().shape()[0] == Tokens
    doAssert o.node.getImpl().shape()[1] == N
    doAssert o.node.getImpl().dtype() == kBf16
    var oRoots = @[o.node]
    o.engine.eval(oRoots.toSpan())
    var gotO = newSeq[bfloat16](Tokens * N)
    copyToHostFromDevice(gotO, o)
    gotO.checkBf16Ulp(refO, 2, name & ": silu·mul (64, 3072)")

    # boundary set at the float32 limits: silu(g) exact at the positive
    # bfloat16 max, -0 at the negative one, subnormal silu underflows
    # to -0 against an f64 reference that rounds to -0 too
    block:
      const
        BTokens = 2
        BN = 6
      let gBig = [(3.3e38'f32).to(bfloat16), (-3.3e38'f32).to(bfloat16),
                  (-100.0'f32).to(bfloat16), (88.0'f32).to(bfloat16),
                  (-88.0'f32).to(bfloat16), (0.0'f32).to(bfloat16)]
      let vBig = [(1.0'f32).to(bfloat16), (1.0'f32).to(bfloat16),
                  (1.0'f32).to(bfloat16), (2.0'f32).to(bfloat16),
                  (2.0'f32).to(bfloat16), (3.0'f32).to(bfloat16)]
      var gbHost = newSeq[bfloat16](BTokens * BN)
      var vbHost = newSeq[bfloat16](BTokens * BN)
      var refB = newSeq[float32](BTokens * BN)
      for t in 0 ..< BTokens:
        for c in 0 ..< BN:
          let g = gBig[c]
          let v = vBig[c]
          gbHost[t * BN + c] = g
          vbHost[t * BN + c] = v
          let g64 = float64(g.to(float32))
          let v64 = float64(v.to(float32))
          refB[t * BN + c] = float32(g64 / (1.0 + exp(-g64)) * v64)
      let gateB = newTensor(any, @[BTokens, BN], kBf16)
      gateB.copyToDeviceFromHost(gbHost)
      let valueB = newTensor(any, @[BTokens, BN], kBf16)
      valueB.copyToDeviceFromHost(vbHost)
      let oB = Tensor(engine: gateB.engine,
        node: tn_silu_mul(gateB.engine, gateB.node, valueB.node))
      var oBRoots = @[oB.node]
      oB.engine.eval(oBRoots.toSpan())
      var gotB = newSeq[bfloat16](BTokens * BN)
      copyToHostFromDevice(gotB, oB)
      gotB.checkBf16Ulp(refB, 2, name & ": silu·mul boundary")

    # face contract: shape, rank and dtype mismatches raise
    block:
      let vShort = newTensor(any, @[Tokens, N - 1], kBf16)
      doAssertRaises ValueError:
        discard tn_silu_mul(any, gate.node, vShort.node)
      let v3d = newTensor(any, @[2, 3, N], kBf16)
      doAssertRaises ValueError:
        discard tn_silu_mul(any, gate.node, v3d.node)
      let vF32 = newTensor(any, @[Tokens, N], kFp32)
      doAssertRaises ValueError:
        discard tn_silu_mul(any, gate.node, vF32.node)

  # ── GatedDenseFFN chained smoke: rmsnorm → gate/up proj → silu·mul →
  # down proj vs reference order at the Qwen3 dense geometry,
  # hidden 1024, intermediate 3072.
  # Intermediates read back: each stage's reference consumes the device's own previous-stage values
  block:
    const
      CTokens = 4
      CHidden = 1024
      CInter = 3072
    var xHost = newSeq[bfloat16](CTokens * CHidden)
    for i in 0 ..< xHost.len:
      xHost[i] = (float32((i * 41) mod 23 - 11) * 0.125'f32).to(bfloat16)
    var wnHost = newSeq[bfloat16](CHidden)
    for i in 0 ..< CHidden:
      wnHost[i] = (float32(i mod 11) * 0.1'f32 + 0.5'f32).to(bfloat16)
    var wgHost = newSeq[bfloat16](CInter * CHidden)
    var wuHost = newSeq[bfloat16](CInter * CHidden)
    for i in 0 ..< CInter * CHidden:
      wgHost[i] = (float32((i * 17) mod 29 - 14) * 0.0625'f32).to(bfloat16)
      wuHost[i] = (float32((i * 13) mod 31 - 15) * 0.0625'f32).to(bfloat16)
    var wdHost = newSeq[bfloat16](CHidden * CInter)
    for i in 0 ..< CHidden * CInter:
      wdHost[i] = (float32((i * 19) mod 27 - 13) * 0.0625'f32).to(bfloat16)

    let x = newTensor(any, @[CTokens, CHidden], kBf16)
    x.copyToDeviceFromHost(xHost)
    let wn = newTensor(any, @[CHidden], kBf16)
    wn.copyToDeviceFromHost(wnHost)
    let wg = newTensor(any, @[CInter, CHidden], kBf16)
    wg.copyToDeviceFromHost(wgHost)
    let wu = newTensor(any, @[CInter, CHidden], kBf16)
    wu.copyToDeviceFromHost(wuHost)
    let wd = newTensor(any, @[CHidden, CInter], kBf16)
    wd.copyToDeviceFromHost(wdHost)

    # reference order: norm, two projections off the normed states, silu·mul, down_proj
    let normed = RmsNorm.init(wn).forward(x)
    let gateProj = Linear.init(wg).forward(normed)
    let upProj = Linear.init(wu).forward(normed)
    let act = Tensor(engine: gateProj.engine,
      node: tn_silu_mul(gateProj.engine, gateProj.node, upProj.node))
    let downOut = Linear.init(wd).forward(act)
    doAssert gateProj.node.getImpl().shape()[0] == CTokens
    doAssert gateProj.node.getImpl().shape()[1] == CInter
    doAssert act.node.getImpl().shape()[1] == CInter
    doAssert downOut.node.getImpl().shape()[0] == CTokens
    doAssert downOut.node.getImpl().shape()[1] == CHidden
    var roots = @[downOut.node]
    downOut.engine.eval(roots.toSpan())

    # norm reference over the device's own input
    var gotNormed = newSeq[bfloat16](CTokens * CHidden)
    copyToHostFromDevice(gotNormed, normed)
    var refNormed = newSeq[float32](CTokens * CHidden)
    for t in 0 ..< CTokens:
      var ss = 0.0'f64
      for i in 0 ..< CHidden:
        let v = float64(xHost[t * CHidden + i].to(float32))
        ss += v * v
      let scale = 1.0 / sqrt(ss / float64(CHidden) + 1e-6)
      for i in 0 ..< CHidden:
        let v = float64(xHost[t * CHidden + i].to(float32))
        let w = float64(wnHost[i].to(float32))
        refNormed[t * CHidden + i] = float32(v * scale * w)
    gotNormed.checkBf16Ulp(refNormed, 2, name & ": FFN chain rmsnorm")

    # projection references over the device's own normed values
    var gotGate = newSeq[bfloat16](CTokens * CInter)
    var gotUp = newSeq[bfloat16](CTokens * CInter)
    copyToHostFromDevice(gotGate, gateProj)
    copyToHostFromDevice(gotUp, upProj)
    var refGate = newSeq[float32](CTokens * CInter)
    var refUp = newSeq[float32](CTokens * CInter)
    for t in 0 ..< CTokens:
      for n in 0 ..< CInter:
        var accG = 0.0'f32
        var accU = 0.0'f32
        for k in 0 ..< CHidden:
          let nv = gotNormed[t * CHidden + k].to(float32)
          accG += nv * wgHost[n * CHidden + k].to(float32)
          accU += nv * wuHost[n * CHidden + k].to(float32)
        refGate[t * CInter + n] = accG
        refUp[t * CInter + n] = accU
    gotGate.checkBf16Ulp(refGate, 4, name & ": FFN chain gate_proj")
    gotUp.checkBf16Ulp(refUp, 4, name & ": FFN chain up_proj")

    # activation reference over the device's own projection values
    var gotAct = newSeq[bfloat16](CTokens * CInter)
    copyToHostFromDevice(gotAct, act)
    var refAct = newSeq[float32](CTokens * CInter)
    for i in 0 ..< CTokens * CInter:
      let g = float64(gotGate[i].to(float32))
      let v = float64(gotUp[i].to(float32))
      refAct[i] = float32(g / (1.0 + exp(-g)) * v)
    gotAct.checkBf16Ulp(refAct, 2, name & ": FFN chain silu·mul")

    # down projection reference over the device's own activation
    var gotDown = newSeq[bfloat16](CTokens * CHidden)
    copyToHostFromDevice(gotDown, downOut)
    var refDown = newSeq[float32](CTokens * CHidden)
    for t in 0 ..< CTokens:
      for n in 0 ..< CHidden:
        var acc = 0.0'f32
        for k in 0 ..< CInter:
          acc += gotAct[t * CInter + k].to(float32) * wdHost[n * CInter + k].to(float32)
        refDown[t * CHidden + n] = acc
    gotDown.checkBf16Ulp(refDown, 4, name & ": FFN chain down_proj")

    # GatedDenseFFN layer port through the erased face: one forward over
    # normed states reproduces the manual chain bit for bit. Both graphs enqueue the same kernels over the same input buffers
    block:
      let ffn = GatedDenseFFN.init(Linear.init(wg), Linear.init(wu), Linear.init(wd))
      let y = ffn.forward(normed)
      doAssert y.node.getImpl().shape()[0] == CTokens
      doAssert y.node.getImpl().shape()[1] == CHidden
      doAssert y.node.getImpl().dtype() == kBf16
      var yRoots = @[y.node]
      y.engine.eval(yRoots.toSpan())
      var gotY = newSeq[bfloat16](CTokens * CHidden)
      copyToHostFromDevice(gotY, y)
      for i in 0 ..< CTokens * CHidden:
        doAssert gotY[i].to(float32) == gotDown[i].to(float32),
          name & ": FFN layer " & $i & ": got " & $gotY[i].to(float32) &
          ", manual chain " & $gotDown[i].to(float32)

    # kGeluTanh branch unported: the reference host op binds libtorch
    block:
      let ffnGelu = GatedDenseFFN.init(Linear.init(wg), Linear.init(wu),
        Linear.init(wd), kGeluTanh)
      doAssertRaises ValueError:
        discard ffnGelu.forward(x)

  # ── rope op, RotaryPositionEmbedding layer, chained q_proj→q_norm→rope ──
  # bf16 q (8, 16, 128) with f32 cos/sin half tables at the Qwen3 geometry
  # (head_dim 128, 16 heads, rope_theta 1e6): judged against an f64 host
  # reference at the kElementwise-class band (2 bf16 ulps)
  block:
    const
      Tokens = 8
      Heads = 16
      Dim = 128
      Half = Dim div 2
      Theta = 1e6
    # independent f64 host tables
    var invFreq = newSeq[float64](Half)
    for j in 0 ..< Half:
      invFreq[j] = pow(Theta, -(2.0 * float64(j)) / float64(Dim))
    var cosHost = newSeq[float32](Tokens * Half)
    var sinHost = newSeq[float32](Tokens * Half)
    for t in 0 ..< Tokens:
      for j in 0 ..< Half:
        let angle = float64(t) * invFreq[j]
        cosHost[t * Half + j] = float32(cos(angle))
        sinHost[t * Half + j] = float32(sin(angle))

    var qHost = newSeq[bfloat16](Tokens * Heads * Dim)
    for i in 0 ..< qHost.len:
      qHost[i] = (float32((i * 37) mod 31 - 15) * 0.25'f32).to(bfloat16)

    let q = newTensor(any, @[Tokens, Heads, Dim], kBf16)
    q.copyToDeviceFromHost(qHost)
    let rotary = RotaryPositionEmbedding.new(any, Dim, Tokens, Theta)
    # layer table construction judged against the independent f64 tables
    block:
      var gotCos = newSeq[float32](Tokens * Half)
      var gotSin = newSeq[float32](Tokens * Half)
      copyToHostFromDevice(gotCos, rotary.cos_cache)
      copyToHostFromDevice(gotSin, rotary.sin_cache)
      for i in 0 ..< Tokens * Half:
        doAssert gotCos[i] == cosHost[i], name & ": cos table " & $i &
          ": got " & $gotCos[i] & ", want " & $cosHost[i]
        doAssert gotSin[i] == sinHost[i], name & ": sin table " & $i &
          ": got " & $gotSin[i] & ", want " & $sinHost[i]

    let o = Tensor(engine: q.engine,
      node: tn_rope(q.engine, q.node, rotary.cos_cache.node, rotary.sin_cache.node))
    doAssert o.node.getImpl().shape() == q.node.getImpl().shape()
    doAssert o.node.getImpl().dtype() == kBf16
    var oRoots = @[o.node]
    o.engine.eval(oRoots.toSpan())
    var gotO = newSeq[bfloat16](qHost.len)
    copyToHostFromDevice(gotO, o)
    # f64 host reference of q*cos + rotateHalf(q)*sin over the reference
    # duplicated half table: entry i reads the table column i mod Half,
    # its rotateHalf partner sitting Half columns away
    var refO = newSeq[float32](qHost.len)
    for t in 0 ..< Tokens:
      for h in 0 ..< Heads:
        for i in 0 ..< Dim:
          let idx = t * Heads * Dim + h * Dim + i
          let partner = if i < Half: i + Half else: i - Half
          let qv = float64(qHost[idx].to(float32))
          let pv = float64(qHost[t * Heads * Dim + h * Dim + partner].to(float32))
          let rot = if i < Half: -pv else: pv
          let c = float64(cosHost[t * Half + i mod Half])
          let s = float64(sinHost[t * Half + i mod Half])
          refO[idx] = float32(qv * c + rot * s)
    gotO.checkBf16Ulp(refO, 2, name & ": rope (8, 16, 128)")

    # rotate_half pairing exact: cos=1/sin=0 leaves q unchanged bit for bit,
    # cos=0/sin=1 swaps the halves with the sign flip on the even columns
    block:
      const
        PTokens = 2
        PHeads = 3
        PDim = 8
        PHalf = PDim div 2
      var pqHost = newSeq[bfloat16](PTokens * PHeads * PDim)
      for i in 0 ..< pqHost.len:
        pqHost[i] = (float32((i * 29) mod 19 - 9) * 0.5'f32).to(bfloat16)
      let pq = newTensor(any, @[PTokens, PHeads, PDim], kBf16)
      pq.copyToDeviceFromHost(pqHost)
      var ones = newSeq[float32](PTokens * PHalf)
      var zeros = newSeq[float32](PTokens * PHalf)
      for i in 0 ..< ones.len:
        ones[i] = 1.0'f32
        zeros[i] = 0.0'f32
      let cosId = newTensor(any, @[PTokens, PHalf], kFp32)
      cosId.copyToDeviceFromHost(ones)
      let sinId = newTensor(any, @[PTokens, PHalf], kFp32)
      sinId.copyToDeviceFromHost(zeros)
      let oid = Tensor(engine: pq.engine,
        node: tn_rope(pq.engine, pq.node, cosId.node, sinId.node))
      var oidRoots = @[oid.node]
      oid.engine.eval(oidRoots.toSpan())
      var gotId = newSeq[bfloat16](pqHost.len)
      copyToHostFromDevice(gotId, oid)
      for i in 0 ..< pqHost.len:
        doAssert gotId[i].to(float32) == pqHost[i].to(float32),
          name & ": rope identity " & $i & ": got " & $gotId[i].to(float32) &
          ", want " & $pqHost[i].to(float32)

      let cosSw = newTensor(any, @[PTokens, PHalf], kFp32)
      cosSw.copyToDeviceFromHost(zeros)
      let sinSw = newTensor(any, @[PTokens, PHalf], kFp32)
      sinSw.copyToDeviceFromHost(ones)
      let osw = Tensor(engine: pq.engine,
        node: tn_rope(pq.engine, pq.node, cosSw.node, sinSw.node))
      var oswRoots = @[osw.node]
      osw.engine.eval(oswRoots.toSpan())
      var gotSw = newSeq[bfloat16](pqHost.len)
      copyToHostFromDevice(gotSw, osw)
      for t in 0 ..< PTokens:
        for h in 0 ..< PHeads:
          for i in 0 ..< PHalf:
            let base = t * PHeads * PDim + h * PDim
            doAssert gotSw[base + i].to(float32) ==
              (-pqHost[base + i + PHalf].to(float32)).to(bfloat16).to(float32),
              name & ": rope swap even " & $i & ": got " &
              $gotSw[base + i].to(float32)
            doAssert gotSw[base + i + PHalf].to(float32) ==
              pqHost[base + i].to(float32),
              name & ": rope swap odd " & $i & ": got " &
              $gotSw[base + i + PHalf].to(float32)

    # face contract: rank, width, dtype and token mismatches raise
    block:
      let q2 = newTensor(any, @[Tokens, Dim], kBf16)
      doAssertRaises ValueError:
        discard tn_rope(any, q2.node, rotary.cos_cache.node, rotary.sin_cache.node)
      let qOdd = newTensor(any, @[Tokens, Heads, Dim + 1], kBf16)
      doAssertRaises ValueError:
        discard tn_rope(any, qOdd.node, rotary.cos_cache.node, rotary.sin_cache.node)
      let cosBf = newTensor(any, @[Tokens, Half], kBf16)
      doAssertRaises ValueError:
        discard tn_rope(any, q.node, cosBf.node, rotary.sin_cache.node)
      let cosShort = newTensor(any, @[Tokens, Half - 1], kFp32)
      doAssertRaises ValueError:
        discard tn_rope(any, q.node, cosShort.node, rotary.sin_cache.node)
      let cosMore = newTensor(any, @[Tokens + 1, Half], kFp32)
      doAssertRaises ValueError:
        discard tn_rope(any, q.node, cosMore.node, rotary.sin_cache.node)

    # ropeByPositions: the cache entries gather per position id
    block:
      let positions = [3, 0, Tokens - 1]
      let (cosSel, sinSel) = rotary.ropeByPositions(positions)
      doAssert cosSel.node.getImpl().shape().len == 2
      doAssert cosSel.node.getImpl().shape()[0] == positions.len
      doAssert cosSel.node.getImpl().shape()[1] == Half
      var selRoots = @[cosSel.node, sinSel.node]
      cosSel.engine.eval(selRoots.toSpan())
      var gotCosSel = newSeq[float32](positions.len * Half)
      var gotSinSel = newSeq[float32](positions.len * Half)
      copyToHostFromDevice(gotCosSel, cosSel)
      copyToHostFromDevice(gotSinSel, sinSel)
      for r in 0 ..< positions.len:
        for j in 0 ..< Half:
          doAssert gotCosSel[r * Half + j] == cosHost[positions[r] * Half + j],
            name & ": ropeByPositions cos " & $r & "," & $j
          doAssert gotSinSel[r * Half + j] == sinHost[positions[r] * Half + j],
            name & ": ropeByPositions sin " & $r & "," & $j
      doAssertRaises ValueError:
        discard rotary.ropeByPositions([Tokens])

    # chained smoke q_proj → q_norm → rope vs reference order at the Qwen3
    # geometry (hidden 1024, 16 q heads, 8 kv heads, head_dim 128):
    # intermediates read back so each stage's reference consumes the device's
    # own previous-stage values
    block:
      const
        CTokens = 4
        CHidden = 1024
        CHeads = 16
        CKvHeads = 8
      var xHost = newSeq[bfloat16](CTokens * CHidden)
      for i in 0 ..< xHost.len:
        xHost[i] = (float32((i * 41) mod 23 - 11) * 0.125'f32).to(bfloat16)
      var wqHost = newSeq[bfloat16](CHeads * Dim * CHidden)
      for i in 0 ..< wqHost.len:
        wqHost[i] = (float32((i * 17) mod 29 - 14) * 0.0625'f32).to(bfloat16)
      var wkHost = newSeq[bfloat16](CKvHeads * Dim * CHidden)
      for i in 0 ..< wkHost.len:
        wkHost[i] = (float32((i * 13) mod 31 - 15) * 0.0625'f32).to(bfloat16)
      var wnHost = newSeq[bfloat16](Dim)
      for i in 0 ..< Dim:
        wnHost[i] = (float32(i mod 11) * 0.1'f32 + 0.5'f32).to(bfloat16)

      let x = newTensor(any, @[CTokens, CHidden], kBf16)
      x.copyToDeviceFromHost(xHost)
      let wq = newTensor(any, @[CHeads * Dim, CHidden], kBf16)
      wq.copyToDeviceFromHost(wqHost)
      let wk = newTensor(any, @[CKvHeads * Dim, CHidden], kBf16)
      wk.copyToDeviceFromHost(wkHost)
      let wn = newTensor(any, @[Dim], kBf16)
      wn.copyToDeviceFromHost(wnHost)
      let rotaryC = RotaryPositionEmbedding.new(any, Dim, CTokens, Theta)

      let qProj = Linear.init(wq).forward(x)
      let kProj = Linear.init(wk).forward(x)
      let q3 = qProj.reshape([CTokens, CHeads, Dim])
      let k3 = kProj.reshape([CTokens, CKvHeads, Dim])
      let qNorm = RmsNorm.init(wn).forward(q3)
      let kNorm = RmsNorm.init(wn).forward(k3)
      let (qRot, kRot) = rotaryC.applyRope(
        qNorm.reshape([1, CTokens, CHeads, Dim]),
        kNorm.reshape([1, CTokens, CKvHeads, Dim]),
        rotaryC.cos_cache, rotaryC.sin_cache)
      doAssert qRot.node.getImpl().shape().len == 4
      doAssert qRot.node.getImpl().shape()[0] == 1
      doAssert qRot.node.getImpl().shape()[1] == CTokens
      doAssert qRot.node.getImpl().shape()[2] == CHeads
      doAssert qRot.node.getImpl().shape()[3] == Dim
      doAssert kRot.node.getImpl().shape()[2] == CKvHeads
      var roots = @[qProj.node, qNorm.node, kNorm.node, qRot.node, kRot.node]
      qRot.engine.eval(roots.toSpan())

      # q_norm reference over the device's own projected values
      var gotQ3 = newSeq[bfloat16](CTokens * CHeads * Dim)
      copyToHostFromDevice(gotQ3, qProj)
      var gotQn = newSeq[bfloat16](CTokens * CHeads * Dim)
      copyToHostFromDevice(gotQn, qNorm)
      var refQn = newSeq[float32](CTokens * CHeads * Dim)
      for t in 0 ..< CTokens:
        for h in 0 ..< CHeads:
          var ss = 0.0'f64
          for i in 0 ..< Dim:
            let v = float64(gotQ3[t * CHeads * Dim + h * Dim + i].to(float32))
            ss += v * v
          let scale = 1.0 / sqrt(ss / float64(Dim) + 1e-6)
          for i in 0 ..< Dim:
            let v = float64(gotQ3[t * CHeads * Dim + h * Dim + i].to(float32))
            let w = float64(wnHost[i].to(float32))
            refQn[t * CHeads * Dim + h * Dim + i] = float32(v * scale * w)
      gotQn.checkBf16Ulp(refQn, 2, name & ": chained q_norm")

      # rope reference over the device's own normed values
      var gotQr = newSeq[bfloat16](CTokens * CHeads * Dim)
      copyToHostFromDevice(gotQr, qRot)
      var refQr = newSeq[float32](CTokens * CHeads * Dim)
      for t in 0 ..< CTokens:
        for h in 0 ..< CHeads:
          for i in 0 ..< Dim:
            let idx = t * CHeads * Dim + h * Dim + i
            let partner = if i < Half: i + Half else: i - Half
            let qv = float64(gotQn[idx].to(float32))
            let pv = float64(gotQn[t * CHeads * Dim + h * Dim + partner].to(float32))
            let rot = if i < Half: -pv else: pv
            let c = float64(cosHost[t * Half + i mod Half])
            let s = float64(sinHost[t * Half + i mod Half])
            refQr[idx] = float32(qv * c + rot * s)
      gotQr.checkBf16Ulp(refQr, 2, name & ": chained rope")

  echo "tattleformers ops (", name, "): all checks passed"
