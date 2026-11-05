# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Gradcheck on Metal: fused backward kernels against f64 host finite differences.
##
## Test that backward() on Metal matches f64 central finite differences
## (step 1e-4) for gelu, softmax, layernorm, cross entropy, gather and where.
## Slice joins through its view scatter. Every eval asserts one submission.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_gradcheck_metal.nim

import std/math

import
  workspace/tensorcery,
  workspace/tensorcery/src/backends/metal

func frand(i: int): float32 =
  ## Deterministic host input in [0, 1) from the golden-ratio low-discrepancy sequence.
  let x = float64(i) * 0.6180339887498949
  float32(x mod 1.0)

func toF64(a: seq[float32]): seq[float64] =
  for x in a:
    result.add float64(x)

proc checkBand(actual: seq[float32], expected: seq[float64], label: string): float64 =
  ## Band check of device f32 gradients against the f64 finite differences.
  ## Returns the largest absolute delta for the report.
  doAssert actual.len == expected.len, label & ": length mismatch"
  var maxDelta = 0.0
  for i in 0 ..< expected.len:
    let band = 1e-3 * abs(expected[i]) + 1e-4
    let delta = abs(float64(actual[i]) - expected[i])
    doAssert delta <= band,
        label & ": element " & $i & " got " & $actual[i] & ", expected " & $expected[i]
    if delta > maxDelta:
      maxDelta = delta
  result = maxDelta

type F64Fn = proc(xs: seq[seq[float64]]): float64 {.closure.}
  ## Host reference: the scalar value of the op graph over the f64 leaf images.

proc centralDiff(f: F64Fn, xs: seq[seq[float64]], leaf, i: int): float64 =
  const h = 1e-4
  var xp = xs
  xp[leaf][i] += h
  let plus = f(xp)
  xp[leaf][i] -= 2 * h
  result = (plus - f(xp)) / (2 * h)

proc expectGrad(leaf: Tensor, f: F64Fn, xs: seq[seq[float64]], li: int, label: string, maxDeltas: var seq[string]) =
  var expected: seq[float64]
  for i in 0 ..< xs[li].len:
    expected.add centralDiff(f, xs, li, i)
  let delta = checkBand(toSeq[float32](grad(leaf)), expected, label)
  maxDeltas.add label & ": max delta " & $delta

proc drainOneSubmission(ctx: Ctx, what: string) =
  ## Evaluates the pending record and asserts the Metal one-submission contract.
  resetSubmissionCount()
  ctx.eval()
  doAssert submissionCount == 1,
      what & ": forward and grad entries must drain as ONE submission, got " & $submissionCount

proc main() =
  var maxDeltas: seq[string]

  block geluGrad:
    let n = 6
    var xdata: seq[float32]
    for i in 0 ..< n:
      xdata.add frand(i + 11) - 0.5
    let ctx = Ctx()
    let x = toGradTensor(ctx, xdata, bkMetal, n)
    let loss = gelu(x)
    backward(loss)
    drainOneSubmission(ctx, "gelu")

    let f = proc(xs: seq[seq[float64]]): float64 =
      var total = 0.0
      for v in xs[0]:
        let u = 0.7978845608028654 * (v + 0.044715 * v * v * v)
        total += 0.5 * v * (1.0 + tanh(u))
      total
    expectGrad(x, f, @[toF64(xdata)], 0, "metal gelu d/dx", maxDeltas)

  block softmaxGrad:
    let rows, d = 3
    var xdata: seq[float32]
    for i in 0 ..< rows * d:
      xdata.add frand(i + 21)
    let ctx = Ctx()
    let x = toGradTensor(ctx, xdata, bkMetal, rows, d)
    let loss = softmax(x)
    backward(loss)
    drainOneSubmission(ctx, "softmax")

    let f = proc(xs: seq[seq[float64]]): float64 =
      var total = 0.0
      for r in 0 ..< rows:
        var m = -1.0e30
        for c in 0 ..< d:
          m = max(m, xs[0][r * d + c])
        var s = 0.0
        for c in 0 ..< d:
          s += exp(xs[0][r * d + c] - m)
        for c in 0 ..< d:
          total += exp(xs[0][r * d + c] - m) / s
      total
    expectGrad(x, f, @[toF64(xdata)], 0, "metal softmax d/dx", maxDeltas)

  block layernormGrad:
    let rows, d = 4
    var xdata, wdata, bdata: seq[float32]
    for i in 0 ..< rows * d:
      xdata.add frand(i + 31)
    for i in 0 ..< d:
      wdata.add frand(i + 51)
      bdata.add frand(i + 61)
    let ctx = Ctx()
    let x = toGradTensor(ctx, xdata, bkMetal, rows, d)
    let w = toGradTensor(ctx, wdata, bkMetal, d)
    let b = toGradTensor(ctx, bdata, bkMetal, d)
    let loss = layernorm(x, w, b, 1e-3)
    backward(loss)
    drainOneSubmission(ctx, "layernorm")

    let f = proc(xs: seq[seq[float64]]): float64 =
      var total = 0.0
      for r in 0 ..< rows:
        var mean = 0.0
        for c in 0 ..< d:
          mean += xs[0][r * d + c]
        mean /= float64(d)
        var vr = 0.0
        for c in 0 ..< d:
          let dv = xs[0][r * d + c] - mean
          vr += dv * dv
        vr /= float64(d)
        let invstd = 1.0 / sqrt(vr + 1e-3)
        for c in 0 ..< d:
          let xh = (xs[0][r * d + c] - mean) * invstd
          total += xh * xs[1][c] + xs[2][c]
      total
    expectGrad(x, f, @[toF64(xdata), toF64(wdata), toF64(bdata)], 0,
        "metal layernorm d/dx", maxDeltas)
    expectGrad(w, f, @[toF64(xdata), toF64(wdata), toF64(bdata)], 1,
        "metal layernorm d/dw", maxDeltas)
    expectGrad(b, f, @[toF64(xdata), toF64(wdata), toF64(bdata)], 2,
        "metal layernorm d/db", maxDeltas)

  block crossEntropyGrad:
    let rows, v = 4
    var ldata: seq[float32]
    for i in 0 ..< rows * v:
      ldata.add frand(i + 71) - 0.25
    var tdata = newSeq[int32](rows)
    tdata[0] = 1
    tdata[1] = 3
    tdata[2] = 0
    tdata[3] = 2
    let ctx = Ctx()
    let logits = toGradTensor(ctx, ldata, bkMetal, rows, v)
    let targets = toTensor(ctx, tdata, bkMetal, rows)
    let loss = crossEntropy(logits, targets)
    backward(loss)
    drainOneSubmission(ctx, "cross entropy")

    let f = proc(xs: seq[seq[float64]]): float64 =
      var total = 0.0
      for r in 0 ..< rows:
        var m = -1.0e30
        for c in 0 ..< v:
          m = max(m, xs[0][r * v + c])
        var s = 0.0
        for c in 0 ..< v:
          s += exp(xs[0][r * v + c] - m)
        let lse = m + ln(s)
        total += lse - xs[0][r * v + int(tdata[r])]
      total
    expectGrad(logits, f, @[toF64(ldata)], 0, "metal cross entropy d/dlogits", maxDeltas)

  block gatherGrad:
    # duplicate indices accumulate through the atomic scatter
    let e = 3
    var wdata: seq[float32]
    for i in 0 ..< 4 * e:
      wdata.add frand(i + 91)
    var idxData = newSeq[int32](4)
    idxData[0] = 0
    idxData[1] = 2
    idxData[2] = 0
    idxData[3] = 2
    let ctx = Ctx()
    let w = toGradTensor(ctx, wdata, bkMetal, 4, e)
    let idx = toTensor(ctx, idxData, bkMetal, 4)
    let loss = gather(w, idx)
    backward(loss)
    drainOneSubmission(ctx, "gather")

    let f = proc(xs: seq[seq[float64]]): float64 =
      var total = 0.0
      for r in 0 ..< 4:
        for c in 0 ..< e:
          total += xs[0][int(idxData[r]) * e + c]
      total
    expectGrad(w, f, @[toF64(wdata)], 0, "metal gather d/dw", maxDeltas)

  block whereGrad:
    let n = 6
    var cdata, adata, bdata: seq[float32]
    for i in 0 ..< n:
      cdata.add (if i mod 2 == 0: 1.0'f32 else: -1.0'f32)
      adata.add frand(i + 101)
      bdata.add frand(i + 111)
    let ctx = Ctx()
    let cond = toGradTensor(ctx, cdata, bkMetal, n)
    let a = toGradTensor(ctx, adata, bkMetal, n)
    let b = toGradTensor(ctx, bdata, bkMetal, n)
    let loss = where(cond, a, b)
    backward(loss)
    drainOneSubmission(ctx, "where")

    let f = proc(xs: seq[seq[float64]]): float64 =
      var total = 0.0
      for i in 0 ..< n:
        if xs[0][i] != 0.0:
          total += xs[1][i]
        else:
          total += xs[2][i]
      total
    expectGrad(a, f, @[toF64(cdata), toF64(adata), toF64(bdata)], 1,
        "metal where d/da", maxDeltas)
    expectGrad(b, f, @[toF64(cdata), toF64(adata), toF64(bdata)], 2,
        "metal where d/db", maxDeltas)

  block sliceGrad:
    let d0, d1, d2 = 2
    var xdata: seq[float32]
    for i in 0 ..< d0 * d1 * d2:
      xdata.add frand(i + 131)
    let ctx = Ctx()
    let x = toGradTensor(ctx, xdata, bkMetal, d0, d1, d2)
    let loss = x.slice(1 .. 1, 0 .. 1)
    backward(loss)
    drainOneSubmission(ctx, "slice")

    let f = proc(xs: seq[seq[float64]]): float64 =
      var total = 0.0
      for b in 0 ..< d1:
        for c in 0 ..< d2:
          total += xs[0][(1 * d1 + b) * d2 + c]
      total
    expectGrad(x, f, @[toF64(xdata)], 0, "metal slice d/dx", maxDeltas)

  block permuteGrad:
    # weighted loss: a uniform cotangent cannot expose a stride error
    let (d0, d1, d2) = (2, 3, 4)
    var xdata, wdata: seq[float32]
    for i in 0 ..< d0 * d1 * d2:
      xdata.add frand(i + 151)
      wdata.add frand(i + 301)
    let ctx = Ctx()
    let x = toGradTensor(ctx, xdata, bkMetal, d0, d1, d2)
    let wts = toTensor(ctx, wdata, bkMetal, d2, d0, d1)
    let loss = x.permute(2, 0, 1) * wts
    backward(loss)
    drainOneSubmission(ctx, "metal permute")

    let f = proc(xs: seq[seq[float64]]): float64 =
      # out axis i reads x axis perm[i], so out[a, b, c] = x[b, c, a]
      var total = 0.0
      for a in 0 ..< d2:
        for b in 0 ..< d0:
          for c in 0 ..< d1:
            total += float64(wdata[a * d0 * d1 + b * d1 + c]) *
                xs[0][b * d1 * d2 + c * d2 + a]
      total
    expectGrad(x, f, @[toF64(xdata)], 0, "metal permute d/dx", maxDeltas)

  block reshapeGrad:
    let (d0, d1, d2) = (2, 3, 4)
    var xdata, wdata: seq[float32]
    for i in 0 ..< d0 * d1 * d2:
      xdata.add frand(i + 161)
      wdata.add frand(i + 311)
    let ctx = Ctx()
    let x = toGradTensor(ctx, xdata, bkMetal, d0, d1, d2)
    let wts = toTensor(ctx, wdata, bkMetal, d0 * d1 * d2)
    let loss = x.reshape(d0 * d1, d2).flatten() * wts
    backward(loss)
    drainOneSubmission(ctx, "metal reshape")

    let f = proc(xs: seq[seq[float64]]): float64 =
      var total = 0.0
      for i in 0 ..< d0 * d1 * d2:
        total += float64(wdata[i]) * xs[0][i]
      total
    expectGrad(x, f, @[toF64(xdata)], 0, "metal reshape d/dx", maxDeltas)

  for line in maxDeltas:
    stdout.writeLine line
  stdout.writeLine "t_gradcheck_metal: OK"

main()
