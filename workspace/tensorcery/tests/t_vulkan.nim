# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan backend: forward op set on the batched submission path.
##
## Test that every Vulkan op matches an independent f64 host reference.
## Test that one eval drains in ONE Vulkan submission and the pure-metadata
## shape ops never enqueue.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_vulkan.nim

import std/math
import std/strutils

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery,
  workspace/tensorcery/src/backends/vulkan

func frand(i: int): float32 =
  ## Deterministic host input in [0, 1) from the golden-ratio low-discrepancy sequence.
  let x = float64(i) * 0.6180339887498949
  float32(x mod 1.0)

func frandSigned(i: int): float32 =
  ## Deterministic host input in [-1, 1) from the golden-ratio low-discrepancy sequence.
  2.0 * frand(i) - 1.0

proc checkBand(actual: seq[float32], expected: seq[float64], label: string) =
  ## Elementwise band check of device f32 output against the f64 host reference.
  doAssert actual.len == expected.len, label & ": length mismatch"
  for i in 0 ..< expected.len:
    let band = 1e-5 * abs(expected[i]) + 1e-5
    doAssert abs(float64(actual[i]) - expected[i]) <= band,
        label & ": element " & $i & " got " & $actual[i] & ", expected " & $expected[i]

proc main() =
  block add:
    let ctx = Ctx()
    let n = 1000
    var adata, bdata: seq[float32]
    for i in 0 ..< n:
      adata.add frandSigned(i)
      bdata.add frandSigned(i + 7)
    let a = toTensor(ctx, adata, bkVulkan, n)
    let b = toTensor(ctx, bdata, bkVulkan, n)
    let c = a + b
    vulkan.resetSubmissionCount()
    ctx.eval()
    doAssert vulkan.submissionCount == 1, "one eval is one Vulkan submission"
    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add float64(adata[i]) + float64(bdata[i])
    checkBand(toSeq[float32](c), expected, "add")

  block chainedAdds:
    let ctx = Ctx()
    let n = 512
    var adata, bdata: seq[float32]
    for i in 0 ..< n:
      adata.add frandSigned(i + 1)
      bdata.add frandSigned(i + 2)
    let a = toTensor(ctx, adata, bkVulkan, n)
    let b = toTensor(ctx, bdata, bkVulkan, n)
    let c = (a + b) + a
    vulkan.resetSubmissionCount()
    ctx.eval()
    doAssert vulkan.submissionCount == 1,
        "chained adds drain in ONE Vulkan submission, got " & $vulkan.submissionCount
    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add 2.0 * float64(adata[i]) + float64(bdata[i])
    checkBand(toSeq[float32](c), expected, "chained adds")

  block addStridedViews:
    ## Both operands are slice views over rank-2 storage. The kernels read them
    ## through their strides.
    let ctx = Ctx()
    var adata, bdata: seq[float32]
    for i in 0 ..< 32:
      adata.add frandSigned(i + 3)
      bdata.add frandSigned(i + 11)
    let a = toTensor(ctx, adata, bkVulkan, 4, 8)
    let b = toTensor(ctx, bdata, bkVulkan, 4, 8)
    let av = a.slice(0 .. 2, 1 .. 6)
    let bv = b.slice(1 .. 3, 0 .. 5)
    let c = av + bv
    vulkan.resetSubmissionCount()
    ctx.eval()
    doAssert vulkan.submissionCount == 1, "one eval is one Vulkan submission"
    var expected: seq[float64]
    for r in 0 ..< 3:
      for col in 0 ..< 6:
        expected.add float64(adata[r * 8 + 1 + col]) + float64(bdata[(r + 1) * 8 + col])
    checkBand(toSeq[float32](c), expected, "add of slice views")

  block shapeOpsNeverEnqueue:
    let ctx = Ctx()
    var adata: seq[float32]
    for i in 0 ..< 24:
      adata.add frandSigned(i)
    let a = toTensor(ctx, adata, bkVulkan, 2, 3, 4)
    vulkan.resetSubmissionCount()
    let r = a.reshape(6, 4)
    let p = a.permute(2, 1, 0)
    let t = a.reshape(6, 4).transpose()
    let s = a.slice(1 .. 1, 0 .. 2, 0 .. 3)
    doAssert vulkan.submissionCount == 0, "shape ops never enqueue"
    let rs = getShape(r)
    doAssert int(rs.len) == 2 and rs[0] == 6 and rs[1] == 4,
        "reshape is pure metadata over the same storage"
    discard p
    discard t
    var expected: seq[float64]
    for j in 0 ..< 3:
      for col in 0 ..< 4:
        expected.add float64(adata[12 + j * 4 + col])
    checkBand(toSeq[float32](s), expected, "slice readback gathers through strides")

  block mulMulScalar:
    let ctx = Ctx()
    let n = 700
    var adata, bdata: seq[float32]
    for i in 0 ..< n:
      adata.add frandSigned(i + 5)
      bdata.add frandSigned(i + 13)
    let a = toTensor(ctx, adata, bkVulkan, n)
    let b = toTensor(ctx, bdata, bkVulkan, n)
    let prod = a * b
    let scaled = prod * 2.5
    let filled = toTensor(ctx, a.getImpl().onesNoGrad(), dtFloat32, devVulkan)
    vulkan.resetSubmissionCount()
    ctx.eval()
    doAssert vulkan.submissionCount == 1,
        "mul, mulScalar and fill drain in ONE Vulkan submission, got " & $vulkan.submissionCount
    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add 2.5 * float64(adata[i]) * float64(bdata[i])
    checkBand(toSeq[float32](scaled), expected, "mul then mulScalar")
    var ones: seq[float64]
    for i in 0 ..< n:
      ones.add 1.0
    checkBand(toSeq[float32](filled), ones, "fill")

  block where:
    let ctx = Ctx()
    let n = 256
    var cdata, adata, bdata: seq[float32]
    for i in 0 ..< n:
      cdata.add frand(i + 31)
      adata.add frandSigned(i + 17)
      bdata.add frandSigned(i + 23)
    let cond = toTensor(ctx, cdata, bkVulkan, n)
    let a = toTensor(ctx, adata, bkVulkan, n)
    let b = toTensor(ctx, bdata, bkVulkan, n)
    let sel = where(cond, a, b)
    vulkan.resetSubmissionCount()
    ctx.eval()
    doAssert vulkan.submissionCount == 1, "one eval is one Vulkan submission"
    var expected: seq[float64]
    for i in 0 ..< n:
      if cdata[i] != 0.0:
        expected.add float64(adata[i])
      else:
        expected.add float64(bdata[i])
    checkBand(toSeq[float32](sel), expected, "where")

  block gather:
    let ctx = Ctx()
    let v, e = 3
    var wdata: seq[float32]
    for i in 0 ..< v * e:
      wdata.add frandSigned(i + 3)
    var idxData: seq[int32]
    idxData.add 2'i32
    idxData.add 0'i32
    idxData.add 2'i32
    let w = toTensor(ctx, wdata, bkVulkan, v, e)
    let idx = toTensor(ctx, idxData, bkVulkan, 3)
    let outT = gather(w, idx)
    vulkan.resetSubmissionCount()
    ctx.eval()
    doAssert vulkan.submissionCount == 1, "one eval is one Vulkan submission"
    var expected: seq[float64]
    for row in 0 ..< 3:
      for col in 0 ..< e:
        expected.add float64(wdata[idxData[row] * e + col])
    checkBand(toSeq[float32](outT), expected, "gather")

  block layernorm:
    let ctx = Ctx()
    let rows, d = 7
    var xdata, wdata, bdata: seq[float32]
    for i in 0 ..< rows * d:
      xdata.add frandSigned(i + 2)
    for i in 0 ..< d:
      wdata.add frandSigned(i + 41)
      bdata.add frandSigned(i + 43)
    let eps = 1e-5
    let x = toTensor(ctx, xdata, bkVulkan, rows, d)
    let w = toTensor(ctx, wdata, bkVulkan, d)
    let b = toTensor(ctx, bdata, bkVulkan, d)
    let outT = layernorm(x, w, b, eps)
    vulkan.resetSubmissionCount()
    ctx.eval()
    doAssert vulkan.submissionCount == 1, "one eval is one Vulkan submission"
    var expected: seq[float64]
    for r in 0 ..< rows:
      var mean = 0.0
      for c in 0 ..< d:
        mean += float64(xdata[r * d + c])
      mean /= float64(d)
      var vr = 0.0
      for c in 0 ..< d:
        let dv = float64(xdata[r * d + c]) - mean
        vr += dv * dv
      vr /= float64(d)
      let scale = 1.0 / sqrt(vr + eps)
      for c in 0 ..< d:
        expected.add (float64(xdata[r * d + c]) - mean) * scale * float64(wdata[c]) + float64(bdata[c])
    checkBand(toSeq[float32](outT), expected, "layernorm")

  block gelu:
    let ctx = Ctx()
    let n = 300
    var adata: seq[float32]
    for i in 0 ..< n:
      adata.add frandSigned(i + 9)
    let a = toTensor(ctx, adata, bkVulkan, n)
    let outT = gelu(a)
    vulkan.resetSubmissionCount()
    ctx.eval()
    doAssert vulkan.submissionCount == 1, "one eval is one Vulkan submission"
    var expected: seq[float64]
    for i in 0 ..< n:
      let v = float64(adata[i])
      let u = 0.7978845608028654 * (v + 0.044715 * v * v * v)
      expected.add 0.5 * v * (1.0 + tanh(u))
    checkBand(toSeq[float32](outT), expected, "gelu")

  block vulkanRefusals:
    let ctx = Ctx()
    let a = toTensor(ctx, @[1.0'f32, 2.0], bkVulkan, 2)
    var refused = false
    try:
      discard a.getImpl().addNoGrad(a.getImpl())
    except AssertionDefect as e:
      refused = "addNoGrad(tensor" in e.msg
    doAssert refused, "grad-chain composition is refused on Vulkan, naming the input"

main()
