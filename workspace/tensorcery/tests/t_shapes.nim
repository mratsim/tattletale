# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shape ops: reshape, permute, transpose, slice, flatten as pure metadata.
##
## Test that shape ops return the right elements in logical order through views
## on CPU and Metal, never enqueue, and never touch the submission count.
## Test that a view reads through its offset and strides when an elementwise op consumes it.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_shapes.nim

import std/math

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery,
  workspace/tensorcery/src/backends/cpu,
  workspace/tensorcery/src/backends/metal

func toFloat64(a: seq[float32]): seq[float64] =
  for x in a:
    result.add float64(x)

func frand(i: int): float32 =
  ## Deterministic host input in [0, 1) from the golden-ratio low-discrepancy sequence.
  let x = float64(i) * 0.6180339887498949
  float32(x mod 1.0)

proc sameShape(t: Tensor, dims: openArray[int]): bool =
  ## Per-axis shape comparison, SmallSeq has no compact `$`.
  if int(t.getShape().len) != dims.len:
    return false
  for i in 0 ..< dims.len:
    if int(t.getShape()[i]) != dims[i]:
      return false
  result = true

proc checkSeq(actual: seq[float32], expected: seq[float64], label: string) =
  doAssert actual.len == expected.len, label & ": length mismatch"
  for i in 0 ..< expected.len:
    doAssert abs(float64(actual[i]) - expected[i]) < 1e-5,
        label & ": element " & $i & " got " & $actual[i] & ", expected " & $expected[i]

proc seq3d(d: seq[float32], d0, d1, d2: int, perm: array[3, int]): seq[float64] =
  ## Host reference: reads `d` shaped (d0, d1, d2) contiguous in the permuted axis order,
  ## output axis 0 reads source axis perm[0], logical order, last axis fastest.
  let dims = [d0, d1, d2]
  for a in 0 ..< dims[perm[0]]:
    for b in 0 ..< dims[perm[1]]:
      for c in 0 ..< dims[perm[2]]:
        var crd: array[3, int]
        crd[perm[0]] = a
        crd[perm[1]] = b
        crd[perm[2]] = c
        result.add float64(d[crd[0] * d1 * d2 + crd[1] * d2 + crd[2]])

proc main() =
  let d0, d1, d2 = 2
  var data: seq[float32]
  for i in 0 ..< d0 * d1 * d2:
    data.add frand(i + 5)

  block cpuReshape:
    let ctx = Ctx()
    let t = toTensor(ctx, data, d0, d1, d2)
    let r = t.reshape(d0 * d1, d2)
    doAssert sameShape(r, [d0 * d1, d2]), "reshape shape: " & $r.getShape()
    var expected: seq[float64]
    for x in data:
      expected.add float64(x)
    checkSeq(toSeq[float32](r), expected, "cpu reshape")
    doAssert ctx.record.len == 1, "reshape records one entry for the walk"

  block cpuPermute:
    let ctx = Ctx()
    let t = toTensor(ctx, data, d0, d1, d2)
    let p = t.permute(2, 0, 1)
    checkSeq(toSeq[float32](p), seq3d(data, d0, d1, d2, [2, 0, 1]),
        "cpu permute(2,0,1)")

  block cpuTranspose:
    var tdata: seq[float32]
    for i in 0 ..< 6:
      tdata.add frand(i + 77)
    let ctx = Ctx()
    let t = toTensor(ctx, tdata, 2, 3)
    let tr = t.transpose()
    doAssert sameShape(tr, [3, 2]), "transpose shape: " & $tr.getShape()
    var expected: seq[float64]
    for j in 0 ..< 3:
      for i in 0 ..< 2:
        expected.add float64(tdata[i * 3 + j])
    checkSeq(toSeq[float32](tr), expected, "cpu transpose")

  block cpuSlice:
    let ctx = Ctx()
    let t = toTensor(ctx, data, d0, d1, d2)
    let s = t.slice(1 .. 1, 0 .. 1)
    doAssert sameShape(s, [1, 2, 2]), "slice shape: " & $s.getShape()
    var expected: seq[float64]
    for b in 0 ..< d1:
      for c in 0 ..< d2:
        expected.add float64(data[(1 * d1 + b) * d2 + c])
    checkSeq(toSeq[float32](s), expected, "cpu slice")
    let f = t.flatten()
    doAssert toSeq[float32](f) == data, "cpu flatten returns the storage"

  block cpuChained:
    # reshape the contiguous storage, then permute the view
    let ctx = Ctx()
    let t = toTensor(ctx, data, d0, d1, d2)
    let v = t.reshape(d0 * d1, d2).permute(1, 0)
    var expected: seq[float64]
    for j in 0 ..< d2:
      for i in 0 ..< d0 * d1:
        expected.add float64(data[i * d2 + j])
    checkSeq(toSeq[float32](v), expected, "cpu reshape+permute")
    var refused = false
    try:
      discard t.permute(2, 1, 0).reshape(4, 2)
    except AssertionDefect:
      refused = true
    doAssert refused, "reshape must refuse a non-contiguous view"

  block cpuViewThroughAdd:
    # a view consumed by an elementwise op reads at its offset through its strides
    let ctx = Ctx()
    let a = toTensor(ctx, data, d0, d1, d2)
    var bdata: seq[float32]
    for i in 0 ..< d0 * d1 * d2:
      bdata.add frand(i + 101)
    let b = toTensor(ctx, bdata, d0, d1, d2)
    let v = a.slice(1 .. 1) + b.slice(1 .. 1)
    doAssert sameShape(v, [1, 2, 2]), "view add shape: " & $v.getShape()
    var expected: seq[float64]
    for i in 0 ..< d1 * d2:
      expected.add float64(data[d1 * d2 + i] + bdata[d1 * d2 + i])
    checkSeq(toSeq[float32](v), expected, "cpu view add")

  block refusals:
    let ctx = Ctx()
    let t = toTensor(ctx, data, d0, d1, d2)
    var refused = false
    try:
      discard t.reshape(7)
    except AssertionDefect:
      refused = true
    doAssert refused, "reshape must refuse an element count change"
    refused = false
    try:
      discard t.permute(0, 0, 1)
    except AssertionDefect:
      refused = true
    doAssert refused, "permute must refuse a non-permutation"

  block metalShapeOpsNeverSubmit:
    let ctx = Ctx()
    let t = toTensor(ctx, data, bkMetal, d0, d1, d2)
    discard t.reshape(d1, d2, d0)
    discard t.permute(2, 0, 1)
    discard t.slice(1 .. 1)
    discard t.flatten()
    doAssert ctx.getPendingCount() == 4, "shape ops record entries for the walk"
    resetSubmissionCount()
    ctx.eval()
    doAssert submissionCount == 0, "shape ops never submit"

  block metalViewValues:
    let ctx = Ctx()
    let t = toTensor(ctx, data, bkMetal, d0, d1, d2)
    let p = t.permute(2, 0, 1)
    checkSeq(toSeq[float32](p), seq3d(data, d0, d1, d2, [2, 0, 1]),
        "metal permute(2,0,1)")
    let s = t.slice(1 .. 1, 0 .. 1)
    var expected: seq[float64]
    for b in 0 ..< d1:
      for c in 0 ..< d2:
        expected.add float64(data[(1 * d1 + b) * d2 + c])
    checkSeq(toSeq[float32](s), expected, "metal slice")

  block cpuSharedViewStorage:
    # D-6: a view shares the parent storage. A view created before an in-place
    # update sees the update on read, and an update through the view reaches the parent.
    let ctx = Ctx()
    let w = toGradTensor(ctx, data, d0 * d1 * d2)
    let v = w.reshape(d1, d2, d0)
    var xdata: seq[float32]
    for i in 0 ..< d0 * d1 * d2:
      xdata.add frand(i + 200)
    let x = toTensor(ctx, xdata, d0 * d1 * d2)
    let loss = w * x
    backward(loss)
    ctx.eval()
    checkSeq(toSeq[float32](v), toFloat64(data), "view before update keeps the parent values")

    sgdUpdate(w, grad(w), -0.5)
    var expectUpdated: seq[float64]
    for i in 0 ..< d0 * d1 * d2:
      expectUpdated.add float64(data[i]) + float64(xdata[i]) * 0.5
    # reading the view must materialize the pending update first (the MLX rule)
    checkSeq(toSeq[float32](v), expectUpdated, "view reads through the parent's in-place update")
    checkSeq(toSeq[float32](w), expectUpdated, "parent reads its own update")

    # update through the contiguous view, parent sees it
    let v2 = w.reshape(4, 2)
    let loss2 = v2 * x.reshape(4, 2)
    backward(loss2)
    sgdUpdate(v2, grad(v2), 0.25)
    ctx.eval()
    var expectV2: seq[float64]
    for i in 0 ..< d0 * d1 * d2:
      expectV2.add expectUpdated[i] - float64(xdata[i]) * 0.25
    checkSeq(toSeq[float32](w), expectV2, "parent reads an update made through the view")
    checkSeq(toSeq[float32](v), expectV2, "earlier view reads an update made through another view")

  block metalSharedViewStorage:
    let ctx = Ctx()
    let w = toGradTensor(ctx, data, bkMetal, d0 * d1 * d2)
    let v = w.reshape(d1, d2, d0)
    var xdata: seq[float32]
    for i in 0 ..< d0 * d1 * d2:
      xdata.add frand(i + 200)
    let x = toTensor(ctx, xdata, bkMetal, d0 * d1 * d2)
    let loss = w * x
    backward(loss)
    ctx.eval()
    checkSeq(toSeq[float32](v), toFloat64(data), "metal view before update keeps the parent values")

    sgdUpdate(w, grad(w), -0.5)
    var expectUpdated: seq[float64]
    for i in 0 ..< d0 * d1 * d2:
      expectUpdated.add float64(data[i]) + float64(xdata[i]) * 0.5
    checkSeq(toSeq[float32](v), expectUpdated, "metal view reads through the parent's in-place update")

  block metalViewThroughAdd:
    let ctx = Ctx()
    let a = toTensor(ctx, data, bkMetal, d0, d1, d2)
    var bdata: seq[float32]
    for i in 0 ..< d0 * d1 * d2:
      bdata.add frand(i + 101)
    let b = toTensor(ctx, bdata, bkMetal, d0, d1, d2)
    let v = a.slice(1 .. 1) + b.slice(1 .. 1)
    resetSubmissionCount()
    ctx.eval()
    doAssert submissionCount == 1, "the elementwise op over views submits once"
    var expected: seq[float64]
    for i in 0 ..< d1 * d2:
      expected.add float64(data[d1 * d2 + i] + bdata[d1 * d2 + i])
    checkSeq(toSeq[float32](v), expected, "metal view add")

  stdout.writeLine "t_shapes: OK"

main()
