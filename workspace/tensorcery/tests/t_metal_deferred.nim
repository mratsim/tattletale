# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal deferred execution: record at enqueue, one eval, one submission.
##
## Test that four enqueued ops drain in ONE Metal submission with one eval and match an f64 host reference.
## Test that the record stores marshaled blob bytes before eval and that eval encodes from them.
##
## Test that reading a pending tensor materializes it with exactly one implicit submission.
## Test that mixed-dtype and cross-device operands refuse naming the combo.
## Test that to() copies Metal to CPU and back, uploads never submit.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_metal_deferred.nim

import std/[math, strutils]

import std/math
import
  workspace/tensorcery,
  workspace/tensorcery/src/backends/metal,
  workspace/tensorcery/src/kernels/k_add,
  workspace/tensorcery/src/kernels/k_mul

func frand(i: int): float32 =
  ## Deterministic host input in [0, 1) from the golden-ratio low-discrepancy sequence.
  let x = float64(i) * 0.6180339887498949
  float32(x mod 1.0)

func checkAgainstF64(actual: seq[float32], expected: seq[float64], label: string) =
  ## Elementwise band check of device f32 output against the f64 host reference.
  doAssert actual.len == expected.len, label & ": length mismatch"
  for i in 0 ..< expected.len:
    let band = 1e-5 * abs(expected[i]) + 1e-6
    doAssert abs(float64(actual[i]) - expected[i]) <= band,
        label & ": element " & $i & " got " & $actual[i] & ", expected " & $expected[i]

proc main() =
  block fourOpsOneSubmission:
    let ctx = Ctx()
    let n = 1024
    var adata, bdata: seq[float32]
    for i in 0 ..< n:
      adata.add frand(i)
      bdata.add frand(i + 7)
    let a = toTensor(ctx, adata, bkMetal, n)
    let b = toTensor(ctx, bdata, bkMetal, n)

    resetSubmissionCount()
    let t1 = a + b
    let t2 = t1 * 2.0
    let t3 = t2 + b
    let t4 = t3 * 2.0
    doAssert ctx.getPendingCount() == 4, "four ops wait in the record before eval"

    ctx.eval()
    doAssert submissionCount == 1,
        "four ops must drain as ONE submission, got " & $submissionCount

    var expected: seq[float64]
    for i in 0 ..< n:
      let e1 = float64(adata[i]) + float64(bdata[i])
      let e2 = e1 * 2.0
      let e3 = e2 + float64(bdata[i])
      expected.add e3 * 2.0
    checkAgainstF64(toSeq[float32](t4), expected, "fourOps")

  block blobMarshalAtEnqueue:
    let ctx = Ctx()
    let n = 256
    var adata, bdata: seq[float32]
    for i in 0 ..< n:
      adata.add frand(i + 100)
      bdata.add frand(i + 200)
    let a = toTensor(ctx, adata, bkMetal, n)
    let b = toTensor(ctx, bdata, bkMetal, n)

    resetSubmissionCount()
    let t1 = a + b
    let t2 = t1 * 2.0
    doAssert ctx.getPendingCount() == 2
    doAssert ctx.record[0].kernel == opAdd
    doAssert ctx.record[1].kernel == opMul
    # record tensor ids: leaves 1 and 2, op outputs 3 and 4
    doAssert ctx.record[0].output == 3
    doAssert ctx.record[1].output == 4
    doAssert ctx.record[0].inputs == [1, 2]
    doAssert ctx.record[1].inputs == [3, 0]
    doAssert ctx.record[0].blob.len == addScalarsSlot + sizeof(AddScalars)
    doAssert ctx.record[1].blob.len == mulScalarsSlot + sizeof(MulScalars)
    var outAddr: uint64
    copyMem(outAddr.addr, unsafeAddr ctx.record[0].blob[0], 8)
    doAssert outAddr != 0, "the record's blob carries the output device address at enqueue, before eval"
    doAssert submissionCount == 0, "marshaling and uploads submit nothing"

    ctx.eval()
    doAssert submissionCount == 1, "eval encodes from the recorded blobs in ONE submission"

    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add (float64(adata[i]) + float64(bdata[i])) * 2.0
    checkAgainstF64(toSeq[float32](t2), expected, "blobRoundTrip")
    doAssert ctx.getPendingCount() == 0

  block readbackMaterializes:
    let ctx = Ctx()
    let n = 512
    var adata, bdata: seq[float32]
    for i in 0 ..< n:
      adata.add frand(i + 300)
      bdata.add frand(i + 400)
    let a = toTensor(ctx, adata, bkMetal, n)
    let b = toTensor(ctx, bdata, bkMetal, n)

    resetSubmissionCount()
    let c = a + b
    doAssert submissionCount == 0, "enqueue submits nothing"
    let vals = toSeq[float32](c)
    doAssert submissionCount == 1, "reading a pending tensor evals exactly once"

    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add float64(adata[i]) + float64(bdata[i])
    checkAgainstF64(vals, expected, "readback")

  block refusals:
    let ctx = Ctx()
    let cpu32 = toTensor(ctx, @[1.0'f32, 2.0'f32], 2)
    let cpu64 = toTensor(ctx, @[1.0'f64, 2.0'f64], 2)

    var msg = ""
    try:
      discard cpu32 + cpu64
    except AssertionDefect as e:
      msg = e.msg
    doAssert "dtFloat32 x dtFloat64" in msg,
        "mixed-dtype refusal must name the combo, got: " & msg

    let metal32 = toTensor(ctx, @[1.0'f32, 2.0'f32], bkMetal, 2)
    msg = ""
    try:
      discard metal32 + cpu32
    except AssertionDefect as e:
      msg = e.msg
    doAssert "devMetal x devCpu" in msg,
        "cross-device refusal must name the combo, got: " & msg

    let otherCtx = Ctx()
    let elsewhere = toTensor(otherCtx, @[1.0'f32, 2.0'f32], bkMetal, 2)
    msg = ""
    try:
      discard metal32 + elsewhere
    except AssertionDefect as e:
      msg = e.msg
    doAssert "different contexts" in msg, "cross-context refusal, got: " & msg

  block toDeviceCopy:
    let ctx = Ctx()
    let host = toTensor(ctx, @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32], 4)

    resetSubmissionCount()
    let m = host.to(bkMetal)
    doAssert m.getDevice() == devMetal
    doAssert submissionCount == 0, "copy uploads never submit"
    doAssert toSeq[float32](m) == @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]

    let back = m.to(bkCpu)
    doAssert back.getDevice() == devCpu
    doAssert toSeq[float32](back) == @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]

    let a = toTensor(ctx, @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32], bkMetal, 4)
    let t = a * 3.0
    let copied = t.to(bkCpu)
    doAssert submissionCount == 1, "copying a pending tensor evals once"
    doAssert toSeq[float32](copied) == @[3.0'f32, 6.0'f32, 9.0'f32, 12.0'f32]

main()
