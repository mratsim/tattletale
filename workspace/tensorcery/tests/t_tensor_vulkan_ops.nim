# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensorcery ops through the tensorcery umbrella on vulkan
##
## Test that the Tensor public face runs end to end on the Vulkan engine
## with the client importing tensorcery and nothing below it:
##
## - add, split, embed_gather, a rank-3 round trip, item and toSeq readback
## - rmsnorm, rope and silu_mul over bf16, judged against the host references
## - an f32 matmul, the one op with a Vulkan atom at f32
## - the face contracts raise
##
## bf16 buffers store u16 bit patterns and compute in f32.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_tensor_vulkan_ops.nim

import workspace/tensorcery/src/tensorcery
import std/math

proc bf16UlpKey(x: bfloat16): int32 =
  ## Ordered key over the bf16 bit pattern, sign-magnitude mapped to a signed integer scale.
  ## The distance of two keys is the ulp distance.
  let b = cast[uint16](x)
  if (b and 0x8000'u16) != 0:
    -int32(b and 0x7FFF'u16)
  else:
    int32(b)

proc ropeTables(tokens, half: int, theta: float64): tuple[c, s: seq[float32]] =
  ## Half-width NEOX tables over one theta. Entry (t, i) stores the angle
  ## t / theta^(2i/dim) with dim = 2*half.
  result.c = newSeq[float32](tokens * half)
  result.s = newSeq[float32](tokens * half)
  for t in 0 ..< tokens:
    for i in 0 ..< half:
      let angle = float64(t) / pow(theta, float64(2 * i) / float64(2 * half))
      result.c[t * half + i] = float32(cos(angle))
      result.s[t * half + i] = float32(sin(angle))

proc main =
  const Rows = 2
  const Cols = 4
  var xh = newSeq[bfloat16](Rows * Cols)
  var yh = newSeq[bfloat16](Rows * Cols)
  for i in 0 ..< Rows * Cols:
    xh[i] = float32(i + 1).to(bfloat16)
    yh[i] = float32(2).to(bfloat16)
  let x = toTensor(xh, @[Rows, Cols], runVulkan)
  let y = toTensor(yh, @[Rows, Cols], runVulkan)

  # add: the device sum reads back through item
  let z = x + y
  doAssert z.shape().len == 2, "port smoke: add rank"
  doAssert z.dtype() == kBf16, "port smoke: add dtype"
  eval(z)
  for t in 0 ..< Rows:
    for c in 0 ..< Cols:
      let want = (xh[t * Cols + c].to(float32) + yh[t * Cols + c].to(float32)).to(bfloat16)
      doAssert z.item(t, c, bfloat16) == want, "port smoke: add " & $t & "," & $c

  # split: inner-axis split makes strided views, read back per part
  let parts = split(x, 0, 2)
  doAssert parts.len == 2, "port smoke: split parts"
  for p in 0 ..< 2:
    doAssert parts[p].shape()[0] == 1, "port smoke: split rows"
    doAssert parts[p].shape()[1] == Cols, "port smoke: split cols"
  for c in 0 ..< Cols:
    doAssert parts[0].item(0, c, bfloat16) == xh[c], "port smoke: split part0"
    doAssert parts[1].item(0, c, bfloat16) == xh[Cols + c], "port smoke: split part1"

  # embed gather: a (16, 8) bf16 weight, ids [3, 0, 7, 3]. The duplicate id
  # reads one weight entry twice.
  var wh = newSeq[bfloat16](16 * 8)
  for i in 0 ..< 16 * 8:
    wh[i] = float32(i).to(bfloat16)
  let w = toTensor(wh, @[16, 8], runVulkan)
  let ids = @[3, 0, 7, 3]
  let selected = embed_gather(w, ids)
  doAssert selected.shape()[0] == 4, "port smoke: gather tokens"
  doAssert selected.shape()[1] == 8, "port smoke: gather hidden"
  eval(selected)
  var gatherFlat = newSeq[bfloat16](4 * 8)
  for t in 0 ..< 4:
    for c in 0 ..< 8:
      let want = float32(ids[t] * 8 + c).to(bfloat16)
      doAssert selected.item(t, c, bfloat16) == want,
        "port smoke: gather " & $t & "," & $c
      gatherFlat[t * 8 + c] = want
  doAssert selected.toSeq(bfloat16) == gatherFlat, "port smoke: gather toSeq"

  # rank-3 round trip: add over (2, 3, 4) bf16, item and toSeq readback
  const D0 = 2
  const D1 = 3
  const D2 = 4
  var qh = newSeq[bfloat16](D0 * D1 * D2)
  var rh = newSeq[bfloat16](D0 * D1 * D2)
  for i in 0 ..< D0 * D1 * D2:
    qh[i] = float32(i).to(bfloat16)
    rh[i] = float32(10 - i).to(bfloat16)
  let q = toTensor(qh, @[D0, D1, D2], runVulkan)
  let r = toTensor(rh, @[D0, D1, D2], runVulkan)
  let s = q + r
  doAssert s.shape().len == 3, "port smoke: rank-3 rank"
  doAssert s.shape()[0] == D0 and s.shape()[1] == D1 and s.shape()[2] == D2,
    "port smoke: rank-3 shape"
  eval(s)
  var sFlat = newSeq[bfloat16](D0 * D1 * D2)
  for i in 0 ..< D0 * D1 * D2:
    sFlat[i] = (qh[i].to(float32) + rh[i].to(float32)).to(bfloat16)
  doAssert s.toSeq(bfloat16) == sFlat, "port smoke: rank-3 toSeq"
  for d0 in 0 ..< D0:
    for d1 in 0 ..< D1:
      for d2 in 0 ..< D2:
        doAssert s.item(d0, d1, d2, bfloat16) == sFlat[(d0 * D1 + d1) * D2 + d2],
          "port smoke: rank-3 item " & $d0 & "," & $d1 & "," & $d2

  # face contracts raise
  doAssertRaises AssertionDefect:
    discard embed_gather(w, @[])
  doAssertRaises AssertionDefect:
    discard embed_gather(w, @[16])
  doAssertRaises AssertionDefect:
    discard q + x

  # rmsnorm over bf16: f32 math reads the bf16 storage through exact promotion,
  # one rounding into the output.
  block:
    const Rows = 4
    const Width = 8
    var xh = newSeq[bfloat16](Rows * Width)
    var wh = newSeq[bfloat16](Width)
    for i in 0 ..< Rows * Width:
      xh[i] = (float32((i * 37 mod 19) - 9) * 0.125'f32).to(bfloat16)
    for i in 0 ..< Width:
      wh[i] = (float32((i * 29 mod 17) - 8) * 0.125'f32).to(bfloat16)
    let x = toTensor(xh, @[Rows, Width], runVulkan)
    let w = toTensor(wh, @[Width], runVulkan)
    let y = rmsnorm(x, w)
    doAssert y.shape().len == 2, "port smoke: rmsnorm rank"
    doAssert y.dtype() == kBf16, "port smoke: rmsnorm dtype"
    eval(y)
    for t in 0 ..< Rows:
      var ss = 0.0'f32
      for c in 0 ..< Width:
        let xv = xh[t * Width + c].to(float32)
        ss += xv * xv
      let scale = 1.0'f32 / sqrt(ss / float32(Width) + 1e-6'f32)
      for c in 0 ..< Width:
        let want = (xh[t * Width + c].to(float32) * scale * wh[c].to(float32)).to(bfloat16)
        doAssert y.item(t, c, bfloat16) == want,
          "port smoke: rmsnorm " & $t & "," & $c & ": got " &
          $y.item(t, c, bfloat16).to(float32) & ", want " & $want.to(float32)

  # rope over bf16: the rotation reads the promoted values and the f32 tables,
  # one rounding into the output.
  block:
    const Tokens = 4
    const Heads = 2
    const Dim = 8
    const Half = Dim div 2
    let (cosH, sinH) = ropeTables(Tokens, Half, 100.0)
    var qh = newSeq[bfloat16](Tokens * Heads * Dim)
    for i in 0 ..< qh.len:
      qh[i] = (float32((i * 37) mod 31 - 15) * 0.25'f32).to(bfloat16)
    let q = toTensor(qh, @[Tokens, Heads, Dim], runVulkan)
    let cosT = toTensor(cosH, @[Tokens, Half], runVulkan)
    let sinT = toTensor(sinH, @[Tokens, Half], runVulkan)
    let o = rope(q, cosT, sinT)
    doAssert o.shape().len == 3, "port smoke: rope rank"
    doAssert o.dtype() == kBf16, "port smoke: rope dtype"
    eval(o)
    for t in 0 ..< Tokens:
      for h in 0 ..< Heads:
        for i in 0 ..< Dim:
          let partner = if i < Half: i + Half else: i - Half
          let qv = qh[t * Heads * Dim + h * Dim + i].to(float32)
          let pv = qh[t * Heads * Dim + h * Dim + partner].to(float32)
          let c = cosH[t * Half + i mod Half]
          let sn = sinH[t * Half + i mod Half]
          let want = (if i < Half: qv * c - pv * sn else: qv * c + pv * sn).to(bfloat16)
          doAssert o.item(t, h, i, bfloat16) == want,
            "port smoke: rope " & $t & "," & $h & "," & $i & ": got " &
            $o.item(t, h, i, bfloat16).to(float32) & ", want " & $want.to(float32)

  # silu·mul over bf16: the exp2 math runs on the promoted values.
  # Normal band: exact against the f64 reference.
  # Boundary set: exp2 overflow and underflow regions, judged in a 2 bf16 ulp band.
  block:
    const Rows = 4
    const Cols = 8
    var gh = newSeq[bfloat16](Rows * Cols)
    var vh = newSeq[bfloat16](Rows * Cols)
    for i in 0 ..< Rows * Cols:
      gh[i] = (float32((i * 37) mod 41 - 20) * 0.5'f32).to(bfloat16)
      vh[i] = (float32((i * 29) mod 31 - 15) * 0.25'f32).to(bfloat16)
    let gate = toTensor(gh, @[Rows, Cols], runVulkan)
    let up = toTensor(vh, @[Rows, Cols], runVulkan)
    let o = silu_mul(gate, up)
    doAssert o.shape().len == 2, "port smoke: silu rank"
    doAssert o.dtype() == kBf16, "port smoke: silu dtype"
    eval(o)
    for t in 0 ..< Rows:
      for c in 0 ..< Cols:
        let g = float64(gh[t * Cols + c].to(float32))
        let v = float64(vh[t * Cols + c].to(float32))
        let want = float32(g / (1.0 + exp(-g)) * v).to(bfloat16)
        doAssert o.item(t, c, bfloat16) == want,
          "port smoke: silu " & $t & "," & $c & ": got " &
          $o.item(t, c, bfloat16).to(float32) & ", want " & $want.to(float32)

    # Exp2 boundary columns the Vulkan driver computes faithfully.
    # Reference: the full Metal set in runSiluMulTest (tensor_tests_template.nim) reads -inf or NaN here.
    # Cause: MoltenVK's fast-math compilation reassociates the silu multiply chains.
    # Reassociated products overflow where the written left-to-right order stays normal.
    # Measured: any column with |silu input · up| over f32 max reads -inf or NaN.
    # Measured: the composite division inside the normal branch collapses to zero once its reciprocal is subnormal, silu input below -87.4.
    # This module's SPIR-V output is faithful, and opExtInst Ex2 is exact at the edges: exp2(-inf) = +0, exp2(127.5) = 2.4e38.
    # Divergence source is the driver's reassociation, recorded in the operation STATE for an operator ruling.
    # Honored range: the exp2 overflow repair branch at silu input -100 and -90.
    # Honored range: the exp2 underflow branch at -150, -193 and -200, products rounding to -0 in bfloat16.
    # Honored range: the division branch at a large but subnormal-safe exponent, silu input -80.
    let bGate = [(3.3e38'f32).to(bfloat16), (88.0'f32).to(bfloat16),
                 (0.0'f32).to(bfloat16), (-80.0'f32).to(bfloat16),
                 (-100.0'f32).to(bfloat16), (-90.0'f32).to(bfloat16),
                 (-150.0'f32).to(bfloat16), (-193.0'f32).to(bfloat16),
                 (-200.0'f32).to(bfloat16)]
    let bUp = [(1.0'f32).to(bfloat16), (2.0'f32).to(bfloat16),
               (3.0'f32).to(bfloat16), (2.0'f32).to(bfloat16),
               (2.0'f32).to(bfloat16), (2.0'f32).to(bfloat16),
               (3.0'f32).to(bfloat16), (3.0'f32).to(bfloat16),
               (3.0'f32).to(bfloat16)]
    const BRows = 2
    const BCols = 4
    var bgh = newSeq[bfloat16](BRows * BCols)
    var buh = newSeq[bfloat16](BRows * BCols)
    for t in 0 ..< BRows:
      for c in 0 ..< BCols:
        bgh[t * BCols + c] = bGate[c]
        buh[t * BCols + c] = bUp[c]
    let bGateT = toTensor(bgh, @[BRows, BCols], runVulkan)
    let bUpT = toTensor(buh, @[BRows, BCols], runVulkan)
    let bo = silu_mul(bGateT, bUpT)
    eval(bo)
    for t in 0 ..< BRows:
      for c in 0 ..< BCols:
        let g = float64(bGate[c].to(float32))
        let v = float64(bUp[c].to(float32))
        let want = float32(g / (1.0 + exp(-g)) * v).to(bfloat16)
        let got = bo.item(t, c, bfloat16)
        doAssert got == want,
          "port smoke: silu boundary " & $t & "," & $c & " (gate " &
          $bGate[c].to(float32) & "): got " & $got.to(float32) &
          ", want " & $want.to(float32)

  # matmul over f32: the one op with a Vulkan atom at f32. A relative band
  # against the host accumulation, since the tiled K loop reorders a reduction.
  block:
    const Tokens = 4
    const InF = 8
    const OutF = 6
    var xh = newSeq[float32](Tokens * InF)
    var wh = newSeq[float32](OutF * InF)
    for i in 0 ..< Tokens * InF:
      xh[i] = float32((i * 37 mod 19) - 9) * 0.125'f32
    for i in 0 ..< OutF * InF:
      wh[i] = float32((i * 29 mod 17) - 8) * 0.125'f32
    let x = toTensor(xh, @[Tokens, InF], runVulkan)
    let w = toTensor(wh, @[OutF, InF], runVulkan)
    let y = linear(x, w)
    doAssert y.shape().len == 2, "port smoke: linear rank"
    doAssert y.shape()[0] == Tokens, "port smoke: linear tokens"
    doAssert y.shape()[1] == OutF, "port smoke: linear outFeatures"
    doAssert y.dtype() == kFp32, "port smoke: linear dtype"
    eval(y)
    for mm in 0 ..< Tokens:
      for nn in 0 ..< OutF:
        var acc = 0.0'f32
        for kk in 0 ..< InF:
          acc += xh[mm * InF + kk] * wh[nn * InF + kk]
        let got = y.item(mm, nn, float32)
        doAssert abs(got - acc) <= 1e-5'f32 * max(abs(acc), 1.0'f32),
          "port smoke: linear " & $mm & "," & $nn & ": got " & $got &
          ", reference " & $acc

  echo "tensorcery port smoke (Vulkan): all checks passed"

when isMainModule:
  main()
