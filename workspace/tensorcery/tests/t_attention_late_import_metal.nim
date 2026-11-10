# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Jagged attention through the tensorcery umbrella only
##
## Test that the attention op dispatches when the client imports tensorcery
## and nothing below it:
##
## - the op registers at the umbrella import. The handler chain
##   instantiates at this module's eval call site
## - the KV metadata type comes through the umbrella's re-export of the face module
## - a decode scenario over the pool slabs matches the f64 host reference
##   inside the bf16 band
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_attention_late_import_metal.nim

from std/math import sqrt, exp
import std/random
import workspace/tensorcery/src/tensorcery

const
  H = 16
  Nkv = 8
  D = 128
  PageSize = 256
  NumLayers = 2
  Bf16Rtol = 1e-2'f64
  Bf16Abstol = 1e-2'f64

proc poolPos(pageId, layer, pos, kvHead: int): int =
  (((pageId * NumLayers + layer) * PageSize + pos) * Nkv + kvHead) * D

proc main =
  # decode: one segment, q_len 1, kv_len 300 over 2 pages, the last page
  # partially covered (300 = 256 + 44), one spare page pure garbage
  const
    NumPages = 3
    QTotal = 1
    KvLen = 300
  var rng = initRand(20261111)
  template randBf16(): bfloat16 =
    rng.rand(-1.0 .. 1.0).to(bfloat16)
  template nanBf16(): bfloat16 =
    cast[bfloat16](uint16(0x7FC0))
  var qH = newSeq[bfloat16](QTotal * H * D)
  for i in 0 ..< qH.len:
    qH[i] = randBf16()
  var kH = newSeq[bfloat16](NumPages * NumLayers * PageSize * Nkv * D)
  var vH = newSeq[bfloat16](kH.len)
  for i in 0 ..< kH.len:
    kH[i] = nanBf16()
    vH[i] = nanBf16()
  for t in 0 ..< KvLen:
    let
      pageIdx = t div PageSize
      base = poolPos(pageIdx, 0, t mod PageSize, 0)
    for h in 0 ..< Nkv:
      for d in 0 ..< D:
        let idx = base + h * D + d
        kH[idx] = randBf16()
        vH[idx] = randBf16()

  # f64 host reference over the same bf16-rounded inputs.
  # Decode attends keys [0, kvLen). GQA kvHead = head div (H div Nkv)
  var refO = newSeq[float64](QTotal * H * D)
  let scale = 1.0 / sqrt(float64(D))
  for h in 0 ..< H:
    let kvHead = h div (H div Nkv)
    var scores = newSeq[float64](KvLen)
    var m = -1.0e300
    for k in 0 ..< KvLen:
      var acc = 0.0'f64
      for d in 0 ..< D:
        let pos = poolPos(k div PageSize, 0, k mod PageSize, kvHead)
        acc += float64(qH[h * D + d].to(float32)) * float64(kH[pos + d].to(float32))
      scores[k] = acc * scale
      m = max(m, scores[k])
    var l = 0.0'f64
    var ps = newSeq[float64](KvLen)
    for k in 0 ..< KvLen:
      ps[k] = exp(scores[k] - m)
      l += ps[k]
    for d in 0 ..< D:
      var acc = 0.0'f64
      for k in 0 ..< KvLen:
        let pos = poolPos(k div PageSize, 0, k mod PageSize, kvHead)
        acc += ps[k] * float64(vH[pos + d].to(float32))
      refO[h * D + d] = acc / l

  let
    q = toTensor(qH, @[QTotal, H, D], runMetal)
    poolK = toTensor(kH, @[NumPages, NumLayers, PageSize, Nkv, D], runMetal)
    poolV = toTensor(vH, @[NumPages, NumLayers, PageSize, Nkv, D], runMetal)
  let kvMeta = JaggedKvMeta(
    offsets: @[0, 2], indices: @[0, 1], seqlens: @[KvLen], batchSize: 1)
  let o = jagged_attention(q, poolK, poolV, @[0, 1], kvMeta,
    float32(scale), 0, H, Nkv)
  doAssert o.shape().len == 3, "late import: attention rank"
  doAssert o.shape()[0] == QTotal, "late import: attention tokens"
  doAssert o.shape()[1] == H, "late import: attention heads"
  doAssert o.shape()[2] == D, "late import: attention dim"
  doAssert o.dtype() == kBf16, "late import: attention dtype"
  eval(o)
  let got = toSeq(o, bfloat16)
  var oBad = 0
  var worst = 0.0'f64
  for i in 0 ..< got.len:
    let g = float64(got[i].to(float32))
    let want = refO[i]
    let bar = Bf16Rtol * abs(want) + Bf16Abstol
    let diff = abs(g - want)
    if not (diff <= bar):
      inc oBad
    elif diff / bar > worst:
      worst = diff / bar
  echo "  decode: worstBar=" & $worst & " mismatches=" & $oBad & "/" & $got.len
  doAssert oBad == 0,
    "late import: " & $oBad & " of " & $got.len & " elements outside the band"
  echo "tensorcery late-import jagged attention (Metal): all checks passed"

when isMainModule:
  main()
