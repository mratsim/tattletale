# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## RoPE through the tensorcery umbrella only
##
## Test that the rope op dispatches when the client imports tensorcery and nothing below it:
##
## - the op registers at the umbrella import. The handler chain
##   instantiates at this module's eval call site
## - the rotated output matches the f32 host reference, one rounding
##   into bfloat16
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_rope_late_import_metal.nim

import std/math
import workspace/tensorcery/src/tensorcery

proc main =
  const Tokens = 4
  const Heads = 2
  const Dim = 8
  const Half = Dim div 2
  var cosH = newSeq[float32](Tokens * Half)
  var sinH = newSeq[float32](Tokens * Half)
  for t in 0 ..< Tokens:
    for j in 0 ..< Half:
      let invFreq = pow(100.0, -(2.0 * float64(j)) / float64(Dim))
      let angle = float64(t) * invFreq
      cosH[t * Half + j] = float32(cos(angle))
      sinH[t * Half + j] = float32(sin(angle))
  var qH = newSeq[bfloat16](Tokens * Heads * Dim)
  for i in 0 ..< qH.len:
    qH[i] = (float32((i * 37) mod 31 - 15) * 0.25'f32).to(bfloat16)
  let q = toTensor(qH, @[Tokens, Heads, Dim], runMetal)
  let cosT = toTensor(cosH, @[Tokens, Half], runMetal)
  let sinT = toTensor(sinH, @[Tokens, Half], runMetal)
  let o = rope(q, cosT, sinT)
  doAssert o.shape().len == 3, "late import: rope rank"
  doAssert o.shape()[0] == Tokens, "late import: rope tokens"
  doAssert o.shape()[1] == Heads, "late import: rope heads"
  doAssert o.shape()[2] == Dim, "late import: rope dim"
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
          "late import: rope " & $t & "," & $h & "," & $i
  echo "tensorcery late-import rope (Metal): all checks passed"

when isMainModule:
  main()
