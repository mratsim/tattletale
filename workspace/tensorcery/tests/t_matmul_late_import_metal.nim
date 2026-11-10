# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Linear projection through the tensorcery umbrella only
##
## Test that the matmul op dispatches when the client imports tensorcery
## and nothing below it:
##
## - the op registers at the umbrella import, the handler chain
##   instantiates at this module's eval call site
## - the projection compares against the sequential f32 host reference
##   rounded once into bfloat16
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_matmul_late_import_metal.nim

import workspace/tensorcery/src/tensorcery

proc main =
  var xf = newSeq[float32](4 * 8)
  var wf = newSeq[float32](6 * 8)
  for i in 0 ..< 4 * 8:
    xf[i] = float32((i * 37 mod 19) - 9) * 0.125'f32
  for i in 0 ..< 6 * 8:
    wf[i] = float32((i * 29 mod 17) - 8) * 0.125'f32
  let x = toTensor(xf, @[4, 8], runMetal, bfloat16)
  let w = toTensor(wf, @[6, 8], runMetal, bfloat16)
  let y = linear(x, w)
  doAssert y.shape().len == 2, "late import: linear rank"
  doAssert y.shape()[0] == 4, "late import: linear tokens"
  doAssert y.shape()[1] == 6, "late import: linear outFeatures"
  eval(y)
  for mm in 0 ..< 4:
    for nn in 0 ..< 6:
      var acc = 0.0'f32
      for kk in 0 ..< 8:
        acc += xf[mm * 8 + kk] * wf[nn * 8 + kk]
      doAssert y.item(mm, nn, bfloat16) == acc.to(bfloat16),
        "late import: linear " & $mm & "," & $nn
  echo "tensorcery late-import linear (Metal): all checks passed"

when isMainModule:
  main()
