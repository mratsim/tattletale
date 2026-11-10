# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## RmsNorm through the tensorcery umbrella only
##
## Test that the rmsnorm op dispatches when the client imports tensorcery
## and nothing below it:
##
## - the op registers at the umbrella import, the handler chain
##   instantiates at this module's eval call site
## - the normalized strips compare against the f32 host reference
##   rounded once into bfloat16
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_rmsnorm_late_import_metal.nim

import std/math
import workspace/tensorcery/src/tensorcery

proc main =
  var xf = newSeq[float32](4 * 8)
  var wf = newSeq[float32](8)
  for i in 0 ..< 4 * 8:
    xf[i] = float32((i * 37 mod 19) - 9) * 0.125'f32
  for i in 0 ..< 8:
    wf[i] = float32((i * 29 mod 17) - 8) * 0.125'f32
  let x = toTensor(xf, @[4, 8], runMetal, bfloat16)
  let w = toTensor(wf, @[8], runMetal, bfloat16)
  let y = rmsnorm(x, w)
  doAssert y.shape().len == 2, "late import: rmsnorm rank"
  doAssert y.shape()[0] == 4, "late import: rmsnorm tokens"
  doAssert y.shape()[1] == 8, "late import: rmsnorm width"
  eval(y)
  for t in 0 ..< 4:
    var ss = 0.0'f32
    for c in 0 ..< 8:
      let x = xf[t * 8 + c]
      ss += x * x
    let scale = 1.0'f32 / sqrt(ss / 8.0'f32 + 1e-6'f32)
    for c in 0 ..< 8:
      let want = (xf[t * 8 + c] * scale * wf[c]).to(bfloat16)
      doAssert y.item(t, c, bfloat16) == want,
        "late import: rmsnorm " & $t & "," & $c
  echo "tensorcery late-import rmsnorm (Metal): all checks passed"

when isMainModule:
  main()
