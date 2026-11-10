# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Silu·mul through the tensorcery umbrella only
##
## Test that the silu·mul op dispatches when the client imports tensorcery and nothing below it:
##
## - the op registers at the umbrella import. The handler chain
##   instantiates at this module's eval call site
## - the output matches the f64 host reference, one rounding into bfloat16
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_silu_mul_late_import_metal.nim

import std/math
import workspace/tensorcery/src/tensorcery

proc main =
  const Rows = 4
  const Cols = 8
  var gh = newSeq[bfloat16](Rows * Cols)
  var vh = newSeq[bfloat16](Rows * Cols)
  for i in 0 ..< Rows * Cols:
    gh[i] = (float32((i * 37) mod 41 - 20) * 0.5'f32).to(bfloat16)
    vh[i] = (float32((i * 29) mod 31 - 15) * 0.25'f32).to(bfloat16)
  let gate = toTensor(gh, @[Rows, Cols], runMetal)
  let up = toTensor(vh, @[Rows, Cols], runMetal)
  let o = silu_mul(gate, up)
  doAssert o.shape().len == 2, "late import: silu rank"
  doAssert o.shape()[0] == Rows, "late import: silu tokens"
  doAssert o.shape()[1] == Cols, "late import: silu cols"
  doAssert o.dtype() == kBf16, "late import: silu dtype"
  eval(o)
  for t in 0 ..< Rows:
    for c in 0 ..< Cols:
      let g = float64(gh[t * Cols + c].to(float32))
      let v = float64(vh[t * Cols + c].to(float32))
      let want = float32(g / (1.0 + exp(-g)) * v).to(bfloat16)
      doAssert o.item(t, c, bfloat16) == want,
        "late import: silu " & $t & "," & $c
  echo "tensorcery late-import silu·mul (Metal): all checks passed"

when isMainModule:
  main()
