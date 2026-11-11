# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Embedding gather through the tensorcery umbrella only
##
## Test that the gather op dispatches when the client imports tensorcery
## and nothing below it.
##
## - the op registers at the umbrella import, the handler chain
##   instantiates at this module's eval call site
## - the gathered entries compare against the weight's own entries read
##   back through the public face
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_embed_gather_late_import_metal.nim

import workspace/tensorcery/src/tensorcery

proc main =
  var wf = newSeq[float32](16 * 8)
  for i in 0 ..< 16 * 8:
    wf[i] = float32(i)
  let w = toTensor(wf, @[16, 8], runMetal, bfloat16)
  let selected = embed_gather(w, @[3, 0, 7, 3])
  doAssert selected.shape().len == 2
  doAssert selected.shape()[0] == 4
  doAssert selected.shape()[1] == 8
  eval(selected)
  for t in 0 ..< 4:
    for c in 0 ..< 8:
      doAssert selected.item(t, c, bfloat16) == w.item([3, 0, 7, 3][t], c, bfloat16),
        "late import: gather out " & $t & "," & $c
  echo "tensorcery late-import gather (Metal): all checks passed"

when isMainModule:
  main()
