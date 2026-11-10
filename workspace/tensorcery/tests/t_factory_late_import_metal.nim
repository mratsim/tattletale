# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensor factories through the tensorcery umbrella only
##
## Test that the host factories and the layout-aware host readback work
## when the client imports tensorcery and nothing below it:
##
## - zeros, arange, eye, full_like and toSeq resolve at the umbrella import
## - a narrowed+transposed view reads back the strided host values, offset included
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_factory_late_import_metal.nim

import workspace/tensorcery/src/tensorcery

proc main =
  var xd: array[16, float32] = [1.0'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16]
  var xq: array[16, bfloat16]
  for i in 0 ..< 16:
    xq[i] = xd[i].to(bfloat16)

  let a = arange(1, 9, 1'f32, runMetal, float32)
  var ga = toSeq(a, float32)
  for i in 0 ..< 8:
    doAssert ga[i] == float32(i + 1), "late import: arange " & $i

  let z = zeros(@[2, 2], runMetal)
  var gz = toSeq(z, bfloat16)
  for i in 0 ..< gz.len:
    doAssert gz[i] == 0'f32.to(bfloat16), "late import: zeros " & $i

  let e = eye(3, runMetal)
  var ge = toSeq(e, bfloat16)
  for i in 0 ..< 3:
    for j in 0 ..< 3:
      doAssert ge[i * 3 + j] == (if i == j: 1'f32.to(bfloat16) else: 0'f32.to(bfloat16)),
        "late import: eye " & $i & "," & $j

  let x = toTensor(xq, @[2, 8], runMetal)
  let f = full_like(x, 4.5'f32)
  var gf = toSeq(f, bfloat16)
  for i in 0 ..< gf.len:
    doAssert gf[i] == 4.5'f32.to(bfloat16), "late import: full_like " & $i

  # narrow columns 2..5 then transpose: (4,2) strides (1,8), offset 2
  let v = transpose(narrow(x, 1, 2, 4))
  var vShape = @[4, 2]
  doAssert v.shape() == vShape.toSpan(), "late import: view gather shape"
  let gv = toSeq(v, bfloat16)
  doAssert gv.len == 8, "late import: view gather numel"
  for i in 0 ..< 4:
    for j in 0 ..< 2:
      doAssert gv[i * 2 + j] == xd[2 + j * 8 + i].to(bfloat16),
        "late import: view gather " & $i & "," & $j

  echo "tensorcery late-import factories (Metal): all checks passed"

when isMainModule:
  main()
