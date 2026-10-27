# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile element-wise binary ops: scalar accumulate, division, max/min, broadcasts
##
## Test that the binary tile ops compute element-wise over host register tiles,
## covering scalar accumulate, division, max/min and the broadcast family
## over size-1 modes.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests workspace/ceramic/tests/tile_ops/t_tile_ops_binary.nim

import std/math
import workspace/ceramic/src/tile_algebra

proc fillSeq(t: var Tile, first: float32) =
  ## Fills the tile's slots with an arithmetic progression
  for i in 0 ..< t.size().toInt():
    t.data[i] = first + float32(i)

proc scalarCase =
  ## In-place scalar += and -=
  var a: tile(float32, (4, 4))
  fillSeq(a, 0.0'f32)
  a += 2.5'f32
  for i in 0 ..< 16:
    doAssert a.data[i] == float32(i) + 2.5'f32, "scalar += at slot " & $i
  a -= 0.5'f32
  for i in 0 ..< 16:
    doAssert a.data[i] == float32(i) + 2.0'f32, "scalar -= at slot " & $i

proc divCase =
  ## Tile / tile in dst form (div) and value form (/.)
  var a: tile(float32, (4, 4))
  var b: tile(float32, (4, 4))
  fillSeq(a, 8.0'f32)
  fillSeq(b, 2.0'f32)
  var d: tile(float32, (4, 4))
  d.divide(a, b)
  for i in 0 ..< 16:
    doAssert d.data[i] == (float32(i) + 8.0'f32) / (float32(i) + 2.0'f32),
      "divide dst form at slot " & $i
  let q = a /. b
  for i in 0 ..< 16:
    doAssert q.data[i] == (float32(i) + 8.0'f32) / (float32(i) + 2.0'f32),
      "/. value form at slot " & $i

proc maxminCase =
  ## Element-wise tile max and min in dst and value form
  var a: tile(float32, (4, 4))
  var b: tile(float32, (4, 4))
  fillSeq(a, 0.0'f32)
  fillSeq(b, 8.0'f32)
  var d: tile(float32, (4, 4))
  d.max(a, b)
  for i in 0 ..< 16:
    doAssert d.data[i] == max(float32(i), float32(i) + 8.0'f32),
      "max dst form at slot " & $i
  let m = min(a, b)
  for i in 0 ..< 16:
    doAssert m.data[i] == min(float32(i), float32(i) + 8.0'f32),
      "min value form at slot " & $i

proc broadcastValueCase =
  ## Broadcast value forms over size-1 modes: (2, (4, 4)) with (1, (4, 1)).
  ## Expected values walk each layout's own strides through crd2idx.
  var a: tile(float32, (2, (4, 4)))
  var b: tile(float32, (1, (4, 1)))
  fillSeq(a, 0.0'f32)
  fillSeq(b, 100.0'f32)
  block:
    let c = a +. b
    for i in 0 ..< 2:
      for j in 0 ..< 4:
        for k in 0 ..< 4:
          let idx = toInt(crd2idx(c.getLayout(), (i, (j, k))))
          doAssert c.data[idx] == float32(toInt(crd2idx(a.getLayout(), (i, (j, k))))) +
            (100.0'f32 + float32(j)),
            "+. broadcast at " & $i & "," & $j & "," & $k
  block:
    let c = a -. b
    for i in 0 ..< 2:
      for j in 0 ..< 4:
        for k in 0 ..< 4:
          let idx = toInt(crd2idx(c.getLayout(), (i, (j, k))))
          doAssert c.data[idx] == float32(toInt(crd2idx(a.getLayout(), (i, (j, k))))) -
            (100.0'f32 + float32(j)),
            "-. broadcast at " & $i & "," & $j & "," & $k
  block:
    let c = a *. b
    for i in 0 ..< 2:
      for j in 0 ..< 4:
        for k in 0 ..< 4:
          let idx = toInt(crd2idx(c.getLayout(), (i, (j, k))))
          doAssert c.data[idx] == float32(toInt(crd2idx(a.getLayout(), (i, (j, k))))) *
            (100.0'f32 + float32(j)),
            "*. broadcast at " & $i & "," & $j & "," & $k
  block:
    let c = a /. b
    for i in 0 ..< 2:
      for j in 0 ..< 4:
        for k in 0 ..< 4:
          let idx = toInt(crd2idx(c.getLayout(), (i, (j, k))))
          doAssert c.data[idx] == float32(toInt(crd2idx(a.getLayout(), (i, (j, k))))) /
            (100.0'f32 + float32(j)),
            "/. broadcast at " & $i & "," & $j & "," & $k

proc broadcastInPlaceCase =
  ## Broadcast in-place forms over size-1 modes
  var a: tile(float32, (2, (4, 4)))
  var b: tile(float32, (1, (4, 1)))
  fillSeq(a, 0.0'f32)
  fillSeq(b, 100.0'f32)
  a +.= b
  for i in 0 ..< 2:
    for j in 0 ..< 4:
      for k in 0 ..< 4:
        let idx = toInt(crd2idx(a.getLayout(), (i, (j, k))))
        doAssert a.data[idx] == float32(idx) + (100.0'f32 + float32(j)),
          "+.= broadcast at " & $i & "," & $j & "," & $k
  a -.= b
  for i in 0 ..< 2:
    for j in 0 ..< 4:
      for k in 0 ..< 4:
        let idx = toInt(crd2idx(a.getLayout(), (i, (j, k))))
        doAssert a.data[idx] == float32(idx),
          "-.= broadcast at " & $i & "," & $j & "," & $k
  a *.= b
  for i in 0 ..< 2:
    for j in 0 ..< 4:
      for k in 0 ..< 4:
        let idx = toInt(crd2idx(a.getLayout(), (i, (j, k))))
        doAssert a.data[idx] == float32(idx) * (100.0'f32 + float32(j)),
          "*.= broadcast at " & $i & "," & $j & "," & $k
  a /.= b
  for i in 0 ..< 2:
    for j in 0 ..< 4:
      for k in 0 ..< 4:
        let idx = toInt(crd2idx(a.getLayout(), (i, (j, k))))
        doAssert a.data[idx] == float32(idx),
          "/.= broadcast at " & $i & "," & $j & "," & $k

proc sameShapeCase =
  ## Same-shape behavior stays undotted: add, sub, mul and the in-place forms
  var a: tile(float32, (4, 4))
  var b: tile(float32, (4, 4))
  fillSeq(a, 1.0'f32)
  fillSeq(b, 2.0'f32)
  let c = a + b
  for i in 0 ..< 16:
    doAssert c.data[i] == 3.0'f32 + 2.0'f32 * float32(i), "same-shape + at slot " & $i
  let d = a - b
  for i in 0 ..< 16:
    doAssert d.data[i] == -1.0'f32, "same-shape - at slot " & $i
  a += b
  for i in 0 ..< 16:
    doAssert a.data[i] == 3.0'f32 + 2.0'f32 * float32(i), "same-shape += at slot " & $i

proc main =
  scalarCase()
  divCase()
  maxminCase()
  broadcastValueCase()
  broadcastInPlaceCase()
  sameShapeCase()
  echo "t_tile_ops_binary: all cases passed"

when isMainModule:
  main()
