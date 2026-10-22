# Test that forEach emits correct nested loops for same-shape tiles.
# Run: nim c -r --hints:off --warnings:off -o:build/wip --nimcache:nimcache/wip workspace/ceramic/tests/tile_algebra/t_foreach.nim

import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra

proc testLayoutLeftIndexFill =
  # rank-2 LayoutLeft, index binds the flat data offset
  var t: tile(float32, (4, 4))
  forEach i, var d in t:
    d = float32(i)
  for k in 0 ..< 16:
    doAssert t.data[k] == float32(k), "fill mismatch at " & $k & ": " & $t.data[k]

proc test3OperandAndInfix =
  var a: tile(float32, (4, 4))
  var b: tile(float32, (4, 4))
  forEach i, var d in a:
    d = float32(i)
  forEach var d in b:
    d = 2.0'f32

  var c: tile(float32, (4, 4))
  add(c, a, b)
  doAssert c.data[5] == 7.0'f32

  let d = a + b
  doAssert d.data[9] == 11.0'f32

  a += b
  doAssert a.data[5] == 7.0'f32

  let e = b - a
  doAssert e.data[9] == -9.0'f32

proc testLayoutRightStrideSort =
  # strides (4, 1): smallest-stride mode (1) is innermost
  var t: tile(float32, make_layout((4, 4), (4, 1)))
  forEach i, var d in t:
    d = float32(i)
  # the fill value is the flat offset, data[k] == k under any
  # loop order. The stride-sorted order with mode 1 innermost is
  # visible in the macro's echoed code
  for k in 0 ..< 16:
    doAssert t.data[k] == float32(k)

proc testSize1ModesSkipped =
  var t: tile(float32, (1, 8))
  forEach i, var d in t:
    d = float32(i)
  for k in 0 ..< 8:
    doAssert t.data[k] == float32(k)

proc testRank3 =
  # LayoutLeft strides (1, 2, 6)
  var t: tile(float32, (2, 3, 4))
  forEach i, var d in t:
    d = float32(i)
  for k in 0 ..< 24:
    doAssert t.data[k] == float32(k)

proc testGenericProcContext =
  # forEach inside a generic proc, `in` desugars to contains
  func scale[T, Sh, St](a: var Tile[T, Sh, St], s: T) =
    forEach var d in a:
      d = d * s

  var t: tile(float32, (4, 4))
  forEach i, var d in t:
    d = float32(i)
  scale(t, 2.0'f32)
  doAssert t.data[5] == 10.0'f32

proc runTests =
  testLayoutLeftIndexFill()
  test3OperandAndInfix()
  testLayoutRightStrideSort()
  testSize1ModesSkipped()
  testRank3()
  testGenericProcContext()

when isMainModule:
  runTests()
  echo "OK: all forEach tests passed"
