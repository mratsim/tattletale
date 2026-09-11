# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}


import defunctional_next_gen

proc eq(a: openArray[char]; b: string): bool =
  if a.len != b.len:
    return false
  for i, c in a:
    if c != b[i]:
      return false
  true

type Chars* = object
  buf*: openArray[char]
  j*: int

func chars*(s: string): Chars {.inline.} =
  Chars(buf: s, j: 0)

func done*(p: Chars): bool {.inline.} =
  p.j >= p.buf.len

proc next*(p: var Chars): char {.inline.} =
  doAssert p.j < p.buf.len, "chars exhausted"
  result = p.buf[p.j]
  inc p.j

iterator items*(p: var Chars): char =
  while not p.done:
    yield p.next()

type Twice* = object

func apply*(t: Twice; x: int): int {.inline.} =
  x * 2

proc nextPullsDemo() =
  var gen = iota(0, 10)
  let a = gen.next()
  let b = gen.next()
  doAssert a == 0 and b == 1, $a & " " & $b

proc mixedPullAndLoopDemo() =
  var gen = iota(0, 10)
  let a = gen.next()
  let b = gen.next()
  var seen: seq[int]
  for i in gen.items():
    seen.add(i)
  doAssert a == 0 and b == 1, $a & " " & $b
  doAssert seen == @[2, 3, 4, 5, 6, 7, 8, 9], $seen
  doAssert gen.done

proc heldStageResumesDemo() =
  var f = filter(iota(0, 10), gt(6))
  var part: seq[int]
  var k = 0
  for x in f.items():
    part.add(x)
    inc k
    if k == 1:
      break
  doAssert part == @[7], $part
  doAssert collect(f.items()) == @[8, 9]
  var g = filter(iota(0, 3), gt(100))
  doAssert g.done     # exact even when NOTHING survives the filter

proc splitCursorDemo() =
  var w = words("alpha beta gamma")
  doAssert w.next().eq("alpha")
  doAssert w.next().eq("beta")
  doAssert collect(w.items()) == @["gamma"]

proc chunksCursorDemo() =
  var c = chunks(chars("abcdefghi"), 4)
  doAssert c.next().eq("abcd")
  doAssert collect(c.items()) == @["efgh", "i"]
  const t = "abcdefghi"
  let view = t.toOpenArray(0, t.high)
  var v = chunks(view, 4)
  doAssert v.next().eq("abcd")
  doAssert collect(v.items()) == @["efgh", "i"]
  var r = chunks(chars("abcdefg"), 3)
  doAssert r.next().eq("abc")
  doAssert r.next().eq("def")
  doAssert r.next().eq("g")
  doAssert r.done

proc sinkOverHeldChainDemo() =
  var m = iota(1, 11).filter(gt(4))
  doAssert collect(m.items()) == @[5, 6, 7, 8, 9, 10]

proc swapDemo() =
  var a = iota(0, 3)
  var b = iota(10, 13)
  swap(a, b)
  doAssert a.next() == 10
  doAssert b.next() == 0

proc main =
  nextPullsDemo()
  mixedPullAndLoopDemo()
  heldStageResumesDemo()
  splitCursorDemo()
  chunksCursorDemo()
  sinkOverHeldChainDemo()
  swapDemo()
  echo "OK: defunctional_next_gen_test"

main()
