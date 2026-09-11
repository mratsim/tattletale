# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import defunctional

proc eq(a: openArray[char]; b: string): bool =
  if a.len != b.len:
    return false
  for i, c in a:
    if c != b[i]:
      return false
  true

proc linesDemo() =
  let doc = "alpha beta\ngamma delta\n\nomega\n"
  doAssert count(lines(doc)) == 4
  var seen = 0
  for l in split(doc, '\n'):
    case seen
    of 0:
      doAssert eq(l, "alpha beta")
    of 1:
      doAssert eq(l, "gamma delta")
    of 2:
      doAssert l.len == 0
    of 3:
      doAssert eq(l, "omega")
    else:
      discard
    inc seen
  doAssert seen == 4

proc wordsDemo() =
  doAssert count(words("a  b c ")) == 4
  var seen = 0
  for w in words("lorem ipsum dolor"):
    case seen
    of 0:
      doAssert eq(w, "lorem")
    of 1:
      doAssert eq(w, "ipsum")
    of 2:
      doAssert eq(w, "dolor")
    else:
      discard
    inc seen
  doAssert seen == 3

proc chunksDemo() =
  const t = "abcdefghi"                # 9 chars, chunk 4 -> 2 full + tail "i"
  doAssert collect(chunks(t, 4)) == @["abcd", "efgh", "i"]
  let view = t.toOpenArray(0, t.high)
  doAssert collect(chunks(view, 4)) == @["abcd", "efgh", "i"]

type LongerThan = object
  n: int
func longerThan(n: int): LongerThan {.inline.} =
  LongerThan(n: n)
func matches*(p: LongerThan; v: openArray[char]): bool {.inline.} =
  v.len > p.n

proc fullCompositionDemo() =
  const t = "abcdefghi"
  let full = chunks(t, 4).filter(longerThan(3))
  doAssert collect(full) == @["abcd", "efgh"]

proc zeroAllocReceipt() =
  let w = chunks("lorem ipsum dolor sit amet", 4)
  let before = getOccupiedMem()
  var n = 0
  for _ in 1 .. 1000:
    n = count(w)
  doAssert n == 7
  doAssert getOccupiedMem() == before, "net memory grew across 1000 drives"

proc main =
  linesDemo()
  wordsDemo()
  chunksDemo()
  fullCompositionDemo()
  zeroAllocReceipt()
  echo "OK: t_string"

main()
