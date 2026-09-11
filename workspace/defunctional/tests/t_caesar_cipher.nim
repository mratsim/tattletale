# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import defunctional

type Rot13Into*[Src: Defunctional] = object
  src: Src
  dst: openArray[char]

func rot13Into*[Src: Defunctional](s: Src; dst: openArray[char]): Rot13Into[Src] {.inline.} =
  Rot13Into[Src](src: s, dst: dst)

iterator items*[Src: Defunctional](p: Rot13Into[Src]): int =
  var p = p
  var i = 0
  for c in p.src:
    if i >= p.dst.len:
      break
    if c in 'a'..'z':
      p.dst[i] = chr((c.ord - 'a'.ord + 13) mod 26 + 'a'.ord)
    elif c in 'A'..'Z':
      p.dst[i] = chr((c.ord - 'A'.ord + 13) mod 26 + 'A'.ord)
    else:
      p.dst[i] = c
    inc i
  yield i

proc rot13SinkDemo() =
  const s = "Hello, World"
  var buf = newString(s.len)
  let r = rot13Into(s, buf)
  var written = -1
  for n in r:
    written = n
  doAssert written == s.len
  doAssert buf == "Uryyb, Jbeyq", buf

proc selfInverseDemo() =
  const s = "The quick brown fox"
  var enc = newString(s.len)
  var dec = newString(s.len)
  for n in rot13Into(s, enc):
    discard n
  for n in rot13Into(enc, dec):
    discard n
  doAssert dec == s, dec

proc bufferBoundDemo() =
  const s = "abcdefghij"                 # 10 chars
  var buf = newString(4)
  var written = -1
  for n in rot13Into(s, buf):
    written = n
  doAssert written == 4
  doAssert buf == "nopq", buf

type Letters = object
  buf: openArray[char]
func letters(s: string): Letters {.inline.} =
  Letters(buf: s)
iterator items*(m: Letters): char =
  for i in 0 ..< m.buf.len:
    yield m.buf[i]

proc machineUpstreamDemo() =
  const doc = "hello world"
  var buf = newString(doc.len)
  var written = -1
  for n in rot13Into(letters(doc), buf):
    written = n
  doAssert written == doc.len
  doAssert buf == "uryyb jbeyq", buf

proc main =
  rot13SinkDemo()
  selfInverseDemo()
  bufferBoundDemo()
  machineUpstreamDemo()
  echo "OK: t_caesar_cipher"

main()
