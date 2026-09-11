# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

# Defunction next-gen
# ======================================================================
#
# The difference is that it uses `next` as generator.
# This allows interleaving generators for example to produce "zip" construct.
# The main issue is that it's more ceremony to write and require mutable inputs.

# #######################################################################
#
#                            Concepts
#
# #######################################################################

# Defunctional
# ----------------------------------------------------------------------

type Defunctional* = concept self
  for e in self.items():
    discard e

template elemType*(D: typedesc[Defunctional]): typedesc =
  typeof(block:
    var container = default(D)
    container.items(), typeOfIter)

# Predicate
# ----------------------------------------------------------------------

type Predicate*[S] = concept p
  var s = default(S)
  for e in s.items():
    p.matches(e) is bool

# Morphism
# ----------------------------------------------------------------------

type Morphism*[S] = concept f
  var s = default(S)
  for e in s.items():
    discard f.apply(e)

# #######################################################################
#
#                          Generators
#
# #######################################################################

# iota
# ----------------------------------------------------------------------

type Iota* = object
  start*: int
  stop*: int
  stride*: int
  cur*: int

func iota*(start: int; stopEx: int; stride = 1): Iota {.inline.} =
  doAssert stride > 0, "iota only counts forward -- reversal is a stage's job"
  Iota(start: start, stop: stopEx, stride: stride, cur: start)

func done*(m: Iota): bool {.inline.} =
  m.cur >= m.stop

proc next*(m: var Iota): int {.inline.} =
  doAssert m.cur < m.stop, "iota exhausted"
  result = m.cur
  inc m.cur, m.stride

iterator items*(m: var Iota): int =
  while not m.done:
    yield m.next()

# split: words / lines
# ----------------------------------------------------------------------

type Split*[delim: static char] = object
  buf*: openArray[char]
  i*: int

func split*(s: string; delim: static char): Split[delim] {.inline.} =
  Split[delim](buf: s, i: 0)

func words*(s: string): Split[' '] {.inline.} =
  Split[' '](buf: s, i: 0)

func lines*(s: string): Split['\n'] {.inline.} =
  Split['\n'](buf: s, i: 0)

func done*(p: Split): bool {.inline.} =
  p.i >= p.buf.len

proc next*[delim: static char](p: var Split[delim]): openArray[char] {.inline.} =
  var j = p.i
  while j < p.buf.len and p.buf[j] != delim:
    inc j
  result = p.buf.toOpenArray(p.i, j - 1)
  p.i = j + 1

iterator items*[delim: static char](p: var Split[delim]): openArray[char] =
  while not p.done:
    yield p.next()

# #######################################################################
#
#                       Materialization Sinks
#
# #######################################################################

template collect*[T](src: iterable[T]): auto =
  ## Consume all items from the pipeline
  ##   when elements are char            -> string
  ##   when elements are T               -> seq[T]
  ##   when elements are openArray[char] -> seq[string]
  when T is openArray[char]:
    var res: seq[string]
    for v in src:
      var s = newString(v.len)
      if v.len > 0:
        copyMem(addr s[0], unsafeAddr v[0], v.len)
      res.add(s)
    res
  elif T is char:
    var res = ""
    for c in src:
      res.add(c)
    res
  else:
    var res: seq[T]
    for x in src:
      res.add(x)
    res

# #######################################################################
#
#                            Collections
#
# #######################################################################

# map
# ----------------------------------------------------------------------

type Map*[Src; F] = object
  src*: Src
  f*: F

func map*[Src; F](s: Src; f: F): Map[Src, F] {.inline.} =
  static:
    doAssert F is Morphism[Src],
      $F & " is not a Morphism for machine " & $Src
  Map[Src, F](src: s, f: f)

func done*(p: Map): bool {.inline.} =
  p.src.done

proc next*(p: var Map): auto {.inline.} =
  mixin apply
  p.f.apply(p.src.next())

iterator items*(p: var Map): auto =
  while not p.done:
    yield p.next()

# gt
# ----------------------------------------------------------------------

type Gt* = object
  than*: int

func gt*(than: int): Gt {.inline.} =
  Gt(than: than)

func matches*(p: Gt; x: int): bool {.inline.} =
  x > p.than

# filter
# ----------------------------------------------------------------------

type Filter*[Src: Defunctional; Pred] = object
  src*: Src
  pred*: Pred
  nextResult*: elemType(Src)
  hasNextResult*: bool

func filter*[Src; Pred](s: Src; pred: Pred): Filter[Src, Pred] {.inline.} =
  static:
    doAssert Pred is Predicate[Src],
      $Pred & " is not a Predicate for machine " & $Src
  var r: Filter[Src, Pred]
  r.src = s
  r.pred = pred
  r

func done*(p: var Filter): bool =
  if p.hasNextResult:
    return false
  while not p.src.done:
    let x = p.src.next()
    if p.pred.matches(x):
      p.nextResult = x
      p.hasNextResult = true
      return false
  true

proc next*(p: var Filter): auto {.inline.} =
  doAssert not p.done
  result = p.nextResult
  p.hasNextResult = false

iterator items*(p: var Filter): auto =
  while not p.done:
    yield p.next()

# chunks
# ----------------------------------------------------------------------

type Chunks*[Src; N: static int] = object
  when Src is openArray:
    buf*: Src
    pos*: int
  else:
    src*: Src
    ring*: array[N, char]
    fill*: int

func chunks*[Src](s: Src; N: static int): Chunks[Src, N] {.inline.} =
  var r: Chunks[Src, N]
  when Src is openArray:
    r.buf = s
  else:
    r.src = s
  r

func done*[Src; N: static int](p: Chunks[Src, N]): bool {.inline.} =
  when Src is openArray:
    p.pos >= p.buf.len
  else:
    p.src.done and p.fill == 0

proc next*[Src; N: static int](p: var Chunks[Src, N]): openArray[char] {.inline.} =
  when Src is openArray:
    doAssert p.pos < p.buf.len, "chunks exhausted"
    let stop =
      if p.pos + N <= p.buf.len:
        p.pos + N
      else:
        p.buf.len
    result = p.buf.toOpenArray(p.pos, stop - 1)
    p.pos = stop
  else:
    doAssert p.fill > 0 or not p.src.done, "chunks exhausted"
    while p.fill < N and not p.src.done:
      p.ring[p.fill] = p.src.next()
      inc p.fill
    result = p.ring.toOpenArray(0, p.fill - 1)
    p.fill = 0

iterator items*[Src; N: static int](p: var Chunks[Src, N]): openArray[char] =
  while not p.done:
    yield p.next()
