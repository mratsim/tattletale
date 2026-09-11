# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

# Defunctional — Building blocks for zero-overhead pipelines
# ======================================================================
#
# Defunctional is a library of composable building blocks
# to build zero-overhead, compile-time fused data processing pipelines.
#
# The main use-case is from a LLM prompt, stream it into
# 1. a chat template renderer
# 2. a tokenizer
#
# all of that driven by pull, with bounded latency and zero-allocation in the hot path.
#
# It works by combining `defunctionalization`, i.e. compute is carried by objects instead of function,
# with Nim compiler's inline iterator fusion.
# This allows both composable objects and full inlining of pipelines.
#
# Furthermore for memory safety it uses Nim escape analysis / borrow checking
# with the {.experimental: "views".} that enable openArray as values.
#
# It is inspired by research in
# tattletale/papers/streams-pipelines-ranges-iterators-coroutines-defunctionalization
#
# ## Defunctionalization and continuations
#
#  - The Best Refactoring You've Never Heard Of
#    James Koppel, Compose Conference 2019
#    https://www.pathsensitive.com/2019/07/the-best-refactoring-youve-never-heard.html
#
# ## Streams and pipelines
#
#  Marc Gravell, 2018
#  - Pipe Dreams, Part 1: Pull, push and the pull-pull pipeline
#    https://blog.marcgravell.com/2018/07/pipe-dreams-part-1.html
#  - Pipe Dreams, Part 2: Push and the pull-push pipeline
#    https://blog.marcgravell.com/2018/07/pipe-dreams-part-2.html
#  - Pipe Dreams, Part 3: The pull-push pipeline and backpressure
#    https://blog.marcgravell.com/2018/07/pipe-dreams-part-3.html
#  - Pipe Dreams, Part 3.1: IO and the async pipeline
#    https://blog.marcgravell.com/2018/07/pipe-dreams-part-31.html
#
#  ### Ranges (D)
#
#  - Programming in D — Ranges
#    Ali Çehreli, 2017
#    https://ddili.org/ders/d.en/ranges.html
#
#  - Iterators Must Go!
#    Andrei Alexandrescu, BoostCon 2009
#
#  - On Iteration
#    Andrei Alexandrescu, 2009
#    https://www.informit.com/articles/article.aspx?p=1407357
#
#  ### Transducers
#
#  - Transducers are coming
#    Rich Hickey, 2014
#    https://cognitect.com/blog/2014/8/6/transducers-are-coming
#
#  - Understanding Transducers Through Python
#    Rob Smallshire, Sixty North, 2016
#
#  ### Coroutines
#
#  - C++ Coroutines
#    Gor Nishanov, CppCon 2015
#    https://github.com/CppCon/CppCon2015/blob/master/Presentations/C%2B%2B%20Coroutines/C%2B%2B%20Coroutines%20-%20Gor%20Nishanov%20-%20CppCon%202015.pdf
#
#  - NanoCoroutines
#    Gor Nishanov, CppCon 2018
#    https://github.com/GorNishanov/await/blob/master/2018_CppCon/NanoCoroutines%20-%20Gor%20Nishanov%20-%20CppCon%202018.pdf
#
#  - Exploiting Coroutines to Attack the Killer "Nanoseconds"
#    Christopher Jonathan, Umar Farooq Minhas, James Hunter, Justin Levandoski, Gor Nishanov, PVLDB 11, 2018
#    https://www.vldb.org/pvldb/vol11/p1702-jonathan.pdf


# #######################################################################
#
#                            Concepts
#
# #######################################################################

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
  for e in S.items():
    p.matches(e) is bool

# Morphism
# ----------------------------------------------------------------------

type Morphism*[S] = concept f
  for e in S.items():
    discard f.apply(e)

# #######################################################################
#
#                            Generators
#
# #######################################################################

# iota
# ----------------------------------------------------------------------

type Iota* = object
  cur: int
  stop: int
  stride: int

func iota*(start: int; stopEx: int; stride = 1): Iota {.inline.} =
  doAssert stride > 0, "iota only counts forward -- reversal is a stage's job"
  Iota(cur: start, stop: stopEx, stride: stride)

iterator items*(m: Iota): int =
  var cur = m.cur
  while cur < m.stop:
    yield cur
    inc cur, m.stride

# split: words / lines
# ----------------------------------------------------------------------

type Split*[delim: static char] = object
  buf: openArray[char]
  i: int

func split*(s: string; delim: static char): Split[delim] {.inline.} =
  Split[delim](buf: s, i: 0)

func words*(s: string): Split[' '] {.inline.} =
  Split[' '](buf: s, i: 0)

func lines*(s: string): Split['\n'] {.inline.} =
  Split['\n'](buf: s, i: 0)

iterator items*[delim: static char](p: Split[delim]): openArray[char] =
  var i = p.i
  while i < p.buf.len:
    var j = i
    while j < p.buf.len and p.buf[j] != delim:
      inc j
    yield p.buf.toOpenArray(i, j - 1)
    i = j + 1

# #######################################################################
#
#                       Materialization Sinks
#
# #######################################################################

template collect*(src: Defunctional): auto =
  ## Consume all items from the pipeline
  ##   when elements are char            -> string
  ##   when elements are T               -> seq[T]
  ##   when elements are openArray[char] -> seq[string]
  when elemType(type(src)) is openArray[char]:
    var res: seq[string]
    for v in src.items:
      var s = newString(v.len)
      if v.len > 0:
        copyMem(addr s[0], unsafeAddr v[0], v.len)
      res.add(s)
    res
  elif elemType(type(src)) is char:
    var res = ""
    for c in src.items:
      res.add(c)
    res
  else:
    var res: seq[elemType(type(src))]
    for x in src.items:
      res.add(x)
    res

# #######################################################################
#
#                            Collections -- Maps
#
# #######################################################################

# map
# ----------------------------------------------------------------------
type Map*[Src: Defunctional; F] = object
  src: Src
  f: F

func map*[Src: Defunctional; F](s: Src; f: F): Map[Src, F] {.inline.} =
  static:
    doAssert F is Morphism[Src],
      $F & " is not a Morphism for machine " & $Src
  Map[Src, F](src: s, f: f)

iterator items*[Src: Defunctional; F](p: Map[Src, F]): auto =
  mixin apply
  for x in p.src:
    yield p.f.apply(x)

# #######################################################################
#
#                            Collections -- Filters
#
# #######################################################################

# filter
# ----------------------------------------------------------------------

type Filter*[Src: Defunctional; Pred] = object
  src: Src
  pred: Pred

func filter*[Src: Defunctional; Pred](s: Src; pred: Pred): Filter[Src, Pred] {.inline.} =
  static:
    doAssert Pred is Predicate[Src],
      $Pred & " is not a Predicate for machine " & $Src
  Filter[Src, Pred](src: s, pred: pred)

iterator items*[Src: Defunctional; Pred](p: Filter[Src, Pred]): auto =
  mixin matches
  for x in p.src:
    if p.pred.matches(x):
      yield x

# gt
# ----------------------------------------------------------------------
type Gt* = object
  than: int

func gt*(than: int): Gt {.inline.} =
  Gt(than: than)

func matches*(p: Gt; x: int): bool {.inline.} =
  x > p.than

# #######################################################################
#
#                            Collections -- Groups
#
# #######################################################################

# chunks
# ----------------------------------------------------------------------

type Chunks*[Src: Defunctional; N: static int] = object
  when Src is openArray:
    buf: Src
  else:
    src: Src
    ring: array[N, char]

func chunks*[Src: Defunctional](s: Src; N: static int): Chunks[Src, N] {.inline.} =
  when Src is openArray:
    Chunks[Src, N](buf: s)
  else:
    Chunks[Src, N](src: s)

iterator items*[Src: Defunctional; N: static int](p: Chunks[Src, N]): openArray[char] =
  when Src is openArray:
    var pos = 0
    while pos < p.buf.len:
      let stop = if pos + N <= p.buf.len:
        pos + N
      else:
        p.buf.len
      yield p.buf.toOpenArray(pos, stop - 1)
      pos = stop
  else:
    var ring: array[N, char]
    var n = 0
    for c in p.src:
      ring[n] = c
      inc n
      if n == N:
        yield ring.toOpenArray(0, N - 1)
        n = 0
    if n > 0:
      yield ring.toOpenArray(0, n - 1)

# #######################################################################
#
#                            Numeric
#
# #######################################################################

# sum
# ----------------------------------------------------------------------

template sum*(a: Defunctional): int =
  var total = 0
  for x in a.items:
    total += x
  total

# count
# ----------------------------------------------------------------------
template count*(a: Defunctional): int =
  var n = 0
  for _ in a.items:
    inc n
  n

# max
# ----------------------------------------------------------------------
template max*(a: Defunctional): int =
  var best: int
  var first = true
  for x in a.items:
    if first:
      best = x
      first = false
    elif x > best:
      best = x
  assert not first, "max over an empty stream"
  best

# min
# ----------------------------------------------------------------------
template min*(a: Defunctional): int =
  var best: int
  var first = true
  for x in a.items:
    if first:
      best = x
      first = false
    elif x < best:
      best = x
  assert not first, "min over an empty stream"
  best

# #######################################################################
#
#                            Text
#
# #######################################################################
