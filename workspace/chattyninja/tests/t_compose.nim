# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Bounded-pull composability proof for the chattyninja engine.
##
## Renders feed a bounded-pull chunk consumer, the ring-window machine shape
## in `workspace/toktoktok/tests/pull_chunks.nim`, adapted to byte windows.
## Suites covered, one span-heavy and one value-heavy:
## - moonlight, 2 emits against 527 B of template
## - qwen3, 26 emits over 4168 B
##
## Checked here:
## - window sizes 7 and 256 deliver byte parity with the recorded bytes and the one-shot `pullAll` render of the same row
## - partial consumption stops mid-piece and resumption from the same driver loses and re-hands no byte
## - scratch through `attachScratch` keeps every windowed render byte-exact
##
## Run:
##   $ nim test_chattyninja

{.experimental: "views".}

import cnj_types, cnj_values, cnj_parse, cnj_engine
import rows

type PullChunks[N: static int] = object
  ## Bounded pull windows over one chattyninja render, the ring-window machine shape
  ## in `workspace/toktoktok/tests/pull_chunks.nim`, adapted to byte windows.
  ##
  ## Contract:
  ## - the tail window carries the remainder when the render length is not a multiple of N
  ## - a full drain equals the whole-render `pullAll`, the next pull after it reports 0
  ## - partial consumption resumes from the driver's fields, no byte re-handed
  ##
  ## Machine and tables stay at the consumer's scope, `Machine.jinja` borrowing the template
  ## text behind the artifact. A `Machine` embedded in another object loses
  ## the borrowed view, so the iterator parameters carry them.
  d: Driver
  buf: array[N, char]

func pullChunks[N: static int](d: Driver): PullChunks[N] =
  ## Builds the windowed pull machine over the driver `d` of a compiled template.
  PullChunks[N](d: d)

iterator items[N: static int](p: var PullChunks[N], m: Machine, t: Tables): openArray[char] =
  ## Yields the render as bounded windows:
  ## - one `pull` call fills the window, the view carries at most N bytes
  ## - the tail window closes the render, a 0 count ends the stream
  while true:
    let n = pull(m, t, p.d, p.buf)
    if n == 0:
      break
    yield p.buf.toOpenArray(0, n - 1)

proc renderChunked[N: static int](m: Machine, t: Tables, ctx: Value, clock: float64): string =
  ## Renders one row through `N`-byte windows, accumulating every window.
  var pc = pullChunks[N](newDriver(ctx, clock))
  for w in pc.items(m, t):
    for c in w:
      result.add c

func pieceRemaining(p: Piece): int =
  ## Bytes of a pending piece not yet delivered, lazy pieces carrying no counted length.
  case p.kind
  of pkNone: 0
  of pkSpan: int(p.hi - p.lo) - p.pos
  of pkStr: p.s.len - p.pos
  of pkScratch: p.shi.int - p.pos
  of pkLazy: 0

template checkWindow(size: static int) =
  ## Checks one row's render through a `size`-byte window against the recorded bytes
  ## and the one-shot render. A static parameter, the window size selects the machine.
  let chunked = renderChunked[size](m, tables, r.context, r.clock)
  doAssert chunked == r.rendered,
      suite & "/" & r.row & ": the " & $size & "-byte window render differs " &
      "from the recorded bytes"
  doAssert chunked == whole,
      suite & "/" & r.row & ": the " & $size & "-byte window render differs " &
      "from the one-shot render"

# Windowed consumption matches the recorded bytes at window sizes 7 and 256,
# and a full drain matches the one-shot render.
# ---------------------------------------------------------------------------
block windowedParity:
  for suite in ["moonlight", "qwen3"]:
    let src = templateSource(suite)
    let (nodes, tables) = parseTemplate(src)
    let m = Machine(jinja: src, nodes: nodes)
    for r in rows(suite):
      let whole = block:
        var d = newDriver(r.context, r.clock)
        pullAll(m, tables, d)
      doAssert whole == r.rendered,
          suite & "/" & r.row & ": the one-shot render differs from the recorded bytes"
      checkWindow(7)
      checkWindow(256)

# Partial consumption stops mid-piece, and resumption from the same driver
# completes the render without losing or re-handing a byte.
# ---------------------------------------------------------------------------
block partialConsumptionResumes:
  for suite in ["moonlight", "qwen3"]:
    let src = templateSource(suite)
    let (nodes, tables) = parseTemplate(src)
    let m = Machine(jinja: src, nodes: nodes)
    let r = rows(suite)[0]
    var pc = pullChunks[7](newDriver(r.context, r.clock))
    var head = ""
    var stoppedMidPiece = false
    for w in pc.items(m, tables):
      for c in w:
        head.add c
      if pc.d.pend.kind != pkNone and pieceRemaining(pc.d.pend) > 0:
        stoppedMidPiece = true
        break
    doAssert stoppedMidPiece,
        suite & ": no window ended inside a pending piece, the mid-piece stop never happened"
    doAssert head.len > 0 and head.len < r.rendered.len,
        suite & ": the early stop covered the whole render"
    doAssert head == r.rendered[0 ..< head.len],
        suite & ": the bytes before the stop diverged from the recording"
    doAssert pc.d.cur == head.len,
        suite & ": cur is " & $pc.d.cur & " but " & $head.len & " bytes were received"
    var tail = ""
    for w in pc.items(m, tables):
      for c in w:
        tail.add c
    doAssert head & tail == r.rendered,
        suite & ": the resumed bytes overlapped or diverged from the recording"

# Scratch attached through `attachScratch` keeps every windowed render byte-exact.
# Corpus rows emit string values, which render as string pieces, so a derived container
# emit forces the scratch-piece drain: a scratch smaller than the repr drains across pulls.
# ---------------------------------------------------------------------------
block windowedScratch:
  for suite in ["moonlight", "qwen3"]:
    let src = templateSource(suite)
    let (nodes, tables) = parseTemplate(src)
    let m = Machine(jinja: src, nodes: nodes)
    for r in rows(suite):
      # `scr` outlives the driver, the scratch pointer must stay valid through the render.
      var scr = newSeq[char](4096)
      var pc = pullChunks[1](newDriver(r.context, r.clock))
      attachScratch(pc.d, scr)
      var got = ""
      for w in pc.items(m, tables):
        for c in w:
          got.add c
      doAssert got == r.rendered,
          suite & "/" & r.row & ": the 1-byte window render with scratch differs " &
          "from the recorded bytes"

  # A derived container emit through the windowed machine. The repr drains as a lazy piece,
  # and the 7-byte window makes it drain across several pulls mid-piece.
  var inner = DictVal()
  dictSet(inner, "alpha", strVal("one"))
  dictSet(inner, "beta", seqVal(@[strVal("x"), strVal("y"), strVal("z")]))
  var cd = DictVal()
  dictSet(cd, "m", dictVal(inner))
  let ctx = dictVal(cd)
  let src = "{{ m }}"
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  let want = renderToString(src, ctx, 0.0)

  var scr = newSeq[char](4096)
  var pc = pullChunks[7](newDriver(ctx, 0.0))
  attachScratch(pc.d, scr)
  var got = ""
  var lazyPulls = 0
  for w in pc.items(m, tables):
    if pc.d.pend.kind == pkLazy:
      inc lazyPulls
    for c in w:
      got.add c
  doAssert got == want, "the windowed lazy-piece drain differs from the string render"
  doAssert lazyPulls >= 3,
      "the lazy piece drained in fewer than three pulls, the mid-piece drain is unobserved"

echo "t_compose: moonlight and qwen3 through window sizes 7, 256 and 1 with scratch, all byte-exact"
