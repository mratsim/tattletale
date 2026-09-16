## Bounded pull machinery for the toktoktok test suites:
## - whole-input one-shot (`pullAll`) and bounded single step (`pullBatch`)
## - the Chunks ring-window machine shape (workspace/defunctional/defunctional.nim) applied to the id stream
## - every row exercises bounded machine consumption and the drain discipline

{.experimental: "views".}

import workspace/toktoktok/src/pipeline

const PullCap* = 4096
## Default window capacity, the bounded-pull batch size of the test suites.

type
  PullChunks*[Src; N: static int] = object
    ## Bounded pull windows over a machine's id stream, the ring-window
    ## machine shape of Chunks (workspace/defunctional/defunctional.nim)
    ## applied to the pipeline's id stream.
    ##
    ## Contract:
    ## - object + ctor + ONE items iterator, each window is a ring view of at most N ids
    ## - the tail window carries the remainder when the stream length is not a multiple of N
    ## - full drain == whole-input one-shot, post-drain empty, partial consumption resumes from the source machine's fields
    src: Src
    ring: array[N, int]

func pullChunks*[Src](s: Src; N: static int): PullChunks[Src, N] {.inline.} =
  ## Builds the windowed pull machine over the source machine `s`.
  ##
  ## Usage:
  ##
  ##   var pc = pipeline.pullChunks(PullCap)
  ##   for window in pc:
  ##     result.add window  # each window is an openArray[int] view
  PullChunks[Src, N](src: s)

iterator items*[Src; N: static int](p: var PullChunks[Src, N]): openArray[int] =
  ## Yields the id stream as bounded windows:
  ## - the ring fills from the source machine, a view yields every N ids
  ## - the tail window closes the stream
  mixin items
  var n = 0
  for id in p.src:
    p.ring[n] = id
    inc n
    if n == N:
      yield p.ring.toOpenArray(0, N - 1)
      n = 0
  if n > 0:
    yield p.ring.toOpenArray(0, n - 1)

proc pullAll*(pipeline: TokPipeline, text: string): seq[int] =
  ## Whole-input encode as a bounded windowed drain, the one-shot
  ## convenience lives in the tests, never in the binding.
  ##
  ## Contract:
  ## - every row exercises bounded machine consumption and the drain discipline
  ## - full drain == whole-input one-shot, post-drain empty
  ## - the flat seq is what the fixture comparison needs, the windows are views
  ##   (no per-window copying beyond that accumulation)
  ##
  ## Run nim test_toktoktok from the worktree root
  pipeline.resetText(text)
  var pc = pipeline.pullChunks(PullCap)
  for window in pc:
    result.add window

proc pullBatch*(p: TokPipeline, cap: int): seq[int] =
  ## One bounded consumption step, at most `cap` ids taken per call,
  ## repeat until drained() reports true.
  ##
  ## Contract:
  ## - the machine keeps its position state (in-flight decision carry) for the next pull
  ## - the cap is a runtime value (the Python binding surface), the step pulls cap windows off a window-of-1 machine
  doAssert cap > 0, "pull capacity must be positive"
  var pc = p.pullChunks(1)
  var taken = 0
  for window in pc:
    result.add window
    inc taken
    if taken == cap:
      break
