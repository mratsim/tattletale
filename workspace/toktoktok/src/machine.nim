# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Machine protocol for toktoktok:
##   composable pull machines.
##
## Every pipeline stage is a machine in the defunctional shape
## (workspace/defunctional on master, PR #97, is the reference library, this branch applies its shape directly):
##
##    type XxxMachine* = object      # value object, inline state ...
##
##   proc xxxMachine*(...): XxxMachine   # ctor
##   iterator items*(m: var XxxMachine): T  # ONE iteration surface
##
## Protocol rules:
##
## - One consumption surface per machine:
##   `items`. No collect proc, no
## pull(dst, machine) overload, no eager buffering of whole outputs,
##   consumers iterate and take what they need, element by element.
## - Zero-alloc hot path:
##   machines hand out views (offsets or openArray) into buffers
## the caller or the machine already owns, per-element work is
##   pointer arithmetic. Scratch buffers live in the machine object,
##   grow once and are reused across `reset` s.
## - Chain depth, at most 3 composed machines in any expression
##   or type chain (Nim issue #9422, deep generic iterator chains explode compile times):
##   deeper pipelines flatten by hand, a driving machine keeps one
##   upstream element in flight instead of stacking adapters.
## - Iteration state:
##   `items` takes the machine by `var` and mutates its fields
##   in place (inline iterators keep state across yields), so
##   a machine must not be copied while a stream is open,
##   and a partially consumed stream resumes from the machine's fields
##   on re-iteration. A machine advances its position state BEFORE
##   each yield:
##     breaking out of a stream mid-iteration leaves every
##   received element consumed and the next re-iteration resumes
##   at the first element not yet received.
## - Streaming feeds:
##   machines that consume a stream (chunked mode) expose `feed`
##   /`finish` procs as state updates, not as extra consumption
##   surfaces. Queued elements stay valid only until the next feed.
##
## SpecialDecision is the element type shared by the special-scan
## machine and its consumers (the pipeline machine).
##
## An ordinary stretch of text or one special-token occurrence,
## as offsets into the scan machine's window.

type
  SpecialDecision* {.final.} = object
    ## One decision of the special-scan machine, as byte offsets
    ## lo (inclusive) .. hi (exclusive) into the scan machine's window
    ## (window offset base exposed alongside). specialId < 0 marks
    ##
    ## ordinary text, specialId >= 0 marks a special token, the id
    ## carried by the dictionary and the offsets covering the matched
    ## token bytes. An empty stretch is lo == hi.
    lo*: int
    hi*: int
    specialId*: int

  IdentityStream* {.final.} = object
    ## Reference byte-stage machine:
    ##   yields every input byte unchanged.
    ## One machine, no upstream, the shape every stage machine follows.
    input: string
    pos: int

proc identityStream*(input: sink string): IdentityStream {.inline.} =
  ## Ctor:
  ##   one identity stage over the whole input.
  IdentityStream(input: input, pos: 0)

iterator items*(m: var IdentityStream): char {.inline.} =
  ## Yields the input bytes in order, unchanged. A partially consumed
  ## stream resumes at the machine's position on re-iteration.
  while m.pos < m.input.len:
    let c = m.input[m.pos]
    inc m.pos
    yield c
