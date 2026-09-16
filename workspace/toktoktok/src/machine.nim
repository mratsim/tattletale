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
## | rule                | contract                                                                                                      |
## | ------------------- | ------------------------------------------------------------------------------------------------------------- |
## | consumption surface | one `items(var)` iterator per machine, no collect proc, no pull overload                                      |
## | zero-alloc hot path | views (offsets or openArray) into caller/machine-owned buffers, scratch grows once and is reused              |
## | chain depth         | at most 3 machines per expression or type chain (Nim issue #9422, deep iterator chains explode compile times) |
## | iteration state     | `items` takes the machine by `var`, in-place field mutation, a stream is never copied while open              |
## | position discipline | position advances before each yield, a partial stream resumes at the first not-yet-received element           |
## | streaming feeds     | `feed`/`finish` are state updates, not consumption surfaces, queued offsets stay valid until the next feed    |

type
  SpecialDecision* {.final.} = object
    ## One decision of the special-scan machine, shared with the pipeline
    ## machine. Byte offsets into the scan machine's window.
    ## - specialId < 0 marks ordinary text, an empty stretch is lo == hi.
    ## - specialId >= 0 marks a special token, the dictionary id.
    ## - lo (inclusive) .. hi (exclusive) index the window, whose offset base is exposed alongside.
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
