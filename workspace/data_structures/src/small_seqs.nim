# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SmallSeq, a sequence whose first `N` elements live inside the object.
##
## - Generic parameters follow `array[N, T]`, capacity first, reading `SmallSeq[5, int32]`.
## - `N` is the inline capacity, not a maximum. Elements past `N` live in one
##   heap block of their own, and the first `N` appends allocate no heap block.
##
## | indices     | condition | element lives at  |
## | ----------- | --------- | ----------------- |
## | `0 ..< N`   | `i < N`   | `arr[i]`          |
## | `N ..< len` | `i >= N`  | `overflow[i - N]` |
##
## `[]` and `[]=` raise `IndexDefect` under `boundChecks`, on by default.
## `-d:danger` or `--checks:off` removes the checks, and an out-of-range index
## may then return garbage or crash.
##
## Layout of `SmallSeq[5, int32]` on arm64 with Nim 2.2.10, sizes 40 and 32
## at `N = 5` and `N = 3`, one cache line for `int32`.
##
## | field      | type                    | offset | bytes |
## | ---------- | ----------------------- | ------ | ----- |
## | `len`      | `int32`                 | 0      | 4     |
## | `cap`      | `int32`                 | 4      | 4     |
## | `arr`      | `array[N, T]`           | 8      | 20    |
## | `overflow` | `ptr UncheckedArray[T]` | 32     | 8     |
##
## - `overflow` is a raw pointer, never a `seq`, which would keep its own length
##   beside the pointer and duplicate `len` for 8 bytes per value.
## - `len` and `cap` are `int32`. With `int` lengths the same object measures 48
##   bytes at `N = 5` and 40 at `N = 3`, the wider pair pushing `arr` forward.
## - An 8-byte-aligned element pads the pair and pushes `arr` forward, giving
##   sizes 64 at `N = 3` and 96 at `N = 5`.

import std/typetraits

type
  SmallSeq*[N: static int; T] = object
    ## Inline-buffer sequence over `T`, the first `N` elements inside the object.
    ##
    ## The zero value is a valid empty sequence, `init` differs only by setting `cap` to `N`.
    ##
    ## Invariant:
    ##   `cap` is 0 or at least `N`, `overflow` is nil exactly while `cap <= N`,
    ##   element `i` sits at `arr[i]` below `N` and `overflow[i - N]` past it.
    ##
    ## A trivial `T` is required. `supportsCopyMem` is false for managed memory and for a type
    ## defining `=destroy` or `=copy`, so `SmallSeq` cannot hold `SmallSeq`, `string`, `seq`
    ## or `ref`, while Nim containers stay free to hold values of such a type.
    len: int32
    cap: int32
    arr: array[N, T]
    overflow: ptr UncheckedArray[T]

const SmallSeqMaxCapacity = high(int32)
  # Buffer growth stops below the `int32` range, `cap` and `len` are that wide.


proc `=destroy`*[N, T](s: var SmallSeq[N, T]) {.raises: [].} =
  when not supportsCopyMem(T):
    {.error: "T must be a trivial type".}
  if s.overflow != nil:
    dealloc(s.overflow)

proc `=sink`*[N, T](dst: var SmallSeq[N, T], src: SmallSeq[N, T]) {.raises: [].} =
  ## Takes over the source tail pointer, no deep copy.
  when not supportsCopyMem(T):
    {.error: "T must be a trivial type".}
  if dst.overflow != nil:
    dealloc(dst.overflow)
  dst.len = src.len
  dst.cap = src.cap
  copyMem(addr dst.arr, unsafeAddr src.arr, sizeof(src.arr))
  dst.overflow = src.overflow

proc `=copy`*[N, T](dst: var SmallSeq[N, T], src {.noalias.}: SmallSeq[N, T]) =
  ## Deep-copies into a fresh tail.
  when not supportsCopyMem(T):
    {.error: "T must be a trivial type".}
  `=destroy`(dst)
  dst.len = src.len
  dst.cap = src.cap
  copyMem(addr dst.arr, unsafeAddr src.arr, sizeof(src.arr))
  dst.overflow = nil
  if src.overflow != nil:
    # The block covers all cap - N slots, the copy covers the live len - N.
    dst.overflow = cast[ptr UncheckedArray[T]](alloc((int(src.cap) - N) * sizeof(T)))
    if src.len > N:
      copyMem(dst.overflow, src.overflow, (int(src.len) - N) * sizeof(T))

func init*[N: static int; T](_: type SmallSeq[N, T]): SmallSeq[N, T] =
  ## Empty sequence, inline buffer only, no allocation.
  SmallSeq[N, T](len: 0, cap: int32(N), overflow: nil)

func len*[N, T](s: SmallSeq[N, T]): int32 {.inline.} =
  ## Returns the number of elements held, inline or spilled.
  s.len

func `[]`*[N, T](s: SmallSeq[N, T], i: SomeInteger): T {.inline.} =
  when compileOption("boundChecks"):
    if unlikely(i < 0 or i >= int(s.len)):
      raise newException(IndexDefect, "index out of range")
  if i < N:
    result = s.arr[i]
  else:
    result = s.overflow[i - N]

func `[]`*[N, T](s: var SmallSeq[N, T], i: SomeInteger): var T {.inline.} =
  when compileOption("boundChecks"):
    if unlikely(i < 0 or i >= int(s.len)):
      raise newException(IndexDefect, "index out of range")
  if i < N:
    result = s.arr[i]
  else:
    result = s.overflow[i - N]

func `[]=`*[N, T](s: var SmallSeq[N, T], i: SomeInteger, v: sink T) {.inline.} =
  when compileOption("boundChecks"):
    if unlikely(i < 0 or i >= int(s.len)):
      raise newException(IndexDefect, "index out of range")
  if i < N:
    s.arr[i] = v
  else:
    s.overflow[i - N] = v

proc grow[N, T](s: var SmallSeq[N, T], newCap: int) =
  # newCap counts total slots, only the tail is ever realloced.
  if newCap > SmallSeqMaxCapacity:
    raise newException(ArithmeticDefect, "SmallSeq capacity would overflow int32")
  s.overflow = cast[ptr UncheckedArray[T]](
    realloc0(s.overflow, (int(s.cap) - N) * sizeof(T), (newCap - N) * sizeof(T)))
  s.cap = int32(newCap)

proc add*[N, T](s: var SmallSeq[N, T], value: sink T) =
  ## Appends `value`, doubling the capacity when the sequence is full.
  if s.len < N:
    s.arr[s.len] = value
  else:
    if s.cap < N:
      # The zero value has no capacity bookkeeping, the inline buffer is still whole.
      s.cap = int32(N)
    if s.len == s.cap:
      s.grow(max(2 * int(s.cap), N + 1))
    s.overflow[int(s.len) - N] = value
  inc s.len

iterator items*[N, T](s: SmallSeq[N, T]): T =
  ## Yields the elements in index order.
  for i in 0 ..< int(s.len):
    yield s[i]

iterator mitems*[N, T](s: var SmallSeq[N, T]): var T =
  ## Yields mutable elements in index order.
  for i in 0 ..< int(s.len):
    yield s[i]

iterator pairs*[N, T](s: SmallSeq[N, T]): (int, T) =
  ## Yields (index, element) pairs in index order.
  for i in 0 ..< int(s.len):
    yield (i, s[i])

iterator mpairs*[N, T](s: var SmallSeq[N, T]): (int, var T) =
  ## Yields (index, mutable element) pairs in index order.
  for i in 0 ..< int(s.len):
    yield (i, s[i])

func `==`*[N, T](s: SmallSeq[N, T], other: openArray[T]): bool =
  if int(s.len) != other.len:
    return false
  for i in 0 ..< other.len:
    if s[i] != other[i]:
      return false
  return true

func clear*[N, T](s: var SmallSeq[N, T]) =
  ## Resets `len` to 0 and keeps the heap block for reuse.
  s.len = 0
