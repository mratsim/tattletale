# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SmallSeq, a sequence whose first `N` elements live inside the object.
##
## - Generic parameters match `array[N, T]`, capacity first, reading `SmallSeq[5, int32]`.
## - `heapCap` counts tail-block slots only, `0` while every element is inline.
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
## | `heapCap`  | `int32`                 | 4      | 4     |
## | `arr`      | `array[N, T]`           | 8      | 20    |
## | `overflow` | `ptr UncheckedArray[T]` | 32     | 8     |
##
## - `overflow` is a raw pointer, never a `seq`, which would keep its own length
##   beside the pointer and duplicate `len` for 8 bytes per value.
## - `len` and `heapCap` are `int32`. With `int` lengths the same object measures 48
##   bytes at `N = 5` and 40 at `N = 3`, the wider pair pushing `arr` forward.
## - An 8-byte-aligned element pads the pair and pushes `arr` forward, giving
##   sizes 64 at `N = 3` and 96 at `N = 5`.

import std/typetraits

type
  SmallSeq*[N: static int; T] = object
    ## Inline-buffer sequence over `T`, the first `N` elements inside the object.
    ##
    ## The zero value is a valid empty sequence, `init` returns it.
    ##
    ## Invariant:
    ##   `heapCap` counts heap slots only, `overflow` is nil exactly while `heapCap` is 0.
    ##   Element `i` sits at `arr[i]` below `N`, at `overflow[i - N]` past it.
    len: int32
    heapCap: int32
    arr: array[N, T]
    overflow: ptr UncheckedArray[T]

const SmallSeqMaxCapacity = high(int32)
  # Buffer growth stops below the `int32` range, `heapCap` and `len` are that wide.

proc allocTailBlock[N: static int, T](elems: int): ptr UncheckedArray[T] =
  ## Fresh tail block of `elems` element slots, zeroed for non-trivial `T` (zero is Nim's moved-from representation).
  when supportsCopyMem(T):
    result = cast[ptr UncheckedArray[T]](alloc(elems * sizeof(T)))
  else:
    result = cast[ptr UncheckedArray[T]](alloc0(elems * sizeof(T)))

proc reallocTailBlock[T](p: ptr UncheckedArray[T], oldElems, newElems: int): ptr UncheckedArray[T] =
  ## Resizes the tail block `p` between element counts, a `nil` block
  ## becoming a fresh zeroed allocation.
  result = cast[ptr UncheckedArray[T]](
    realloc0(p, oldElems * sizeof(T), newElems * sizeof(T)))

proc `=destroy`*[N, T](s: var SmallSeq[N, T]) =
  when supportsCopyMem(T):
    if s.overflow != nil:
      dealloc(s.overflow)
  else:
    for i in 0 ..< int(s.len):
      `=destroy`(s[i])
    if s.overflow != nil:
      dealloc(s.overflow)

proc `=sink`*[N, T](dst: var SmallSeq[N, T], src: SmallSeq[N, T]) =
  ## Takes over the source elements, no deep copy.
  ##
  ## - the source's length, heap capacity and tail pointer are read into locals
  ##   before the first `dst` write
  ## - the move lowering is free to zero the source at any point
  let srcLen = src.len
  let srcHeapCap = src.heapCap
  let srcOverflow = src.overflow
  when supportsCopyMem(T):
    if dst.overflow != nil:
      dealloc(dst.overflow)
    dst.len = srcLen
    dst.heapCap = srcHeapCap
    copyMem(addr dst.arr, unsafeAddr src.arr, sizeof(src.arr))
    dst.overflow = srcOverflow
  else:
    # A slot the source does not refill reads back as `default(T)`, a slot
    # it refills is overwritten by the move assignment itself.
    for i in 0 ..< N:
      if i < srcLen:
        dst.arr[i] = move unsafeAddr(src.arr[i])[]
      elif i < int(dst.len):
        dst.arr[i] = default(T)
    if dst.overflow != nil:
      for i in N ..< int(dst.len):
        dst.overflow[i - N] = default(T)
      dealloc(dst.overflow)
    dst.len = srcLen
    dst.heapCap = srcHeapCap
    dst.overflow = nil
    if srcOverflow != nil:
      dst.overflow = allocTailBlock[N, T](int(srcHeapCap))
      for i in N ..< srcLen:
        dst.overflow[i - N] = move unsafeAddr(srcOverflow[i - N])[]

proc `=copy`*[N, T](dst: var SmallSeq[N, T], src {.noalias.}: SmallSeq[N, T]) =
  ## Deep-copies `src`, reusing `dst`'s tail block when it covers `src`'s heap.
  if addr(dst) == unsafeAddr(src):
    # Self-assignment would destroy the old elements and then read them
    # back from the destroyed slots, so the guard keeps a self-copy a no-op.
    return
  when supportsCopyMem(T):
    copyMem(addr dst.arr, unsafeAddr src.arr, sizeof(src.arr))
    if dst.overflow != nil and int(dst.heapCap) >= int(src.heapCap):
      if src.len > N:
        copyMem(dst.overflow, src.overflow, (int(src.len) - N) * sizeof(T))
      dst.len = src.len
    else:
      if src.overflow != nil:
        if dst.overflow == nil:
          dst.overflow = allocTailBlock[N, T](int(src.heapCap))
        else:
          dst.overflow = reallocTailBlock(dst.overflow, int(dst.heapCap),
              int(src.heapCap))
        if src.len > N:
          copyMem(dst.overflow, src.overflow, (int(src.len) - N) * sizeof(T))
      elif dst.overflow != nil:
        dealloc(dst.overflow)
        dst.overflow = nil
      dst.len = src.len
      dst.heapCap = src.heapCap
  else:
    for i in 0 ..< N:
      if i < int(src.len):
        dst.arr[i] = src.arr[i]
      elif i < int(dst.len):
        dst.arr[i] = default(T)
    if src.overflow != nil:
      if dst.overflow != nil and int(dst.heapCap) >= int(src.heapCap):
        # A refilled slot's assignment destroys its prior content, a slot
        # the source does not refill falls back to `default(T)`.
        for i in N ..< int(dst.len):
          if i >= int(src.len):
            dst.overflow[i - N] = default(T)
        for i in N ..< int(src.len):
          dst.overflow[i - N] = src.overflow[i - N]
        dst.len = src.len
      else:
        if dst.overflow != nil:
          for i in N ..< int(dst.len):
            dst.overflow[i - N] = default(T)
          dealloc(dst.overflow)
        dst.overflow = allocTailBlock[N, T](int(src.heapCap))
        for i in N ..< int(src.len):
          dst.overflow[i - N] = src.overflow[i - N]
        dst.len = src.len
        dst.heapCap = src.heapCap
    else:
      if dst.overflow != nil:
        for i in N ..< int(dst.len):
          dst.overflow[i - N] = default(T)
        dealloc(dst.overflow)
      dst.overflow = nil
      dst.len = src.len
      dst.heapCap = src.heapCap

func init*[N: static int; T](_: type SmallSeq[N, T]): SmallSeq[N, T] =
  ## Empty sequence, no allocation.
  SmallSeq[N, T](len: 0, heapCap: 0, overflow: nil)

proc initFrom*[N: static int; T](_: type SmallSeq[N, T], vals: openArray[T]): SmallSeq[N, T] =
  ## Sequence seeded with `vals`.
  ##
  ## Returns:
  ## - a sequence holding `vals`, inline while `vals.len <= N`
  ## - a tail block presized to `vals.len` otherwise, one allocation, no growth doubling
  result = SmallSeq[N, T](len: 0, heapCap: 0, overflow: nil)
  for i in 0 ..< min(vals.len, N):
    result.arr[i] = vals[i]
  if vals.len > N:
    result.overflow = allocTailBlock[N, T](vals.len - N)
    for i in N ..< vals.len:
      result.overflow[i - N] = vals[i]
    result.heapCap = int32(vals.len - N)
  result.len = int32(vals.len)

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

proc grow[N, T](s: var SmallSeq[N, T], newTotal: int) =
  # `newTotal` counts inline and heap slots, only the tail is ever realloced.
  if newTotal > SmallSeqMaxCapacity:
    raise newException(ArithmeticDefect, "SmallSeq capacity would overflow int32")
  when supportsCopyMem(T):
    s.overflow = reallocTailBlock(s.overflow, int(s.heapCap), newTotal - N)
  else:
    let p = allocTailBlock[N, T](newTotal - N)
    if s.overflow != nil:
      for i in N ..< int(s.len):
        p[i - N] = move s.overflow[i - N]
      dealloc(s.overflow)
    s.overflow = p
  s.heapCap = int32(newTotal - N)

proc add*[N, T](s: var SmallSeq[N, T], value: sink T) =
  ## Appends `value`, doubling the capacity when full.
  if s.len < N:
    s.arr[s.len] = value
  else:
    if s.len == N + int(s.heapCap):
      s.grow(max(2 * (N + int(s.heapCap)), N + 1))
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
  ## Element-wise comparison against `other`.
  if int(s.len) != other.len:
    return false
  for i in 0 ..< other.len:
    if s[i] != other[i]:
      return false
  return true

func `$`*[N, T](s: SmallSeq[N, T]): string =
  ## Elements as `[a, b, c]`.
  result = "["
  for i, v in s.pairs:
    if i > 0:
      result.add ", "
    result.addQuoted(v)
  result.add "]"

func clear*[N, T](s: var SmallSeq[N, T]) =
  ## Resets `len` to 0, destroying the elements, and keeps the heap block.
  when not supportsCopyMem(T):
    for i in 0 ..< int(s.len):
      reset(s[i])
  s.len = 0
